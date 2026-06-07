"""Прогон конвейера ВериДок по эталонному набору и подсчёт метрик качества.

Этот модуль загружает eval/datasets/ground_truth.json (схему генерирует
eval/seed.py), прогоняет run_check по каждому уникальному проверяемому документу
и сравнивает находки с эталоном.

Ожидаемая схема ground_truth.json (список записей; имена ключей читаем мягко,
допуская синонимы — это устойчиво к мелким расхождениям с seed.py):
    [
      {
        "check_path":     "<путь к проверяемому документу .pdf/.docx/.pptx>",
        "source_path":    "<путь к доверенному источнику этого документа>",
        "kind":           "contradicted" | "internal_conflict",
        "location":       {"kind": "page"|"slide"|"paragraph", "index": int},
        "tampered_value": "<подменённое значение, ожидаемое в тексте находки>"
      },
      ...
    ]
Каждая запись описывает один внедрённый дефект, который пайплайн должен поймать.
Допустимые синонимы ключей: check_path ~ {doc_path, document, check, doc};
source_path ~ {source, src, source_file}; kind ~ {verdict, type, defect};
location ~ {loc} (а index — также из плоского поля page/slide/paragraph/index);
tampered_value ~ {tampered, value, wrong_value, expected_value}.

Положительная находка (positive) — Finding с verdict in
{"contradicted", "internal_conflict"}.

Правило сопоставления (ленивое, осознанно нестрогое — приоритет recall):
GT-запись считается ПОЙМАННОЙ, если найдётся ещё не использованная positive-
находка такого же kind (contradicted/internal_conflict) И выполняется хотя бы
одно из условий:
  (а) совпадает location.index находки и GT (та же страница/слайд/абзац);
  (б) tampered_value (без учёта регистра и пробелов) содержится в тексте claim
      или в одной из цитат evidence находки.
Жадно: GT-записи обрабатываются по порядку, каждая находка матчится максимум к
одной GT-записи, каждая GT-запись — максимум к одной находке.

Метрики на документ и суммарно:
  recall    = пойманные_GT / всего_GT
  precision = матченные_positive (= пойманные_GT) / всего_positive
  f1        = 2 * P * R / (P + R)
  avg_time  = средний report.elapsed_sec по успешно пройденным документам

Результаты печатаются таблицей и сохраняются в eval/results.json. Недоступность
моделей/Qdrant (исключение в run_check) ловится: документ помечается failed,
прогон продолжается.

Запуск:
    python -m eval.evaluate
"""

import json
import logging
import os
import shutil
import tempfile
import time
from collections import OrderedDict

from veridoc.pipeline import run_check

logger = logging.getLogger(__name__)

# Каталог этого модуля — пути к данным считаем относительно него.
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = os.path.join(_HERE, "datasets")
_GROUND_TRUTH = os.path.join(_DATASETS, "ground_truth.json")
_RESULTS = os.path.join(_HERE, "results.json")

# Вердикты-«позитивы»: находка такого вердикта считается срабатыванием детектора.
_POSITIVE_VERDICTS = {"contradicted", "internal_conflict"}

# Синонимы ключей GT-записи — читаем схему мягко.
_CHECK_KEYS = ("check_path", "doc_path", "document", "check", "doc")
_SOURCE_KEYS = ("source_path", "source", "src", "source_file")
_KIND_KEYS = ("kind", "verdict", "type", "defect")
_LOCATION_KEYS = ("location", "loc")
_INDEX_KEYS = ("index", "page", "slide", "paragraph")
_TAMPERED_KEYS = (
    "tampered_value",
    "tampered",
    "value",
    "wrong_value",
    "expected_value",
)


def _first(record: dict, keys: tuple[str, ...]):
    """Вернуть значение первого присутствующего ключа из keys (иначе None)."""
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _record_index(record: dict):
    """Достать location.index из записи: из вложенного location либо плоского поля."""
    location = _first(record, _LOCATION_KEYS)
    if isinstance(location, dict):
        idx = location.get("index")
        if idx is not None:
            return idx
    flat = _first(record, _INDEX_KEYS)
    return flat


def _norm(text) -> str:
    """Нормализовать строку для нестрогого сравнения: lower + схлопнутые пробелы."""
    if text is None:
        return ""
    return " ".join(str(text).split()).lower()


def load_ground_truth(path: str = _GROUND_TRUTH) -> list[dict]:
    """Загрузить ground_truth.json и нормализовать записи к единой форме.

    Каждая нормализованная запись:
        {"check_path", "source_path", "kind", "index", "tampered_value", "raw"}.
    Допускает как голый список записей, так и обёртку {"records": [...]}.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # Достаём список из типичных обёрток.
        data = (
            data.get("records")
            or data.get("items")
            or data.get("ground_truth")
            or data.get("data")
            or []
        )
    if not isinstance(data, list):
        raise ValueError(f"ground_truth.json: ожидался список записей, получено {type(data).__name__}")

    records: list[dict] = []
    for raw in data:
        if not isinstance(raw, dict):
            continue
        kind = _first(raw, _KIND_KEYS)
        records.append(
            {
                "check_path": _first(raw, _CHECK_KEYS),
                "source_path": _first(raw, _SOURCE_KEYS),
                "kind": kind if kind in _POSITIVE_VERDICTS else kind,
                "index": _record_index(raw),
                "tampered_value": _first(raw, _TAMPERED_KEYS),
                "raw": raw,
            }
        )
    logger.info("Загружено GT-записей: %d из %s", len(records), path)
    return records


def _resolve_path(path: str) -> str:
    """Преобразовать путь/имя файла GT к абсолютному.

    В ground_truth.json doc/source — голые имена файлов, лежащих в eval/datasets/.
    Поэтому относительные пути ищем сначала в datasets/, затем в eval/, затем в
    текущем каталоге. Абсолютные пути возвращаем как есть.
    """
    if not path:
        return path
    if os.path.isabs(path):
        return path
    for base in (_DATASETS, _HERE):
        candidate = os.path.join(base, path)
        if os.path.exists(candidate):
            return candidate
    return os.path.abspath(path)


def _finding_index(finding) -> object:
    """location.index находки (по claim.location)."""
    location = getattr(finding.claim, "location", None) or {}
    return location.get("index")


def _finding_haystack(finding) -> str:
    """Нормализованный текст для поиска tampered_value: claim.text + цитаты."""
    parts = [getattr(finding.claim, "text", "") or ""]
    for ev in getattr(finding, "evidence", None) or []:
        parts.append(getattr(ev, "quote", "") or "")
    reasoning = getattr(finding, "reasoning", "") or ""
    parts.append(reasoning)
    return _norm(" ".join(parts))


def _match_findings(gt_records: list[dict], positives: list) -> dict:
    """Сопоставить GT-записи документа с positive-находками (жадно, 1:1).

    Возвращает словарь с результатами матчинга: списком per-GT исходов, числами
    пойманных GT и матченных находок.
    """
    used = [False] * len(positives)
    per_gt: list[dict] = []
    caught = 0

    for gt in gt_records:
        gt_kind = gt["kind"]
        gt_index = gt["index"]
        gt_value = _norm(gt["tampered_value"])
        match_i = None
        match_reason = None

        for i, finding in enumerate(positives):
            if used[i]:
                continue
            if finding.verdict != gt_kind:
                continue
            # Условие (а): совпал индекс местоположения.
            by_index = (
                gt_index is not None
                and _finding_index(finding) is not None
                and str(_finding_index(finding)) == str(gt_index)
            )
            # Условие (б): подменённое значение встречается в тексте находки.
            by_value = bool(gt_value) and gt_value in _finding_haystack(finding)
            if by_index or by_value:
                match_i = i
                match_reason = "index" if by_index else "value"
                break

        if match_i is not None:
            used[match_i] = True
            caught += 1
        per_gt.append(
            {
                "kind": gt_kind,
                "index": gt_index,
                "tampered_value": gt["tampered_value"],
                "caught": match_i is not None,
                "match_reason": match_reason,
            }
        )

    matched_positives = sum(used)
    return {
        "per_gt": per_gt,
        "caught": caught,
        "matched_positives": matched_positives,
        "used": used,
    }


def _make_sources_dir(source_path: str) -> str:
    """Создать временный каталог с единственным источником (его копией).

    Возвращает путь к каталогу; вызывающий обязан удалить его сам.
    """
    tmp = tempfile.mkdtemp(prefix="veridoc_src_")
    if source_path and os.path.isfile(source_path):
        shutil.copy2(source_path, os.path.join(tmp, os.path.basename(source_path)))
    else:
        logger.warning("Источник не найден или не задан: %r", source_path)
    return tmp


def _prf1(caught: int, total_gt: int, total_positive: int) -> tuple[float, float, float]:
    """Посчитать (precision, recall, f1) с защитой от деления на ноль."""
    recall = caught / total_gt if total_gt else 0.0
    precision = caught / total_positive if total_positive else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def evaluate(gt_path: str = _GROUND_TRUTH, results_path: str = _RESULTS) -> dict:
    """Прогнать пайплайн по эталону и вернуть структурный отчёт с метриками.

    Группирует GT-записи по проверяемому документу; на каждый документ запускает
    run_check с временным каталогом источников (только его источник). Считает
    per-document и агрегированные метрики, печатает таблицу, сохраняет results.json.
    Исключения run_check (нет моделей/Qdrant) ловятся: документ помечается failed.
    """
    started = time.perf_counter()
    records = load_ground_truth(gt_path)

    # Группируем GT по проверяемому документу, сохраняя порядок появления.
    by_doc: "OrderedDict[str, dict]" = OrderedDict()
    for rec in records:
        check = rec["check_path"]
        if not check:
            logger.warning("GT-запись без check_path пропущена: %r", rec["raw"])
            continue
        group = by_doc.setdefault(
            check, {"check_path": check, "source_path": rec["source_path"], "records": []}
        )
        # Если источник в группе ещё не зафиксирован — берём из текущей записи.
        if not group["source_path"] and rec["source_path"]:
            group["source_path"] = rec["source_path"]
        group["records"].append(rec)

    doc_results: list[dict] = []
    total_gt = 0
    total_caught = 0
    total_positive = 0
    elapsed_list: list[float] = []
    failed_docs = 0

    for check, group in by_doc.items():
        gt_records = group["records"]
        check_path = _resolve_path(check)
        source_path = _resolve_path(group["source_path"]) if group["source_path"] else None
        doc_name = os.path.basename(check)

        entry: dict = {
            "doc_name": doc_name,
            "check_path": check_path,
            "source_path": source_path,
            "gt_total": len(gt_records),
        }

        if not os.path.isfile(check_path):
            logger.error("Документ не найден: %s — помечаю failed", check_path)
            entry.update(
                {
                    "status": "failed",
                    "error": f"check file not found: {check_path}",
                    "positives": 0,
                    "caught": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "elapsed_sec": None,
                    "per_gt": [
                        {
                            "kind": r["kind"],
                            "index": r["index"],
                            "tampered_value": r["tampered_value"],
                            "caught": False,
                            "match_reason": None,
                        }
                        for r in gt_records
                    ],
                }
            )
            doc_results.append(entry)
            total_gt += len(gt_records)
            failed_docs += 1
            continue

        sources_dir = _make_sources_dir(source_path)
        try:
            logger.info("run_check: %s (источник %s)", doc_name, os.path.basename(source_path) if source_path else "—")
            report = run_check(check_path, sources_dir)
        except Exception as exc:  # noqa: BLE001 — устойчивость к недоступным моделям/Qdrant
            logger.exception("run_check упал на %s: %s — помечаю failed", doc_name, exc)
            entry.update(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "positives": 0,
                    "caught": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "elapsed_sec": None,
                    "per_gt": [
                        {
                            "kind": r["kind"],
                            "index": r["index"],
                            "tampered_value": r["tampered_value"],
                            "caught": False,
                            "match_reason": None,
                        }
                        for r in gt_records
                    ],
                }
            )
            doc_results.append(entry)
            total_gt += len(gt_records)
            failed_docs += 1
            continue
        finally:
            shutil.rmtree(sources_dir, ignore_errors=True)

        positives = [f for f in report.findings if f.verdict in _POSITIVE_VERDICTS]
        match = _match_findings(gt_records, positives)

        doc_positive = len(positives)
        doc_caught = match["caught"]
        precision, recall, f1 = _prf1(doc_caught, len(gt_records), doc_positive)

        entry.update(
            {
                "status": "ok",
                "positives": doc_positive,
                "caught": doc_caught,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "elapsed_sec": round(report.elapsed_sec, 3),
                "per_gt": match["per_gt"],
            }
        )
        doc_results.append(entry)

        total_gt += len(gt_records)
        total_caught += doc_caught
        total_positive += doc_positive
        elapsed_list.append(report.elapsed_sec)

    # Агрегированные метрики (по успешно пройденным документам для precision/recall
    # суммируем уже посчитанные числа; failed-документы дают свои GT в знаменатель recall).
    agg_precision, agg_recall, agg_f1 = _prf1(total_caught, total_gt, total_positive)
    avg_time = sum(elapsed_list) / len(elapsed_list) if elapsed_list else 0.0

    summary = {
        "documents": len(by_doc),
        "documents_ok": len(by_doc) - failed_docs,
        "documents_failed": failed_docs,
        "gt_total": total_gt,
        "positives_total": total_positive,
        "caught_total": total_caught,
        "precision": round(agg_precision, 4),
        "recall": round(agg_recall, 4),
        "f1": round(agg_f1, 4),
        "avg_time_sec": round(avg_time, 3),
        "eval_elapsed_sec": round(time.perf_counter() - started, 3),
    }

    report_obj = {"summary": summary, "documents": doc_results}

    _print_table(doc_results, summary)
    _save_results(report_obj, results_path)

    logger.info(
        "Оценка завершена: docs=%d (failed=%d), recall=%.3f, precision=%.3f, f1=%.3f, avg_time=%.2f с",
        summary["documents"],
        summary["documents_failed"],
        summary["recall"],
        summary["precision"],
        summary["f1"],
        summary["avg_time_sec"],
    )
    return report_obj


def _print_table(doc_results: list[dict], summary: dict) -> None:
    """Напечатать таблицу по документам и итоговую сводку."""
    headers = ("Документ", "Статус", "GT", "Pos", "Поймано", "Recall", "Precision", "F1", "Время, с")
    rows: list[tuple[str, ...]] = []
    for d in doc_results:
        elapsed = "-" if d["elapsed_sec"] is None else f"{d['elapsed_sec']:.2f}"
        rows.append(
            (
                d["doc_name"],
                d["status"],
                str(d["gt_total"]),
                str(d["positives"]),
                str(d["caught"]),
                f"{d['recall']:.3f}",
                f"{d['precision']:.3f}",
                f"{d['f1']:.3f}",
                elapsed,
            )
        )

    limits = (40, 8, 4, 4, 8, 7, 9, 6, 9)
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = min(max(widths[i], len(cell)), limits[i])

    def _fmt(cells: tuple[str, ...]) -> str:
        out = []
        for i, cell in enumerate(cells):
            text = cell if len(cell) <= widths[i] else cell[: widths[i] - 1] + "…"
            out.append(text.ljust(widths[i]))
        return " | ".join(out)

    print("\nРезультаты оценки ВериДок")
    print(_fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(_fmt(row))

    print("\nИтог:")
    print(f"  документов:           {summary['documents']} (ok={summary['documents_ok']}, failed={summary['documents_failed']})")
    print(f"  всего GT:             {summary['gt_total']}")
    print(f"  всего positive:       {summary['positives_total']}")
    print(f"  поймано GT:           {summary['caught_total']}")
    print(f"  recall:               {summary['recall']:.3f}")
    print(f"  precision:            {summary['precision']:.3f}")
    print(f"  f1:                   {summary['f1']:.3f}")
    print(f"  среднее время/док:    {summary['avg_time_sec']:.2f} с")


def _save_results(report_obj: dict, path: str) -> None:
    """Сохранить отчёт об оценке в JSON (UTF-8, человекочитаемо)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report_obj, f, ensure_ascii=False, indent=2)
    logger.info("Результаты сохранены: %s", path)


def main() -> None:
    """Точка входа: настроить логирование и запустить оценку."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    evaluate()


if __name__ == "__main__":
    main()
