"""Оценка ВериДок на синтетическом корпусе: реальные прогоны run_check + метрики.

Читает манифест ``eval/datasets/ground_truth.json`` (его генерирует ``seed.py``),
прогоняет ``run_check`` по каждому проверяемому документу с временным каталогом,
содержащим ТОЛЬКО его источник, и сравнивает находки с внесёнными ошибками.

ПРАВИЛО СОПОСТАВЛЕНИЯ (детекция; то же задокументировано в README §9):
  Находка-«позитив» — это Finding с verdict ∈ {contradicted, internal_conflict}.
  Внесённая ошибка e (kind ∈ {contradicted, internal_conflict}) считается
  ПОЙМАННОЙ (TP), если найдётся ещё не использованная находка-позитив f, для
  которой выполнено хотя бы одно:
    (а) location.index ошибки совпадает с одним из «покрытых» индексов находки —
        её claim.location ЛИБО location любого её evidence (для internal_conflict
        это место второго, конфликтующего утверждения);
    (б) tampered_value ошибки (норм.) встречается в «стоге» находки — тексте
        утверждения, цитатах evidence или reasoning.
  Вердикт-метка НЕ обязана точно совпадать: и contradicted, и internal_conflict
  засчитываются как детекция (одна и та же ошибка может всплыть обоими способами).
  Сопоставление жадное 1:1: каждая находка матчится максимум к одной ошибке.

ДЕДУП ДВОЙНОГО ПОДСЧЁТА: оставшиеся (не сматченные) позитивы, чьи покрытые индексы
совпадают с локацией уже пойманной ошибки, — это ПОВТОРНАЯ детекция той же ошибки
(напр. значение помечено и как internal_conflict, и как contradicted), а не ложное
срабатывание. Они отбрасываются и НЕ идут в FP.

NOT_FOUND — отдельная диагностика (НЕ входит в recall/precision): для каждого
not_found-кейса проверяем, что на его локации сервис НЕ выдал ложный позитив.

ЧИСТЫЕ документы: внесённых ошибок нет, поэтому любой позитив на них — FP.

Метрики: recall = TP/(TP+FN); precision = TP/(TP+FP); F1; плюс разбивка по
форматам (P/R/F1) и по типам ошибок (recall — FP не типизированы), время
(среднее/медиана) и ускорение vs ручная проверка (--manual-min-per-doc).

Запуск:
    python eval/evaluate.py --manual-min-per-doc 25
    LLM_MODEL=<тег> python eval/evaluate.py --manual-min-per-doc 25 --model <тег>
"""

import argparse
import csv
import json
import logging
import os
import shutil
import statistics
import tempfile
import time

from config import settings
from veridoc.pipeline import run_check

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = os.path.join(_HERE, "datasets")
_GROUND_TRUTH = os.path.join(_DATASETS, "ground_truth.json")
_RESULTS_DIR = os.path.join(_HERE, "results")

_POSITIVE = {"contradicted", "internal_conflict"}


# --- утилиты сопоставления --------------------------------------------------

def _norm(text) -> str:
    if text is None:
        return ""
    return " ".join(str(text).split()).lower().replace("«", "").replace("»", "")


def _finding_indices(f) -> set:
    """Индексы локаций, которые находка «покрывает»: claim + все evidence."""
    idx = set()
    loc = getattr(f.claim, "location", None) or {}
    if loc.get("index") is not None:
        idx.add(loc["index"])
    for ev in getattr(f, "evidence", None) or []:
        evloc = getattr(ev, "location", None) or {}
        if evloc.get("index") is not None:
            idx.add(evloc["index"])
    return idx


def _finding_haystack(f) -> str:
    parts = [getattr(f.claim, "text", "") or "", getattr(f, "reasoning", "") or ""]
    for ev in getattr(f, "evidence", None) or []:
        parts.append(getattr(ev, "quote", "") or "")
    return _norm(" ".join(parts))


def _matches(gt_error, finding, finding_idx, finding_hay) -> str | None:
    """Вернуть причину матча ('index'|'value') или None."""
    gi = gt_error["location"]["index"]
    if gi in finding_idx:
        return "index"
    val = _norm(gt_error["tampered_value"])
    if val and val in finding_hay:
        return "value"
    return None


# --- прогон одного документа ------------------------------------------------

def _make_sources_dir(source_path: str) -> str:
    tmp = tempfile.mkdtemp(prefix="veridoc_src_")
    if source_path and os.path.isfile(source_path):
        shutil.copy2(source_path, os.path.join(tmp, os.path.basename(source_path)))
    else:
        logger.warning("Источник не найден: %r", source_path)
    return tmp


def _evaluate_doc(doc_meta, gt_errors):
    """Прогнать один документ и вернуть (rows, stats) для матчинга/диагностики.

    rows — записи для errors.csv (TP/FP/FN/not_found). stats — счётчики.
    """
    check_path = os.path.join(_DATASETS, doc_meta["doc"])
    source_path = os.path.join(_DATASETS, doc_meta["source"])
    fmt = doc_meta["format"]

    sources_dir = _make_sources_dir(source_path)
    try:
        report = run_check(check_path, sources_dir)
    finally:
        shutil.rmtree(sources_dir, ignore_errors=True)

    findings = report.findings
    positives = [f for f in findings if f.verdict in _POSITIVE]
    pos_idx = [_finding_indices(f) for f in positives]
    pos_hay = [_finding_haystack(f) for f in positives]
    used = [False] * len(positives)

    detect_errors = [e for e in gt_errors if e["kind"] in _POSITIVE]
    nf_errors = [e for e in gt_errors if e["kind"] == "not_found"]

    rows = []
    tp = fn = 0
    matched_indices = set()  # локации пойманных ошибок (для дедупа FP)

    # 1) Жадно матчим внесённые ошибки.
    for e in detect_errors:
        hit_i = hit_reason = None
        for i, f in enumerate(positives):
            if used[i]:
                continue
            reason = _matches(e, f, pos_idx[i], pos_hay[i])
            if reason:
                hit_i, hit_reason = i, reason
                break
        if hit_i is not None:
            used[hit_i] = True
            tp += 1
            matched_indices.add(e["location"]["index"])
            matched_indices |= pos_idx[hit_i]
            rows.append(_row(doc_meta, e, "TP", positives[hit_i], hit_reason))
        else:
            fn += 1
            rows.append(_row(doc_meta, e, "FN", None, None))

    # 2) Оставшиеся позитивы: дедуп (повтор детекции) либо FP.
    fp = 0
    for i, f in enumerate(positives):
        if used[i]:
            continue
        if pos_idx[i] & matched_indices:
            # повторная детекция уже пойманной ошибки — не считаем FP
            continue
        fp += 1
        rows.append(_row_fp(doc_meta, f))

    # 3) not_found-диагностика: на локации не должно быть ложного позитива.
    nf_pass = nf_total = 0
    for e in nf_errors:
        nf_total += 1
        gi = e["location"]["index"]
        false_alarm = any(gi in pos_idx[i] for i in range(len(positives)))
        # value-match тоже считаем ложным позитивом для not_found
        if not false_alarm:
            val = _norm(e["tampered_value"])
            false_alarm = any(val and val in pos_hay[i] for i in range(len(positives)))
        outcome = "NF_FALSE_ALARM" if false_alarm else "NF_OK"
        if not false_alarm:
            nf_pass += 1
        rows.append(_row(doc_meta, e, outcome, None, None))

    stats = {
        "doc": doc_meta["doc"], "format": fmt, "clean": doc_meta["clean"],
        "tp": tp, "fp": fp, "fn": fn,
        "positives": len(positives), "findings": len(findings),
        "nf_pass": nf_pass, "nf_total": nf_total,
        "elapsed_sec": round(report.elapsed_sec, 3),
    }
    return rows, stats


def _row(doc_meta, e, outcome, finding, reason):
    return {
        "doc": doc_meta["doc"], "format": doc_meta["format"],
        "error_type": e["error_type"], "subtype": e["subtype"],
        "gt_kind": e["kind"], "location_index": e["location"]["index"],
        "tampered_value": e["tampered_value"], "outcome": outcome,
        "finding_verdict": finding.verdict if finding else "",
        "match_reason": reason or "",
    }


def _row_fp(doc_meta, finding):
    loc = getattr(finding.claim, "location", None) or {}
    return {
        "doc": doc_meta["doc"], "format": doc_meta["format"],
        "error_type": "", "subtype": "", "gt_kind": "",
        "location_index": loc.get("index", ""),
        "tampered_value": (getattr(finding.claim, "text", "") or "")[:80],
        "outcome": "FP", "finding_verdict": finding.verdict, "match_reason": "",
    }


# --- агрегация и метрики ----------------------------------------------------

def _prf(tp, fp, fn):
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4)


def evaluate(gt_path=_GROUND_TRUTH, model=None, manual_min_per_doc=None,
             results_path=None):
    """Прогнать корпус, посчитать метрики, записать results.json/errors.csv/metrics.md."""
    started = time.perf_counter()
    with open(gt_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    documents = manifest["documents"]
    errors = manifest["errors"]
    model = model or settings.llm_model

    by_doc_errors = {}
    for e in errors:
        by_doc_errors.setdefault(e["doc"], []).append(e)

    all_rows = []
    doc_stats = []
    for dm in documents:
        logger.info("run_check: %s (%s)", dm["doc"], dm["format"])
        doc_errors = by_doc_errors.get(dm["doc"], [])
        try:
            rows, stats = _evaluate_doc(dm, doc_errors)
        except Exception as exc:  # noqa: BLE001 — устойчивость к транзиентным сбоям
            # Один упавший документ (нет Qdrant/таймаут LLM) не должен валить весь
            # прогон: его внесённые ошибки засчитываются как FN, прогон продолжается.
            logger.exception("run_check упал на %s: %s — документ помечен failed", dm["doc"], exc)
            rows = [_row(dm, e, "FN", None, None)
                    for e in doc_errors if e["kind"] in _POSITIVE]
            stats = {
                "doc": dm["doc"], "format": dm["format"], "clean": dm["clean"],
                "tp": 0, "fp": 0,
                "fn": sum(1 for e in doc_errors if e["kind"] in _POSITIVE),
                "positives": 0, "findings": 0,
                "nf_pass": 0,
                "nf_total": sum(1 for e in doc_errors if e["kind"] == "not_found"),
                "elapsed_sec": 0.0, "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        all_rows.extend(rows)
        doc_stats.append(stats)

    # --- агрегаты ---
    def agg(rows_filter):
        tp = sum(1 for r in all_rows if r["outcome"] == "TP" and rows_filter(r))
        fp = sum(1 for r in all_rows if r["outcome"] == "FP" and rows_filter(r))
        fn = sum(1 for r in all_rows if r["outcome"] == "FN" and rows_filter(r))
        return tp, fp, fn

    tp, fp, fn = agg(lambda r: True)
    precision, recall, f1 = _prf(tp, fp, fn)

    by_format = {}
    for fmt in sorted({d["format"] for d in documents}):
        t, p, n = agg(lambda r, fmt=fmt: r["format"] == fmt)
        pr, rc, f = _prf(t, p, n)
        by_format[fmt] = {"tp": t, "fp": p, "fn": n, "precision": pr,
                          "recall": rc, "f1": f}

    by_type = {}
    for et in ("numeric", "claim", "internal"):
        t = sum(1 for r in all_rows if r["outcome"] == "TP" and r["error_type"] == et)
        n = sum(1 for r in all_rows if r["outcome"] == "FN" and r["error_type"] == et)
        rc = round(t / (t + n), 4) if (t + n) else 0.0
        by_type[et] = {"tp": t, "fn": n, "recall": rc}

    nf_pass = sum(s["nf_pass"] for s in doc_stats)
    nf_total = sum(s["nf_total"] for s in doc_stats)
    times = [s["elapsed_sec"] for s in doc_stats]
    avg_t = round(statistics.mean(times), 3) if times else 0.0
    med_t = round(statistics.median(times), 3) if times else 0.0

    speedup = None
    manual_sec = None
    if manual_min_per_doc is not None and avg_t > 0:
        manual_sec = manual_min_per_doc * 60.0
        speedup = round(manual_sec / avg_t, 1)

    summary = {
        "model": model,
        "documents": len(documents),
        "clean_docs": sum(1 for d in documents if d["clean"]),
        "errors_total": tp + fn,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "by_format": by_format,
        "by_type": by_type,
        "not_found_diagnostic": {
            "passed": nf_pass, "total": nf_total,
            "pass_rate": round(nf_pass / nf_total, 4) if nf_total else None,
        },
        "time": {"avg_sec": avg_t, "median_sec": med_t},
        "manual_min_per_doc": manual_min_per_doc,
        "manual_sec_per_doc": manual_sec,
        "speedup_vs_manual": speedup,
        "seed": manifest.get("meta", {}).get("seed"),
        "eval_elapsed_sec": round(time.perf_counter() - started, 1),
    }
    report_obj = {"summary": summary, "documents": doc_stats}

    # --- запись артефактов ---
    # Имена errors.csv/metrics.md наследуют суффикс модели из results_path, чтобы
    # прогоны разных моделей НЕ затирали артефакты друг друга (results.json →
    # errors.csv/metrics.md; results_qwen3-coder.json → errors_qwen3-coder.csv/…).
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    if results_path is None:
        results_path = os.path.join(_HERE, "results.json")
    base = os.path.splitext(os.path.basename(results_path))[0]
    suffix = base[len("results"):] if base.startswith("results") else "_" + base
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(report_obj, fh, ensure_ascii=False, indent=2)

    errors_csv = os.path.join(_HERE, f"errors{suffix}.csv")
    with open(errors_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "doc", "format", "error_type", "subtype", "gt_kind",
            "location_index", "tampered_value", "outcome",
            "finding_verdict", "match_reason"])
        w.writeheader()
        w.writerows(all_rows)

    metrics_md = os.path.join(_RESULTS_DIR, f"metrics{suffix}.md")
    _write_metrics_md(summary, manifest, metrics_md)
    _print_summary(summary)
    logger.info("Артефакты: %s, %s, %s", results_path, errors_csv, metrics_md)
    return report_obj


def _write_metrics_md(s, manifest, path):
    """Финальная таблица для слайдов 07–08."""
    m = manifest.get("meta", {})
    lines = []
    lines.append("# ВериДок — метрики оценки (для слайдов 07–08)\n")
    lines.append(f"**Модель:** `{s['model']}` · **seed:** {s['seed']} · "
                 f"**temperature:** 0.0 (детерминизм)\n")
    lines.append(f"**Корпус:** {s['documents']} документов "
                 f"(из них чистых: {s['clean_docs']}), внесённых ошибок-детекций: "
                 f"{s['errors_total']}; форматы DOCX/PPTX/PDF.\n")
    lines.append("## Итоговые метрики\n")
    lines.append("| Метрика | Значение |")
    lines.append("|---|---|")
    lines.append(f"| Recall (обнаружено ошибок) | **{s['recall']:.3f}** ({round(s['recall']*100)}%) |")
    lines.append(f"| Precision (точность) | **{s['precision']:.3f}** ({round(s['precision']*100)}%) |")
    lines.append(f"| F1 | **{s['f1']:.3f}** |")
    lines.append(f"| Среднее время / документ | {s['time']['avg_sec']:.1f} с |")
    lines.append(f"| Медианное время / документ | {s['time']['median_sec']:.1f} с |")
    if s["speedup_vs_manual"]:
        lines.append(f"| Ручная проверка (оценка) | {s['manual_min_per_doc']} мин/док |")
        lines.append(f"| **Ускорение vs ручная** | **×{s['speedup_vs_manual']}** |")
    lines.append("")
    lines.append("## По форматам\n")
    lines.append("| Формат | Recall | Precision | F1 | TP/FP/FN |")
    lines.append("|---|---|---|---|---|")
    for fmt, v in s["by_format"].items():
        lines.append(f"| {fmt} | {v['recall']:.3f} | {v['precision']:.3f} | "
                     f"{v['f1']:.3f} | {v['tp']}/{v['fp']}/{v['fn']} |")
    lines.append("")
    lines.append("## По типам ошибок (recall)\n")
    lines.append("| Тип | Recall | TP/FN |")
    lines.append("|---|---|---|")
    for et, v in s["by_type"].items():
        lines.append(f"| {et} | {v['recall']:.3f} | {v['tp']}/{v['fn']} |")
    lines.append("\n> Precision по типам не приводится: ложные срабатывания (FP) не "
                 "привязаны к типу внесённой ошибки. P/R/F1 по форматам и общие — приводятся.\n")
    nf = s["not_found_diagnostic"]
    lines.append("## Диагностика not_found (отдельно от recall/precision)\n")
    if nf["total"]:
        lines.append(f"Корректно не выдал ложный contradicted: **{nf['passed']}/{nf['total']}** "
                     f"({round((nf['pass_rate'] or 0)*100)}%).\n")
    else:
        lines.append("not_found-кейсы в наборе отсутствуют.\n")
    lines.append("## Размер выборки и ограничения\n")
    lines.append(f"- N = {s['documents']} документов, {s['errors_total']} внесённых ошибок "
                 f"(+ {nf['total']} not_found-кейсов).")
    lines.append("- Набор **синтетический**: ошибки внесены программно по фиксированному "
                 f"seed={s['seed']}; повтор генерации даёт тот же корпус.")
    lines.append("- Все вызовы LLM при `temperature=0`. Числа получены реальными прогонами "
                 "`run_check` (без заглушек).")
    lines.append("- Локальный стек: LLM по OpenAI-совместимому эндпоинту (Ollama), эмбеддер "
                 "Qwen3-Embedding-8B, векторное хранилище Qdrant.")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _print_summary(s):
    print("\n" + "=" * 64)
    print(f"ВериДок — оценка (модель: {s['model']})")
    print("=" * 64)
    print(f"  документов:        {s['documents']} (чистых: {s['clean_docs']})")
    print(f"  TP/FP/FN:          {s['tp']}/{s['fp']}/{s['fn']}")
    print(f"  recall:            {s['recall']:.3f}")
    print(f"  precision:         {s['precision']:.3f}")
    print(f"  f1:                {s['f1']:.3f}")
    print(f"  время ср/медиана:  {s['time']['avg_sec']:.1f} / {s['time']['median_sec']:.1f} с")
    nf = s["not_found_diagnostic"]
    if nf["total"]:
        print(f"  not_found:         {nf['passed']}/{nf['total']} корректно")
    if s["speedup_vs_manual"]:
        print(f"  ускорение vs ручн: ×{s['speedup_vs_manual']} (ручная ~{s['manual_min_per_doc']} мин/док)")
    fmt_str = ", ".join("{}: R={:.2f}".format(k, v["recall"]) for k, v in s["by_format"].items())
    type_str = ", ".join("{}: {:.2f}".format(k, v["recall"]) for k, v in s["by_type"].items())
    print(f"  по форматам:       {fmt_str}")
    print(f"  по типам (recall): {type_str}")
    print("=" * 64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Оценка ВериДок на корпусе.")
    parser.add_argument("--manual-min-per-doc", type=float, default=None,
                        help="оценка времени ручной проверки одного документа (мин)")
    parser.add_argument("--model", default=None, help="метка модели для отчёта")
    parser.add_argument("--gt", default=_GROUND_TRUTH, help="путь к ground_truth.json")
    parser.add_argument("--results", default=None,
                        help="путь к results.json (по умолчанию eval/results.json)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    evaluate(gt_path=args.gt, model=args.model,
             manual_min_per_doc=args.manual_min_per_doc, results_path=args.results)


if __name__ == "__main__":
    main()
