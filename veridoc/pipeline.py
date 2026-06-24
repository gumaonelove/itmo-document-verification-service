"""Оркестратор полной проверки документа ВериДок.

Связывает все этапы конвейера в единую функцию run_check: парсинг → индексация
источников → извлечение утверждений → поиск + верификация → согласованность →
сборка отчёта. Если источники не предоставлены, все утверждения помечаются
not_found, но внутренние противоречия всё равно ищутся.

CLI (python -m veridoc.pipeline doc.pdf --sources dir/) печатает таблицу находок
и сводку по вердиктам — это критерий приёмки M1.
"""

import argparse
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from config import settings
from veridoc.consistency import check_consistency
from veridoc.embedder import Embedder
from veridoc.extract import extract_claims
from veridoc.models import Finding, Report
from veridoc.parsing import parse_document
from veridoc.store import QdrantStore
from veridoc.verify import verify_claim

logger = logging.getLogger(__name__)

# Расширения файлов, принимаемых как источники.
_SOURCE_EXTS = (".pdf", ".docx", ".pptx")

# Все возможные вердикты — для инициализации счётчиков в stats нулями.
_VERDICTS = ("supported", "contradicted", "not_found", "internal_conflict")


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Нарезать текст на чанки по символам с перекрытием.

    Шаг окна — (size - overlap); пустые/пробельные чанки отбрасываются. При
    некорректных параметрах (size <= 0 или overlap >= size) откатываемся на
    шаг в один символ, чтобы не зациклиться.
    """
    if not text:
        return []
    step = size - overlap
    if size <= 0 or step <= 0:
        step = max(size, 1)
    chunks: list[str] = []
    for start in range(0, len(text), step):
        piece = text[start : start + size].strip()
        if piece:
            chunks.append(piece)
    return chunks


def _list_source_files(sources_dir: str) -> list[str]:
    """Вернуть отсортированные пути к файлам-источникам поддерживаемых типов."""
    if not sources_dir or not os.path.isdir(sources_dir):
        return []
    files: list[str] = []
    for name in sorted(os.listdir(sources_dir)):
        path = os.path.join(sources_dir, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in _SOURCE_EXTS:
            files.append(path)
    return files


def _index_sources(sources_dir: str, embedder: Embedder, store: QdrantStore) -> int:
    """Распарсить, нарезать, закодировать и проиндексировать все источники.

    Возвращает число проиндексированных чанков. Для каждого источника текст —
    конкатенация его сегментов; source_id берётся из doc_id (sentence-уровень
    сегмента), chunk_id = f"{source_id}#c{k}".
    """
    t0 = time.perf_counter()
    files = _list_source_files(sources_dir)
    chunks_payload: list[dict] = []
    counter = 0  # сквозной счётчик для уникальных chunk_id между сегментами/файлами

    for file in files:
        src_segments = parse_document(file)
        if not src_segments:
            continue
        source_id = src_segments[0].doc_id
        # Чанкуем КАЖДЫЙ сегмент отдельно — так чанк сохраняет место (страницу/
        # слайд/абзац) сегмента-источника, и его можно показать в отчёте.
        pieces_with_loc: list[tuple[str, dict]] = []
        for seg in src_segments:
            for piece in _chunk_text(seg.text, settings.chunk_size, settings.chunk_overlap):
                pieces_with_loc.append((piece, seg.location))
        if not pieces_with_loc:
            continue
        vectors = embedder.embed_documents([p for p, _ in pieces_with_loc])
        for (piece, location), vector in zip(pieces_with_loc, vectors):
            chunks_payload.append(
                {
                    "chunk_id": f"{source_id}#c{counter}",
                    "source_id": source_id,
                    "text": piece,
                    "location": location,
                    "vector": vector,
                }
            )
            counter += 1

    store.index_sources(chunks_payload)
    logger.info(
        "Индексация: %d источников → %d чанков за %.2f с",
        len(files),
        len(chunks_payload),
        time.perf_counter() - t0,
    )
    return len(chunks_payload)


def run_check(doc_path: str, sources_dir: str | None) -> Report:
    """Прогнать документ через весь конвейер и вернуть отчёт.

    doc_path — проверяемый документ; sources_dir — каталог доверенных источников
    (или None). Без источников каждое утверждение получает not_found, но поиск
    внутренних противоречий выполняется в любом случае.
    """
    started = time.perf_counter()

    # 1) Парсинг проверяемого документа.
    t0 = time.perf_counter()
    segments = parse_document(doc_path)
    logger.info(
        "Этап parse: %d сегментов за %.2f с", len(segments), time.perf_counter() - t0
    )

    # 2) Индексация источников (если есть подходящие файлы).
    store: QdrantStore | None = None
    embedder: Embedder | None = None
    source_files = _list_source_files(sources_dir) if sources_dir else []
    if source_files:
        embedder = Embedder()
        store = QdrantStore()
        # recreate=True: каждая проверка индексирует ТОЛЬКО свои источники, чтобы
        # поиск не подмешивал чанки прошлых документов/прогонов.
        store.ensure_collection(recreate=True)
        _index_sources(sources_dir, embedder, store)
    else:
        logger.info("Этап index: источники не предоставлены — пропуск")

    # 3) Извлечение утверждений.
    t0 = time.perf_counter()
    claims = extract_claims(segments)
    logger.info(
        "Этап extract: %d утверждений за %.2f с", len(claims), time.perf_counter() - t0
    )

    # 4) Поиск + верификация по каждому утверждению.
    # Запросы независимы → выполняем ПАРАЛЛЕЛЬНО (embed_query + search + verify
    # на утверждение). ThreadPoolExecutor.map сохраняет порядок, результат
    # детерминирован. Без источников — мгновенный not_found без LLM/поиска.
    t0 = time.perf_counter()
    if store is not None and embedder is not None and claims:

        def _retrieve_and_verify(claim) -> Finding:
            qv = embedder.embed_query(claim.text)
            evs = store.search(qv, settings.top_k)
            return verify_claim(claim, evs)

        workers = max(1, min(settings.llm_concurrency, len(claims)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            findings = list(executor.map(_retrieve_and_verify, claims))
    else:
        findings = [
            Finding(
                claim=claim,
                verdict="not_found",
                reasoning="Источники не предоставлены",
            )
            for claim in claims
        ]
    logger.info(
        "Этап retrieve+verify: %d находок за %.2f с",
        len(findings),
        time.perf_counter() - t0,
    )

    # 5) Поиск внутренних противоречий.
    # ПРИМ.: намеренно НЕ дедуплицируем contradicted-находки, пересекающиеся с
    # internal_conflict. Противоречие источнику — первичный сигнал, и его сокрытие
    # ради «чистоты» precision однажды прячет реальную ошибку (проверено на eval:
    # дедуп терял настоящий contradicted на project). Лишняя пометка корректна.
    t0 = time.perf_counter()
    conflicts = check_consistency(claims)
    findings += conflicts
    logger.info(
        "Этап consistency: %d конфликтов за %.2f с",
        len(conflicts),
        time.perf_counter() - t0,
    )

    # 6) Сводная статистика: всего + по каждому вердикту.
    stats: dict = {"total": len(findings)}
    for verdict in _VERDICTS:
        stats[verdict] = 0
    for finding in findings:
        stats[finding.verdict] = stats.get(finding.verdict, 0) + 1

    # 7) Сборка отчёта.
    elapsed_sec = time.perf_counter() - started
    report = Report(
        doc_name=os.path.basename(doc_path),
        generated_at=datetime.now().isoformat(timespec="seconds"),
        findings=findings,
        stats=stats,
        elapsed_sec=elapsed_sec,
    )
    logger.info(
        "Проверка завершена: %d находок, %s, за %.2f с",
        stats["total"],
        ", ".join(f"{v}={stats[v]}" for v in _VERDICTS),
        elapsed_sec,
    )
    return report


def _location_str(location: dict) -> str:
    """Человекочитаемое местоположение находки: '<kind> <index>'."""
    kind = location.get("kind", "?")
    index = location.get("index", "?")
    return f"{kind} {index}"


def _print_report(report: Report) -> None:
    """Напечатать таблицу находок и сводку в консоль (CLI-вывод M1)."""
    headers = ("Утверждение", "Вердикт", "Где", "Обоснование")
    rows: list[tuple[str, str, str, str]] = []
    for finding in report.findings:
        claim = finding.claim
        where = _location_str(claim.location)
        # Для внутренних конфликтов поясняем, с каким утверждением спор.
        reasoning = finding.reasoning or ""
        if finding.conflict_with:
            suffix = f" (конфликт с {finding.conflict_with})"
            reasoning = (reasoning + suffix) if reasoning else suffix.strip()
        rows.append((claim.text, finding.verdict, where, reasoning))

    # Ширины колонок с разумным ограничением, чтобы таблица не разъезжалась.
    limits = (60, 18, 14, 60)
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

    print(f"\nДокумент: {report.doc_name}   ({report.generated_at})")
    print(_fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(_fmt(row))

    stats = report.stats
    print("\nСводка:")
    print(f"  всего находок:        {stats.get('total', 0)}")
    print(f"  supported:            {stats.get('supported', 0)}")
    print(f"  contradicted:         {stats.get('contradicted', 0)}")
    print(f"  not_found:            {stats.get('not_found', 0)}")
    print(f"  internal_conflict:    {stats.get('internal_conflict', 0)}")
    print(f"  время выполнения:     {report.elapsed_sec:.2f} с")


def main() -> None:
    """Точка входа CLI: python -m veridoc.pipeline doc.ext [--sources DIR]."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        prog="veridoc.pipeline",
        description="ВериДок: проверка достоверности документа по доверенным источникам.",
    )
    parser.add_argument("doc_path", help="путь к проверяемому документу (.pdf/.docx/.pptx)")
    parser.add_argument(
        "--sources",
        dest="sources",
        default=None,
        help="каталог с доверенными источниками (.pdf/.docx/.pptx)",
    )
    args = parser.parse_args()

    report = run_check(args.doc_path, args.sources)
    _print_report(report)


if __name__ == "__main__":
    main()
