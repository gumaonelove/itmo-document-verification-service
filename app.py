"""Streamlit-интерфейс ВериДок.

Запуск: streamlit run app.py

Пользователь загружает проверяемый документ и доверенные источники (pdf/docx/pptx),
нажимает «Проверить» — файлы сохраняются во временную папку, вызывается
veridoc.pipeline.run_check(doc_path, sources_dir), результат показывается как
сводка по вердиктам и таблица находок с цветовой пометкой строк.
"""

import logging
import os
import tempfile

import pandas as pd
import streamlit as st

from config import settings
from veridoc.pipeline import run_check

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Допустимые типы загружаемых файлов (без точки — как ждёт file_uploader).
ALLOWED_TYPES = ["pdf", "docx", "pptx"]

# Подписи и порядок вердиктов для сводки.
VERDICT_LABELS = {
    "supported": "Подтверждено",
    "contradicted": "Опровергнуто",
    "not_found": "Не найдено",
    "internal_conflict": "Внутренний конфликт",
}

# Цвета строк таблицы по вердикту (мягкие фоновые тона).
VERDICT_COLORS = {
    "supported": "#d4edda",        # зелёный
    "contradicted": "#f8d7da",     # красный
    "internal_conflict": "#f8d7da",  # красный
    "not_found": "#fff3cd",        # жёлтый
}


def _save_uploaded(uploaded_file, dst_dir: str) -> str:
    """Сохранить один загруженный файл в каталог dst_dir, вернуть путь к нему."""
    path = os.path.join(dst_dir, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def _location_label(location) -> str:
    """Человекочитаемое «Где» из location {"kind", "index"}."""
    if not isinstance(location, dict):
        return "—"
    kind = location.get("kind")
    index = location.get("index")
    if kind == "page":
        return f"стр. {index}"
    if kind == "slide":
        return f"слайд {index}"
    if kind == "paragraph":
        return f"абзац {index}"
    return "—"


def _reasoning_cell(finding) -> str:
    """Текст обоснования: рассуждение + цитата (или ссылка на конфликтующий claim)."""
    parts = []
    if finding.reasoning:
        parts.append(finding.reasoning)
    # Опорные цитаты из источников.
    for ev in finding.evidence:
        if ev.quote:
            parts.append(f"«{ev.quote}» [{ev.source_id}]")
    # Для внутреннего конфликта — ссылка на противоречащее утверждение.
    if finding.verdict == "internal_conflict" and finding.conflict_with:
        parts.append(f"Конфликтует с: {finding.conflict_with}")
    return "\n".join(parts) if parts else "—"


def _findings_dataframe(findings) -> pd.DataFrame:
    """Собрать таблицу находок и скрытую колонку вердикта для раскраски."""
    rows = []
    for f in findings:
        rows.append(
            {
                "Утверждение": f.claim.text,
                "Вердикт": VERDICT_LABELS.get(f.verdict, f.verdict),
                "Где": _location_label(f.claim.location),
                "Обоснование (цитата)": _reasoning_cell(f),
                "_verdict": f.verdict,  # служебная колонка для Styler
            }
        )
    return pd.DataFrame(rows)


def _style_rows(row: pd.Series) -> list[str]:
    """Покрасить строку DataFrame по её вердикту (для Styler.apply, axis=1)."""
    color = VERDICT_COLORS.get(row["_verdict"], "")
    style = f"background-color: {color}" if color else ""
    return [style] * len(row)


def _render_summary(report) -> None:
    """Сводка сверху: метрики по вердиктам и затраченное время."""
    stats = report.stats or {}

    # Считаем по факту из находок — надёжнее, чем угадывать ключи stats.
    counts = {v: 0 for v in VERDICT_LABELS}
    for f in report.findings:
        counts[f.verdict] = counts.get(f.verdict, 0) + 1

    total = stats.get("total", len(report.findings))

    cols = st.columns(len(VERDICT_LABELS) + 2)
    cols[0].metric("Всего", total)
    for i, (verdict, label) in enumerate(VERDICT_LABELS.items(), start=1):
        # Приоритет — значение из stats, иначе посчитанное.
        value = stats.get(verdict, counts.get(verdict, 0))
        cols[i].metric(label, value)
    cols[-1].metric("Время, с", f"{report.elapsed_sec:.1f}")


def _render_legend() -> None:
    """Легенда цветов вердиктов."""
    st.markdown(
        """
        **Легенда:**
        <span style="background-color:#d4edda;padding:2px 8px;border-radius:4px;">Подтверждено</span>
        <span style="background-color:#f8d7da;padding:2px 8px;border-radius:4px;">Опровергнуто / Внутренний конфликт</span>
        <span style="background-color:#fff3cd;padding:2px 8px;border-radius:4px;">Не найдено</span>
        """,
        unsafe_allow_html=True,
    )


def _render_report(report) -> None:
    """Показать сводку, легенду и таблицу находок."""
    st.subheader("Сводка")
    _render_summary(report)

    st.subheader("Находки")
    _render_legend()

    if not report.findings:
        st.info("Проверяемых утверждений не найдено.")
        return

    df = _findings_dataframe(report.findings)
    styler = (
        df.style.apply(_style_rows, axis=1)
        .hide(axis="columns", subset=["_verdict"])  # прячем служебную колонку
    )
    st.dataframe(styler, use_container_width=True, hide_index=True)


def main() -> None:
    """Точка входа Streamlit-приложения."""
    st.set_page_config(page_title="ВериДок", layout="wide")
    st.title("ВериДок — проверка достоверности документов")
    st.caption(
        f"Локальная проверка по доверенным источникам. Модель: {settings.llm_model}."
    )

    doc_file = st.file_uploader(
        "Проверяемый документ",
        type=ALLOWED_TYPES,
        accept_multiple_files=False,
    )
    source_files = st.file_uploader(
        "Доверенные источники",
        type=ALLOWED_TYPES,
        accept_multiple_files=True,
    )

    if st.button("Проверить", type="primary"):
        if doc_file is None:
            st.warning("Загрузите проверяемый документ.")
            return
        if not source_files:
            st.warning(
                "Источники не загружены — будет выполнена только проверка "
                "внутренней согласованности документа."
            )

        # Временная папка: документ в корне, источники — в подпапке sources/.
        workdir = tempfile.mkdtemp(prefix="veridoc_")
        sources_dir = os.path.join(workdir, "sources")
        os.makedirs(sources_dir, exist_ok=True)

        doc_path = _save_uploaded(doc_file, workdir)
        for sf in source_files or []:
            _save_uploaded(sf, sources_dir)

        logger.info(
            "Запуск проверки: документ %s, источников %d",
            doc_path,
            len(source_files or []),
        )

        try:
            with st.spinner("Идёт проверка документа…"):
                report = run_check(doc_path, sources_dir)
        except Exception as exc:  # noqa: BLE001 — показать ошибку пользователю
            logger.exception("Ошибка при проверке документа")
            st.error(f"Не удалось выполнить проверку: {exc}")
            return

        st.success(f"Проверка завершена за {report.elapsed_sec:.1f} с.")
        _render_report(report)


if __name__ == "__main__":
    main()
