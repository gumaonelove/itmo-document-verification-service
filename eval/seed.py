"""Генерация датасета для оценки ВериДок.

Создаёт 3–5 ПАР документов: «источник» (правильные факты) и «проверяемый»
(с внесёнными ошибками), а также общий ground_truth.json — список ожидаемых
находок. Покрываются все четыре типа внесений:

  (1) изменение числа;
  (2) замена даты;
  (3) утверждение, противоречащее источнику;
  (4) внутреннее расхождение (одно число названо по-разному в двух местах
      одного проверяемого документа).

И все три формата: DOCX (python-docx), PPTX (python-pptx), PDF (pymupdf).

Генерация ДЕТЕРМИНИРОВАНА: тексты заданы литералами, порядок фиксирован,
случайности нет. Индексы location в ground_truth согласованы с парсером
veridoc.parsing (1-based; для DOCX счётчик абзацев сквозной и включает пустые
абзацы, поэтому индексы вычисляются по позиции в плоском списке абзацев).

Запуск:
    python -m eval.seed
    # или
    python eval/seed.py
"""

import json
import logging
import os
import time

logger = logging.getLogger(__name__)

# Каталог с результатами — рядом с этим файлом, в datasets/.
DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
GROUND_TRUTH_PATH = os.path.join(DATASETS_DIR, "ground_truth.json")


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------

def _write_docx(path: str, paragraphs: list[str]) -> None:
    """Записать DOCX: по одному абзацу на элемент списка (порядок сохранён).

    Парсер veridoc.parsing нумерует абзацы сквозным 1-based счётчиком, считая
    и пустые абзацы. Поэтому индекс location для абзаца с текстом paragraphs[i]
    равен i + 1 (учитывая, что в новом документе python-docx нет «нулевого»
    лишнего абзаца — каждый add_paragraph создаёт ровно один параграф)."""
    from docx import Document

    doc = Document()
    for text in paragraphs:
        doc.add_paragraph(text)
    doc.save(path)


def _docx_index(paragraphs: list[str], needle: str) -> int:
    """1-based индекс первого абзаца, содержащего подстроку needle.

    Совпадает с location.index, который выставит парсер для этого абзаца.
    """
    for i, text in enumerate(paragraphs, start=1):
        if needle in text:
            return i
    raise ValueError(f"Подстрока не найдена в абзацах: {needle!r}")


# ---------------------------------------------------------------------------
# PPTX
# ---------------------------------------------------------------------------

def _write_pptx(path: str, slides: list[list[str]]) -> None:
    """Записать PPTX: один слайд на элемент slides; элемент — список строк.

    Первая строка слайда становится заголовком, остальные — телом (буллеты).
    Слайды нумеруются 1-based, как в парсере (location.index = номер слайда).
    """
    from pptx import Presentation
    from pptx.util import Inches

    prs = Presentation()
    blank = prs.slide_layouts[6]  # пустой макет, добавляем текст сами

    for lines in slides:
        slide = prs.slides.add_slide(blank)
        title_text = lines[0] if lines else ""
        body_lines = lines[1:]

        title_box = slide.shapes.add_textbox(
            Inches(0.5), Inches(0.3), Inches(9.0), Inches(1.0)
        )
        title_box.text_frame.text = title_text

        body_box = slide.shapes.add_textbox(
            Inches(0.5), Inches(1.5), Inches(9.0), Inches(5.0)
        )
        tf = body_box.text_frame
        for j, line in enumerate(body_lines):
            para = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
            para.text = line

    prs.save(path)


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def _write_pdf(path: str, pages: list[list[str]]) -> None:
    """Записать PDF: один лист на элемент pages; элемент — список строк.

    Текст вставляется построчно через page.insert_text. Используется встроенный
    шрифт "china-ss": в отличие от "helv", он покрывает кириллицу и корректно
    извлекается обратно (get_text), что важно для русского содержимого. Страницы
    нумеруются 1-based (location.index = номер страницы), как в парсере PDF.
    """
    import pymupdf

    fontsize = 11.0
    left = 72.0
    top = 72.0
    line_step = 20.0

    doc = pymupdf.open()
    try:
        for lines in pages:
            page = doc.new_page()  # стандартный A4 (ширина 595 пунктов)
            # Шрифт china-ss моноширинный: один глиф = fontsize пунктов по
            # горизонтали. insert_text не переносит и обрезает текст по краю
            # листа, поэтому считаем лимит символов на строку и переносим сами.
            usable = page.rect.width - left - 56.0  # запас справа
            max_chars = int(usable // fontsize)
            y = top
            for line in lines:
                for chunk in _wrap_line(line, max_chars):
                    page.insert_text(
                        (left, y), chunk, fontsize=fontsize, fontname="china-ss"
                    )
                    y += line_step
        doc.save(path)
    finally:
        doc.close()


def _wrap_line(line: str, max_chars: int) -> list[str]:
    """Перенести строку по словам в пределах max_chars символов на строку.

    Пустая строка сохраняется как одна пустая часть (вертикальный отступ).
    Перенос только по пробелам — слова целиком, поэтому короткие подменённые
    значения (имя, число) не разрываются между строками.
    """
    if not line.strip():
        return [""]
    words = line.split(" ")
    out: list[str] = []
    cur = ""
    for w in words:
        candidate = w if not cur else f"{cur} {w}"
        if len(candidate) <= max_chars:
            cur = candidate
        else:
            if cur:
                out.append(cur)
            cur = w
    if cur:
        out.append(cur)
    return out


# ---------------------------------------------------------------------------
# Содержимое пар: для каждой пары — функция, которая пишет файлы и
# возвращает список записей ground_truth.
# ---------------------------------------------------------------------------

def _pair_finance_docx() -> list[dict]:
    """Пара 1 (DOCX): финансовый отчёт.

    Вносим: (1) изменение числа выручки и (4) внутреннее расхождение —
    итоговая премия названа по-разному в двух абзацах проверяемого документа.
    """
    name = "finance"
    src_path = os.path.join(DATASETS_DIR, f"source_{name}.docx")
    chk_path = os.path.join(DATASETS_DIR, f"check_{name}.docx")

    source = [
        "Финансовый отчёт компании «Орбита» за 2024 год",
        "Выручка компании за 2024 год составила 152 миллиона рублей.",
        "Чистая прибыль по итогам года достигла 18 миллионов рублей.",
        "Совокупный премиальный фонд сотрудников за год составил 7 миллионов рублей.",
        "Отчёт подготовлен финансовым отделом и утверждён советом директоров.",
        "Премиальный фонд в размере 7 миллионов рублей распределён между подразделениями.",
    ]
    # В проверяемом: выручка занижена (152 -> 140); премия в двух местах
    # названа по-разному (7 vs 9) — внутреннее расхождение.
    check = [
        "Финансовый отчёт компании «Орбита» за 2024 год",
        "Выручка компании за 2024 год составила 140 миллионов рублей.",
        "Чистая прибыль по итогам года достигла 18 миллионов рублей.",
        "Совокупный премиальный фонд сотрудников за год составил 7 миллионов рублей.",
        "Отчёт подготовлен финансовым отделом и утверждён советом директоров.",
        "Премиальный фонд в размере 9 миллионов рублей распределён между подразделениями.",
    ]

    _write_docx(src_path, source)
    _write_docx(chk_path, check)

    doc_name = f"check_{name}.docx"
    src_name = f"source_{name}.docx"
    return [
        {
            "doc": doc_name,
            "source": src_name,
            "kind": "contradicted",
            "location": {
                "kind": "paragraph",
                "index": _docx_index(check, "Выручка компании"),
            },
            "tampered_value": "140 миллионов рублей",
            "note": "Изменено число: выручка 152 -> 140 миллионов рублей (тип number).",
        },
        {
            "doc": doc_name,
            "source": src_name,
            "kind": "internal_conflict",
            "location": {
                "kind": "paragraph",
                "index": _docx_index(check, "распределён между подразделениями"),
            },
            "tampered_value": "9 миллионов рублей",
            "note": (
                "Внутреннее расхождение: премиальный фонд назван 7 млн в одном "
                "абзаце и 9 млн в другом (тип number)."
            ),
        },
    ]


def _pair_project_pptx() -> list[dict]:
    """Пара 2 (PPTX): план проекта.

    Вносим: (2) замену даты запуска и (3) утверждение, противоречащее
    источнику (статус бюджета).
    """
    name = "project"
    src_path = os.path.join(DATASETS_DIR, f"source_{name}.pptx")
    chk_path = os.path.join(DATASETS_DIR, f"check_{name}.pptx")

    source = [
        ["План проекта «Маяк»", "Краткий обзор ключевых вех проекта."],
        [
            "Сроки",
            "Старт проекта: 1 марта 2025 года.",
            "Дата запуска в эксплуатацию: 15 сентября 2025 года.",
        ],
        [
            "Команда и бюджет",
            "Руководитель проекта: Анна Воробьёва.",
            "Утверждённый бюджет проекта: 12 миллионов рублей.",
            "Бюджет проекта согласован и не превышен.",
        ],
    ]
    # В проверяемом: дата запуска смещена (15 сентября -> 30 ноября);
    # добавлено противоречащее утверждение о превышении бюджета.
    check = [
        ["План проекта «Маяк»", "Краткий обзор ключевых вех проекта."],
        [
            "Сроки",
            "Старт проекта: 1 марта 2025 года.",
            "Дата запуска в эксплуатацию: 30 ноября 2025 года.",
        ],
        [
            "Команда и бюджет",
            "Руководитель проекта: Анна Воробьёва.",
            "Утверждённый бюджет проекта: 12 миллионов рублей.",
            "Бюджет проекта превышен на 3 миллиона рублей.",
        ],
    ]

    _write_pptx(src_path, source)
    _write_pptx(chk_path, check)

    doc_name = f"check_{name}.pptx"
    src_name = f"source_{name}.pptx"
    return [
        {
            "doc": doc_name,
            "source": src_name,
            "kind": "contradicted",
            "location": {"kind": "slide", "index": 2},
            "tampered_value": "30 ноября 2025 года",
            "note": "Заменена дата: запуск 15 сентября -> 30 ноября 2025 (тип date).",
        },
        {
            "doc": doc_name,
            "source": src_name,
            "kind": "contradicted",
            "location": {"kind": "slide", "index": 3},
            "tampered_value": "Бюджет проекта превышен на 3 миллиона рублей.",
            "note": (
                "Вставлено утверждение, противоречащее источнику (в источнике "
                "бюджет не превышен) (тип statement)."
            ),
        },
    ]


def _pair_hr_pdf() -> list[dict]:
    """Пара 3 (PDF): кадровая записка.

    Вносим: (3) утверждение, противоречащее источнику (имя сотрудника), и
    (1) изменение числа (оклад).
    """
    name = "hr"
    src_path = os.path.join(DATASETS_DIR, f"source_{name}.pdf")
    chk_path = os.path.join(DATASETS_DIR, f"check_{name}.pdf")

    source = [
        [
            "Кадровая записка № 47",
            "",
            "На должность ведущего аналитика назначен Игорь Смирнов.",
            "Должностной оклад сотрудника составляет 95000 рублей в месяц.",
            "Дата выхода на работу: 10 февраля 2025 года.",
            "Испытательный срок: три месяца.",
        ],
    ]
    # В проверяемом: имя заменено (Игорь Смирнов -> Павел Смирнов);
    # оклад изменён (95000 -> 110000).
    check = [
        [
            "Кадровая записка № 47",
            "",
            "На должность ведущего аналитика назначен Павел Смирнов.",
            "Должностной оклад сотрудника составляет 110000 рублей в месяц.",
            "Дата выхода на работу: 10 февраля 2025 года.",
            "Испытательный срок: три месяца.",
        ],
    ]

    _write_pdf(src_path, source)
    _write_pdf(chk_path, check)

    doc_name = f"check_{name}.pdf"
    src_name = f"source_{name}.pdf"
    return [
        {
            "doc": doc_name,
            "source": src_name,
            "kind": "contradicted",
            "location": {"kind": "page", "index": 1},
            "tampered_value": "Павел Смирнов",
            "note": (
                "Заменено имя: Игорь Смирнов -> Павел Смирнов, что противоречит "
                "источнику (тип name)."
            ),
        },
        {
            "doc": doc_name,
            "source": src_name,
            "kind": "contradicted",
            "location": {"kind": "page", "index": 1},
            "tampered_value": "110000 рублей",
            "note": "Изменено число: оклад 95000 -> 110000 рублей (тип number).",
        },
    ]


def _pair_estimate_docx() -> list[dict]:
    """Пара 4 (DOCX): смета договора.

    Вносим: (4) внутреннее расхождение (общая сумма договора названа
    по-разному в двух абзацах) и (1) изменение числа (срок гарантии),
    противоречащее источнику.
    """
    name = "estimate"
    src_path = os.path.join(DATASETS_DIR, f"source_{name}.docx")
    chk_path = os.path.join(DATASETS_DIR, f"check_{name}.docx")

    source = [
        "Смета по договору № 18 на поставку оборудования",
        "Общая сумма договора составляет 480 тысяч рублей.",
        "Стоимость монтажных работ включена в общую сумму и равна 60 тысяч рублей.",
        "Гарантийный срок на оборудование составляет 24 месяца.",
        "Итоговая сумма к оплате по договору — 480 тысяч рублей.",
    ]
    # В проверяемом: гарантийный срок изменён (24 -> 36); общая сумма названа
    # по-разному в двух абзацах (480 vs 520) — внутреннее расхождение.
    check = [
        "Смета по договору № 18 на поставку оборудования",
        "Общая сумма договора составляет 480 тысяч рублей.",
        "Стоимость монтажных работ включена в общую сумму и равна 60 тысяч рублей.",
        "Гарантийный срок на оборудование составляет 36 месяцев.",
        "Итоговая сумма к оплате по договору — 520 тысяч рублей.",
    ]

    _write_docx(src_path, source)
    _write_docx(chk_path, check)

    doc_name = f"check_{name}.docx"
    src_name = f"source_{name}.docx"
    return [
        {
            "doc": doc_name,
            "source": src_name,
            "kind": "contradicted",
            "location": {
                "kind": "paragraph",
                "index": _docx_index(check, "Гарантийный срок"),
            },
            "tampered_value": "36 месяцев",
            "note": "Изменено число: гарантийный срок 24 -> 36 месяцев (тип number).",
        },
        {
            "doc": doc_name,
            "source": src_name,
            "kind": "internal_conflict",
            "location": {
                "kind": "paragraph",
                "index": _docx_index(check, "Итоговая сумма к оплате"),
            },
            "tampered_value": "520 тысяч рублей",
            "note": (
                "Внутреннее расхождение: общая сумма договора названа 480 тыс. "
                "в одном абзаце и 520 тыс. в другом (тип number)."
            ),
        },
    ]


# Порядок пар фиксирован для детерминированности.
_PAIR_BUILDERS = [
    _pair_finance_docx,
    _pair_project_pptx,
    _pair_hr_pdf,
    _pair_estimate_docx,
]


def main() -> None:
    """Создать все пары документов и ground_truth.json, напечатать сводку."""
    logging.basicConfig(level=logging.INFO)
    start = time.perf_counter()

    os.makedirs(DATASETS_DIR, exist_ok=True)

    ground_truth: list[dict] = []
    for build in _PAIR_BUILDERS:
        entries = build()
        ground_truth.extend(entries)
        logger.info("Пара готова: %s (%d находок)", build.__name__, len(entries))

    with open(GROUND_TRUTH_PATH, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, ensure_ascii=False, indent=2)

    elapsed = time.perf_counter() - start

    # Сводка по типам внесений и форматам.
    kinds: dict[str, int] = {}
    for e in ground_truth:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1

    logger.info(
        "Готово: %d пар, %d находок за %.2f c",
        len(_PAIR_BUILDERS), len(ground_truth), elapsed,
    )
    logger.info("Каталог: %s", DATASETS_DIR)
    logger.info("ground_truth.json: %s", GROUND_TRUTH_PATH)
    logger.info("Находки по вердиктам: %s", kinds)

    print("=" * 60)
    print("Сводка генерации датасета ВериДок")
    print("=" * 60)
    print(f"Пар документов:        {len(_PAIR_BUILDERS)}")
    print(f"Записей ground_truth:  {len(ground_truth)}")
    print(f"По вердиктам:          {kinds}")
    print(f"Каталог:               {DATASETS_DIR}")
    print(f"ground_truth.json:     {GROUND_TRUTH_PATH}")
    print(f"Время:                 {elapsed:.2f} c")
    print("=" * 60)


if __name__ == "__main__":
    main()
