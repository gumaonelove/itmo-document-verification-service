"""Шаблоны промптов и сборщики финальных строк.

Шаблоны содержат литеральные фигурные скобки JSON-примеров, поэтому подстановка
делается через str.replace по явным плейсхолдерам ({segment_text} и т.п.), а НЕ
через str.format — иначе скобки JSON ломают форматирование.

Все вызовы LLM выполняются при temperature=0.0.
"""

EXTRACT_PROMPT = """Ты выделяешь из текста атомарные ПРОВЕРЯЕМЫЕ утверждения для сверки с источниками.
Верни СТРОГО JSON-массив без markdown. Каждый элемент:
{"text": "<одно самодостаточное утверждение>",
 "type": "number|date|name|statement",
 "raw_value": "<число или дата как в тексте, иначе null>"}
Извлекай ВСЕ проверяемые факты, а не только числа:
- number — числовые показатели, суммы, проценты;
- date — даты и сроки;
- name — имена людей, должности, названия организаций;
- statement — фактические утверждения и статусы (например «согласовано», «бюджет превышен»).
Правила: одно утверждение = один факт; не выдумывай; если проверяемых фактов нет — верни [].
ТЕКСТ:
<<<{segment_text}>>>"""

VERIFY_PROMPT = """Проверь утверждение по приведённым фрагментам источников.
Верни СТРОГО JSON: {"verdict":"supported|contradicted|not_found",
"source_id":"<id или null>","quote":"<дословная цитата ≤30 слов или null>",
"reasoning":"<кратко по-русски>"}
supported — источник подтверждает; contradicted — источник прямо противоречит;
not_found — во фрагментах нет ответа. Цитата — только из приведённых фрагментов.
УТВЕРЖДЕНИЕ: {claim_text}
ФРАГМЕНТЫ:
{evidence_block}"""

CONSISTENCY_PROMPT = """Найди ВЗАИМНО ПРОТИВОРЕЧАЩИЕ утверждения (одна величина/дата с разными значениями).
Верни СТРОГО JSON-массив пар: [{"a":"<id>","b":"<id>","reason":"<кратко>"}]. Если нет — [].
УТВЕРЖДЕНИЯ:
{claims_block}"""


def extract_prompt(segment_text: str) -> str:
    """Промпт извлечения утверждений для одного сегмента."""
    return EXTRACT_PROMPT.replace("{segment_text}", segment_text)


def verify_prompt(claim_text: str, evidence_block: str) -> str:
    """Промпт верификации одного утверждения.

    evidence_block — заранее собранная строка вида "[source_id] текст...".
    """
    return VERIFY_PROMPT.replace("{claim_text}", claim_text).replace(
        "{evidence_block}", evidence_block
    )


def consistency_prompt(claims_block: str) -> str:
    """Промпт поиска внутренних противоречий.

    claims_block — строки вида "[id] text (raw_value)".
    """
    return CONSISTENCY_PROMPT.replace("{claims_block}", claims_block)
