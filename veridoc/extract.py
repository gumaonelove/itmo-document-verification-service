"""Извлечение проверяемых утверждений из сегментов документа.

Для каждого сегмента LLM возвращает JSON-массив атомарных фактов; здесь они
превращаются в Claim со сквозной нумерацией. MVP-приоритет — надёжно вытаскивать
числа (number) и даты (date), для них сохраняется raw_value.
"""

import json
import logging
import time

from config import settings
from veridoc import prompts
from veridoc.llm import chat_json
from veridoc.models import Claim, Segment

logger = logging.getLogger(__name__)

# Допустимые типы утверждений; всё прочее нормализуется в "statement".
_VALID_TYPES = {"number", "date", "name", "statement"}
# Типы, для которых имеет смысл хранить исходное значение (для consistency-этапа).
_RAW_VALUE_TYPES = {"number", "date"}


def extract_claims(segments: list[Segment]) -> list[Claim]:
    """Извлечь проверяемые утверждения из всех сегментов.

    Для каждого сегмента шлём один запрос к LLM и ждём JSON-массив объектов
    вида {"text", "type", "raw_value"}. Сегменты с не-списочным ответом
    пропускаются с warning. На сегмент берём не более
    ``settings.max_claims_per_segment`` утверждений.
    """
    started = time.perf_counter()
    claims: list[Claim] = []
    counter = 0  # сквозной счётчик для id вида C0001

    for segment in segments:
        messages = [
            {"role": "user", "content": prompts.extract_prompt(segment.text)}
        ]
        # Неустранимо битый JSON от LLM не должен валить извлечение всего
        # документа — пропускаем проблемный сегмент, как и не-списочный ответ.
        try:
            data = chat_json(messages)
        except json.JSONDecodeError:
            logger.warning(
                "Сегмент %s: LLM вернул невалидный JSON — пропускаю", segment.id
            )
            continue

        if not isinstance(data, list):
            logger.warning(
                "Сегмент %s: ответ LLM не список (%s) — пропускаю",
                segment.id,
                type(data).__name__,
            )
            continue

        # Ограничиваем число утверждений на сегмент.
        for item in data[: settings.max_claims_per_segment]:
            if not isinstance(item, dict):
                continue

            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                continue  # без непустого text утверждение бессмысленно

            raw_type = item.get("type")
            claim_type = raw_type if raw_type in _VALID_TYPES else "statement"

            # raw_value храним только для чисел и дат.
            raw_value = None
            if claim_type in _RAW_VALUE_TYPES:
                rv = item.get("raw_value")
                if rv is not None:
                    raw_value = str(rv)

            counter += 1
            claims.append(
                Claim(
                    id=f"C{counter:04d}",
                    text=text.strip(),
                    type=claim_type,
                    segment_id=segment.id,
                    location=segment.location,
                    raw_value=raw_value,
                )
            )

    elapsed = time.perf_counter() - started
    logger.info(
        "Извлечено утверждений: %d из %d сегментов за %.2f с",
        len(claims),
        len(segments),
        elapsed,
    )
    return claims
