"""Верификация одного утверждения по найденным фрагментам источников.

Этап «поиск → верификация»: на вход — утверждение и релевантные фрагменты
доверенных источников, на выход — Finding с вердиктом и опорной цитатой.
Единственный внешний вызов — chat_json; всё остальное — детерминированная сборка.
Все LLM-вызовы выполняются при temperature=0.0 (задаётся в llm.py).
"""

import json
import logging
import time

from config import settings
from veridoc import prompts
from veridoc.llm import chat_json
from veridoc.models import Claim, Evidence, Finding

logger = logging.getLogger(__name__)

# Допустимые вердикты модели на этапе верификации (internal_conflict — другой этап).
_VALID_VERDICTS = {"supported", "contradicted", "not_found"}


def verify_claim(claim: Claim, evidences: list[Evidence]) -> Finding:
    """Проверить утверждение по списку найденных фрагментов источников.

    Если фрагментов нет или лучший из них ниже порога близости — not_found без
    обращения к LLM. Иначе собираем evidence_block, спрашиваем LLM и формируем
    Finding с опорной цитатой.
    """
    t0 = time.perf_counter()

    # Нет фрагментов или все ниже порога → ответа в источниках нет.
    if not evidences or max(e.score for e in evidences) < settings.score_threshold:
        logger.info(
            "verify_claim %s: not_found (фрагментов: %d) за %.3f с",
            claim.id,
            len(evidences),
            time.perf_counter() - t0,
        )
        return Finding(
            claim=claim,
            verdict="not_found",
            evidence=[],
            reasoning="Нет релевантных фрагментов в источниках",
        )

    # Лучший фрагмент — запасной источник цитаты/идентификаторов.
    best = max(evidences, key=lambda e: e.score)
    # Карта source_id → Evidence для сопоставления ответа модели с фрагментом.
    by_source = {e.source_id: e for e in evidences}

    evidence_block = "\n".join(f"[{e.source_id}] {e.quote}" for e in evidences)
    messages = [
        {"role": "user", "content": prompts.verify_prompt(claim.text, evidence_block)}
    ]

    # chat_json при неустранимом сбое разбора бросает json.JSONDecodeError.
    # Один битый ответ LLM не должен валить всю проверку — трактуем как not_found.
    try:
        data = chat_json(messages)
    except json.JSONDecodeError:
        logger.warning(
            "verify_claim %s: LLM вернул невалидный JSON — not_found", claim.id
        )
        return Finding(
            claim=claim,
            verdict="not_found",
            evidence=[],
            reasoning="LLM вернул невалидный JSON",
        )
    # Дополнительно страхуемся от валидного, но не-dict ответа.
    if not isinstance(data, dict):
        data = {}

    verdict = data.get("verdict")
    if verdict not in _VALID_VERDICTS:
        verdict = "not_found"

    reasoning = data.get("reasoning") or ""

    # not_found — опорной цитаты нет, evidence оставляем пустым.
    if verdict == "not_found":
        logger.info(
            "verify_claim %s: not_found за %.3f с",
            claim.id,
            time.perf_counter() - t0,
        )
        return Finding(
            claim=claim,
            verdict=verdict,
            evidence=[],
            reasoning=reasoning,
        )

    # Сопоставляем выбранный моделью source_id с найденным фрагментом; иначе — лучший.
    chosen_id = data.get("source_id")
    matched = by_source.get(chosen_id, best)
    source_id = chosen_id or matched.source_id
    quote = data.get("quote") or matched.quote

    evidence = [
        Evidence(
            source_id=source_id,
            chunk_id=matched.chunk_id,
            quote=quote,
            score=matched.score,
        )
    ]

    logger.info(
        "verify_claim %s: %s за %.3f с",
        claim.id,
        verdict,
        time.perf_counter() - t0,
    )
    return Finding(
        claim=claim,
        verdict=verdict,
        evidence=evidence,
        reasoning=reasoning,
    )
