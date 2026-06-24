"""Поиск внутренних противоречий между утверждениями документа.

Этап «согласованность»: берём числовые и датовые утверждения и спрашиваем у LLM,
какие пары противоречат друг другу (одна величина/дата с разными значениями).
Каждый конфликт оформляется как Finding с verdict="internal_conflict".
"""

import json
import logging
import time

from config import settings
from veridoc import prompts
from veridoc.llm import chat_json
from veridoc.models import Claim, Evidence, Finding

logger = logging.getLogger(__name__)


def check_consistency(claims: list[Claim]) -> list[Finding]:
    """Найти взаимно противоречащие числовые/датовые утверждения.

    Отбираем claim с type in {"number", "date"}; если их меньше двух — проверять
    нечего. Просим LLM вернуть пары противоречий, дедуплицируем и превращаем
    каждую валидную пару в Finding(verdict="internal_conflict").
    """
    started = time.perf_counter()

    candidates = [c for c in claims if c.type in {"number", "date"}]
    if len(candidates) < 2:
        logger.info(
            "Согласованность: кандидатов %d (<2) — пропуск", len(candidates)
        )
        return []

    # Строки вида "[<id>] <text> (<raw_value>)" для промпта.
    claims_block = "\n".join(
        f"[{c.id}] {c.text} ({c.raw_value})" for c in candidates
    )
    messages = [{"role": "user", "content": prompts.consistency_prompt(claims_block)}]

    # Битый JSON от LLM — не падаем, считаем, что конфликтов не найдено.
    try:
        data = chat_json(messages)
    except json.JSONDecodeError:
        logger.warning("Согласованность: LLM вернул невалидный JSON — конфликтов нет")
        return []
    if not isinstance(data, list):
        logger.warning(
            "Согласованность: LLM вернул не список (%s) — конфликтов нет",
            type(data).__name__,
        )
        return []

    by_id: dict[str, Claim] = {c.id: c for c in candidates}

    findings: list[Finding] = []
    seen: set[frozenset[str]] = set()  # дедуп пар по frozenset({a, b})
    for pair in data:
        if not isinstance(pair, dict):
            continue
        a = pair.get("a")
        b = pair.get("b")
        reason = pair.get("reason", "")
        # Оба id известны и различны.
        if a not in by_id or b not in by_id or a == b:
            continue
        key = frozenset({a, b})
        if key in seen:
            continue
        seen.add(key)

        claim_a = by_id[a]
        claim_b = by_id[b]
        reason_text = reason if isinstance(reason, str) else str(reason)
        # Конфликтующее утверждение кладём отдельным фрагментом-evidence: в UI
        # колонка «Обоснование» покажет ИМЕННО различающееся утверждение (claim_b),
        # не дублируя «Утверждение» (claim_a) и «Где» (его место). Текст claim_b
        # содержит словесное значение, поэтому оценка по value остаётся устойчивой.
        partner = Evidence(
            source_id=f"конфликт с {claim_b.id}",
            chunk_id=claim_b.id,
            quote=claim_b.text,
            score=0.0,
            location=claim_b.location,
        )
        findings.append(
            Finding(
                claim=claim_a,
                verdict="internal_conflict",
                evidence=[partner],
                reasoning=reason_text,
                conflict_with=claim_b.id,
            )
        )

    elapsed = time.perf_counter() - started
    logger.info(
        "Согласованность: кандидатов %d, конфликтов %d за %.2f с",
        len(candidates),
        len(findings),
        elapsed,
    )
    return findings
