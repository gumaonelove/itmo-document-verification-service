"""Доменные модели ВериДок.

Единый источник правды для контрактов между модулями: парсинг → извлечение →
поиск → верификация → согласованность → отчёт. Все остальные модули импортируют
типы отсюда и не переопределяют их.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

# Местоположение находки в документе.
# {"kind": "page"|"slide"|"paragraph", "index": int}
Location = dict


@dataclass
class Segment:
    """Атомарный кусок исходного документа с привязкой к месту."""

    id: str
    doc_id: str
    text: str
    location: Location


@dataclass
class Claim:
    """Извлечённое атомарное проверяемое утверждение."""

    id: str
    text: str                       # переформулированное самодостаточное утверждение
    type: Literal["number", "date", "name", "statement"]
    segment_id: str
    location: Location
    raw_value: Optional[str] = None  # извлечённое число/дата как строка (для consistency)


@dataclass
class Evidence:
    """Фрагмент доверенного источника, найденный поиском."""

    source_id: str
    chunk_id: str
    quote: str                      # ≤ 30 слов из источника
    score: float


@dataclass
class Finding:
    """Итог проверки одного утверждения."""

    claim: Claim
    verdict: Literal["supported", "contradicted", "not_found", "internal_conflict"]
    evidence: list[Evidence] = field(default_factory=list)
    reasoning: str = ""
    conflict_with: Optional[str] = None   # id другого Claim (для internal_conflict)


@dataclass
class Report:
    """Полный отчёт о проверке документа."""

    doc_name: str
    generated_at: str
    findings: list[Finding]
    stats: dict                     # {"total": int, "supported": int, "contradicted": int, ...}
    elapsed_sec: float
