"""OpenAI-совместимый клиент ВериДок.

Единая точка обращения к локальному LLM-эндпойнту. Все вызовы — при
temperature=0.0 по умолчанию (см. settings). Клиент создаётся лениво и переиспользуется.

Использование:
    from veridoc.llm import chat, chat_json
    text = chat([{"role": "user", "content": "..."}])
    data = chat_json([{"role": "user", "content": "...верни JSON..."}])

chat_json внутри вызывает chat(), поэтому в тестах достаточно подменить chat монкипатчем.
"""

import json
import logging
import re
import time
from typing import Any, Optional

from openai import OpenAI

from config import settings

logger = logging.getLogger(__name__)

# Ленивый одиночный клиент.
_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    """Вернуть единственный экземпляр OpenAI-клиента, создав его при первом вызове."""
    global _client
    if _client is None:
        _client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
        logger.info("LLM-клиент создан: base_url=%s, model=%s", settings.llm_base_url, settings.llm_model)
    return _client


def chat(
    messages: list[dict],
    *,
    temperature: float = settings.llm_temperature,
    max_tokens: int = settings.llm_max_tokens,
) -> str:
    """Один синхронный chat-запрос; вернуть текст ответа.

    messages — список вида [{"role": "...", "content": "..."}].
    """
    client = get_client()
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = resp.choices[0].message.content or ""
    logger.info("chat: %d сообщений → %d символов за %.2f с", len(messages), len(content), time.perf_counter() - t0)
    return content


# Markdown-ограждения: ```json ... ``` , ``` ... ``` , тройные кавычки.
_FENCE_RE = re.compile(
    r"^\s*(?:```+|''')[ \t]*[A-Za-z0-9_+-]*[ \t]*\r?\n(.*?)\r?\n\s*(?:```+|''')\s*$",
    re.DOTALL,
)


def _strip_fences(text: str) -> str:
    """Снять markdown-ограждения (json-блоки, тройные кавычки) вокруг ответа."""
    s = text.strip()
    m = _FENCE_RE.match(s)
    if m:
        return m.group(1).strip()
    # Подстраховка: одиночные строки-ограждения без переноса распознаём по краям.
    for fence in ("```json", "```", "'''"):
        if s.startswith(fence):
            s = s[len(fence):]
            break
    for fence in ("```", "'''"):
        if s.endswith(fence):
            s = s[: -len(fence)]
            break
    return s.strip()


def _extract_json_substring(text: str) -> Optional[str]:
    """Вырезать подстроку от первого '[' или '{' до парного закрывающего символа.

    Скобки внутри строковых литералов JSON игнорируются. Возвращает None,
    если сбалансированную подстроку найти не удалось.
    """
    start = -1
    open_ch = ""
    for i, ch in enumerate(text):
        if ch in "[{":
            start = i
            open_ch = ch
            break
    if start < 0:
        return None
    close_ch = "]" if open_ch == "[" else "}"

    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _try_parse(text: str) -> Any:
    """Распарсить JSON: сначала со снятыми ограждениями, затем по вырезанной подстроке.

    Бросает json.JSONDecodeError, если ничего не распарсилось.
    """
    cleaned = _strip_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        substr = _extract_json_substring(cleaned)
        if substr is not None:
            return json.loads(substr)
        raise


def chat_json(
    messages: list[dict],
    *,
    retries: int = 2,
    max_tokens: int = settings.llm_max_tokens,
) -> Any:
    """Chat-запрос с разбором ответа как JSON (list или dict).

    Вызывает chat() (его можно подменить в тестах), снимает markdown-ограждения и
    пытается json.loads. При неудаче — извлекает сбалансированную JSON-подстроку.
    Если не вышло, добавляет уточняющее сообщение и повторяет до `retries` раз.
    После исчерпания попыток логирует сырой ответ и пробрасывает json.JSONDecodeError.
    """
    t0 = time.perf_counter()
    msgs = list(messages)
    raw = ""
    last_err: Optional[json.JSONDecodeError] = None

    for attempt in range(retries + 1):
        raw = chat(msgs, max_tokens=max_tokens)
        try:
            data = _try_parse(raw)
            logger.info("chat_json: распарсено с попытки %d за %.2f с", attempt + 1, time.perf_counter() - t0)
            return data
        except json.JSONDecodeError as err:
            last_err = err
            logger.warning("chat_json: попытка %d не распарсилась: %s", attempt + 1, err)
            if attempt < retries:
                msgs = msgs + [
                    {"role": "user", "content": "Верни строго валидный JSON, без markdown и пояснений."}
                ]

    logger.error("chat_json: не удалось распарсить JSON после %d попыток. Сырой ответ: %r", retries + 1, raw)
    if last_err is not None:
        raise last_err
    raise json.JSONDecodeError("не удалось распарсить JSON", raw, 0)
