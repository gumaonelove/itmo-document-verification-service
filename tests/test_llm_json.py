"""Тесты veridoc.llm.chat_json без сети.

chat_json внутри зовёт chat(); подменяем veridoc.llm.chat монкипатчем и проверяем
разбор JSON: снятие markdown-ограждений, чистый объект, мусор до/после, повтор
после неудачи и проброс json.JSONDecodeError при исчерпании попыток.
"""

import json

import pytest

from veridoc.llm import chat_json

# Произвольные сообщения: содержимое неважно, chat всё равно подменён.
MESSAGES = [{"role": "user", "content": "верни JSON"}]


def test_strips_markdown_fence_returns_list(monkeypatch):
    """Массив в ```json ... ``` → ограждение снимается, возвращается list."""

    def fake(messages, *, max_tokens=None, **kwargs):
        return '```json\n[{"id": 1}, {"id": 2}]\n```'

    monkeypatch.setattr("veridoc.llm.chat", fake)

    result = chat_json(MESSAGES)
    assert result == [{"id": 1}, {"id": 2}]


def test_plain_json_object_returns_dict(monkeypatch):
    """Чистый JSON-объект без ограждений → возвращается dict."""

    def fake(messages, *, max_tokens=None, **kwargs):
        return '{"verdict": "supported", "score": 0.9}'

    monkeypatch.setattr("veridoc.llm.chat", fake)

    result = chat_json(MESSAGES)
    assert result == {"verdict": "supported", "score": 0.9}


def test_extracts_json_with_surrounding_text(monkeypatch):
    """Посторонний текст до и после JSON → подстрока извлекается и парсится."""

    def fake(messages, *, max_tokens=None, **kwargs):
        return (
            "Конечно, вот ответ:\n"
            '{"a": [1, 2], "b": "текст с } скобкой"}\n'
            "Надеюсь, это помогло."
        )

    monkeypatch.setattr("veridoc.llm.chat", fake)

    result = chat_json(MESSAGES)
    # Скобка внутри строкового литерала не должна сбивать извлечение подстроки.
    assert result == {"a": [1, 2], "b": "текст с } скобкой"}


def test_retries_then_succeeds(monkeypatch):
    """Сначала мусор, затем валидный JSON → chat_json повторяет и парсит успешно."""
    calls = {"n": 0}

    def fake(messages, *, max_tokens=None, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return "это вообще не JSON, простите"
        return '[{"ok": true}]'

    monkeypatch.setattr("veridoc.llm.chat", fake)

    result = chat_json(MESSAGES)
    assert result == [{"ok": True}]
    # Ровно два обращения: первое — мусор, второе — валидный ответ.
    assert calls["n"] == 2


def test_always_garbage_raises_after_retries(monkeypatch):
    """chat всегда возвращает мусор → json.JSONDecodeError после исчерпания retries."""
    calls = {"n": 0}

    def fake(messages, *, max_tokens=None, **kwargs):
        calls["n"] += 1
        return "никакого JSON тут нет"

    monkeypatch.setattr("veridoc.llm.chat", fake)

    with pytest.raises(json.JSONDecodeError):
        chat_json(MESSAGES, retries=2)

    # retries=2 → ровно три попытки (1 исходная + 2 повтора).
    assert calls["n"] == 3
