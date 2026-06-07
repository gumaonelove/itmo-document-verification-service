# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ВериДок — локальный сервис проверки достоверности корпоративных документов (ITMO 1st-year practice project, MVP). По загруженному документу (PDF/DOCX/PPTX) он извлекает атомарные проверяемые утверждения, сверяет каждое с доверенными источниками через RAG (вердикт supported / contradicted / not_found с цитатой), ищет внутренние противоречия и показывает отчёт в Streamlit. Интерфейс и документы — **русские**. Всё считается **локально**, без облачных API. Полное ТЗ и инструкции по запуску — в [README.md](README.md).

## Commands

Всё запускается **из корня репозитория** (там лежат `config.py` и `app.py`; модули пакета делают `from config import settings`, что требует корень в `sys.path`).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                       # правка эндпойнтов при необходимости

streamlit run app.py                       # веб-демо (UI)
python -m veridoc.pipeline <doc> --sources <dir>   # CLI-прогон одного документа (M1)

python eval/seed.py                        # сгенерировать датасет с внесёнными ошибками
python eval/evaluate.py                    # метрики recall/precision/F1/время → eval/results.json

pytest                                     # тесты
pytest tests/test_llm_json.py::test_retries_then_succeeds   # один тест
```

Тесты в [tests/](tests/) **офлайновые** — им не нужны LLM/эмбеддер/Qdrant, только `pydantic-settings`, `openai`, `python-docx`, `python-pptx`, `pytest` (лёгкое подмножество `requirements.txt`; `torch`/`sentence-transformers`/`qdrant-client`/`streamlit` для них не требуются). Для реального прогона `run_check` нужны поднятый OpenAI-совместимый LLM-эндпойнт и Qdrant — см. README.

## Architecture

Конвейер собран в [veridoc/pipeline.py](veridoc/pipeline.py) `run_check(doc_path, sources_dir)`:

```
ИСТОЧНИКИ (папка) ─► parse → _chunk_text → Embedder.embed_documents → QdrantStore (фаза индексации)
ДОКУМЕНТ ─► parse_document → extract_claims (LLM) ─┬─► embed_query → QdrantStore.search → verify_claim (LLM) → Finding
                                                   └─► check_consistency (LLM, по number/date) → Finding(internal_conflict)
                                          → assemble Report → Streamlit / CLI
```

Слои и их ответственность ([veridoc/](veridoc/)): `models.py` — единственный источник правды для dataclasses (Segment/Claim/Evidence/Finding/Report); `parsing.py` — PDF (PyMuPDF4LLM, fallback pymupdf), DOCX (python-docx), PPTX (python-pptx); `llm.py` — OpenAI-совместимый клиент; `embedder.py` — sentence-transformers; `store.py` — Qdrant; `extract.py`/`verify.py`/`consistency.py` — три LLM-этапа; `prompts.py` — шаблоны. Конфигурация — [config.py](config.py) (pydantic-settings, объект `settings`).

### Conventions and gotchas (требуют чтения нескольких файлов)

- **LLM — раннер-агностичен.** Код общается с OpenAI-совместимым `/v1/chat/completions` по `LLM_BASE_URL`/`LLM_MODEL`; конкретный сервинг (mlx-vlm, Ollama и т.п.) — деталь окружения, в коде её нет.
- **Эмбеддинги — только через `sentence-transformers`, НЕ через `mlx-lm`** (тот не поддерживает embedding-модели). Запросы кодируются с инструкцией (`embed_query` использует `prompt_name="query"` с ручным фолбэк-префиксом), документы — без неё.
- **Промпты подставляются через `str.replace`, а НЕ `.format`** ([prompts.py](veridoc/prompts.py)): шаблоны содержат литеральные фигурные скобки JSON-примеров, которые сломали бы `.format`.
- **`chat_json` после `retries` бросает `json.JSONDecodeError`** (а не возвращает безопасное значение). `extract.py`/`verify.py`/`consistency.py` ловят его и деградируют мягко (пропуск сегмента / `not_found` / пустой список), поэтому один битый ответ LLM не валит весь прогон — в т.ч. в CLI.
- **Каждая находка несёт `location`** (`{"kind": "page"|"slide"|"paragraph", "index": int}`); форматирование «Где» в UI и CLI завязано на это.
- **Память (M4 Max 48 ГБ): индексация и верификация — разные фазы.** При смене `EMBED_DIM` коллекция Qdrant пересоздаётся (`QdrantStore.ensure_collection`). Qdrant — по `QDRANT_URL` либо embedded по `QDRANT_PATH`.
- **Детерминизм:** все LLM-вызовы при `temperature=0.0` (из `settings`).

При изменении контрактов правьте `models.py`/`config.py`/`prompts.py` как источник правды — на них завязаны все остальные модули.
