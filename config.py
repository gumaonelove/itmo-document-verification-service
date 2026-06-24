"""Конфигурация ВериДок.

Значения читаются из переменных окружения (и файла .env). Имена переменных —
заглавными, как в .env.example (LLM_BASE_URL, EMBED_DIM, …); pydantic-settings
сопоставляет их с полями без учёта регистра.

Использование в модулях:
    from config import settings
    settings.llm_base_url, settings.embed_dim, ...
"""

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM: OpenAI-совместимый HTTP-эндпойнт (раннер — деталь окружения) ---
    # Система МОДЕЛЬ-АГНОСТИЧНА: меняется одной переменной LLM_MODEL/LLM_BASE_URL.
    # Дефолт — qwen3-coder:30b (лучший результат на корпусе оценки: recall/precision/F1 = 1.0);
    # альтернатива — лёгкая русско-специализированная T-lite-it-2.1 (см. MODEL_COMPARISON.md).
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen3-coder:30b"
    # Локальные серверы ключ игнорируют, но openai SDK требует непустое значение.
    llm_api_key: str = "not-needed"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1500
    # Параллелизм независимых LLM-вызовов (extract по сегментам, verify по
    # утверждениям). Ollama обслуживает конкурентные запросы; на Apple Silicon
    # практический потолок ~2 (GPU насыщается). Порядок результатов сохраняется.
    llm_concurrency: int = 4

    # --- Эмбеддинги ---
    # Бэкенд: "ollama" (OpenAI-совместимый /v1/embeddings, дефолт — модель уже
    # есть в Ollama) или "sentence-transformers" (локальная модель Qwen3-Embedding
    # на MPS, EMBED_MODEL=Qwen/Qwen3-Embedding-8B). В обоих случаях это одна и та
    # же модель Qwen3-Embedding-8B (dim 4096).
    embed_backend: str = "ollama"
    embed_model: str = "qwen3-embedding:8b"
    embed_dim: int = 4096
    embed_device: str = "mps"
    # Эндпойнт для embed_backend="ollama".
    embed_base_url: str = "http://localhost:11434/v1"

    # --- Qdrant: HTTP по url ИЛИ embedded по qdrant_path ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_path: Optional[str] = None  # если задан — embedded (on-disk) режим вместо url
    qdrant_collection: str = "veridoc_sources"

    # --- Чанкинг и поиск ---
    chunk_size: int = 800              # символов
    chunk_overlap: int = 150
    top_k: int = 5
    max_claims_per_segment: int = 12
    # Сколько верхних по score фрагментов реально подаётся в verify (полным
    # текстом). Меньше фрагментов → меньше префилл LLM и меньше шума; для проверки
    # факта обычно хватает 2-3. Снижает время на длинных источниках.
    verify_evidence_k: int = 3

    # Порог косинусной близости: если max score найденного < порога → not_found.
    score_threshold: float = 0.3


settings = Settings()
