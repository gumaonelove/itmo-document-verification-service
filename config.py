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
    llm_base_url: str = "http://localhost:8080/v1"
    llm_model: str = "Qwen3.6-35B-A3B"
    # Локальные серверы ключ игнорируют, но openai SDK требует непустое значение.
    llm_api_key: str = "not-needed"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1500

    # --- Эмбеддинги: sentence-transformers на MPS ---
    embed_model: str = "Qwen/Qwen3-Embedding-8B"
    embed_dim: int = 4096
    embed_device: str = "mps"

    # --- Qdrant: HTTP по url ИЛИ embedded по qdrant_path ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_path: Optional[str] = None  # если задан — embedded (on-disk) режим вместо url
    qdrant_collection: str = "veridoc_sources"

    # --- Чанкинг и поиск ---
    chunk_size: int = 800              # символов
    chunk_overlap: int = 150
    top_k: int = 5
    max_claims_per_segment: int = 12

    # Порог косинусной близости: если max score найденного < порога → not_found.
    score_threshold: float = 0.3


settings = Settings()
