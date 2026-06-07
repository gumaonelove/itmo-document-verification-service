"""Эмбеддер ВериДок поверх sentence-transformers.

Кодирует тексты в плотные нормированные векторы (для косинусной близости).
Документы кодируются без инструкции, а запросы — с инструкцией, как того
требует семейство Qwen3-Embedding: запросу нужен префикс "Instruct: ...\nQuery: ".
"""

import logging
import time

from sentence_transformers import SentenceTransformer

from config import settings

logger = logging.getLogger(__name__)

# Размер батча при кодировании документов (компромисс памяти/скорости).
_BATCH_SIZE = 16

# Ручная инструкция для запросов — фолбэк, если модель не знает prompt_name="query".
_QUERY_INSTRUCT = (
    "Instruct: Given a query, retrieve relevant passages that answer it\nQuery: "
)


class Embedder:
    """Обёртка над SentenceTransformer с раздельным кодированием документов и запросов."""

    def __init__(
        self,
        model: str = settings.embed_model,
        device: str = settings.embed_device,
    ) -> None:
        t0 = time.perf_counter()
        self.model = SentenceTransformer(model, device=device)
        logger.info(
            "Загружена модель эмбеддингов %s на %s за %.2f c",
            model,
            device,
            time.perf_counter() - t0,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Кодирует список документов БЕЗ инструкции, нормируя векторы (косинус)."""
        t0 = time.perf_counter()
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=_BATCH_SIZE,
        ).tolist()
        dim = len(vectors[0]) if vectors else 0
        logger.info(
            "Закодировано %d документов, размерность %d за %.2f c",
            len(vectors),
            dim,
            time.perf_counter() - t0,
        )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Кодирует запрос С инструкцией (требование Qwen3-Embedding), нормируя вектор.

        Сначала пробуем штатный prompt_name="query"; если модель его не знает —
        падаем на ручной префикс _QUERY_INSTRUCT и обычный encode.
        """
        t0 = time.perf_counter()
        try:
            vector = self.model.encode(
                text,
                prompt_name="query",
                normalize_embeddings=True,
            )
        except Exception:
            # Модель не зарегистрировала prompt "query" — добавляем инструкцию вручную.
            logger.info("prompt_name='query' недоступен, используем ручной префикс")
            vector = self.model.encode(
                _QUERY_INSTRUCT + text,
                normalize_embeddings=True,
            )
        result = vector.tolist()
        logger.info(
            "Закодирован запрос, размерность %d за %.2f c",
            len(result),
            time.perf_counter() - t0,
        )
        return result
