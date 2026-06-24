"""Эмбеддер ВериДок с двумя бэкендами.

- "sentence-transformers": локальная модель Qwen3-Embedding на MPS;
- "ollama": OpenAI-совместимый /v1/embeddings (напр. qwen3-embedding:8b).

Бэкенд выбирается ``settings.embed_backend``. Тяжёлые зависимости импортируются
лениво: для бэкенда "ollama" не нужны ни torch, ни sentence-transformers.

Документы кодируются БЕЗ инструкции, а запросы — С инструкцией, как требует
семейство Qwen3-Embedding (запросу нужен префикс "Instruct: ...\\nQuery: ").
Векторы L2-нормируются для косинусной близости.
"""

import logging
import math
import time

from config import settings

logger = logging.getLogger(__name__)

# Размер батча при кодировании документов (компромисс памяти/скорости).
_BATCH_SIZE = 16

# Ручная инструкция для запросов — для Ollama и как фолбэк для sentence-transformers.
_QUERY_INSTRUCT = (
    "Instruct: Given a query, retrieve relevant passages that answer it\nQuery: "
)


def _l2_normalize(vec: list[float]) -> list[float]:
    """L2-нормировка вектора (для согласованной косинусной близости)."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return list(vec)
    return [x / norm for x in vec]


class Embedder:
    """Кодирует тексты в плотные векторы; бэкенд — sentence-transformers или Ollama."""

    def __init__(
        self,
        model: str = settings.embed_model,
        device: str = settings.embed_device,
    ) -> None:
        self.backend = settings.embed_backend
        self.model_name = model
        t0 = time.perf_counter()

        if self.backend == "ollama":
            # OpenAI-совместимый клиент к Ollama; sentence-transformers не нужен.
            from openai import OpenAI

            self._client = OpenAI(
                base_url=settings.embed_base_url, api_key=settings.llm_api_key
            )
            self._st = None
            logger.info(
                "Эмбеддер: бэкенд ollama, модель %s, эндпойнт %s",
                model,
                settings.embed_base_url,
            )
        else:
            from sentence_transformers import SentenceTransformer

            self._st = SentenceTransformer(model, device=device)
            self._client = None
            logger.info(
                "Эмбеддер: бэкенд sentence-transformers, модель %s на %s за %.2f c",
                model,
                device,
                time.perf_counter() - t0,
            )

    def _ollama_embed(self, inputs: list[str]) -> list[list[float]]:
        """Закодировать батч строк через Ollama, нормируя векторы."""
        resp = self._client.embeddings.create(model=self.model_name, input=inputs)
        # Порядок ответа должен совпадать с входом; страхуемся сортировкой по index.
        items = sorted(resp.data, key=lambda d: d.index)
        return [_l2_normalize(list(d.embedding)) for d in items]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Кодирует список документов БЕЗ инструкции, нормируя векторы (косинус)."""
        t0 = time.perf_counter()
        if not texts:
            return []

        if self.backend == "ollama":
            vectors: list[list[float]] = []
            for i in range(0, len(texts), _BATCH_SIZE):
                vectors.extend(self._ollama_embed(texts[i : i + _BATCH_SIZE]))
        else:
            vectors = self._st.encode(
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
        """Кодирует запрос С инструкцией (требование Qwen3-Embedding), нормируя вектор."""
        t0 = time.perf_counter()

        if self.backend == "ollama":
            # Ollama не применяет инструкцию автоматически — добавляем префикс вручную.
            vector = self._ollama_embed([_QUERY_INSTRUCT + text])[0]
        else:
            # Сначала пробуем штатный prompt_name="query"; иначе — ручной префикс.
            try:
                vector = self._st.encode(
                    text, prompt_name="query", normalize_embeddings=True
                ).tolist()
            except Exception:
                logger.info("prompt_name='query' недоступен, используем ручной префикс")
                vector = self._st.encode(
                    _QUERY_INSTRUCT + text, normalize_embeddings=True
                ).tolist()

        logger.info(
            "Закодирован запрос, размерность %d за %.2f c",
            len(vector),
            time.perf_counter() - t0,
        )
        return vector
