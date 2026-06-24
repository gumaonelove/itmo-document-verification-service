"""Обёртка над Qdrant: индексация чанков источников и векторный поиск.

Один экземпляр QdrantStore владеет соединением с коллекцией. Поддерживаются два
режима (выбор — деталь окружения через config): embedded on-disk (если задан
settings.qdrant_path) и HTTP по url. Размерность вектора фиксируется на старте
коллекции; при несовпадении коллекция пересоздаётся.
"""

import logging
import time
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from config import settings
from veridoc.models import Evidence

logger = logging.getLogger(__name__)

# Размер батча для upsert — чтобы не слать тысячи точек одним запросом.
_BATCH_SIZE = 256


class QdrantStore:
    """Хранилище векторов доверенных источников поверх Qdrant."""

    def __init__(
        self,
        url: str = settings.qdrant_url,
        collection: str = settings.qdrant_collection,
        dim: int = settings.embed_dim,
    ) -> None:
        self.collection = collection
        self.dim = dim
        # Embedded-режим имеет приоритет: если задан путь — работаем on-disk.
        if settings.qdrant_path:
            self.client = QdrantClient(path=settings.qdrant_path)
            logger.info("QdrantStore: embedded режим, path=%s", settings.qdrant_path)
        else:
            self.client = QdrantClient(url=url)
            logger.info("QdrantStore: HTTP режим, url=%s", url)

    def ensure_collection(self, recreate: bool = False) -> None:
        """Создать коллекцию при отсутствии; пересоздать при несовпадении размерности.

        recreate=True пересоздаёт коллекцию безусловно (удалив старую) — так каждая
        проверка стартует с ЧИСТОЙ коллекции и поиск не находит фрагменты из других
        документов/прошлых прогонов (иначе коллекция накапливает чужие чанки).
        """
        params = VectorParams(size=self.dim, distance=Distance.COSINE)

        exists = False
        try:
            exists = self.client.collection_exists(self.collection)
        except Exception:  # noqa: BLE001 — старый клиент без collection_exists
            try:
                self.client.get_collection(self.collection)
                exists = True
            except Exception:  # noqa: BLE001
                exists = False

        if recreate:
            if exists:
                self.client.delete_collection(self.collection)
            self.client.create_collection(self.collection, vectors_config=params)
            logger.info("Коллекция '%s' пересоздана начисто (dim=%d)", self.collection, self.dim)
            return

        if not exists:
            self.client.create_collection(self.collection, vectors_config=params)
            logger.info("Коллекция '%s' создана (dim=%d)", self.collection, self.dim)
            return

        # Коллекция есть — проверим размерность вектора.
        current_dim = self._current_dim()
        if current_dim is not None and current_dim != self.dim:
            logger.info(
                "Коллекция '%s': размерность %s != %d → пересоздание",
                self.collection,
                current_dim,
                self.dim,
            )
            self.client.delete_collection(self.collection)
            self.client.create_collection(self.collection, vectors_config=params)
        else:
            logger.info(
                "Коллекция '%s' уже существует (dim=%s) — без изменений",
                self.collection,
                current_dim,
            )

    def _current_dim(self) -> int | None:
        """Размерность вектора существующей коллекции или None, если не определить."""
        try:
            info = self.client.get_collection(self.collection)
            vectors = info.config.params.vectors
            # vectors может быть VectorParams (один вектор) или dict именованных.
            if isinstance(vectors, dict):
                first = next(iter(vectors.values()), None)
                return getattr(first, "size", None)
            return getattr(vectors, "size", None)
        except Exception:  # noqa: BLE001 — не смогли прочитать конфиг → не трогаем
            return None

    def index_sources(self, chunks: list[dict]) -> None:
        """Проиндексировать чанки источников.

        Каждый chunk = {"chunk_id", "source_id", "text", "vector"}. Id точки —
        детерминированный UUID5 от chunk_id (Qdrant требует unsigned int или UUID).
        Upsert идёт батчами.
        """
        t0 = time.perf_counter()
        if not chunks:
            logger.info("index_sources: пустой вход — нечего индексировать")
            return

        points: list[PointStruct] = []
        for ch in chunks:
            chunk_id = ch["chunk_id"]
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))
            points.append(
                PointStruct(
                    id=point_id,
                    vector=ch["vector"],
                    payload={
                        "source_id": ch["source_id"],
                        "text": ch["text"],
                        "chunk_id": chunk_id,
                        "location": ch.get("location"),
                    },
                )
            )

        for start in range(0, len(points), _BATCH_SIZE):
            batch = points[start : start + _BATCH_SIZE]
            self.client.upsert(collection_name=self.collection, points=batch)

        logger.info(
            "index_sources: проиндексировано %d чанков за %.2f c",
            len(points),
            time.perf_counter() - t0,
        )

    def search(
        self,
        query_vector: list[float],
        top_k: int = settings.top_k,
    ) -> list[Evidence]:
        """Найти top_k ближайших чанков. quote — полный текст чанка-кандидата."""
        t0 = time.perf_counter()
        # query_points — актуальный API qdrant-client (бывший .search() удалён
        # в свежих версиях клиента). Возвращает QueryResponse с .points.
        hits = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        ).points

        results: list[Evidence] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                Evidence(
                    source_id=payload["source_id"],
                    chunk_id=payload.get("chunk_id", ""),
                    quote=payload["text"],
                    score=hit.score,
                    location=payload.get("location"),
                )
            )

        logger.info(
            "search: найдено %d кандидатов (top_k=%d) за %.3f c",
            len(results),
            top_k,
            time.perf_counter() - t0,
        )
        return results
