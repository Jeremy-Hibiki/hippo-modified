from __future__ import annotations

import logging
from collections.abc import Sequence
from typing_extensions import override
from uuid import uuid4

import numpy as np

from ..embedding_model import BaseEmbeddingModel
from ..utils.config_utils import BaseConfig
from ..utils.misc_utils import compute_mdhash_id, load_hit_stopwords
from .base import BaseEmbeddingStore

logger = logging.getLogger(__name__)


class MilvusEmbeddingStore(BaseEmbeddingStore):
    def __init__(
        self,
        global_config: BaseConfig,
        embedding_model: BaseEmbeddingModel,
        namespace: str,
        uri: str = "http://localhost:19530",
        token: str = "",
        db_name: str = "default",
        collection_prefix: str = "hippo",
        enable_hybrid_search: bool = True,
    ):
        try:
            from pymilvus import AsyncMilvusClient, Collection, MilvusClient
        except ImportError:
            raise ImportError("Please install `pymilvus` to use MilvusEmbeddingStore") from None

        self.global_config = global_config
        self._embedding_model = embedding_model
        self._namespace = namespace
        self._db_name = db_name
        self._collection_name = f"{collection_prefix}_{namespace}"
        self._enable_hybrid_search = enable_hybrid_search

        self._client = MilvusClient(uri=uri, token=token, db_name=db_name, alias=uuid4().hex)
        self._async_client = AsyncMilvusClient(uri=uri, token=token, db_name=db_name, alias=uuid4().hex)

        self._setup_collection()

        self._col = Collection(self._collection_name, using=self._client._using)

        logger.info(f"Initialized MilvusEmbeddingStore for namespace: {namespace}")

    @property
    def client(self):
        return self._client

    @property
    def async_client(self):
        return self._async_client

    def _setup_collection(self):
        try:
            if self.client.has_collection(collection_name=self._collection_name):
                logger.info(f"Using existing collection: {self._collection_name}")
            else:
                self._create_collection()
        except Exception as e:
            logger.error(f"Failed to setup collection: {e}")
            raise

    def _create_collection(self):
        from pymilvus import DataType

        schema = self.client.create_schema(enable_dynamic_field=True)
        index_params = self.client.prepare_index_params()

        schema.add_field(
            field_name="hash_id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=128,
        )
        index_params.add_index(field_name="hash_id", index_type="AUTOINDEX")

        schema.add_field(
            field_name="embedding",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.global_config.milvus_dense_embedding_dim,
        )
        index_params.add_index(field_name="embedding", metric_type="IP", index_type="FLAT")

        try:
            from pymilvus import Function, FunctionType

            schema.add_field(
                field_name="content",
                datatype=DataType.VARCHAR,
                max_length=65535,
                enable_analyzer=True,
                analyzer_params={
                    "tokenizer": "jieba",
                    "filter": [
                        "asciifolding",
                        "cnalphanumonly",
                        "lowercase",
                        {"type": "stemmer", "language": "english"},
                        {"type": "stop", "stop_words": ["_english_", *load_hit_stopwords()]},
                    ],
                },
                enable_match=True,
            )

            schema.add_field(field_name="sparse_embedding", datatype=DataType.SPARSE_FLOAT_VECTOR)
            schema.add_function(
                Function(
                    name="bm25",
                    function_type=FunctionType.BM25,
                    input_field_names=["content"],
                    output_field_names=["sparse_embedding"],
                )
            )
            index_params.add_index(
                field_name="sparse_embedding",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
            )
        except ImportError:
            logger.warning("Current pymilvus version does not support native bm25")
            schema.add_field(
                field_name="content",
                datatype=DataType.VARCHAR,
                max_length=65535,
            )

        self.client.create_collection(
            collection_name=self._collection_name,
            schema=schema,
            index_params=index_params,
        )

    def get_row(self, hash_id: str) -> dict:
        results = self.client.get(ids=[hash_id], collection_name=self._collection_name)

        if not results:
            raise KeyError(f"Hash ID {hash_id} not found")

        return results[0]

    async def async_get_row(self, hash_id: str) -> dict:
        results = await self.async_client.get(ids=[hash_id], collection_name=self._collection_name)

        if not results:
            raise KeyError(f"Hash ID {hash_id} not found")

        return results[0]

    def get_rows(self, hash_ids: Sequence[str], dtype=np.float32) -> dict[str, dict]:
        if not hash_ids:
            return {}

        results = self.client.get(ids=list(hash_ids), collection_name=self._collection_name)

        return {result["hash_id"]: result for result in results}

    async def async_get_rows(self, hash_ids: Sequence[str], dtype=np.float32) -> dict[str, dict]:
        if not hash_ids:
            return {}

        results = await self.async_client.get(ids=list(hash_ids), collection_name=self._collection_name)

        return {result["hash_id"]: result for result in results}

    def get_hash_id(self, text: str) -> str:
        return compute_mdhash_id(text, prefix=self._namespace + "-")

    def get_all_ids(self) -> list[str]:
        iterator = self._col.query_iterator(
            batch_size=2048,
            expr="hash_id is not null",
            output_fields=["hash_id"],
        )
        results = []
        while result := iterator.next():
            results += result
        iterator.close()

        return list({result["hash_id"] for result in results})

    def get_all_id_to_rows(self) -> dict:
        iterator = self._col.query_iterator(
            batch_size=2048,
            expr="hash_id is not null",
            output_fields=["hash_id", "content"],
        )
        results = []
        while result := iterator.next():
            results += result
        iterator.close()

        return {result["hash_id"]: result for result in results}

    def get_embeddings(self, hash_ids: Sequence[str], dtype=np.float32) -> list[np.ndarray]:
        if not hash_ids:
            return []

        results = self.client.query(
            ids=list(hash_ids),
            collection_name=self._collection_name,
            output_fields=["hash_id", "embedding"],
        )

        # 按原始顺序排列结果
        id_to_embedding = {result["hash_id"]: np.array(result["embedding"], dtype=dtype) for result in results}

        return [id_to_embedding[hash_id] for hash_id in hash_ids if hash_id in id_to_embedding]

    def insert_strings(self, texts: Sequence[str]) -> dict | None:
        nodes_dict = {}
        for text in texts:
            nodes_dict[compute_mdhash_id(text, prefix=self._namespace + "-")] = {"content": text}

        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return None  # Nothing to insert.

        embeddings = self._embedding_model.batch_encode(list(texts))
        for hash_id, embedding in zip(all_hash_ids, embeddings, strict=True):
            nodes_dict[hash_id]["hash_id"] = hash_id
            nodes_dict[hash_id]["embedding"] = embedding.tolist()
        data = list(nodes_dict.values())
        self.client.upsert(
            collection_name=self._collection_name,
            data=data,
        )
        return None

    @override
    async def async_insert_strings(self, texts: Sequence[str]) -> dict | None:
        nodes_dict = {}
        for text in texts:
            nodes_dict[compute_mdhash_id(text, prefix=self._namespace + "-")] = {"content": text}

        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return None  # Nothing to insert.

        embeddings = await self._embedding_model.async_batch_encode(list(texts))
        for hash_id, embedding in zip(all_hash_ids, embeddings, strict=True):
            nodes_dict[hash_id]["hash_id"] = hash_id
            nodes_dict[hash_id]["embedding"] = embedding.tolist()
        data = list(nodes_dict.values())
        self.async_client.upsert(
            collection_name=self._collection_name,
            data=data,
        )
        return None

    def search(
        self,
        query_text: str,
        instruction: str = "",
        top_k: int = 10,
    ) -> list[tuple[dict, float]]:
        if self.global_config.embedding_use_instruction:
            text_to_embed = self.global_config.embedding_instruction_format.format(
                instruction=instruction, text=query_text
            )
        else:
            text_to_embed = query_text
        query_embedding = self._embedding_model.batch_encode([text_to_embed])[0]
        if not self._enable_hybrid_search:
            results = self.client.search(
                self._collection_name,
                data=[query_embedding],
                anns_field="embedding",
                limit=top_k,
                output_fields=["hash_id", "content", "embedding"],
            )
        else:
            try:
                from pymilvus import AnnSearchRequest, RRFRanker
            except ImportError:
                raise ImportError("Please upgrade `pymilvus` for hybrid search") from None

            dense_req = AnnSearchRequest(
                [query_embedding],
                "embedding",
                {"metric_type": "IP"},
                limit=top_k * 4,
            )
            sparse_req = AnnSearchRequest(
                [query_text],
                "sparse_embedding",
                {"metric_type": "BM25"},
                limit=top_k * 4,
            )
            try:
                results = self.client.hybrid_search(
                    self._collection_name,
                    [dense_req, sparse_req],
                    ranker=RRFRanker(),
                    limit=top_k,
                    output_fields=["hash_id", "content", "embedding"],
                )
            except Exception as e:
                logger.error(f"Hybrid search failed: {e}")
                raise e
        return [(hit["entity"], float(hit["distance"])) for hit in results[0]]

    @override
    async def async_search(
        self,
        query_text: str,
        instruction: str = "",
        top_k: int = 10,
    ) -> list[tuple[dict, float]]:
        if self.global_config.embedding_use_instruction:
            text_to_embed = self.global_config.embedding_instruction_format.format(
                instruction=instruction, text=query_text
            )
        else:
            text_to_embed = query_text
        query_embedding = (await self._embedding_model.async_batch_encode([text_to_embed]))[0]
        if not self._enable_hybrid_search:
            results = await self.async_client.search(
                self._collection_name,
                data=[query_embedding],
                anns_field="embedding",
                limit=top_k,
                output_fields=["hash_id", "content", "embedding"],
            )
        else:
            try:
                from pymilvus import AnnSearchRequest, RRFRanker
            except ImportError:
                raise ImportError("Please upgrade `pymilvus` for hybrid search") from None

            dense_req = AnnSearchRequest(
                [query_embedding],
                "embedding",
                {"metric_type": "IP"},
                limit=top_k * 4,
            )
            sparse_req = AnnSearchRequest(
                [query_text],
                "sparse_embedding",
                {"metric_type": "BM25"},
                limit=top_k * 4,
            )
            try:
                results = await self.async_client.hybrid_search(
                    self._collection_name,
                    [dense_req, sparse_req],
                    ranker=RRFRanker(),
                    limit=top_k,
                    output_fields=["hash_id", "content", "embedding"],
                )
            except Exception as e:
                logger.error(f"Hybrid search failed: {e}")
                raise e
        return [(hit["entity"], float(hit["distance"])) for hit in results[0]]

    def internal_cross_knn(self, top_k: int) -> dict[str, tuple[list[str], list[float]]]:
        all_ids = self.get_all_ids()
        all_rows = self.get_all_id_to_rows()
        if len(all_rows) == 0:
            return {}

        vecs = self._embedding_model.batch_encode([v["content"] for v in all_rows.values()]).astype(np.float32)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

        similarity = np.dot(vecs, vecs.T)

        actual_k = min(top_k, similarity.shape[1])

        results = {}
        for i in range(len(all_ids)):
            query_id = all_ids[i]

            # 获取该query的相似度分数，按降序排序
            query_similarities = similarity[i]
            topk_indices = np.argsort(query_similarities)[-actual_k:][::-1]  # 最大的k个，降序
            topk_scores = query_similarities[topk_indices]

            # 转换indices为key_ids
            query_topk_key_ids = [all_ids[idx] for idx in topk_indices]

            results[query_id] = (query_topk_key_ids, topk_scores.tolist())

        return results
