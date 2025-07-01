from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from pymilvus import FunctionType

from hippo_modified.embedding_model import BaseEmbeddingModel
from hippo_modified.embedding_store.base import BaseEmbeddingStore
from hippo_modified.utils.misc_utils import compute_mdhash_id, load_hit_stopwords

logger = logging.getLogger(__name__)


class MilvusEmbeddingStore(BaseEmbeddingStore):
    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        namespace: str,
        uri: str = "http://localhost:19530",
        token: str = "",
        db_name: str = "default",
        collection_prefix: str = "hippo",
        enable_hybrid_search: bool = True,
    ):
        try:
            from pymilvus import MilvusClient
        except ImportError:
            raise ImportError("Please install `pymilvus` to use MilvusEmbeddingStore") from None

        self._embedding_model = embedding_model
        self._namespace = namespace
        self._db_name = db_name
        self._collection_name = f"{collection_prefix}_{namespace}"
        self._enable_hybrid_search = enable_hybrid_search

        self._client = MilvusClient(uri=uri, token=token)

        self._setup_collection()

        logger.info(f"Initialized MilvusEmbeddingStore for namespace: {namespace}")

    @property
    def client(self):
        return self._client

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
        from pymilvus import DataType, Function

        schema = self.client.create_schema(enable_dynamic_field=True)
        index_params = self.client.prepare_index_params()

        schema.add_field(
            field_name="hash_id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=128,
        )
        schema.add_field(
            field_name="embedding",
            datatype=DataType.FLOAT_VECTOR,
            dim=self._embedding_model.embedding_dim,
        )
        index_params.add_index(field_name="embedding", metric_type="IP", index_type="FLAT")

        if self._enable_hybrid_search:
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
        else:
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

    def get_rows(self, hash_ids: Sequence[str], dtype=np.float32) -> dict[str, dict]:
        if not hash_ids:
            return {}

        results = self.client.get(ids=list(hash_ids), collection_name=self._collection_name)

        return {result["hash_id"]: result for result in results}

    def get_hash_id(self, text: str) -> str:
        return compute_mdhash_id(text, prefix=self._namespace + "-")

    def get_all_ids(self) -> list[str]:
        from pymilvus import Collection

        iterator = Collection(self._collection_name).query_iterator(
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
        from pymilvus import Collection

        iterator = Collection(self._collection_name).query_iterator(
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
            return  # Nothing to insert.

        embeddings = self._embedding_model.batch_encode(list(texts))
        for hash_id, embedding in zip(all_hash_ids, embeddings, strict=True):
            nodes_dict[hash_id]["embedding"] = embedding.tolist()

        self.client.upsert(
            collection_name=self._collection_name,
            data=nodes_dict,
        )

    def search(
        self,
        query_text: str,
        target_ids: Sequence[str] = None,
        instruction: str = "",
    ) -> np.ndarray:
        # [TODO]
        pass

    def internal_cross_knn(
        self,
        top_k: int,
        query_batch_size: int = 1000,
        key_batch_size: int = 10000,
    ) -> dict:
        pass
