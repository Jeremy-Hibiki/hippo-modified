from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..utils.misc_utils import compute_mdhash_id
from .base import BaseEmbeddingStore

if TYPE_CHECKING:
    from ..embedding_model import BaseEmbeddingModel
    from ..utils.config_utils import BaseConfig

logger = logging.getLogger(__name__)


class DataFrameEmbeddingStore(BaseEmbeddingStore):
    def __init__(
        self,
        global_config: BaseConfig,
        embedding_model: BaseEmbeddingModel,
        db_filename: str,
        batch_size: int,
        namespace: str,
    ):
        """
        Initializes the class with necessary configurations and sets up the working directory.

        Parameters:
        global_config
        embedding_model: The model used for embeddings.
        db_filename: The directory path where data will be stored or retrieved.
        batch_size: The batch size used for processing.
        namespace: A unique identifier for data segregation.

        Functionality:
        - Assigns the provided parameters to instance variables.
        - Checks if the directory specified by `db_filename` exists.
          - If not, creates the directory and logs the operation.
        - Constructs the filename for storing data in a parquet file format.
        - Calls the method `_load_data()` to initialize the data loading process.
        """
        self.global_config = global_config
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.namespace = namespace

        Path(db_filename).mkdir(parents=True, exist_ok=True)

        self.filename = Path(db_filename) / f"vdb_{self.namespace}.parquet"
        self._load_data()

    def insert_strings(self, texts: Sequence[str], metadatas: Sequence[dict[str, Any]] | None = None) -> dict | None:
        nodes_dict = {}

        # Normalize metadatas to match texts length
        if metadatas is None:
            metadatas = [{}] * len(texts)
        elif len(metadatas) != len(texts):
            raise ValueError(f"Length mismatch: {len(texts)} texts vs {len(metadatas)} metadatas")

        for text, metadata in zip(texts, metadatas, strict=True):
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {"content": text, "metadata": metadata}

        # Get all hash_ids from the input dictionary.
        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return None  # Nothing to insert.

        existing = self.hash_id_to_row.keys()

        # Filter out the missing hash_ids.
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]

        logger.info(
            f"Inserting {len(missing_ids)} new records, {len(all_hash_ids) - len(missing_ids)} records already exist."
        )

        if not missing_ids:
            return {}  # All records already exist.

        # Prepare the texts to encode from the "content" field.
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]
        metadatas_to_encode = [nodes_dict[hash_id]["metadata"] for hash_id in missing_ids]

        assert self.embedding_model is not None, "Embedding model is not initialized."

        missing_embeddings = self.embedding_model.batch_encode(texts_to_encode)

        self._upsert(missing_ids, texts_to_encode, missing_embeddings, metadatas_to_encode)

        return None

    def _load_data(self) -> None:
        if Path(self.filename).exists():
            df = pd.read_parquet(self.filename)
            self.hash_ids, self.texts, self.embeddings = (
                df["hash_id"].values.tolist(),
                df["content"].values.tolist(),
                df["embedding"].values.tolist(),
            )

            # Load metadata column if it exists (backward compatibility)
            if "metadata" in df.columns:
                self.metadatas = df["metadata"].apply(lambda x: json.loads(x) if pd.notna(x) else {}).tolist()
            else:
                self.metadatas = [{}] * len(self.hash_ids)  # Backward compat

            self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
            self.hash_id_to_row = {
                h: {"hash_id": h, "content": t, "metadata": m}
                for h, t, m in zip(self.hash_ids, self.texts, self.metadatas, strict=False)
            }
            self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
            self.text_to_hash_id = {self.texts[idx]: h for idx, h in enumerate(self.hash_ids)}
            assert len(self.hash_ids) == len(self.texts) == len(self.embeddings)
            logger.info(f"Loaded {len(self.hash_ids)} records from {self.filename}")
        else:
            self.hash_ids, self.texts, self.embeddings, self.metadatas = [], [], [], []
            self.hash_id_to_idx, self.hash_id_to_row = {}, {}

    def _save_data(self) -> None:
        # Serialize metadata to JSON strings for parquet storage
        data_to_save = pd.DataFrame({
            "hash_id": self.hash_ids,
            "content": self.texts,
            "embedding": self.embeddings,
            "metadata": [json.dumps(m) for m in self.metadatas],
        })
        data_to_save.to_parquet(self.filename, index=False)
        self.hash_id_to_row = {
            h: {"hash_id": h, "content": t, "metadata": m}
            for h, t, m in zip(self.hash_ids, self.texts, self.metadatas, strict=False)
        }
        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
        self.text_to_hash_id = {self.texts[idx]: h for idx, h in enumerate(self.hash_ids)}
        logger.info(f"Saved {len(self.hash_ids)} records to {self.filename}")

    def _upsert(self, hash_ids, texts, embeddings, metadatas=None) -> None:
        self.embeddings.extend(embeddings)
        self.hash_ids.extend(hash_ids)
        self.texts.extend(texts)
        self.metadatas.extend(metadatas if metadatas else [{}] * len(hash_ids))

        logger.info("Saving new records.")
        self._save_data()

    def get_row(self, hash_id) -> dict:
        return self.hash_id_to_row[hash_id]

    def get_rows(self, hash_ids, dtype=np.float32):
        if not hash_ids:
            return {}

        results = {id: self.hash_id_to_row[id] for id in hash_ids}

        return results

    def get_all_ids(self) -> list[str]:
        return deepcopy(self.hash_ids)

    def get_all_id_to_rows(self) -> dict:
        return deepcopy(self.hash_id_to_row)

    def get_all_texts(self) -> set[str]:
        return set(row["content"] for row in self.hash_id_to_row.values())

    def search(
        self,
        query_text: str,
        instruction: str = "",
        top_k: int = 10,
        metadata_filters: dict[str, Any] | None = None,
    ):
        if self.global_config.embedding_use_instruction:
            text_to_embed = self.global_config.embedding_instruction_format.format(
                instruction=instruction, text=query_text
            )
        else:
            text_to_embed = query_text
        query_embedding = self.embedding_model.batch_encode([text_to_embed])[0]
        similarity = np.dot(query_embedding, np.array(self.embeddings).T)

        # Apply metadata filters before ranking
        if metadata_filters:
            filtered_indices = [
                idx for idx, metadata in enumerate(self.metadatas) if self._matches_filters(metadata, metadata_filters)
            ]
            if not filtered_indices:
                return []  # No matches after filtering
            # Only compute similarity for filtered items
            similarity = np.array([similarity[idx] for idx in filtered_indices])
            topk_indices_in_filtered = np.argsort(similarity)[-min(top_k, len(filtered_indices)) :][::-1]
            topk_indices = [filtered_indices[idx] for idx in topk_indices_in_filtered]
        else:
            topk_indices = np.argsort(similarity)[-top_k:][::-1]

        topk_scores = similarity[topk_indices]
        topk_key_ids = [self.hash_ids[idx] for idx in topk_indices]
        return [
            (self.hash_id_to_row[key_id], float(score))
            for key_id, score in zip(topk_key_ids, topk_scores, strict=True)
        ]

    def _matches_filters(self, metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
        """Check if metadata matches all filter criteria."""
        for key, value in filters.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True

    def internal_cross_knn(self, top_k: int) -> dict[str, tuple[list[str], list[float]]]:
        all_ids = self.get_all_ids()
        all_rows = self.get_all_id_to_rows()
        if len(all_rows) == 0:
            return {}

        vecs = self.embedding_model.batch_encode([v["content"] for v in all_rows.values()]).astype(np.float32)
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

    def delete(self, hash_ids: Sequence[str]):
        if not hash_ids:
            return

        # 将输入的hash_ids转换为集合，便于快速查找
        hash_ids_to_delete = set(hash_ids)

        # 找出实际存在的要删除的hash_ids
        existing_hash_ids = set(self.hash_ids)
        actual_deletions = hash_ids_to_delete.intersection(existing_hash_ids)

        if not actual_deletions:
            logger.info("No matching records found for deletion.")
            return

        logger.info(f"Deleting {len(actual_deletions)} records.")

        # 找出要保留的记录的索引
        indices_to_keep = [i for i, hash_id in enumerate(self.hash_ids) if hash_id not in actual_deletions]

        # 1. 使用保留的索引重建主要列表
        self.hash_ids = [self.hash_ids[i] for i in indices_to_keep]
        self.texts = [self.texts[i] for i in indices_to_keep]
        self.embeddings = [self.embeddings[i] for i in indices_to_keep]
        self.metadatas = [self.metadatas[i] for i in indices_to_keep]

        # 2. 重建所有查找字典
        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
        self.text_to_hash_id = {self.texts[idx]: h for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_row = {
            h: {"hash_id": h, "content": t, "metadata": m}
            for h, t, m in zip(self.hash_ids, self.texts, self.metadatas, strict=False)
        }

        # 3. 保存更新后的数据到文件
        self._save_data()

        logger.info(f"Successfully deleted {len(actual_deletions)} records. Remaining records: {len(self.hash_ids)}")
