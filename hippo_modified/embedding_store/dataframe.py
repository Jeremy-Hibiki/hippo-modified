import logging
import os
from collections.abc import Sequence
from copy import deepcopy

import numpy as np
import pandas as pd

from hippo_modified.embedding_model import BaseEmbeddingModel
from hippo_modified.embedding_store.base import BaseEmbeddingStore
from hippo_modified.utils.misc_utils import compute_mdhash_id

logger = logging.getLogger(__name__)


class DataFrameEmbeddingStore(BaseEmbeddingStore):
    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        db_filename: str,
        batch_size: int,
        namespace: str,
    ):
        """
        Initializes the class with necessary configurations and sets up the working directory.

        Parameters:
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
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.namespace = namespace

        if not os.path.exists(db_filename):
            logger.info(f"Creating working directory: {db_filename}")
            os.makedirs(db_filename, exist_ok=True)

        self.filename = os.path.join(db_filename, f"vdb_{self.namespace}.parquet")
        self._load_data()

    def insert_strings(self, texts: Sequence[str]) -> dict | None:
        nodes_dict = {}

        for text in texts:
            # if text == "":
            #    continue
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {"content": text}

        # Get all hash_ids from the input dictionary.
        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return  # Nothing to insert.

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

        assert self.embedding_model is not None, "Embedding model is not initialized."

        missing_embeddings = self.embedding_model.batch_encode(texts_to_encode)

        self._upsert(missing_ids, texts_to_encode, missing_embeddings)

    def _load_data(self) -> None:
        if os.path.exists(self.filename):
            df = pd.read_parquet(self.filename)
            self.hash_ids, self.texts, self.embeddings = (
                df["hash_id"].values.tolist(),
                df["content"].values.tolist(),
                df["embedding"].values.tolist(),
            )
            self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
            self.hash_id_to_row = {
                h: {"hash_id": h, "content": t} for h, t in zip(self.hash_ids, self.texts, strict=False)
            }
            self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
            self.text_to_hash_id = {self.texts[idx]: h for idx, h in enumerate(self.hash_ids)}
            assert len(self.hash_ids) == len(self.texts) == len(self.embeddings)
            logger.info(f"Loaded {len(self.hash_ids)} records from {self.filename}")
        else:
            self.hash_ids, self.texts, self.embeddings = [], [], []
            self.hash_id_to_idx, self.hash_id_to_row = {}, {}

    def _save_data(self) -> None:
        data_to_save = pd.DataFrame({"hash_id": self.hash_ids, "content": self.texts, "embedding": self.embeddings})
        data_to_save.to_parquet(self.filename, index=False)
        self.hash_id_to_row = {
            h: {"hash_id": h, "content": t}
            for h, t, e in zip(self.hash_ids, self.texts, self.embeddings, strict=False)
        }
        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
        self.text_to_hash_id = {self.texts[idx]: h for idx, h in enumerate(self.hash_ids)}
        logger.info(f"Saved {len(self.hash_ids)} records to {self.filename}")

    def _upsert(self, hash_ids, texts, embeddings) -> None:
        self.embeddings.extend(embeddings)
        self.hash_ids.extend(hash_ids)
        self.texts.extend(texts)

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
        target_ids: Sequence[str] = None,
        instruction: str = "",
    ) -> np.ndarray:
        query_embedding = self.embedding_model.batch_encode([query_text])[0]
        return np.dot(query_embedding, np.array(self.embeddings).T)

    def internal_cross_knn(self, top_k: int) -> dict:
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
