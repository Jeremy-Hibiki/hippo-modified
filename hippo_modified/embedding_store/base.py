from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt


class BaseEmbeddingStore(ABC):
    @abstractmethod
    def insert_strings(self, texts: Sequence[str]) -> dict | None:
        pass

    async def async_insert_strings(self, texts: Sequence[str]) -> dict | None:
        return self.insert_strings(texts)

    @abstractmethod
    def get_row(self, hash_id: str) -> dict:
        pass

    async def async_get_row(self, hash_id: str) -> dict:
        return self.get_row(hash_id)

    @abstractmethod
    def get_rows(self, hash_ids: Sequence[str], dtype: npt.DTypeLike = np.float32) -> dict[str, dict]:
        pass

    async def async_get_rows(self, hash_ids: Sequence[str], dtype: npt.DTypeLike = np.float32) -> dict[str, dict]:
        return self.get_rows(hash_ids, dtype)

    @abstractmethod
    def get_all_ids(self) -> list[str]:
        pass

    @abstractmethod
    def get_all_id_to_rows(self) -> dict[str, dict]:
        pass

    @abstractmethod
    def search(
        self,
        query_text: str,
        instruction: str = "",
        top_k: int = 10,
    ) -> list[tuple[dict, float]]:
        """
        Searches for the top_k most similar texts to the query_text.

        Args:
            query_text: The text to search for.
            instruction: An optional instruction for the embedding model.
            top_k: The number of results to return.

        Returns:
            A list of tuples, where each tuple contains the a document and
            its similarity score. The list is sorted by score in descending order.
        """
        pass

    async def async_search(
        self,
        query_text: str,
        instruction: str = "",
        top_k: int = 10,
    ) -> list[tuple[dict, float]]:
        """
        Async version of search method.
        Default implementation falls back to sync search method.
        Subclasses can override this for true async behavior.

        Args:
            query_text: The text to search for.
            instruction: An optional instruction for the embedding model.
            top_k: The number of results to return.

        Returns:
            A list of tuples, where each tuple contains the a document and
            its similarity score. The list is sorted by score in descending order.
        """
        return self.search(query_text, instruction, top_k)

    @abstractmethod
    def internal_cross_knn(self, top_k: int) -> dict[str, tuple[list[str], list[float]]]:
        """
        执行KNN搜索，基于已存储的ID

        Args:
            top_k: 返回的近邻数量

        Returns:
            字典，键为query_id，值为(top_k_key_ids, top_k_scores)的元组
        """
        pass
