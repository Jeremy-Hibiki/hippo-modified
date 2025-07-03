from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt


class BaseEmbeddingStore(ABC):
    @abstractmethod
    def insert_strings(self, texts: Sequence[str]) -> dict | None:
        pass

    @abstractmethod
    def get_row(self, hash_id: str) -> dict:
        pass

    @abstractmethod
    def get_rows(self, hash_ids: Sequence[str], dtype: npt.DTypeLike = np.float32) -> dict[str, dict]:
        pass

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
    ) -> np.ndarray:
        pass

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
