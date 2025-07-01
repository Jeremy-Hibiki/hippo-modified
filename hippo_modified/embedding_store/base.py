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
        target_ids: Sequence[str] = None,
        instruction: str = "",
    ) -> np.ndarray:
        """
        在当前向量库进行相似度计算

        Args:
            query_text: 查询文本
            target_ids: 目标ID列表，如果为None则计算与所有存储的向量的相似度
            instruction: 嵌入指令，用于指导向量化过程

        Returns:
            相似度分数数组
        """
        pass

    @abstractmethod
    def internal_cross_knn(self, top_k: int) -> dict:
        """
        执行KNN搜索，基于已存储的ID

        Args:
            top_k: 返回的近邻数量

        Returns:
            字典，键为query_id，值为(top_k_key_ids, top_k_scores)的元组
        """
        pass
