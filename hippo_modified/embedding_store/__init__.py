import os.path

from ..embedding_model.base import BaseEmbeddingModel
from ..utils.config_utils import BaseConfig
from .base import BaseEmbeddingStore
from .dataframe import DataFrameEmbeddingStore
from .milvus import MilvusEmbeddingStore


def create_embedding_store(
    embedding_model: BaseEmbeddingModel,
    db_name: str,
    global_config: BaseConfig,
    batch_size: int,
    namespace: str = "default",
) -> BaseEmbeddingStore:
    """
    根据配置创建相应的嵌入存储实例

    参数:
    embedding_model: 嵌入模型实例
    config: 配置对象
    namespace: 命名空间（默认为"default"）

    返回:
    BaseEmbeddingStore: 嵌入存储实例
    """
    if global_config.embedding_store_type == "dataframe":
        return DataFrameEmbeddingStore(
            global_config=global_config,
            embedding_model=embedding_model,
            namespace=namespace,
            batch_size=batch_size,
            db_filename=db_name,
        )

    elif global_config.embedding_store_type == "milvus":
        return MilvusEmbeddingStore(
            global_config=global_config,
            embedding_model=embedding_model,
            namespace=namespace,
            uri=global_config.milvus_uri,
            token=global_config.milvus_token,
            db_name=os.path.split(db_name)[-1],
            collection_prefix=global_config.milvus_collection_prefix,
            enable_hybrid_search=global_config.milvus_enable_hybrid_search,
        )

    else:
        raise ValueError(f"不支持的嵌入存储类型: {global_config.embedding_store_type}")


__all__ = ["BaseEmbeddingStore", "DataFrameEmbeddingStore", "MilvusEmbeddingStore", "create_embedding_store"]
