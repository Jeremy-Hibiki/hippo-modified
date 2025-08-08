# from .TextEmbeddingsInference import TextEmbeddingsInferenceModel
from __future__ import annotations

from typing import Literal

from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig
from .OpenAI import OpenAIEmbeddingModel

logger = get_logger(__name__)


def get_embedding_model_class(
    embedding_type: Literal["openai"] = "openai",
    embedding_model_name: str = "nvidia/NV-Embed-v2",
):
    if embedding_type == "openai":
        return OpenAIEmbeddingModel
    raise ValueError(f"Unknown embedding model name: {embedding_model_name}")


__all__ = [
    "BaseEmbeddingModel",
    "EmbeddingConfig",
    "OpenAIEmbeddingModel",
    "get_embedding_model_class",
]
