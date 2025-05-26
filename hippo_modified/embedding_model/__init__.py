# from .TextEmbeddingsInference import TextEmbeddingsInferenceModel
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig
from .Contriever import ContrieverModel
from .GritLM import GritLMEmbeddingModel
from .NVEmbedV2 import NVEmbedV2EmbeddingModel
from .OpenAI import OpenAIEmbeddingModel

logger = get_logger(__name__)


def _get_embedding_model_class(embedding_model_name: str = "nvidia/NV-Embed-v2"):
    if "GritLM" in embedding_model_name:
        return GritLMEmbeddingModel
    elif "NV-Embed-v2" in embedding_model_name:
        return NVEmbedV2EmbeddingModel
    elif "contriever" in embedding_model_name:
        return ContrieverModel
    elif "bge" in embedding_model_name or "text-embedding" in embedding_model_name:
        return OpenAIEmbeddingModel
    raise ValueError(f"Unknown embedding model name: {embedding_model_name}")


__all__ = [
    "BaseEmbeddingModel",
    "ContrieverModel",
    "EmbeddingConfig",
    "GritLMEmbeddingModel",
    "NVEmbedV2EmbeddingModel",
    "OpenAIEmbeddingModel",
    "_get_embedding_model_class",
]
