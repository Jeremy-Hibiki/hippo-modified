import json
import multiprocessing
import threading
from dataclasses import asdict, dataclass, field
from typing import Any
from typing_extensions import Self

import numpy as np
import numpy.typing as npt

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class EmbeddingConfig:
    _data: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __getattr__(self, key: str) -> Any:
        # Define patterns to ignore for Jupyter/IPython-related attributes
        ignored_prefixes = ("_ipython_", "_repr_")
        if any(key.startswith(prefix) for prefix in ignored_prefixes):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

        if key in self._data:
            return self._data[key]

        logger.error(f"'{self.__class__.__name__}' object has no attribute '{key}'")
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        if key == "_data":
            super().__setattr__(key, value)
        else:
            self._data[key] = value

    def __delattr__(self, key: str) -> None:
        if key in self._data:
            del self._data[key]
        else:
            logger.error(f"'{self.__class__.__name__}' object has no attribute '{key}'")
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __getitem__(self, key: str) -> Any:
        """Allow dict-style key lookup."""
        if key in self._data:
            return self._data[key]
        logger.error(f"'{key}' not found in configuration.")
        raise KeyError(f"'{key}' not found in configuration.")

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dict-style key assignment."""
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        """Allow dict-style key deletion."""
        if key in self._data:
            del self._data[key]
        else:
            logger.error(f"'{key}' not found in configuration.")
            raise KeyError(f"'{key}' not found in configuration.")

    def __contains__(self, key: str) -> bool:
        """Allow usage of 'in' to check for keys."""
        return key in self._data

    def batch_upsert(self, updates: dict[str, Any]) -> None:
        """Update existing attributes or add new ones from the given dictionary."""
        self._data.update(updates)

    def to_dict(self) -> dict[str, Any]:
        """Export the configuration as a JSON-serializable dictionary."""
        return self._data

    def to_json(self) -> str:
        """Export the configuration as a JSON string."""
        return json.dumps(self._data)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> Self:
        """Create an LLMConfig instance from a dictionary."""
        instance = cls()
        instance.batch_upsert(config_dict)
        return instance

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Create an LLMConfig instance from a JSON string."""
        instance = cls()
        instance.batch_upsert(json.loads(json_str))
        return instance

    def __str__(self) -> str:
        """Provide a user-friendly string representation of the configuration."""
        return json.dumps(self._data, indent=4)


class BaseEmbeddingModel:
    global_config: BaseConfig
    embedding_model_name: str  # Class name indicating which embedding model to use.
    embedding_config: EmbeddingConfig

    embedding_dim: int  # Need subclass to init

    def __init__(self, global_config: BaseConfig | None = None) -> None:
        if global_config is None:
            logger.info("global config is not given. Using the default ExperimentConfig instance.")
            self.global_config = BaseConfig()
        else:
            self.global_config = global_config
        logger.info(f"Loading {self.__class__.__name__} with global_config: {asdict(self.global_config)}")

        self.embedding_model_name = self.global_config.embedding_model_name

        logger.info(f"Init {self.__class__.__name__}'s embedding_model_name with: {self.embedding_model_name}")

    def batch_encode(self, texts: list[str], **kwargs) -> npt.NDArray:
        raise NotImplementedError

    async def async_batch_encode(self, texts: list[str], **kwargs) -> npt.NDArray:
        return self.batch_encode(texts, **kwargs)

    def get_query_doc_scores(self, query_vec: np.ndarray, doc_vecs: np.ndarray):
        # """
        # @param query_vec: query vector
        # @param doc_vecs: doc matrix
        # @return: a matrix of query-doc scores
        # """
        return np.dot(query_vec, doc_vecs.T)


class EmbeddingCache:
    """A multiprocessing-safe global cache for storing embeddings."""

    _manager = multiprocessing.Manager()
    _cache = _manager.dict()  # Shared dictionary for multiprocessing
    _lock = threading.Lock()  # Thread-safe lock for concurrent access

    @classmethod
    def get(cls, content):
        """Retrieve the embedding if cached."""
        return cls._cache.get(content)

    @classmethod
    def set(cls, content, embedding):
        """Store an embedding in the cache."""
        with cls._lock:  # Ensures thread safety
            cls._cache[content] = embedding

    @classmethod
    def contains(cls, content):
        """Check if the embedding exists in cache."""
        return content in cls._cache

    @classmethod
    def clear(cls):
        """Clear the entire cache."""
        with cls._lock:
            cls._cache.clear()
