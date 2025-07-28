import os

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseLLM
from .openai_gpt import CacheOpenAI

logger = get_logger(__name__)


def get_llm_class(config: BaseConfig):
    if config.llm_base_url is not None and "localhost" in config.llm_base_url and os.getenv("OPENAI_API_KEY") is None:
        os.environ["OPENAI_API_KEY"] = "sk-"
    return CacheOpenAI.from_experiment_config(config)


__all__ = ["BaseLLM", "CacheOpenAI", "get_llm_class"]
