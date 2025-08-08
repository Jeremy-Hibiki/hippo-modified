from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger

if TYPE_CHECKING:
    from ..utils.llm_utils import TextChatMessage

logger = get_logger(__name__)


@dataclass
class LLMConfig:
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
    def from_dict(cls, config_dict: dict[str, Any]) -> LLMConfig:
        """Create an LLMConfig instance from a dictionary."""
        instance = cls()
        instance.batch_upsert(config_dict)
        return instance

    @classmethod
    def from_json(cls, json_str: str) -> LLMConfig:
        """Create an LLMConfig instance from a JSON string."""
        instance = cls()
        instance.batch_upsert(json.loads(json_str))
        return instance

    def __str__(self) -> str:
        """Provide a user-friendly string representation of the configuration."""
        return json.dumps(self._data, indent=4)


class BaseLLM(ABC):
    """Abstract base class for LLMs."""

    global_config: BaseConfig
    llm_name: str  # Class name indicating which LLM model to use.
    llm_config: LLMConfig  # Store LLM specific config, init and handled by specifc LLM

    def __init__(self, global_config: BaseConfig | None = None) -> None:
        if global_config is None:
            logger.info("global config is not given. Using the default ExperimentConfig instance.")
            self.global_config = BaseConfig()
        else:
            self.global_config = global_config
        logger.info(f"Loading {self.__class__.__name__} with global_config: {asdict(self.global_config)}")

        self.llm_name = self.global_config.llm_name
        logger.info(f"Init {self.__class__.__name__}'s llm_name with: {self.llm_name}")

    @abstractmethod
    def _init_llm_config(self) -> None:
        """
        Each LLM model should extract its own running parameters from global_config and raise exception if any mandatory parameter is not defined in global_config.
        This function must init `self.llm_config`.
        """
        pass

    def batch_upsert_llm_config(self, updates: dict[str, Any]) -> None:
        """
        Upsert self.llm_config with attribute-value pairs specified by a dict.

        Args:
            updates (Dict[str, Any]): a dict to be integrated into self.llm_config.

        Returns:
            None
        """

        self.llm_config.batch_upsert(updates=updates)
        logger.info(
            f"Updated {self.__class__.__name__}'s llm_config with {updates} to eventually obtain llm_config as: {self.llm_config}"
        )

    @abstractmethod
    def infer(
        self,
        messages: list[TextChatMessage],
        **kwargs,
    ) -> tuple[list[TextChatMessage] | str, dict, bool]:
        """
        Perform synchronous inference using the LLM.

        Args:
            messages (List[TextChatMessage]): Input chat history for the LLM.

        Returns:
            Tuple[List[TextChatMessage], dict]: The list of n (number of choices) LLM response message (a single dict of role + content), and additional metadata (all input params including input chat) as a dictionary.
        """
        pass

    async def async_infer(
        self,
        messages: list[TextChatMessage],
        **kwargs,
    ) -> tuple[list[TextChatMessage] | str, dict, bool]:
        """
        Perform asynchronous inference using the LLM.
        """
        return self.infer(messages, **kwargs)

    def batch_infer(
        self, batch_chat: list[list[TextChatMessage]]
    ) -> tuple[list[list[TextChatMessage] | str], list[dict]]:
        """
        Perform batched synchronous inference using the LLM.

        Args:
            batch_chat (List[List[TextChatMessage]]): Input chat history batch for the LLM.

        Returns:
            Tuple[List[List[TextChatMessage]], List[dict]]: The batch list of length-n (number of choices) list of LLM response message (a single dict of role + content), and corresponding batch of additional metadata (all input params including input chat) as a list of dictionaries.
        """

        responses: list[list[TextChatMessage] | str] = []
        metadatas: list[dict] = []
        for chat in batch_chat:
            res, metadata, _ = self.infer(chat)
            responses.append(res)
            metadatas.append(metadata)

        return responses, metadatas


# # Example usage
# if __name__ == "__main__":
#     config = LLMConfig()
#     config.batch_upsert({"learning_rate": 0.001, "batch_size": 32})
#     print(config.to_dict())

#     config.optimizer = "adam"
#     print(config.to_dict())

#     json_config = config.to_json()
#     print(json_config)

#     new_config = LLMConfig.from_json(json_config)
#     print(new_config.to_dict())

#     dict_config = {"dropout": 0.5, "epochs": 10}
#     another_config = LLMConfig.from_dict(dict_config)
#     print(another_config.to_dict())
