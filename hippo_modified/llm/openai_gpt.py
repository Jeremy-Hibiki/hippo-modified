import hashlib
import inspect
import json
import os
import sqlite3
from collections.abc import Callable
from copy import deepcopy
from typing import TypeVar

import aiosqlite
import httpx
import openai
import wrapt
from filelock import AsyncFileLock, FileLock
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
from packaging import version
from tenacity import retry, stop_after_attempt, wait_fixed

from ..utils.config_utils import BaseConfig
from ..utils.llm_utils import TextChatMessage
from ..utils.logging_utils import get_logger
from .base import BaseLLM, LLMConfig

logger = get_logger(__name__)

_R = TypeVar("_R")


def cache_response(func: Callable[..., _R]) -> Callable[..., _R]:
    def _build_cache_key(instance, args, kwargs):
        messages = args[0] if args else kwargs.get("messages")
        if messages is None:
            raise ValueError("Missing required 'messages' parameter for caching.")

        gen_params = getattr(instance, "llm_config", {}).generate_params if hasattr(instance, "llm_config") else {}
        model = kwargs.get("model", gen_params.get("model"))
        seed = kwargs.get("seed", gen_params.get("seed"))
        temperature = kwargs.get("temperature", gen_params.get("temperature"))
        max_completion_tokens = kwargs.get("max_completion_tokens", gen_params.get("max_completion_tokens"))

        key_data = {
            "messages": messages,
            "model": model,
            "seed": seed,
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()
        return key_hash

    @wrapt.decorator
    def sync_wrapper(wrapped, instance, args, kwargs):
        if not instance.cache_llm_response:
            message, metadata = wrapped(*args, **kwargs)
            return message, metadata, False

        # get messages from args or kwargs
        messages = args[0] if args else kwargs.get("messages")
        if messages is None:
            raise ValueError("Missing required 'messages' parameter for caching.")

        key_hash = _build_cache_key(instance, args, kwargs)

        # the file name of lock, ensure mutual exclusion when accessing concurrently
        lock_file = instance.cache_file_name + ".lock"

        # Try to read from SQLite cache
        with FileLock(lock_file), sqlite3.connect(instance.cache_file_name) as conn:
            c = conn.cursor()
            # if the table does not exist, create it
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            conn.commit()  # commit to save the table creation
            c.execute("SELECT message, metadata FROM cache WHERE key = ?", (key_hash,))
            row = c.fetchone()
            if row is not None:
                message, metadata_str = row
                metadata = json.loads(metadata_str)
                # return cached result and mark as hit
                return message, metadata, True

        # if cache miss, call the original function to get the result
        result = wrapped(*args, **kwargs)
        message, metadata = result
        if not isinstance(message, str):
            return message, metadata, False

        # insert new result into cache
        with FileLock(lock_file), sqlite3.connect(instance.cache_file_name) as conn:
            c = conn.cursor()
            # make sure the table exists again (if it doesn't exist, it would be created)
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            metadata_str = json.dumps(metadata)
            c.execute(
                "INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?, ?, ?)",
                (key_hash, message, metadata_str),
            )
            conn.commit()

        return message, metadata, False

    @wrapt.decorator
    async def async_wrapper(wrapped, instance, args, kwargs):
        if not instance.cache_llm_response:
            message, metadata = await wrapped(*args, **kwargs)
            return message, metadata, False

        # get messages from args or kwargs
        messages = args[0] if args else kwargs.get("messages")
        if messages is None:
            raise ValueError("Missing required 'messages' parameter for caching.")

        key_hash = _build_cache_key(instance, args, kwargs)

        # the file name of lock, ensure mutual exclusion when accessing concurrently
        lock_file = instance.cache_file_name + ".lock"

        # Try to read from SQLite cache (using aiosqlite for async)
        async with AsyncFileLock(lock_file), aiosqlite.connect(instance.cache_file_name) as conn:
            # if the table does not exist, create it
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            await conn.commit()  # commit to save the table creation

            async with conn.execute("SELECT message, metadata FROM cache WHERE key = ?", (key_hash,)) as cursor:
                row = await cursor.fetchone()

            if row is not None:
                message, metadata_str = row
                metadata = json.loads(metadata_str)
                # return cached result and mark as hit
                return message, metadata, True

        # if cache miss, call the original async function to get the result
        result = await wrapped(*args, **kwargs)
        message, metadata = result
        if not isinstance(message, str):
            return message, metadata, False

        # insert new result into cache
        async with AsyncFileLock(lock_file), aiosqlite.connect(instance.cache_file_name) as conn:
            # make sure the table exists again (if it doesn't exist, it would be created)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            metadata_str = json.dumps(metadata)
            await conn.execute(
                "INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?, ?, ?)",
                (key_hash, message, metadata_str),
            )
            await conn.commit()

        return message, metadata, False

    if inspect.iscoroutinefunction(func):
        return async_wrapper(func)
    else:
        return sync_wrapper(func)


def dynamic_retry_decorator(func: Callable[..., _R]) -> Callable[..., _R]:
    @wrapt.decorator
    def sync_wrapper(wrapped, instance, args, kwargs):
        max_retries = getattr(instance, "max_retries", 5)
        dynamic_retry = retry(stop=stop_after_attempt(max_retries), wait=wait_fixed(1))
        decorated_func = dynamic_retry(wrapped)
        return decorated_func(*args, **kwargs)

    @wrapt.decorator
    async def async_wrapper(wrapped, instance, args, kwargs):
        max_retries = getattr(instance, "max_retries", 5)
        dynamic_retry = retry(stop=stop_after_attempt(max_retries), wait=wait_fixed(1))
        decorated_func = dynamic_retry(wrapped)
        return await decorated_func(*args, **kwargs)

    # 根据被装饰函数的类型返回对应的装饰器
    if inspect.iscoroutinefunction(func):
        return async_wrapper(func)
    else:
        return sync_wrapper(func)


class CacheOpenAI(BaseLLM):
    """OpenAI LLM implementation."""

    @classmethod
    def from_experiment_config(cls, global_config: BaseConfig) -> "CacheOpenAI":
        config_dict = global_config.__dict__
        config_dict["max_retries"] = global_config.max_retry_attempts
        cache_dir = os.path.join(global_config.save_dir, "llm_cache")
        return cls(cache_dir=cache_dir, global_config=global_config)

    def __init__(
        self,
        cache_dir: str,
        global_config: BaseConfig,
        cache_filename: str = None,
        high_throughput: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.cache_dir = cache_dir
        self.global_config = global_config

        self.llm_name = global_config.llm_name
        self.llm_base_url = global_config.llm_base_url
        self.cache_llm_response = global_config.cache_llm_response

        os.makedirs(self.cache_dir, exist_ok=True)
        if cache_filename is None:
            cache_filename = f"{self.llm_name.replace('/', '_')}_cache.sqlite"
        self.cache_file_name = os.path.join(self.cache_dir, cache_filename)

        self._init_llm_config()
        if high_throughput:
            limits = httpx.Limits(max_connections=500, max_keepalive_connections=100)
            client = httpx.Client(limits=limits, timeout=httpx.Timeout(5 * 60))
            async_client = httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(5 * 60))
        else:
            client = None
            async_client = None

        self.max_retries = kwargs.get("max_retries", 2)

        if self.global_config.azure_endpoint is None:
            self.openai_client = OpenAI(base_url=self.llm_base_url, http_client=client, max_retries=self.max_retries)
            self.async_openai_client = AsyncOpenAI(
                base_url=self.llm_base_url, http_client=async_client, max_retries=self.max_retries
            )
        else:
            self.openai_client = AzureOpenAI(
                api_version=self.global_config.azure_endpoint.split("api-version=")[1],
                azure_endpoint=self.global_config.azure_endpoint,
                max_retries=self.max_retries,
            )
            self.async_openai_client = AsyncAzureOpenAI(
                api_version=self.global_config.azure_endpoint.split("api-version=")[1],
                azure_endpoint=self.global_config.azure_endpoint,
                max_retries=self.max_retries,
            )

    def _init_llm_config(self) -> None:
        config_dict = self.global_config.__dict__

        config_dict["llm_name"] = self.global_config.llm_name
        config_dict["llm_base_url"] = self.global_config.llm_base_url
        config_dict["generate_params"] = {
            "model": self.global_config.llm_name,
            "max_completion_tokens": config_dict.get("max_completion_tokens", 65536),
            # "max_new_tokens": config_dict.get("max_new_tokens", 8192),
            "n": config_dict.get("num_gen_choices", 1),
            "seed": config_dict.get("seed", 0),
            "temperature": config_dict.get("temperature", 0.0),
        }

        self.llm_config = LLMConfig.from_dict(config_dict=config_dict)
        logger.info(f"Init {self.__class__.__name__}'s llm_config: {self.llm_config}")

    @cache_response
    @dynamic_retry_decorator
    def infer(self, messages: list[TextChatMessage], **kwargs) -> tuple[str, dict]:
        params = deepcopy(self.llm_config.generate_params)
        if kwargs:
            params.update(kwargs)
        logger.info(f"Calling OpenAI GPT API with:\n{params}")
        params["messages"] = messages

        if (
            "gpt" not in params["model"] or version.parse(openai.__version__) < version.parse("1.45.0")
        ):  # if we use vllm to call openai api or if we use openai but the version is too old to use 'max_completion_tokens' argument
            # TODO strange version change in openai protocol, but our current vllm version not changed yet
            params["max_tokens"] = params.pop("max_completion_tokens")

        response = self.openai_client.chat.completions.create(**params)
        stream = params.pop("stream", False)
        if stream:
            return response, {"prompt_tokens": 0, "completion_tokens": 0, "finish_reason": "0"}
        response_message = response.choices[0].message.content
        assert isinstance(response_message, str), "response_message should be a string"

        metadata = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "finish_reason": response.choices[0].finish_reason,
        }

        return response_message, metadata

    @cache_response
    @dynamic_retry_decorator
    async def async_infer(self, messages: list[TextChatMessage], **kwargs) -> tuple[str, dict]:
        params = deepcopy(self.llm_config.generate_params)
        if kwargs:
            params.update(kwargs)
        logger.info(f"Calling OpenAI GPT API with:\n{params}")
        params["messages"] = messages

        if (
            "gpt" not in params["model"] or version.parse(openai.__version__) < version.parse("1.45.0")
        ):  # if we use vllm to call openai api or if we use openai but the version is too old to use 'max_completion_tokens' argument
            # TODO strange version change in openai protocol, but our current vllm version not changed yet
            params["max_tokens"] = params.pop("max_completion_tokens")

        response = await self.async_openai_client.chat.completions.create(**params)
        stream = params.pop("stream", False)
        if stream:
            return response, {"prompt_tokens": 0, "completion_tokens": 0, "finish_reason": "0"}
        response_message = response.choices[0].message.content
        assert isinstance(response_message, str), "response_message should be a string"

        metadata = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "finish_reason": response.choices[0].finish_reason,
        }

        return response_message, metadata
