from __future__ import annotations

import ast
import asyncio
import difflib
import json
import logging
import re
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, TypeAdapter

from .prompts.filter_default_prompt import best_dspy_prompt
from .utils.misc_utils import flatten_list

if TYPE_CHECKING:
    from .AsyncHippoRAG import AsyncHippoRAG
    from .HippoRAG import HippoRAG

logger = logging.getLogger(__name__)


class Fact(BaseModel):
    fact: list[list[str]] = Field(
        description="A list of facts, each fact is a list of 3 strings: [subject, predicate, object]"
    )


class DSPyFilter:
    def __init__(self, hipporag: HippoRAG | AsyncHippoRAG):
        """
        Initializes the object with the necessary configurations and templates for processing input and output messages.

        Parameters:
        hipporag : An object that provides the global configuration and the LLM model required for inference.

        Attributes:
        dspy_file_path : The file path for reranking as specified in the global configuration.
        one_input_template : A string template for formatting the input message with placeholders for specific fields.
        one_output_template : A string template for formatting the output message with specific fields.
        message_template : A template generated using the specified dspy file path.
        llm_infer_fn : A function reference for making inferences using the provided LLM model.
        model_name : The name of the language model as specified in the global configuration.
        default_gen_kwargs : A dictionary for storing the default generation keyword arguments.
        """
        dspy_file_path = hipporag.global_config.rerank_dspy_file_path
        self.one_input_template = """[[ ## question ## ]]\n{question}\n\n[[ ## fact_before_filter ## ]]\n{fact_before_filter}\n\nRespond with the corresponding output fields, starting with the field `[[ ## fact_after_filter ## ]]` (must be formatted as a valid Python Fact), and then ending with the marker for `[[ ## completed ## ]]`."""
        self.one_output_template = """[[ ## fact_after_filter ## ]]\n{fact_after_filter}\n\n[[ ## completed ## ]]"""
        self.rerank_num = 5
        self.message_template = self.make_template(dspy_file_path, num=self.rerank_num)
        self.llm_infer_fn = hipporag.llm_model.infer
        self.llm_async_infer_fn = hipporag.llm_model.async_infer
        self.model_name = hipporag.global_config.llm_name
        self.default_gen_kwargs: dict[str, Any] = {}

    def make_template(self, dspy_file_path: str | None, num: int = 5) -> list[dict[str, str]]:
        if dspy_file_path is not None:
            with open(dspy_file_path) as f:
                dspy_saved = json.load(f)
        else:
            dspy_saved = best_dspy_prompt

        system_prompt = dspy_saved["prog"]["system"]
        message_template = [
            {"role": "system", "content": system_prompt},
        ]
        demos = dspy_saved["prog"]["demos"]
        for demo in demos:
            message_template.append(
                {
                    "role": "user",
                    "content": self.one_input_template.format(
                        question=demo["question"], fact_before_filter=demo["fact_before_filter"]
                    ),
                }
            )
            message_template.append(
                {
                    "role": "assistant",
                    "content": self.one_output_template.format(fact_after_filter=demo["fact_after_filter"]),
                }
            )
        return message_template

    def parse_filter(self, response: str) -> list[list[str]]:
        sections: list[tuple[str | None, list[str]]] = [(None, [])]
        field_header_pattern = re.compile("\\[\\[ ## (\\w+) ## \\]\\]")
        for line in response.splitlines():
            match = field_header_pattern.match(line.strip())
            if match:
                sections.append((match.group(1), []))
            else:
                sections[-1][1].append(line)

        sections2 = [(k, "\n".join(v).strip()) for k, v in sections]
        parsed = []
        for k, value in sections2:
            if k == "fact_after_filter":
                try:
                    # fields[k] = parse_value(v, signature.output_fields[k].annotation) if _parse_values else v
                    try:
                        parsed_value = json.loads(value)
                    except json.JSONDecodeError:
                        try:
                            parsed_value = ast.literal_eval(value)
                        except (ValueError, SyntaxError):
                            parsed_value = value
                    parsed = TypeAdapter(Fact).validate_python(parsed_value).fact
                except Exception as e:
                    logger.error(
                        f"Error parsing field {k}: {e}.\n\n\t\tOn attempting to parse the value\n```\n{value}\n```"
                    )

        return parsed

    def llm_call(self, question: str, fact_before_filter: str):
        # make prompt
        messages = deepcopy(self.message_template)
        messages.append(
            {
                "role": "user",
                "content": self.one_input_template.format(question=question, fact_before_filter=fact_before_filter),
            }
        )
        # call openai

        if self.default_gen_kwargs.get("max_completion_tokens") is None:
            self.default_gen_kwargs["max_completion_tokens"] = 512

        response = self.llm_infer_fn(messages=messages, model=self.model_name, **self.default_gen_kwargs)

        if len(response) > 1:
            return response[0]
        return response

    async def async_llm_call(self, question: str, fact_before_filter: str):
        # make prompt
        messages = deepcopy(self.message_template)
        messages.append(
            {
                "role": "user",
                "content": self.one_input_template.format(question=question, fact_before_filter=fact_before_filter),
            }
        )
        # call openai

        if self.default_gen_kwargs.get("max_completion_tokens") is None:
            self.default_gen_kwargs["max_completion_tokens"] = 512

        response = await self.llm_async_infer_fn(messages=messages, model=self.model_name, **self.default_gen_kwargs)

        if len(response) > 1:
            return response[0]
        return response

    def __call__(self, *args, **kwargs):
        return self.rerank(*args, **kwargs)

    def rerank(
        self,
        query: str,
        candidate_items: list[tuple],
        candidate_indices: list[int],
        len_after_rerank: int,
        batch_size: int = 10,
    ) -> tuple[list[int], list[tuple], dict]:
        candidate = [list(candidate_item) for candidate_item in candidate_items]
        sorted_candidate_indices = []
        sorted_candidate_items = []
        # logger.info('===' * 10 + " start " + '===' * 10, candidate, sep='\r\n')
        while len(candidate) > 0:
            batch_candidate = candidate[:batch_size]
            # candidate = candidate[batch_size:]
            fact_before_filter = {"fact": batch_candidate}
            try:
                # prediction = self.program(question=query, fact_before_filter=json.dumps(fact_before_filter))
                response = self.llm_call(query, json.dumps(fact_before_filter))
                # logger.info(response, sep='\r\n')
                generated_facts = self.parse_filter(response)
                candidate = candidate[batch_size:]
            except Exception as e:
                logger.error("exception", e)
                if batch_size > 6:
                    batch_size -= 1
                    continue
                else:
                    candidate = candidate[batch_size:]
                generated_facts = []

            result_indices = []
            for generated_fact in generated_facts:
                closest_matched_fact = difflib.get_close_matches(
                    str(generated_fact), [str(i) for i in candidate_items], n=1, cutoff=0.0
                )[0]
                try:
                    result_indices.append(candidate_items.index(ast.literal_eval(closest_matched_fact)))
                except Exception as e:
                    logger.info("result_indices exception", e)

            sorted_candidate_indices.extend([candidate_indices[i] for i in result_indices])
            sorted_candidate_items.extend([candidate_items[i] for i in result_indices])
        # logger.info([], '===' * 10 + " end " + '===' * 10, sep='\r\n')
        return (
            sorted_candidate_indices[:len_after_rerank],
            sorted_candidate_items[:len_after_rerank],
            {"confidence": None},
        )

    async def async_rerank(
        self,
        query: str,
        candidate_items: list[tuple],
        candidate_indices: list[int],
        len_after_rerank: int,
        batch_size: int = 10,
        concurrency: int = 4,
    ) -> tuple[list[int], list[tuple], dict]:
        candidate_lists: list[list[str]] = [list(candidate_item) for candidate_item in candidate_items]

        semaphore = asyncio.Semaphore(concurrency)

        async def _process_one_batch(batch: list[list[str]]):
            """Process one batch with local retry logic that reduces batch_size on JSON errors."""
            local_batch_size = len(batch)
            generated_facts: list[list[str]] = []

            # Retry loop which progressively decreases the batch size in case of parsing errors
            while True:
                fact_before_filter = {"fact": batch[:local_batch_size]}
                try:
                    async with semaphore:
                        response = await self.async_llm_call(query, json.dumps(fact_before_filter))
                    generated_facts = self.parse_filter(response)
                except Exception as e:
                    logger.error("exception", e)

                # Break if we got valid facts or cannot shrink further
                if generated_facts or local_batch_size <= 6:
                    break

                # Otherwise reduce the batch and retry
                local_batch_size -= 1
            return generated_facts

        tasks = [
            asyncio.create_task(_process_one_batch(candidate_lists[i : i + batch_size]))
            for i in range(0, len(candidate_lists), batch_size)
        ]

        generated_facts = flatten_list(await asyncio.gather(*tasks, return_exceptions=False))
        closest_matched_indices = [
            candidate_items.index(
                ast.literal_eval(
                    difflib.get_close_matches(
                        str(generated_fact),
                        [str(i) for i in candidate_items],
                        n=1,
                        cutoff=0.0,
                    )[0]
                )
            )
            for generated_fact in generated_facts
        ]

        sorted_candidate_indices = [candidate_indices[i] for i in closest_matched_indices]
        sorted_candidate_items = [candidate_items[i] for i in closest_matched_indices]

        return (
            sorted_candidate_indices[:len_after_rerank],
            sorted_candidate_items[:len_after_rerank],
            {"confidence": None},
        )
