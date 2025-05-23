from __future__ import annotations

from typing import Any, Callable, List, Optional, Union
from urllib.parse import urljoin

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.rerank import ReRankEndEvent, ReRankStartEvent
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

DEFAULT_URL = "http://127.0.0.1:8080"

dispatcher = get_dispatcher(__name__)


class TextEmbeddingsInferenceRerank(BaseNodePostprocessor):
    base_url: str = Field(
        default=DEFAULT_URL,
        description="Base URL for the text embeddings service.",
    )
    timeout: float = Field(
        default=60.0,
        description="Timeout in seconds for the request.",
    )
    batch_size: int = Field(
        default=32,
        description="Batch size for the inference.",
    )
    top_n: int = Field(default=2, description="Top N nodes to return.")
    raw_score: bool = Field(
        default=False,
        description="Whether to use raw rerank score or sigmoid of the rerank score.",
    )
    truncate: bool = Field(
        default=True,
        description="Whether to truncate the input to satisfy the 'context length limit' on the query and the documents.",
    )
    keep_retrieval_score: bool = Field(
        default=False,
        description="Whether to keep the retrieval score in metadata.",
    )
    auth_token: Optional[Union[str, Callable[[str], str]]] = Field(
        default=None,
        description="Authentication token or authentication token generating function for authenticated requests",
    )

    _client: Any = PrivateAttr()

    def __init__(
        self,
        base_url: str = DEFAULT_URL,
        timeout: float = 60.0,
        batch_size: int = 32,
        top_n: int = 2,
        raw_score: bool = False,
        truncate_text: bool = True,
        keep_retrieval_score: bool = False,
        auth_token: Optional[Union[str, Callable[[str], str]]] = None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            timeout=timeout,
            batch_size=batch_size,
            top_n=top_n,
            raw_score=raw_score,
            truncate_text=truncate_text,
            keep_retrieval_score=keep_retrieval_score,
            auth_token=auth_token,
        )  # type: ignore

        try:
            import httpx
        except ImportError:
            raise ImportError(  # noqa: B904
                "Cannot import httpx package, please `pip install httpx`."
            )
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={"Content-Type": "application/json"},
        )

        if self.auth_token is not None:
            if callable(self.auth_token):
                self._client.headers.update({"Authorization": self.auth_token(self.base_url)})
            else:
                self._client.headers.update({"Authorization": self.auth_token})

    @classmethod
    def class_name(cls) -> str:
        return "TextEmbeddingsInferenceRerank"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        dispatcher.event(
            ReRankStartEvent(
                query=query_bundle,
                nodes=nodes,
                top_n=self.top_n,
                model_name=urljoin(self.base_url, "/rerank"),
            )
        )

        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        if len(nodes) == 0 or self.top_n == 0:
            return []

        batched_list: List[List[NodeWithScore]] = []
        for i in range(0, len(nodes), self.batch_size):
            batched_list.append(nodes[i : i + self.batch_size])

        with self.callback_manager.event(
            CBEventType.RERANKING,
            payload={
                EventPayload.NODES: nodes,
                EventPayload.QUERY_STR: query_bundle.query_str,
                EventPayload.TOP_K: self.top_n,
            },
        ) as event:
            new_nodes: List[NodeWithScore] = []
            for batch in batched_list:
                new_nodes += self._rerank_single_batch(batch, query_bundle.query_str)
            top_n_nodes = new_nodes[: self.top_n]
            event.on_end(payload={EventPayload.NODES: top_n_nodes})

        dispatcher.event(ReRankEndEvent(nodes=top_n_nodes))
        return top_n_nodes

    def _rerank_single_batch(
        self,
        nodes: List[NodeWithScore],
        query_str: str,
    ) -> List[NodeWithScore]:
        texts = [node.node.get_content() for node in nodes]

        results = self._client.post(
            "/rerank",
            json={
                "query": query_str,
                "texts": texts,
                "truncate": self.truncate,
                "raw_score": self.raw_score,
                "return_text": False,
            },
        ).json()
        print(results)
        new_nodes: List[NodeWithScore] = []
        for result in results:
            score = result["score"]

            orig_node: NodeWithScore = nodes[result["index"]]
            if self.keep_retrieval_score:
                orig_node.node.metadata["retrieval_score"] = orig_node.score

            new_node_with_score = NodeWithScore(
                node=orig_node.node,
                score=score,
            )
            new_nodes.append(new_node_with_score)
        return new_nodes
