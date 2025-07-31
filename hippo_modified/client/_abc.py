from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeAlias, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Awaitable

    import numpy.typing as npt

    from ..utils.misc_utils import QuerySolution
    from ..utils.typing import NotGiven

    _T = TypeVar("_T")
    MaybeAwaitable: TypeAlias = _T | Awaitable[_T]


@runtime_checkable
class HippoRAGProtocol(Protocol):
    def index(self, docs: list[str]) -> MaybeAwaitable[None]: ...

    def delete(
        self,
        docs_to_delete: list[str] | None = None,
        doc_ids_to_delete: list[str] | None = None,
    ) -> MaybeAwaitable[None]: ...

    def retrieve(
        self,
        queries: list[str],
        num_to_retrieve: int | NotGiven = ...,
        num_to_link: int | NotGiven = ...,
        passage_node_weight: float | NotGiven = ...,
        pike_node_weight: float | NotGiven = ...,
        rerank_batch_num: int = 10,
        rerank_file_path: str | NotGiven = ...,
        atom_query_num: int = 5,
    ) -> MaybeAwaitable[list[QuerySolution]]: ...

    def graph_search_with_fact_entities(
        self,
        query: str,
        link_top_k: int,
        top_k_facts: list[tuple[str, str, str]],
        fact_scores_dict: dict[str, float],
        passage_node_weight: float = 0.05,
        pike_node_weight: float = 1.0,
    ) -> MaybeAwaitable[tuple[npt.NDArray, npt.NDArray, float]]: ...

    def rerank_facts(
        self,
        query: str,
        batch_size: int = 10,
    ) -> MaybeAwaitable[tuple[list[tuple[str, str, str]], dict[str, float], dict]]: ...
