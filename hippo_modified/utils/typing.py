from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Literal, TypeVar
from typing_extensions import override

from pydantic import BaseModel, Field

ListTriple = Annotated[list[str], Field(min_length=3, max_length=3)]
TupleTriple = tuple[str, str, str]
Triple = TupleTriple | ListTriple

_T = TypeVar("_T")


# Sentinel class used until PEP 0661 is accepted
class NotGiven:
    """
    A sentinel singleton class used to distinguish omitted keyword arguments
    from those passed in with the value None (which may have different behavior).

    For example:

    ```py
    def get(timeout: Union[int, NotGiven, None] = NotGiven()) -> Response: ...


    get(timeout=1)  # 1s timeout
    get(timeout=None)  # No timeout
    get()  # Default timeout behavior, which may not be statically known at the method definition.
    ```
    """

    def __bool__(self) -> Literal[False]:
        return False

    @override
    def __repr__(self) -> str:
        return "NOT_GIVEN"


NOT_GIVEN = NotGiven()


class OpenIEDocItem(BaseModel):
    idx: str
    passage: str
    extracted_triples: Sequence[Triple] = Field(default_factory=list)
    extracted_entities: list[str] = Field(default_factory=list)


class OpenIEResult(BaseModel):
    docs: list[OpenIEDocItem] = Field(default_factory=list)
    avg_ent_words: float = Field(default=0.0)
    avg_ent_chars: float = Field(default=0.0)
