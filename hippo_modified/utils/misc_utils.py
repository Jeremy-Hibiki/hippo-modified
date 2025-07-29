import itertools
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
from typing import Any, TypeVar
from typing_extensions import overload

import numpy as np
import regex

from .llm_utils import filter_invalid_triples
from .typing import ListTriple, OpenIEDocItem, Triple, TupleTriple

logger = logging.getLogger(__name__)


_T = TypeVar("_T")


@dataclass
class NerRawOutput:
    chunk_id: str
    response: str | None
    unique_entities: list[str]
    metadata: dict[str, Any]


@dataclass
class TripleRawOutput:
    chunk_id: str
    response: str | None
    triples: list[ListTriple]
    metadata: dict[str, Any]


@dataclass
class QuerySolution:
    question: str
    docs: list[str]
    doc_scores: np.ndarray | None = None
    answer: str | None = None

    def to_dict(self):
        return {
            "question": self.question,
            "answer": self.answer,
            "docs": self.docs[:5],
            "doc_scores": [round(v, 4) for v in self.doc_scores.tolist()[:5]] if self.doc_scores is not None else None,
        }


NON_ALPHA_NUM_CJK_PATTERN = regex.compile(r"[^a-zA-Z0-9\p{Han}]")


@overload
def text_processing(text: str) -> str: ...


@overload
def text_processing(text: list[str]) -> list[str]: ...


def text_processing(text: str | list[str]) -> list[str] | str:
    if isinstance(text, list):
        return [text_processing(t) for t in text]  # type: ignore
    if not isinstance(text, str):
        text = str(text)
    # return re.sub("[^A-Za-z0-9 ]", " ", text.lower()).strip()
    return NON_ALPHA_NUM_CJK_PATTERN.sub("", text.lower()).strip()


def reformat_openie_results(
    corpus_openie_results: list[OpenIEDocItem],
) -> tuple[dict[str, NerRawOutput], dict[str, TripleRawOutput]]:
    ner_output_dict = {
        chunk_item.idx: NerRawOutput(
            chunk_id=chunk_item.idx,
            response=None,
            metadata={},
            unique_entities=list(np.unique(chunk_item.extracted_entities)),
        )
        for chunk_item in corpus_openie_results
    }
    triple_output_dict = {
        chunk_item.idx: TripleRawOutput(
            chunk_id=chunk_item.idx,
            response=None,
            metadata={},
            triples=filter_invalid_triples(triples=chunk_item.extracted_triples),
        )
        for chunk_item in corpus_openie_results
    }

    return ner_output_dict, triple_output_dict


def extract_entity_nodes(chunk_triples: list[list[Triple]]) -> tuple[list[str], list[list[str]]]:
    chunk_triple_entities = []  # a list of lists of unique entities from each chunk's triples
    for triples in chunk_triples:
        triple_entities = set()
        for t in triples:
            if len(t) == 3:
                triple_entities.update([t[0], t[2]])
            else:
                logger.warning(f"During graph construction, invalid triple is found: {t}")
        chunk_triple_entities.append(list(triple_entities))
    graph_nodes = list(np.unique([ent for ents in chunk_triple_entities for ent in ents]))
    return graph_nodes, chunk_triple_entities


def flatten_facts(chunk_triples: Sequence[Sequence[Triple]]) -> list[TupleTriple]:
    graph_triples: list[TupleTriple] = []  # a list of unique relation triple (in tuple) from all chunks
    for triples in chunk_triples:
        graph_triples.extend([tuple(t) for t in triples])  # type: ignore
    graph_triples = list(set(graph_triples))
    return graph_triples


def min_max_normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    Compute the MD5 hash of the given content string and optionally prepend a prefix.

    Args:
        content (str): The input string to be hashed.
        prefix (str, optional): A string to prepend to the resulting hash. Defaults to an empty string.

    Returns:
        str: A string consisting of the prefix followed by the hexadecimal representation of the MD5 hash.
    """
    return prefix + md5(content.encode()).hexdigest()


def load_hit_stopwords():
    return (Path(__file__).parent / "hit_stopwords.txt").read_text(encoding="utf-8").splitlines()


def flatten_list(a: list[list[_T]]) -> list[_T]:
    return list(itertools.chain.from_iterable(a))
