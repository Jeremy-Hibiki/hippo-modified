import logging
from argparse import ArgumentTypeError
from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
from typing import Any, Literal

import numpy as np
import regex

from .llm_utils import filter_invalid_triples
from .typing import Triple

logger = logging.getLogger(__name__)


@dataclass
class NerRawOutput:
    chunk_id: str
    response: str
    unique_entities: list[str]
    metadata: dict[str, Any]


@dataclass
class TripleRawOutput:
    chunk_id: str
    response: str
    triples: list[list[str]]
    metadata: dict[str, Any]


@dataclass
class LinkingOutput:
    score: np.ndarray
    type: Literal["node", "dpr"]


@dataclass
class QuerySolution:
    question: str
    docs: list[str]
    doc_scores: np.ndarray = None
    answer: str = None

    def to_dict(self):
        return {
            "question": self.question,
            "answer": self.answer,
            "docs": self.docs[:5],
            "doc_scores": [round(v, 4) for v in self.doc_scores.tolist()[:5]] if self.doc_scores is not None else None,
        }


NON_ALPHA_NUM_CJK_PATTERN = regex.compile(r"[^a-zA-Z0-9\p{Han}]")


def text_processing(text: str | list[str]) -> list[str] | str:
    if isinstance(text, list):
        return [text_processing(t) for t in text]
    if not isinstance(text, str):
        text = str(text)
    # return re.sub("[^A-Za-z0-9 ]", " ", text.lower()).strip()
    return NON_ALPHA_NUM_CJK_PATTERN.sub("", text.lower()).strip()


def reformat_openie_results(corpus_openie_results) -> tuple[dict[str, NerRawOutput], dict[str, TripleRawOutput]]:
    ner_output_dict = {
        chunk_item["idx"]: NerRawOutput(
            chunk_id=chunk_item["idx"],
            response=None,
            metadata={},
            unique_entities=list(np.unique(chunk_item["extracted_entities"])),
        )
        for chunk_item in corpus_openie_results
    }
    triple_output_dict = {
        chunk_item["idx"]: TripleRawOutput(
            chunk_id=chunk_item["idx"],
            response=None,
            metadata={},
            triples=filter_invalid_triples(triples=chunk_item["extracted_triples"]),
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


def flatten_facts(chunk_triples: list[Triple]) -> list[Triple]:
    graph_triples = []  # a list of unique relation triple (in tuple) from all chunks
    for triples in chunk_triples:
        graph_triples.extend([tuple(t) for t in triples])
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


def all_values_of_same_length(data: dict) -> bool:
    """
    Return True if all values in 'data' have the same length or data is an empty dict,
    otherwise return False.
    """
    # Get an iterator over the dictionary's values
    value_iter = iter(data.values())

    # Get the length of the first sequence (handle empty dict case safely)
    try:
        first_length = len(next(value_iter))
    except StopIteration:
        # If the dictionary is empty, treat it as all having "the same length"
        return True

    # Check that every remaining sequence has this same length
    return all(len(seq) == first_length for seq in value_iter)


def string_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )


def load_hit_stopwords():
    return (Path(__file__).parent / "hit_stopwords.txt").read_text(encoding="utf-8").splitlines()
