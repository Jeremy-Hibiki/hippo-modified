from __future__ import annotations

import ast
import asyncio
import logging
import time
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ..prompts.linking import get_query_instruction
from ..utils.misc_utils import (
    QuerySolution,
    compute_mdhash_id,
    extract_entity_nodes,
    flatten_facts,
    min_max_normalize,
    reformat_openie_results,
    text_processing,
)
from ..utils.typing import NOT_GIVEN, NotGiven, OpenIEDocItem, Triple
from ._abc import HippoRAGProtocol
from .base import BaseHippoRAG

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..utils.config_utils import BaseConfig

logger = logging.getLogger(__name__)


class AsyncHippoRAG(BaseHippoRAG, HippoRAGProtocol):
    def __init__(
        self,
        global_config: BaseConfig | None = None,
        save_dir: str | NotGiven = NOT_GIVEN,
        llm_model_name: str | NotGiven = NOT_GIVEN,
        llm_base_url: str | NotGiven = NOT_GIVEN,
        embedding_model_name: str | NotGiven = NOT_GIVEN,
        embedding_base_url: str | NotGiven = NOT_GIVEN,
        azure_endpoint: str | NotGiven = NOT_GIVEN,
        azure_embedding_endpoint: str | NotGiven = NOT_GIVEN,
    ):
        super().__init__(
            global_config,
            save_dir,
            llm_model_name,
            llm_base_url,
            embedding_model_name,
            embedding_base_url,
            azure_endpoint,
            azure_embedding_endpoint,
        )
        self.rerank_filter_fn = self.rerank_filter.async_rerank

        self._prepare_lock = asyncio.Lock()

    async def index(self, docs: list[str]):
        """
        Indexes the given documents based on the HippoRAG 2 framework which generates an OpenIE knowledge graph
        based on the given documents and encodes passages, entities and facts separately for later retrieval.

        Parameters:
            docs : List[str]
                A list of documents to be indexed.
        """

        logger.info("Indexing Documents")

        logger.info("Performing OpenIE")

        await self.chunk_embedding_store.async_insert_strings(docs)
        chunk_to_rows = self.chunk_embedding_store.get_all_id_to_rows()

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(list(chunk_to_rows.keys()))
        new_openie_rows: dict[str, dict] = {k: chunk_to_rows[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)  # type: ignore
            self.merge_openie_results(
                all_openie_info,
                new_openie_rows,
                new_ner_results_dict,
                new_triple_results_dict,
            )

        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)

        assert len(chunk_to_rows) == len(ner_results_dict) == len(triple_results_dict), (
            f"{len(chunk_to_rows)}, {len(ner_results_dict)}, {len(triple_results_dict)}"
        )

        # prepare data_store
        chunk_ids = list(chunk_to_rows.keys())

        chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in chunk_ids]
        entity_nodes, chunk_triple_entities = extract_entity_nodes(chunk_triples)  # type: ignore
        facts = flatten_facts(chunk_triples)  # type: ignore

        logger.info("Encoding Entities")
        await self.entity_embedding_store.async_insert_strings(entity_nodes)

        logger.info("Encoding Facts")
        await self.fact_embedding_store.async_insert_strings([str(fact) for fact in facts])

        logger.info("Constructing Graph")

        self.node_to_node_stats = {}
        self.ent_node_to_chunk_ids = {}

        self.add_fact_edges(chunk_ids, chunk_triples)  # type: ignore
        num_new_chunks = self.add_passage_edges(chunk_ids, chunk_triple_entities)

        if num_new_chunks > 0:
            logger.info(f"Found {num_new_chunks} new chunks to save into graph.")
            self.add_synonymy_edges()

            self.augment_graph()
            self.save_igraph()

    async def delete(
        self,
        docs_to_delete: list[str] | None = None,
        doc_ids_to_delete: list[str] | None = None,
    ):
        """
        Deletes the given documents from all data structures within the HippoRAG class.
        Note that triples and entities which are indexed from chunks that are not being removed will not be removed.

        Parameters:
            docs : List[str]
                A list of documents to be deleted.
        """

        # Making sure that all the necessary structures have been built.
        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        if TYPE_CHECKING:
            assert self.ent_node_to_chunk_ids is not None

        merged_docs_to_delete = docs_to_delete or []
        if doc_ids_to_delete:
            merged_docs_to_delete += [
                chunk["content"] for chunk in self.chunk_embedding_store.get_rows(doc_ids_to_delete).values()
            ]

        if not merged_docs_to_delete:
            logger.warning("No documents to delete")
            return

        all_chunks = self.chunk_embedding_store.get_all_id_to_rows()
        chunk_ids_to_delete = [
            chunk_id for chunk_id, row in all_chunks.items() if row["content"] in merged_docs_to_delete
        ]

        # Find triples in chunks to delete
        all_openie_info, _ = self.load_existing_openie(chunk_ids_to_delete)
        triples_to_delete: list[Sequence[Triple]] = []

        all_openie_info_with_deletes: list[OpenIEDocItem] = []

        for openie_doc in all_openie_info:
            if openie_doc.idx in chunk_ids_to_delete:
                triples_to_delete.append(openie_doc.extracted_triples)
            else:
                all_openie_info_with_deletes.append(openie_doc)

        flat_triples_to_delete = flatten_facts(triples_to_delete)

        # Filter out triples that appear in unaltered chunks
        true_triples_to_delete: list[Triple] = []

        for triple in flat_triples_to_delete:
            proc_triple = tuple(text_processing(list(triple)))

            doc_ids = self.proc_triples_to_docs[str(proc_triple)]

            non_deleted_docs = doc_ids.difference(chunk_ids_to_delete)

            if len(non_deleted_docs) == 0:
                true_triples_to_delete.append(triple)

        processed_true_triples_to_delete = [[text_processing(list(triple)) for triple in true_triples_to_delete]]
        entities_to_delete, _ = extract_entity_nodes(processed_true_triples_to_delete)  # type: ignore
        processed_true_triples_to_delete = flatten_facts(processed_true_triples_to_delete)  # type: ignore

        all_facts = self.fact_embedding_store.get_all_id_to_rows()
        triple_ids_to_delete = set([
            fact_id
            for fact_id, fact in all_facts.items()
            if fact["content"] in list(map(str, processed_true_triples_to_delete))
        ])

        # Filter out entities that appear in unaltered chunks
        all_entities = self.entity_embedding_store.get_all_id_to_rows()
        ent_ids_to_delete = set([
            entity_id for entity_id, entity in all_entities.items() if entity["content"] in entities_to_delete
        ])

        filtered_ent_ids_to_delete = []

        for ent_node in ent_ids_to_delete:
            doc_ids = self.ent_node_to_chunk_ids[ent_node]

            non_deleted_docs = doc_ids.difference(chunk_ids_to_delete)

            if len(non_deleted_docs) == 0:
                filtered_ent_ids_to_delete.append(ent_node)

        # Find atomic queries to delete
        all_queries = self.query_embedding_store.get_all_id_to_rows()
        all_query_to_query_ids: dict[str, list[str]] = {}
        for query_id, query in all_queries.items():
            if query["content"] not in self.query_to_chunk_ids:
                all_query_to_query_ids[query["content"]] = []
            all_query_to_query_ids[query["content"]].append(query_id)

        all_query_ids_to_delete: list[str] = []
        for q, chunk_id_list in self.query_to_chunk_ids.items():
            if any(chunk_id in chunk_ids_to_delete for chunk_id in chunk_id_list):
                all_query_ids_to_delete.extend(all_query_to_query_ids[q])

        logger.info(f"Deleting {len(chunk_ids_to_delete)} Chunks")
        logger.info(f"Deleting {len(triple_ids_to_delete)} Triples")
        logger.info(f"Deleting {len(filtered_ent_ids_to_delete)} Entities")
        logger.info(f"Deleting {len(all_query_ids_to_delete)} Queries")

        self.save_openie_results(all_openie_info_with_deletes)

        await asyncio.gather(
            self.entity_embedding_store.async_delete(filtered_ent_ids_to_delete),
            self.fact_embedding_store.async_delete(list(triple_ids_to_delete)),
            self.chunk_embedding_store.async_delete(chunk_ids_to_delete),
            self.query_embedding_store.async_delete(all_query_ids_to_delete),
        )

        # Delete Nodes from Graph
        self.graph.delete_vertices(list(filtered_ent_ids_to_delete) + list(chunk_ids_to_delete))
        self.save_igraph()

        self.ready_to_retrieve = False

    async def retrieve(
        self,
        queries: list[str],
        num_to_retrieve: int | NotGiven = NOT_GIVEN,
        num_to_link: int | NotGiven = NOT_GIVEN,
        passage_node_weight: float | NotGiven = NOT_GIVEN,
        pike_node_weight: float | NotGiven = NOT_GIVEN,
        rerank_batch_num: int = 10,
        rerank_file_path: str | NotGiven = NOT_GIVEN,
        atom_query_num: int = 5,
    ) -> list[QuerySolution]:
        """
        Performs retrieval using the HippoRAG 2 framework, which consists of several steps:
        - Fact Retrieval
        - Recognition Memory for improved fact selection
        - Dense passage scoring
        - Personalized PageRank based re-ranking

        Parameters:
            queries: List[str]
                A list of query strings for which documents are to be retrieved.
            num_to_retrieve: int, optional
                The maximum number of documents to retrieve for each query. If not specified, defaults to
                the `retrieval_top_k` value defined in the global configuration.

        Returns:
            List[QuerySolution]
                A list of QuerySolution objects, each containing
                the retrieved documents and their scores for the corresponding query.

        Notes
        -----
        - Long queries with no relevant facts after reranking will default to results from dense passage retrieval.
        """
        retrieve_start_time = time.time()  # Record start time

        if isinstance(num_to_retrieve, NotGiven):
            num_to_retrieve = self.global_config.retrieval_top_k

        if not isinstance(num_to_link, NotGiven):
            self.global_config.linking_top_k = num_to_link

        if not isinstance(passage_node_weight, NotGiven):
            self.passage_node_weight = passage_node_weight
        else:
            self.passage_node_weight = self.global_config.passage_node_weight

        if not isinstance(pike_node_weight, NotGiven):
            self.pike_node_weight = pike_node_weight
        else:
            self.pike_node_weight = self.global_config.passage_node_weight

        if not isinstance(rerank_file_path, NotGiven):
            self.global_config.rerank_dspy_file_path = rerank_file_path

        if not isinstance(atom_query_num, NotGiven):
            self.atom_query_num = atom_query_num

        if not self.ready_to_retrieve:
            async with self._prepare_lock:
                if not self.ready_to_retrieve:
                    self.prepare_retrieval_objects()

        retrieval_results = []

        for q_idx, query in enumerate(queries):
            rerank_start = time.time()
            top_k_facts, fact_scores_dict, _rerank_log = await self.rerank_facts(
                query,
                batch_size=rerank_batch_num,
            )
            rerank_end = time.time()

            rerank_time = rerank_end - rerank_start

            if len(top_k_facts) == 0:
                logger.info("No facts found after reranking")
                sorted_doc_ids, sorted_doc_scores, ppr_time = (
                    np.array([], dtype=np.intp),
                    np.array([], dtype=np.float64),
                    0.0,
                )
            else:
                sorted_doc_ids, sorted_doc_scores, ppr_time = await self.graph_search_with_fact_entities(
                    query=query,
                    link_top_k=self.global_config.linking_top_k,
                    top_k_facts=top_k_facts,
                    fact_scores_dict=fact_scores_dict,
                    passage_node_weight=self.passage_node_weight,
                    pike_node_weight=self.pike_node_weight,
                )

            top_k_docs = [
                v["content"]
                for v in (
                    await self.chunk_embedding_store.async_get_rows(
                        [self.passage_node_keys[idx] for idx in sorted_doc_ids[:num_to_retrieve]],
                    )
                ).values()
            ]

            retrieval_results.append(
                QuerySolution(
                    question=query,
                    docs=top_k_docs,
                    doc_scores=sorted_doc_scores[:num_to_retrieve],
                )
            )

            retrieve_end_time = time.time()  # Record end time

            retrieval_time = retrieve_end_time - retrieve_start_time

            logger.info(f"Retrieval Time {retrieval_time:.2f}s")
            logger.info(f"  LLM Rerank Time {rerank_time:.2f}s")
            logger.info(f"  PPR Time {ppr_time:.2f}s")
            logger.info(f"  Misc Time {retrieval_time - (rerank_time + ppr_time):.2f}s")

        return retrieval_results

    async def graph_search_with_fact_entities(
        self,
        query: str,
        link_top_k: int,
        top_k_facts: list[tuple[str, str, str]],
        fact_scores_dict: dict[str, float],
        passage_node_weight: float = 0.05,
        pike_node_weight: float = 1.0,
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.float64], float]:
        """
        Computes document scores based on fact-based similarity and relevance using personalized
        PageRank (PPR) and dense retrieval models. This function combines the signal from the relevant
        facts identified with passage similarity and graph-based search for enhanced result ranking.

        Parameters:
            query (str): The input query string for which similarity and relevance computations
                need to be performed.
            link_top_k (int): The number of top phrases to include from the linking score map for
                downstream processing.
            top_k_facts (List[Tuple]): A list of top-ranked facts, where each fact is represented
                as a tuple of its subject, predicate, and object.
            fact_scores_dict (Dict[str, float]): A dictionary mapping fact string to its score.
            passage_node_weight (float): Default weight to scale passage scores in the graph.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
                - The first array corresponds to document IDs sorted based on their scores.
                - The second array consists of the PPR scores associated with the sorted document IDs.
        """
        if TYPE_CHECKING:
            assert self.ent_node_to_chunk_ids is not None

        # Assigning phrase weights based on selected facts from previous steps.
        linking_score_map = {}  # from phrase to the average scores of the facts that contain the phrase
        phrase_scores: dict[
            str, list[float]
        ] = {}  # store all fact scores for each phrase regardless of whether they exist in the knowledge graph or not
        phrase_weights = np.zeros(len(self.graph.vs["name"]))
        passage_weights = np.zeros(len(self.graph.vs["name"]))
        pike_passage_weights = np.zeros(len(self.graph.vs["name"]))

        for rank, f in enumerate(top_k_facts):
            subject_phrase = f[0].lower()
            object_phrase = f[2].lower()
            fact_score = fact_scores_dict.get(str(f), 0.0)
            for phrase in [subject_phrase, object_phrase]:
                phrase_key = compute_mdhash_id(content=phrase, prefix="entity-")
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)

                if phrase_id is not None:
                    phrase_weights[phrase_id] = fact_score

                    if len(self.ent_node_to_chunk_ids.get(phrase_key, set())) > 0:
                        phrase_weights[phrase_id] /= len(self.ent_node_to_chunk_ids[phrase_key])

                if phrase not in phrase_scores:
                    phrase_scores[phrase] = []
                phrase_scores[phrase].append(fact_score)

        # calculate average fact score for each phrase
        for phrase, scores in phrase_scores.items():
            linking_score_map[phrase] = float(np.mean(scores))

        if link_top_k:
            phrase_weights, linking_score_map = self.get_top_k_weights(
                link_top_k, phrase_weights, linking_score_map
            )  # at this stage, the length of linking_scope_map is determined by link_top_k

        # Get pike chunk according to chosen atom query
        atom_query_results = await self.query_embedding_store.async_search(query, top_k=self.atom_query_num)
        query_sorted_scores = [score for _, score in atom_query_results]
        normalized_query_sorted_scores = min_max_normalize(np.array(query_sorted_scores))
        if len(atom_query_results) != 0:
            for i, (query_dict, _) in enumerate(atom_query_results):
                query_dpr_score = normalized_query_sorted_scores[i]
                atom_query = query_dict["content"]
                chunk_ids = self.query_to_chunk_ids.get(atom_query, [])
                for chunk_id in chunk_ids:
                    # passage_node_key = self.passage_node_keys[chunk_id]
                    passage_node_id = self.node_name_to_vertex_idx[chunk_id]
                    pike_passage_weights[passage_node_id] = query_dpr_score * pike_node_weight

        # Get passage scores according to chosen dense retrieval model
        dpr_top_passages = await self.chunk_embedding_store.async_search(
            query_text=query,
            instruction=get_query_instruction("query_to_passage"),
            top_k=len(self.passage_node_keys),
        )
        passage_scores_list = [score for _, score in dpr_top_passages]
        normalized_dpr_sorted_scores = min_max_normalize(np.array(passage_scores_list))

        for i, (passage_dict, _) in enumerate(dpr_top_passages):
            passage_dpr_score = normalized_dpr_sorted_scores[i]
            passage_node_id = self.node_name_to_vertex_idx[passage_dict["hash_id"]]
            passage_weights[passage_node_id] = passage_dpr_score * passage_node_weight
            passage_node_text = passage_dict["content"]
            linking_score_map[passage_node_text] = passage_dpr_score * passage_node_weight

        # Combining phrase and passage scores into one array for PPR
        # Combining pike passage scores
        node_weights = phrase_weights + passage_weights + pike_passage_weights

        # Recording top 30 facts in linking_score_map
        if len(linking_score_map) > 30:
            linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:30])

        assert sum(node_weights) > 0, f"No phrases found in the graph for the given facts: {top_k_facts}"

        # Running PPR algorithm based on the passage and phrase weights previously assigned
        ppr_start = time.time()
        ppr_sorted_doc_ids, ppr_sorted_doc_scores = self.run_ppr(node_weights, damping=self.global_config.damping)
        ppr_end = time.time()

        ppr_time = ppr_end - ppr_start

        assert len(ppr_sorted_doc_ids) == len(self.passage_node_idxs), (
            f"Doc prob length {len(ppr_sorted_doc_ids)} != corpus length {len(self.passage_node_idxs)}"
        )

        return ppr_sorted_doc_ids, ppr_sorted_doc_scores, ppr_time

    async def rerank_facts(
        self,
        query: str,
        batch_size: int = 10,
    ) -> tuple[list[tuple[str, str, str]], dict[str, float], dict]:
        """
        Args:

        Returns:
            top_k_facts:
            fact_scores (dict):
            rerank_log (dict): {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
                - candidate_facts (list): list of link_top_k facts (each fact is a relation triple in tuple data type).
                - top_k_facts:
        """
        # load args
        link_top_k: int = self.global_config.linking_top_k

        try:
            # Get the top k facts by score
            top_facts_from_search = await self.fact_embedding_store.async_search(
                query_text=query,
                instruction=get_query_instruction("query_to_fact"),
                top_k=link_top_k,
            )
            if not top_facts_from_search:
                logger.warning("No facts available for reranking. Returning empty lists.")
                return [], {}, {"facts_before_rerank": [], "facts_after_rerank": []}

            fact_row_dict = [fact_dict for fact_dict, score in top_facts_from_search]
            candidate_facts: list[tuple] = [ast.literal_eval(f["content"]) for f in fact_row_dict]
            initial_fact_scores = {f["content"]: score for f, score in top_facts_from_search}

            logger.info(f"query: {query}")
            logger.info(f"candidate_facts: {candidate_facts}")
            # Rerank the facts
            top_k_facts: list[tuple] = (
                await self.rerank_filter_fn(
                    query,
                    candidate_facts,
                    list(range(len(candidate_facts))),
                    len_after_rerank=link_top_k,
                    batch_size=batch_size,
                )
            )[1]
            logger.info(f"top_k_facts: {top_k_facts}")
            rerank_log = {
                "facts_before_rerank": candidate_facts,
                "facts_after_rerank": top_k_facts,
            }

            # The scores are not reranked, we use the original scores from vector search
            fact_scores = {str(fact): initial_fact_scores.get(str(fact), 0.0) for fact in top_k_facts}

            return top_k_facts, fact_scores, rerank_log

        except Exception as e:
            logger.error(f"Error in rerank_facts: {e!s}")
            return (
                [],
                {},
                {"facts_before_rerank": [], "facts_after_rerank": [], "error": str(e)},
            )
