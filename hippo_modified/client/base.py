from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar, cast

import igraph as ig
import numpy as np
import regex as re
from tqdm import tqdm

from ..embedding_model import get_embedding_model_class
from ..embedding_store import create_embedding_store
from ..information_extraction import OpenIE
from ..llm import get_llm_class
from ..prompts.prompt_template_manager import PromptTemplateManager
from ..rerank import DSPyFilter
from ..utils.config_utils import BaseConfig
from ..utils.misc_utils import (
    NerRawOutput,
    TripleRawOutput,
    compute_mdhash_id,
)
from ..utils.typing import NOT_GIVEN, NotGiven, OpenIEDocItem, OpenIEResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    _SCT = TypeVar("_SCT", bound=np.generic)
    _Array1D: TypeAlias = np.ndarray[tuple[int], np.dtype[_SCT]]


logger = logging.getLogger(__name__)


class BaseHippoRAG:
    node_name_to_vertex_idx: dict[str, int]
    passage_node_idxs: list[int]

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
    ) -> None:
        """
        Initializes an async version of the class and its related components.

        Attributes:
            global_config (BaseConfig): The global configuration settings for the instance. An instance
                of BaseConfig is used if no value is provided.
            saving_dir (str): The directory where specific HippoRAG instances will be stored. This defaults
                to `outputs` if no value is provided.
            llm_model (BaseLLM): The language model used for processing based on the global
                configuration settings.
            openie (Union[OpenIE, VLLMOfflineOpenIE]): The Open Information Extraction module
                configured in either online or offline mode based on the global settings.
            graph: The graph instance initialized by the `initialize_graph` method.
            embedding_model (BaseEmbeddingModel): The embedding model associated with the current
                configuration.
            chunk_embedding_store (EmbeddingStore): The embedding store handling chunk embeddings.
            entity_embedding_store (EmbeddingStore): The embedding store handling entity embeddings.
            fact_embedding_store (EmbeddingStore): The embedding store handling fact embeddings.
            prompt_template_manager (PromptTemplateManager): The manager for handling prompt templates
                and roles mappings.
            openie_results_path (str): The file path for storing Open Information Extraction results
                based on the dataset and LLM name in the global configuration.
            rerank_filter (Optional[DSPyFilter]): The filter responsible for reranking information
                when a rerank file path is specified in the global configuration.
            ready_to_retrieve (bool): A flag indicating whether the system is ready for retrieval
                operations.

        Parameters:
            global_config: The global configuration object. Defaults to None, leading to initialization
                of a new BaseConfig object.
            working_dir: The directory for storing working files. Defaults to None, constructing a default
                directory based on the class name and timestamp.
            llm_model_name: LLM model name, can be inserted directly as well as through configuration file.
            embedding_model_name: Embedding model name, can be inserted directly as well as through configuration file.
            llm_base_url: LLM URL for a deployed LLM model, can be inserted directly as well as through configuration file.
        """
        if global_config is None:
            self.global_config = BaseConfig()
        else:
            self.global_config = global_config

        # Overwriting Configuration if Specified
        if not isinstance(save_dir, NotGiven):
            self.global_config.save_dir = save_dir

        if not isinstance(llm_model_name, NotGiven):
            self.global_config.llm_name = llm_model_name

        if not isinstance(embedding_model_name, NotGiven):
            self.global_config.embedding_model_name = embedding_model_name

        if not isinstance(llm_base_url, NotGiven):
            self.global_config.llm_base_url = llm_base_url

        if not isinstance(embedding_base_url, NotGiven):
            self.global_config.embedding_base_url = embedding_base_url

        if not isinstance(azure_endpoint, NotGiven):
            self.global_config.azure_endpoint = azure_endpoint

        if not isinstance(azure_embedding_endpoint, NotGiven):
            self.global_config.azure_embedding_endpoint = azure_embedding_endpoint

        print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self.global_config).items()])
        logger.info(f"HippoRAG init with config:\n  {print_config}\n")

        # LLM and embedding model specific working directories are created under every specified saving directories
        llm_label = self.global_config.llm_name.replace("/", "_")
        embedding_label = self.global_config.embedding_model_name.replace("/", "_")
        self.working_dir = Path(self.global_config.save_dir) / f"{llm_label}_{embedding_label}"

        if not Path(self.working_dir).exists():
            logger.info(f"Creating working directory: {self.working_dir}")
            self.working_dir.mkdir(parents=True, exist_ok=True)

        self.llm_model = get_llm_class(self.global_config)

        self.openie = OpenIE(llm_model=self.llm_model)

        self.graph = self.initialize_graph()

        self.embedding_model = get_embedding_model_class(
            embedding_type=self.global_config.embedding_type,
            embedding_model_name=self.global_config.embedding_model_name,
        )(
            global_config=self.global_config,
            embedding_model_name=self.global_config.embedding_model_name,
        )

        self.chunk_embedding_store = create_embedding_store(
            embedding_model=self.embedding_model,
            db_name=str(Path(self.working_dir) / "chunk_embeddings"),
            batch_size=self.global_config.embedding_batch_size,
            namespace="chunk",
            global_config=self.global_config,
        )
        self.entity_embedding_store = create_embedding_store(
            embedding_model=self.embedding_model,
            db_name=str(Path(self.working_dir) / "entity_embeddings"),
            batch_size=self.global_config.embedding_batch_size,
            namespace="entity",
            global_config=self.global_config,
        )
        self.fact_embedding_store = create_embedding_store(
            embedding_model=self.embedding_model,
            db_name=str(Path(self.working_dir) / "fact_embeddings"),
            batch_size=self.global_config.embedding_batch_size,
            namespace="fact",
            global_config=self.global_config,
            enable_hybrid_search=self.global_config.milvus_enable_hybrid_search,
        )

        self.query_embedding_store = create_embedding_store(
            embedding_model=self.embedding_model,
            db_name=str(Path(self.working_dir) / "query_embeddings"),
            batch_size=self.global_config.embedding_batch_size,
            namespace="query",
            global_config=self.global_config,
            enable_hybrid_search=self.global_config.milvus_enable_hybrid_search,
        )

        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={"system": "system", "user": "user", "assistant": "assistant"}
        )

        self.openie_results_path = Path(
            self.global_config.save_dir,
            f"openie_results_ner_{self.global_config.llm_name.replace('/', '_')}.json",
        )

        self.rerank_filter = DSPyFilter(self)

        self.ready_to_retrieve = False

        self.node_to_node_stats: dict[tuple[str, str], float] = {}
        self.ent_node_to_chunk_ids: dict[str, set[str]] | None = None

    def initialize_graph(self) -> ig.Graph:
        """
        Initializes a graph using a Pickle file if available or creates a new graph.

        The function attempts to load a pre-existing graph stored in a Pickle file. If the file
        is not present or the graph needs to be created from scratch, it initializes a new directed
        or undirected graph based on the global configuration. If the graph is loaded successfully
        from the file, pertinent information about the graph (number of nodes and edges) is logged.

        Returns:
            ig.Graph: A pre-loaded or newly initialized graph.

        Raises:
            None
        """
        self._graph_pickle_filename = Path(self.working_dir) / "graph.pickle"

        preloaded_graph = None

        if not self.global_config.force_index_from_scratch and Path(self._graph_pickle_filename).exists():
            preloaded_graph = cast("ig.Graph", ig.Graph.Read_Pickle(self._graph_pickle_filename))

        if preloaded_graph is None:
            return ig.Graph(directed=self.global_config.is_directed_graph)
        else:
            logger.info(
                f"Loaded graph from {self._graph_pickle_filename} with {preloaded_graph.vcount()} nodes, {preloaded_graph.ecount()} edges"
            )
            return preloaded_graph

    def add_fact_edges(
        self,
        chunk_ids: list[str],
        chunk_triples: Sequence[Sequence[Sequence[str] | tuple[str, str, str]]],
        chunk_metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        """
        Adds fact edges from given triples to the graph.

        The method processes chunks of triples, computes unique identifiers
        for entities and relations, and updates various internal statistics
        to build and maintain the graph structure. Entities are uniquely
        identified and linked based on their relationships.

        Parameters:
            chunk_ids: List[str]
                A list of unique identifiers for the chunks being processed.
            chunk_triples: List[Tuple]
                A list of tuples representing triples to process. Each triple
                consists of a subject, predicate, and object.
            chunk_metadatas: Optional[List[Dict]]
                Metadata for each chunk.

        Raises:
            Does not explicitly raise exceptions within the provided function logic.
        """
        if TYPE_CHECKING:
            assert self.ent_node_to_chunk_ids is not None

        # Normalize metadatas
        if chunk_metadatas is None:
            chunk_metadatas = [{}] * len(chunk_ids)

        # Store chunk metadata for later use in graph
        if not hasattr(self, "chunk_metadata_map"):
            self.chunk_metadata_map = {}

        for chunk_id, metadata in zip(chunk_ids, chunk_metadatas, strict=False):
            if chunk_id in self.chunk_metadata_map:
                # Merge metadata if chunk already exists
                self.chunk_metadata_map[chunk_id].update(metadata)
            else:
                self.chunk_metadata_map[chunk_id] = metadata.copy()

        current_graph_nodes = set(self.graph.vs["name"]) if "name" in self.graph.vs else set()

        logger.info("Adding OpenIE triples to graph.")

        for chunk_key, triples in zip(chunk_ids, chunk_triples, strict=False):
            entities_in_chunk = set()

            if chunk_key not in current_graph_nodes:
                for triple in triples:
                    triple = tuple(triple)
                    if triple[0] == "" or triple[2] == "":
                        logger.info(f"skip {triple}")
                        continue

                    node_key = compute_mdhash_id(content=triple[0], prefix=("entity-"))
                    node_2_key = compute_mdhash_id(content=triple[2], prefix=("entity-"))

                    self.node_to_node_stats[node_key, node_2_key] = (
                        self.node_to_node_stats.get((node_key, node_2_key), 0.0) + 1
                    )
                    self.node_to_node_stats[node_2_key, node_key] = (
                        self.node_to_node_stats.get((node_2_key, node_key), 0.0) + 1
                    )

                    entities_in_chunk.add(node_key)
                    entities_in_chunk.add(node_2_key)

                for node in entities_in_chunk:
                    self.ent_node_to_chunk_ids[node] = self.ent_node_to_chunk_ids.get(node, set()).union(
                        set([chunk_key])
                    )

    def add_passage_edges(
        self,
        chunk_ids: list[str],
        chunk_triple_entities: list[list[str]],
        chunk_metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> int:
        """
        Adds edges connecting passage nodes to phrase nodes in the graph.

        This method is responsible for iterating through a list of chunk identifiers
        and their corresponding triple entities. It calculates and adds new edges
        between the passage nodes (defined by the chunk identifiers) and the phrase
        nodes (defined by the computed unique hash IDs of triple entities). The method
        also updates the node-to-node statistics map and keeps count of newly added
        passage nodes.

        Parameters:
            chunk_ids : List[str]
                A list of identifiers representing passage nodes in the graph.
            chunk_triple_entities : List[List[str]]
                A list of lists where each sublist contains entities (strings) associated
                with the corresponding chunk in the chunk_ids list.
            chunk_metadatas: Optional[List[Dict]]
                Metadata for each chunk.

        Returns:
            int
                The number of new passage nodes added to the graph.
        """

        # Normalize metadatas
        if chunk_metadatas is None:
            chunk_metadatas = [{}] * len(chunk_ids)

        # Store metadata for passage nodes
        if not hasattr(self, "chunk_metadata_map"):
            self.chunk_metadata_map = {}

        for chunk_id, metadata in zip(chunk_ids, chunk_metadatas, strict=False):
            if chunk_id in self.chunk_metadata_map:
                # Merge metadata if chunk already exists
                self.chunk_metadata_map[chunk_id].update(metadata)
            else:
                self.chunk_metadata_map[chunk_id] = metadata.copy()

        current_graph_nodes = set(self.graph.vs["name"]) if "name" in self.graph.vs.attribute_names() else set()

        num_new_chunks = 0

        logger.info("Connecting passage nodes to phrase nodes.")

        for idx, chunk_key in tqdm(enumerate(chunk_ids)):
            if chunk_key not in current_graph_nodes:
                for chunk_ent in chunk_triple_entities[idx]:
                    node_key = compute_mdhash_id(chunk_ent, prefix="entity-")

                    self.node_to_node_stats[chunk_key, node_key] = 1.0

                num_new_chunks += 1

        return num_new_chunks

    def add_synonymy_edges(self) -> None:
        """
        Adds synonymy edges between similar nodes in the graph to enhance connectivity by identifying and linking synonym entities.

        This method performs key operations to compute and add synonymy edges. It first retrieves embeddings for all nodes, then conducts
        a nearest neighbor (KNN) search to find similar nodes. These similar nodes are identified based on a score threshold, and edges
        are added to represent the synonym relationship.

        Attributes:
            entity_id_to_row: dict (populated within the function). Maps each entity ID to its corresponding row data, where rows
                              contain `content` of entities used for comparison.
            entity_embedding_store: Manages retrieval of texts and embeddings for all rows related to entities.
            global_config: Configuration object that defines parameters such as `synonymy_edge_topk`, `synonymy_edge_sim_threshold`,
                           `synonymy_edge_query_batch_size`, and `synonymy_edge_key_batch_size`.
            node_to_node_stats: dict. Stores scores for edges between nodes representing their relationship.

        """
        logger.info("Expanding graph with synonymy edges")

        self.entity_id_to_row = self.entity_embedding_store.get_all_id_to_rows()

        logger.info(f"Performing KNN retrieval for each phrase nodes ({len(self.entity_id_to_row)}).")

        query_node_key2knn_node_keys = self.entity_embedding_store.internal_cross_knn(
            top_k=self.global_config.synonymy_edge_topk,
        )

        num_synonym_triple = 0
        synonym_candidates = []  # [(node key, [(synonym node key, corresponding score), ...]), ...]

        for node_key in tqdm(query_node_key2knn_node_keys.keys(), total=len(query_node_key2knn_node_keys)):
            synonyms = []

            entity = self.entity_id_to_row[node_key]["content"]

            if len(re.sub(r"[^A-Za-z0-9]", "", entity)) > 2:
                nns = query_node_key2knn_node_keys[node_key]

                num_nns = 0
                for nn, score in zip(nns[0], nns[1], strict=False):
                    if score < self.global_config.synonymy_edge_sim_threshold or num_nns > 100:
                        break

                    nn_phrase = self.entity_id_to_row[nn]["content"]

                    if nn != node_key and nn_phrase != "":
                        sim_edge = (node_key, nn)
                        synonyms.append((nn, score))
                        num_synonym_triple += 1

                        self.node_to_node_stats[sim_edge] = score  # Need to seriously discuss on this
                        num_nns += 1

            synonym_candidates.append((node_key, synonyms))

    def merge_openie_results(
        self,
        all_openie_info: list[OpenIEDocItem],
        chunks_to_save: dict[str, dict],
        ner_results_dict: dict[str, NerRawOutput],
        triple_results_dict: dict[str, TripleRawOutput],
    ) -> list[OpenIEDocItem]:
        """
        Merges OpenIE extraction results with corresponding passage and metadata.

        This function integrates the OpenIE extraction results, including named-entity
        recognition (NER) entities and triples, with their respective text passages
        using the provided chunk keys. The resulting merged data is appended to
        the `all_openie_info` list containing dictionaries with combined and organized
        data for further processing or storage.

        Parameters:
            all_openie_info (List[dict]): A list to hold dictionaries of merged OpenIE
                results and metadata for all chunks.
            chunks_to_save (Dict[str, dict]): A dict of chunk identifiers (keys) to process
                and merge OpenIE results to dictionaries with `hash_id` and `content` keys.
            ner_results_dict (Dict[str, NerRawOutput]): A dictionary mapping chunk keys
                to their corresponding NER extraction results.
            triple_results_dict (Dict[str, TripleRawOutput]): A dictionary mapping chunk
                keys to their corresponding OpenIE triple extraction results.

        Returns:
            List[OpenIEDocItem]: The `all_openie_info` list containing dictionaries with merged
            OpenIE results, metadata, and the passage content for each chunk.

        """

        for chunk_key, row in chunks_to_save.items():
            passage = row["content"]
            metadata = row.get("metadata", {})  # Extract metadata
            chunk_openie_info = OpenIEDocItem(
                idx=chunk_key,
                passage=passage,
                extracted_entities=ner_results_dict[chunk_key].unique_entities,
                extracted_triples=triple_results_dict[chunk_key].triples,
                metadata=metadata,  # Include metadata
            )
            all_openie_info.append(chunk_openie_info)

        return all_openie_info

    def save_openie_results(self, all_openie_info: list[OpenIEDocItem]):
        """
        Computes statistics on extracted entities from OpenIE results and saves the aggregated data in a
        JSON file. The function calculates the average character and word lengths of the extracted entities
        and writes them along with the provided OpenIE information to a file.

        Parameters:
            all_openie_info : List[dict]
                List of dictionaries, where each dictionary represents information from OpenIE, including
                extracted entities.
        """

        sum_phrase_chars = sum([len(e) for chunk in all_openie_info for e in chunk.extracted_entities])
        sum_phrase_words = sum([len(e.split()) for chunk in all_openie_info for e in chunk.extracted_entities])
        num_phrases = sum([len(chunk.extracted_entities) for chunk in all_openie_info])

        if len(all_openie_info) > 0:
            # Avoid division by zero if there are no phrases
            if num_phrases > 0:
                avg_ent_chars = round(sum_phrase_chars / num_phrases, 4)
                avg_ent_words = round(sum_phrase_words / num_phrases, 4)
            else:
                avg_ent_chars = 0
                avg_ent_words = 0

            openie_res = OpenIEResult(
                docs=all_openie_info,
                avg_ent_chars=avg_ent_chars,
                avg_ent_words=avg_ent_words,
            )

            self.openie_results_path.write_bytes(openie_res.model_dump_json().encode())
            logger.info(f"OpenIE results saved to {self.openie_results_path}")

    def augment_graph(self) -> None:
        """
        Provides utility functions to augment a graph by adding new nodes and edges.
        It ensures that the graph structure is extended to include additional components,
        and logs the completion status along with printing the updated graph information.
        """

        self.add_new_nodes()
        self.add_new_edges()

        logger.info("Graph construction completed!")
        logger.info(self.get_graph_info())

    def add_new_nodes(self) -> None:
        """
        Adds new nodes to the graph from entity and passage embedding stores based on their attributes.

        This method identifies and adds new nodes to the graph by comparing existing nodes
        in the graph and nodes retrieved from the entity embedding store and the passage
        embedding store. The method checks attributes and ensures no duplicates are added.
        New nodes are prepared and added in bulk to optimize graph updates.
        """

        existing_nodes = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()}

        entity_to_row = self.entity_embedding_store.get_all_id_to_rows()
        passage_to_row = self.chunk_embedding_store.get_all_id_to_rows()

        node_to_rows = entity_to_row
        node_to_rows.update(passage_to_row)

        new_nodes: dict[str, list[Any]] = {}
        for node_id, node in node_to_rows.items():
            node["name"] = node_id
            # Ensure metadata field exists
            if "metadata" not in node:
                node["metadata"] = {}

            # Add chunk_metadata_map metadata to passage nodes
            if (
                node_id.startswith("chunk-")
                and hasattr(self, "chunk_metadata_map")
                and node_id in self.chunk_metadata_map
            ):
                node["metadata"].update(self.chunk_metadata_map[node_id])

            if node_id not in existing_nodes:
                for k, v in node.items():
                    if k not in new_nodes:
                        new_nodes[k] = []
                    new_nodes[k].append(v)

        if len(new_nodes) > 0:
            self.graph.add_vertices(n=len(next(iter(new_nodes.values()))), attributes=new_nodes)

    def add_new_edges(self) -> None:
        """
        Processes edges from `node_to_node_stats` to add them into a graph object while
        managing adjacency lists, validating edges, and logging invalid edge cases.
        """

        graph_adj_list = defaultdict[str, dict[str, float]](dict)
        graph_inverse_adj_list = defaultdict[str, dict[str, float]](dict)
        edge_source_node_keys = []
        edge_target_node_keys = []
        edge_metadata = []
        for edge, weight in self.node_to_node_stats.items():
            if edge[0] == edge[1]:
                continue
            graph_adj_list[edge[0]][edge[1]] = weight
            graph_inverse_adj_list[edge[1]][edge[0]] = weight

            edge_source_node_keys.append(edge[0])
            edge_target_node_keys.append(edge[1])
            edge_metadata.append({"weight": weight})

        valid_edges, valid_weights = [], {"weight": []}
        current_node_ids = set(self.graph.vs["name"])
        for source_node_id, target_node_id, edge_d in zip(
            edge_source_node_keys, edge_target_node_keys, edge_metadata, strict=False
        ):
            if source_node_id in current_node_ids and target_node_id in current_node_ids:
                valid_edges.append((source_node_id, target_node_id))
                weight = edge_d.get("weight", 1.0)
                valid_weights["weight"].append(weight)
            else:
                logger.warning(f"Edge {source_node_id} -> {target_node_id} is not valid.")
        self.graph.add_edges(valid_edges, attributes=valid_weights)

    def save_igraph(self) -> None:
        logger.info(f"Writing graph with {len(self.graph.vs())} nodes, {len(self.graph.es())} edges")
        self.graph.write_pickle(self._graph_pickle_filename)
        logger.info("Saving graph completed!")

    def get_graph_info(self) -> dict:
        """
        Obtains detailed information about the graph such as the number of nodes,
        triples, and their classifications.

        This method calculates various statistics about the graph based on the
        stores and node-to-node relationships, including counts of phrase and
        passage nodes, total nodes, extracted triples, triples involving passage
        nodes, synonymy triples, and total triples.

        Returns:
            Dict
                A dictionary containing the following keys and their respective values:
                - num_phrase_nodes: The number of unique phrase nodes.
                - num_passage_nodes: The number of unique passage nodes.
                - num_total_nodes: The total number of nodes (sum of phrase and passage nodes).
                - num_extracted_triples: The number of unique extracted triples.
                - num_triples_with_passage_node: The number of triples involving at least one
                  passage node.
                - num_synonymy_triples: The number of synonymy triples (distinct from extracted
                  triples and those with passage nodes).
                - num_total_triples: The total number of triples.
        """
        graph_info = {}

        # get # of phrase nodes
        phrase_nodes_keys = self.entity_embedding_store.get_all_ids()
        graph_info["num_phrase_nodes"] = len(set(phrase_nodes_keys))

        # get # of passage nodes
        passage_nodes_keys = self.chunk_embedding_store.get_all_ids()
        graph_info["num_passage_nodes"] = len(set(passage_nodes_keys))

        # get # of total nodes
        graph_info["num_total_nodes"] = graph_info["num_phrase_nodes"] + graph_info["num_passage_nodes"]

        # get # of extracted triples
        graph_info["num_extracted_triples"] = len(self.fact_embedding_store.get_all_ids())

        num_triples_with_passage_node = 0
        passage_nodes_set = set(passage_nodes_keys)
        num_triples_with_passage_node = sum(
            1
            for node_pair in self.node_to_node_stats
            if node_pair[0] in passage_nodes_set or node_pair[1] in passage_nodes_set
        )
        graph_info["num_triples_with_passage_node"] = num_triples_with_passage_node

        graph_info["num_synonymy_triples"] = (
            len(self.node_to_node_stats) - graph_info["num_extracted_triples"] - num_triples_with_passage_node
        )

        # get # of total triples
        graph_info["num_total_triples"] = len(self.node_to_node_stats)

        return graph_info

    def get_top_k_weights(
        self,
        link_top_k: int,
        all_phrase_weights: _Array1D[np.float64],
        linking_score_map: dict[str, float],
    ) -> tuple[_Array1D[np.float64], dict[str, float]]:
        """
        This function filters the all_phrase_weights to retain only the weights for the
        top-ranked phrases in terms of the linking_score_map. It also filters linking scores
        to retain only the top `link_top_k` ranked nodes. Non-selected phrases in phrase
        weights are reset to a weight of 0.0.

        Args:
            link_top_k (int): Number of top-ranked nodes to retain in the linking score map.
            all_phrase_weights (np.ndarray): An array representing the phrase weights, indexed
                by phrase ID.
            linking_score_map (Dict[str, float]): A mapping of phrase content to its linking
                score, sorted in descending order of scores.

        Returns:
            Tuple[np.ndarray, Dict[str, float]]: A tuple containing the filtered array
            of all_phrase_weights with unselected weights set to 0.0, and the filtered
            linking_score_map containing only the top `link_top_k` phrases.
        """
        # choose top ranked nodes in linking_score_map
        linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:link_top_k])

        # only keep the top_k phrases in all_phrase_weights
        top_k_phrases = set(linking_score_map.keys())
        top_k_phrases_keys = set([
            compute_mdhash_id(content=top_k_phrase, prefix="entity-") for top_k_phrase in top_k_phrases
        ])

        for phrase_key in self.node_name_to_vertex_idx:
            if phrase_key not in top_k_phrases_keys:
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)
                if phrase_id is not None:
                    all_phrase_weights[phrase_id] = 0.0

        assert np.count_nonzero(all_phrase_weights) == len(linking_score_map.keys())
        return all_phrase_weights, linking_score_map

    def run_ppr(
        self,
        reset_prob: npt.NDArray[np.float64],
        damping: float = 0.5,
        metadata_filters: dict[str, Any] | None = None,
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.float64]]:
        """
        Runs Personalized PageRank (PPR) on a graph and computes relevance scores for
        nodes corresponding to document passages. The method utilizes a damping
        factor for teleportation during rank computation and can take a reset
        probability array to influence the starting state of the computation.

        Parameters:
            reset_prob (np.ndarray): A 1-dimensional array specifying the reset
                probability distribution for each node. The array must have a size
                equal to the number of nodes in the graph. NaNs or negative values
                within the array are replaced with zeros.
            damping (float): A scalar specifying the damping factor for the
                computation. Defaults to 0.5 if not provided or set to `None`.
            metadata_filters (dict, optional): Metadata filters to create a subgraph
                for PPR. Only nodes matching the filters will be included.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays. The
                first array represents the sorted node IDs of document passages based
                on their relevance scores in descending order. The second array
                contains the corresponding relevance scores of each document passage
                in the same order.
        """

        if damping is None:
            damping = 0.5  # for potential compatibility
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)

        # Determine if we need to create a subgraph based on metadata filters
        if metadata_filters:
            subgraph_result = self._create_subgraph_by_metadata(metadata_filters)
            if subgraph_result is not None:
                valid_passage_idxs, valid_entity_idxs, valid_chunk_ids = subgraph_result
                valid_node_idxs = valid_passage_idxs | valid_entity_idxs

                # Use igraph's induced_subgraph API to create an induced subgraph
                # Use 'auto' to let igraph choose the best implementation
                logger.info(
                    f"Creating subgraph for PPR: {len(valid_passage_idxs)} passages, "
                    f"{len(valid_entity_idxs)} entities from {len(valid_chunk_ids)} chunks "
                    f"(total nodes: {len(valid_node_idxs)})"
                )

                subgraph: ig.Graph = self.graph.induced_subgraph(list(valid_node_idxs), implementation="auto")

                # Map original node indices to subgraph indices
                original_to_subgraph_idx = {orig_idx: i for i, orig_idx in enumerate(valid_node_idxs)}

                # Filter reset_prob to only include nodes in subgraph
                subgraph_reset_prob = np.array([
                    reset_prob[orig_idx] if orig_idx < len(reset_prob) else 0.0 for orig_idx in valid_node_idxs
                ])

                # Run PPR on the subgraph
                pagerank_scores: list[float] = subgraph.personalized_pagerank(
                    vertices=range(len(subgraph.vs)),
                    damping=damping,
                    directed=False,
                    weights="weight",
                    reset=subgraph_reset_prob,
                    implementation="prpack",
                )

                # Get passage indices that are in the subgraph
                subgraph_passage_idxs = [idx for idx in self.passage_node_idxs if idx in valid_passage_idxs]

                # Map passage indices to subgraph indices
                subgraph_passage_to_subgraph_idx = {
                    orig_idx: original_to_subgraph_idx[orig_idx] for orig_idx in subgraph_passage_idxs
                }

                # Extract scores for passage nodes in the correct order
                doc_scores: npt.NDArray[np.float64] = np.array([
                    pagerank_scores[subgraph_passage_to_subgraph_idx[idx]] for idx in subgraph_passage_idxs
                ])

                # Get the original indices in the full passage_node_idxs array
                sorted_doc_indices = np.argsort(doc_scores)[::-1]
                sorted_doc_ids = np.array(
                    [self.passage_node_idxs.index(subgraph_passage_idxs[idx]) for idx in sorted_doc_indices],
                    dtype=np.intp,
                )
                sorted_doc_scores = doc_scores[sorted_doc_indices]

                logger.info(f"PPR on subgraph returned {len(sorted_doc_ids)} results")

                return sorted_doc_ids, sorted_doc_scores

        # Run PPR on the full graph (no metadata filters or no matching chunks)
        pagerank_scores: list[float] = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=damping,
            directed=False,
            weights="weight",
            reset=reset_prob,
            implementation="prpack",
        )

        doc_scores: npt.NDArray[np.float64] = np.array([pagerank_scores[idx] for idx in self.passage_node_idxs])
        sorted_doc_ids = np.argsort(doc_scores)[::-1]
        sorted_doc_scores = doc_scores[sorted_doc_ids.tolist()]

        return sorted_doc_ids, sorted_doc_scores

    def _create_subgraph_by_metadata(
        self,
        metadata_filters: dict[str, Any] | None,
    ) -> tuple[set[int], set[int], set[str]] | None:
        """
        Creates a subgraph based on metadata filters.

        Args:
            metadata_filters: Dict of metadata key-value pairs to filter by

        Returns:
            Tuple of (valid_passage_idxs, valid_entity_idxs, valid_chunk_ids) if filters provided,
            None if no filters (use full graph)
        """
        if not metadata_filters:
            return None

        # Filter chunk IDs by metadata
        all_chunk_ids = list(self.chunk_embedding_store.get_all_ids())
        valid_chunk_ids = {
            chunk_id for chunk_id in all_chunk_ids if self._chunk_matches_metadata(chunk_id, metadata_filters)
        }

        if not valid_chunk_ids:
            logger.warning(f"No chunks match metadata filters: {metadata_filters}")
            return None

        # Get passage node indices for valid chunks
        valid_passage_idxs = {
            self.node_name_to_vertex_idx[chunk_id]
            for chunk_id in valid_chunk_ids
            if chunk_id in self.node_name_to_vertex_idx
        }

        # Find all entity nodes connected to valid passage nodes
        valid_entity_idxs = set()
        for passage_idx in valid_passage_idxs:
            # Get neighbors of this passage node
            neighbors = self.graph.neighbors(passage_idx)
            for neighbor_idx in neighbors:
                neighbor_name = self.graph.vs[neighbor_idx]["name"]
                # Only include entity nodes (starting with "entity-")
                if neighbor_name.startswith("entity-"):
                    valid_entity_idxs.add(neighbor_idx)

        logger.info(
            f"Created subgraph: {len(valid_passage_idxs)} passages, "
            f"{len(valid_entity_idxs)} entities from {len(valid_chunk_ids)} chunks"
        )

        return valid_passage_idxs, valid_entity_idxs, valid_chunk_ids

    def _chunk_matches_metadata(self, chunk_id: str, filters: dict[str, Any]) -> bool:
        """Check if a chunk matches metadata filters."""
        try:
            chunk_row = self.chunk_embedding_store.get_row(chunk_id)
            metadata = chunk_row.get("metadata", {})
            return all(metadata.get(key) == value for key, value in filters.items())
        except KeyError:
            return False
