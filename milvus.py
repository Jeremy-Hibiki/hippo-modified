import logging
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, cast
from typing_extensions import override

import numpy as np
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.storage.kvstore.types import DEFAULT_BATCH_SIZE
from llama_index.core.utils import iter_batch
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    DEFAULT_DOC_ID_KEY,
    DEFAULT_EMBEDDING_KEY,
    metadata_dict_to_node,
    node_to_metadata_dict,
)
from llama_index.vector_stores.milvus import IndexManagement, MilvusVectorStore
from llama_index.vector_stores.milvus.base import MILVUS_ID_FIELD
from llama_index.vector_stores.milvus.utils import BaseSparseEmbeddingFunction, get_default_sparse_embedding_function
from pymilvus import AnnSearchRequest, AsyncMilvusClient, Collection, DataType, MilvusClient
from pymilvus.client.types import LoadState
from transformers import PreTrainedTokenizerBase

try:
    from pymilvus import RRFRanker, WeightedRanker
except Exception:
    WeightedRanker, RRFRanker = None, None

logger = logging.getLogger(__name__)

DEFAULT_BM25_FIELD = "bm25_sparse"

dispatcher = get_dispatcher()


class MilvusWithBM25Function(MilvusVectorStore):
    enable_bm25: bool = False
    bm25_field: str = DEFAULT_BM25_FIELD
    analyzer_params: dict | None = None

    def __init__(
        self,
        uri: str = "./milvus_llamaindex.db",
        token: str = "",
        collection_name: str = "llamacollection",
        dim: int | None = None,
        embedding_field: str = DEFAULT_EMBEDDING_KEY,
        doc_id_field: str = DEFAULT_DOC_ID_KEY,
        similarity_metric: str = "IP",
        consistency_level: str = "Session",
        overwrite: bool = False,
        text_key: str | None = None,
        output_fields: list[str] | None = None,
        index_config: dict | None = None,
        search_config: dict | None = None,
        collection_properties: dict | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        enable_bm25: bool = False,
        bm25_field: str = DEFAULT_BM25_FIELD,
        analyzer_params: dict | None = None,
        enable_sparse: bool = False,
        sparse_embedding_function: BaseSparseEmbeddingFunction | None = None,
        hybrid_ranker: str = "RRFRanker",
        hybrid_ranker_params: dict = {},
        index_management: IndexManagement = IndexManagement.CREATE_IF_NOT_EXISTS,
        scalar_field_names: list[str] | None = None,
        scalar_field_types: list[DataType] | None = None,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super(BasePydanticVectorStore, self).__init__(
            collection_name=collection_name,
            dim=dim,
            embedding_field=embedding_field,
            doc_id_field=doc_id_field,
            consistency_level=consistency_level,
            overwrite=overwrite,
            text_key=text_key,
            output_fields=output_fields or [],
            index_config=index_config if index_config else {},
            search_config=search_config if search_config else {},
            collection_properties=collection_properties,
            batch_size=batch_size,
            enable_bm25=enable_bm25,
            bm25_field=bm25_field,
            analyzer_params=analyzer_params,
            enable_sparse=enable_sparse,
            sparse_embedding_function=sparse_embedding_function,
            hybrid_ranker=hybrid_ranker,
            hybrid_ranker_params=hybrid_ranker_params,
            index_management=index_management,
            scalar_field_names=scalar_field_names,
            scalar_field_types=scalar_field_types,
        )

        if enable_bm25 and not text_key:
            raise ValueError("text_key is required to enable builtin BM25 function.")

        # Select the similarity metric
        similarity_metrics_map = {
            "ip": "IP",
            "l2": "L2",
            "euclidean": "L2",
            "cosine": "COSINE",
        }
        self.similarity_metric = similarity_metrics_map.get(similarity_metric.lower(), "L2")
        # Connect to Milvus instance
        self._milvusclient = MilvusClient(
            uri=uri,
            token=token,
            **kwargs,  # pass additional arguments such as server_pem_path
        )
        self._async_milvusclient = AsyncMilvusClient(
            uri=uri,
            token=token,
            **kwargs,  # pass additional arguments such as server_pem_path
        )
        # Delete previous collection if overwriting
        if overwrite and collection_name in self.client.list_collections():
            self.client.drop_collection(collection_name)

        # Create the collection if it does not exist
        if collection_name not in self.client.list_collections():
            if dim is None:
                raise ValueError("Dim argument required for collection creation.")
            if self.enable_sparse is False:
                # Check if custom index should be created
                if index_config is not None and self.index_management is not IndexManagement.NO_VALIDATION:
                    try:
                        # Create a schema according to LlamaIndex Schema.
                        schema = self._create_schema()
                        schema.verify()

                        # Prepare index
                        index_params = self.client.prepare_index_params()
                        index_type = index_config["index_type"]
                        index_params.add_index(
                            field_name=embedding_field,
                            index_type=index_type,
                            metric_type=self.similarity_metric,
                        )
                        if enable_bm25:
                            index_params.add_index(
                                field_name=bm25_field,
                                index_type="AUTOINDEX",
                                metric_type="BM25",
                            )

                        # Using private method exposed by pymilvus client, in order to avoid creating indexes twice
                        # Reason: create_collection in pymilvus only checks schema and ignores index_config setup
                        # https://github.com/milvus-io/pymilvus/issues/2265
                        self.client._create_collection_with_schema(
                            collection_name=collection_name,
                            schema=schema,
                            index_params=index_params,
                            dimemsion=dim,
                            primary_field=MILVUS_ID_FIELD,
                            vector_field=embedding_field,
                            id_type="string",
                            max_length=65_535,
                            consistency_level=consistency_level,
                        )
                        self._collection = Collection(collection_name, using=self.client._using)
                    except Exception as e:
                        logger.error("Error creating collection with index_config")
                        raise NotImplementedError("Error creating collection with index_config") from e
                else:
                    schema = self._create_schema()
                    schema.verify()

                    self.client.create_collection(
                        collection_name=collection_name,
                        dimension=dim,
                        primary_field_name=MILVUS_ID_FIELD,
                        vector_field_name=embedding_field,
                        id_type="string",
                        metric_type=self.similarity_metric,
                        max_length=65_535,
                        schema=schema,
                        consistency_level=consistency_level,
                    )
                    self._collection = Collection(collection_name, using=self.client._using)

                    # Check if we have to create an index here to avoid duplicity of indexes
                    self._create_index_if_required()
            else:
                try:
                    _ = DataType.SPARSE_FLOAT_VECTOR
                except Exception as e:
                    logger.error("Hybrid retrieval is only supported in Milvus 2.4.0 or later.")
                    raise NotImplementedError("Hybrid retrieval requires Milvus 2.4.0 or later.") from e
                self._create_hybrid_index(collection_name)
        else:
            self._collection = Collection(collection_name, using=self.client._using)

        # Set properties
        if collection_properties:
            if self.client.get_load_state(collection_name)["state"] == LoadState.Loaded:
                self._collection.release()
                self._collection.set_properties(properties=collection_properties)
                self._collection.load()
            else:
                self._collection.set_properties(properties=collection_properties)

        if enable_sparse is True and sparse_embedding_function is None:
            logger.warning("Sparse embedding function is not provided, using default.")
            self.sparse_embedding_function = get_default_sparse_embedding_function()
        elif enable_sparse is True and sparse_embedding_function is not None:
            self.sparse_embedding_function = sparse_embedding_function
        else:
            pass

        logger.debug(f"Successfully created a new collection: {self.collection_name}")

    @override
    def _create_schema(self):
        schema = super()._create_schema()
        if self.enable_bm25:
            try:
                from pymilvus import Function, FunctionType
            except ImportError:
                raise ImportError(
                    "Builtin function is supported in pymilvus 2.5.0 or later. Please install it by `pip install -U pymilvus`"
                ) from None
            assert self.text_key is not None
            analyzer_params = self.analyzer_params or {"type": "standard"}
            schema.add_field(
                field_name=self.text_key,
                datatype=DataType.VARCHAR,
                max_length=65_535,
                enable_analyzer=True,
                enable_match=True,
                analyzer_params=analyzer_params,
            )
            schema.add_field(field_name=self.bm25_field, datatype=DataType.SPARSE_FLOAT_VECTOR)
            schema.add_function(
                Function(
                    name=f"{self.text_key}_bm25",
                    input_field_names=[self.text_key],
                    output_field_names=[self.bm25_field],
                    function_type=FunctionType.BM25,
                )
            )
        return schema

    @override
    def _create_index_if_required(self) -> None:
        if self.index_management == IndexManagement.NO_VALIDATION:
            return

        if self.enable_bm25:
            self._create_bm25_index()

        if self.enable_sparse is False:
            self._create_dense_index()
        else:
            self._create_hybrid_index(self.collection_name)

    @override
    def _create_dense_index(self) -> None:
        """
        Create or recreate the dense vector index.

        This method handles the creation of the dense vector index based on
        the current index management strategy and the state of the collection.
        """
        index_exists = self._collection.has_index(index_name=self.embedding_field)

        if (not index_exists and self.index_management == IndexManagement.CREATE_IF_NOT_EXISTS) or (
            index_exists and self.overwrite
        ):
            if index_exists:
                self._collection.release()
                self._collection.drop_index()

            base_params: dict[str, Any] = self.index_config.copy()
            index_type: str = base_params.pop("index_type", "FLAT")
            index_params: dict[str, str | dict[str, Any]] = {
                "params": base_params,
                "metric_type": self.similarity_metric,
                "index_type": index_type,
            }
            self._collection.create_index(self.embedding_field, index_params=index_params)
            self._collection.load()

    def _create_bm25_index(self) -> None:
        coll = cast(Collection, self._collection)
        index_exists = coll.has_index(index_name=self.bm25_field)
        if (not index_exists and self.index_management == IndexManagement.CREATE_IF_NOT_EXISTS) or (
            index_exists and self.overwrite
        ):
            if index_exists:
                coll.release()
                coll.drop_index(index_name=self.bm25_field)

            coll.create_index(
                self.bm25_field,
                index_params=dict(
                    index_type="AUTOINDEX",
                    metric_type="BM25",
                ),
            )

    @override
    def _create_hybrid_index(self, collection_name: str) -> None:
        """
        Create or recreate the hybrid (dense and sparse) vector index.

        Args:
            collection_name (str): The name of the collection to create the index for.
        """
        # Check if the collection exists, if not, create it
        if collection_name not in self.client.list_collections():
            schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
            schema.add_field(
                field_name="id",
                datatype=DataType.VARCHAR,
                max_length=65535,
                is_primary=True,
            )
            schema.add_field(
                field_name=self.embedding_field,
                datatype=DataType.FLOAT_VECTOR,
                dim=self.dim,
            )
            schema.add_field(
                field_name=self.sparse_embedding_field,
                datatype=DataType.SPARSE_FLOAT_VECTOR,
            )
            if self.enable_bm25:
                try:
                    from pymilvus import Function, FunctionType
                except ImportError:
                    raise ImportError(
                        "Builtin function is supported in pymilvus 2.5.0 or later. Please install it by `pip install -U pymilvus`"
                    ) from None
                assert self.text_key is not None
                analyzer_params = self.analyzer_params or {"type": "standard"}
                schema.add_field(
                    field_name=self.text_key,
                    datatype=DataType.VARCHAR,
                    max_length=65_535,
                    enable_analyzer=True,
                    enable_match=True,
                    analyzer_params=analyzer_params,
                )
                schema.add_field(
                    field_name=self.bm25_field,
                    datatype=DataType.SPARSE_FLOAT_VECTOR,
                )
                schema.add_function(
                    Function(
                        name=f"{self.text_key}_bm25",
                        input_field_names=[self.text_key],
                        output_field_names=[self.bm25_field],
                        function_type=FunctionType.BM25,
                    )
                )
            self.client.create_collection(collection_name=collection_name, schema=schema)

        # Initialize or get the collection
        self._collection = Collection(collection_name, using=self.client._using)

        dense_index_exists = self._collection.has_index(index_name=self.embedding_field)
        sparse_index_exists = self._collection.has_index(index_name=self.sparse_embedding_field)
        bm25_index_exists = (not self.enable_bm25) or self._collection.has_index(index_name=self.bm25_field)

        if (
            (not dense_index_exists or not sparse_index_exists or not bm25_index_exists)
            and self.index_management == IndexManagement.CREATE_IF_NOT_EXISTS
        ) or (dense_index_exists and sparse_index_exists and bm25_index_exists and self.overwrite):
            if dense_index_exists:
                self._collection.release()
                self._collection.drop_index(index_name=self.embedding_field)
            if sparse_index_exists:
                self._collection.drop_index(index_name=self.sparse_embedding_field)
            if bm25_index_exists and self.enable_bm25:
                self._collection.drop_index(index_name=self.bm25_field)

            # Create sparse index
            sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
            self._collection.create_index(self.sparse_embedding_field, sparse_index)

            # Create dense index
            base_params = self.index_config.copy()
            index_type = base_params.pop("index_type", "FLAT")
            dense_index = {
                "params": base_params,
                "metric_type": self.similarity_metric,
                "index_type": index_type,
            }
            self._collection.create_index(self.embedding_field, dense_index)

            if self.enable_bm25:
                bm25_index = {"index_type": "AUTOINDEX", "metric_type": "BM25"}
                self._collection.create_index(self.bm25_field, bm25_index)

        self._collection.load()

    @override
    def add(self, nodes: Sequence[BaseNode], **add_kwargs: Any) -> list[str]:
        """Add the embeddings and their nodes into Milvus.

        Args:
            nodes (List[BaseNode]): List of nodes with embeddings
                to insert.

        Raises:
            MilvusException: Failed to insert data.

        Returns:
            List[str]: List of ids inserted.
        """
        insert_list = []
        insert_ids = []

        if self.enable_sparse is True and self.sparse_embedding_function is None:
            logging.fatal("sparse_embedding_function is None when enable_sparse is True.")

        # Process that data we are going to insert
        for node in cast(list[TextNode], nodes):
            if self.text_key:
                entry = node_to_metadata_dict(node, remove_text=True)
                entry[self.text_key] = node.text
            else:
                entry = node_to_metadata_dict(node)
            entry[MILVUS_ID_FIELD] = node.node_id
            entry[self.embedding_field] = node.embedding

            if self.enable_sparse is True:
                entry[self.sparse_embedding_field] = self.sparse_embedding_function.encode_documents([node.text])[0]

            insert_ids.append(node.node_id)
            insert_list.append(entry)

        # Insert the data into milvus
        for insert_batch in iter_batch(insert_list, self.batch_size):
            self.client.insert(self.collection_name, insert_batch)
        if add_kwargs.get("force_flush", False):
            self.client.flush(self.collection_name)
        return insert_ids

    @override
    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query_embedding (List[float]): query embedding
            similarity_top_k (int): top k most similar nodes
            doc_ids (Optional[List[str]]): list of doc_ids to filter by
            node_ids (Optional[List[str]]): list of node_ids to filter by
            output_fields (Optional[List[str]]): list of fields to return
            embedding_field (Optional[str]): name of embedding field
        """
        if query.mode == VectorStoreQueryMode.DEFAULT:
            pass
        elif query.mode == VectorStoreQueryMode.HYBRID:
            if self.enable_sparse is False:
                raise ValueError("QueryMode is HYBRID, but enable_sparse is False.")
        elif query.mode == VectorStoreQueryMode.MMR or query.mode == VectorStoreQueryMode.SEMANTIC_HYBRID:
            pass
        else:
            raise ValueError(f"Milvus does not support {query.mode} yet.")

        string_expr, output_fields = self._prepare_before_search(query, **kwargs)

        # Perform the search
        if query.mode == VectorStoreQueryMode.DEFAULT:
            nodes, similarities, ids = self._default_search(query, string_expr, output_fields, **kwargs)

        elif query.mode == VectorStoreQueryMode.MMR:
            nodes, similarities, ids = self._mmr_search(query, string_expr, output_fields, **kwargs)

        elif query.mode == VectorStoreQueryMode.HYBRID:
            nodes, similarities, ids = self._hybrid_search(query, string_expr, output_fields)

        elif query.mode == VectorStoreQueryMode.SEMANTIC_HYBRID:
            nodes, similarities, ids = self._semantic_hybrid_search(query, string_expr, output_fields)

        else:
            raise ValueError(f"Milvus does not support {query.mode} yet.")

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    def _semantic_hybrid_search(
        self, query: VectorStoreQuery, string_expr: str, output_fields: list[str]
    ) -> tuple[list[BaseNode], list[float], list[str]]:
        """
        Perform hybrid search.
        """
        if self.enable_bm25:
            try:
                from pymilvus import Function, FunctionType  # noqa: F401
            except ImportError:
                raise ImportError(
                    "BM25 search is not supported in this version of Milvus. Please upgrade to Milvus 2.5.0 or later."
                ) from None
        dense_search_params = {
            "metric_type": self.similarity_metric,
            "params": self.search_config,
        }
        dense_emb = query.query_embedding
        dense_req = AnnSearchRequest(
            data=[dense_emb],
            anns_field=self.embedding_field,
            param=dense_search_params,
            limit=query.similarity_top_k,
            expr=string_expr,  # Apply metadata filters to dense search
        )
        reqs = [dense_req]
        if self.enable_bm25:
            bm25_search_params = {"metric_type": "BM25"}
            bm25_req = AnnSearchRequest(
                data=[query.query_str],
                anns_field=self.bm25_field,
                param=bm25_search_params,
                limit=query.similarity_top_k,
                expr=string_expr,  # Apply metadata filters to bm25 search
            )
            reqs.append(bm25_req)

        if self.enable_sparse:
            sparse_emb = self.sparse_embedding_function.encode_queries([query.query_str])[0]
            sparse_search_params = {"metric_type": "IP"}
            sparse_req = AnnSearchRequest(
                data=[sparse_emb],
                anns_field=self.sparse_embedding_field,
                param=sparse_search_params,
                limit=query.similarity_top_k,
                expr=string_expr,  # Apply metadata filters to sparse search
            )
            reqs.append(sparse_req)

        if WeightedRanker is None or RRFRanker is None:
            logger.error("Hybrid retrieval is only supported in Milvus 2.4.0 or later.")
            raise ValueError("Hybrid retrieval is only supported in Milvus 2.4.0 or later.")
        if self.hybrid_ranker == "WeightedRanker":
            if self.hybrid_ranker_params == {}:
                self.hybrid_ranker_params = {"weights": [1.0, 1.0, 1.0]}
            ranker = WeightedRanker(*self.hybrid_ranker_params["weights"])
        elif self.hybrid_ranker == "RRFRanker":
            if self.hybrid_ranker_params == {}:
                self.hybrid_ranker_params = {"k": 60}
            ranker = RRFRanker(self.hybrid_ranker_params["k"])
        else:
            raise ValueError(f"Unsupported ranker: {self.hybrid_ranker}")

        if not hasattr(self.client, "hybrid_search"):
            raise ValueError(
                "Your pymilvus version does not support hybrid search. please update it by `pip install -U pymilvus`"
            )
        res = self.client.hybrid_search(
            self.collection_name,
            reqs,
            ranker=ranker,
            limit=query.similarity_top_k,
            output_fields=output_fields,
        )
        logger.debug(
            f"Successfully searched embedding in collection: {self.collection_name} Num Results: {len(res[0])}"
        )
        nodes, similarities, ids = self._parse_from_milvus_results(res)
        return nodes, similarities, ids

    @override
    def _parse_from_milvus_results(self, results: list) -> tuple[list[BaseNode], list[float], list[str]]:
        """
        Parses the results from Milvus and returns a list of nodes, similarities and ids.
        Only parse the first result since we are only searching for one query.
        """
        if len(results) > 1:
            logger.warning("More than one result found in Milvus search. Only parsing the first result.")
        nodes = []
        similarities = []
        ids = []
        # Parse the results
        for hit in results[0]:
            if not self.text_key:
                node = metadata_dict_to_node(
                    {
                        "_node_content": hit["entity"].get("_node_content", None),
                        "_node_type": hit["entity"].get("_node_type", None),
                    }
                )
            else:
                try:
                    text = hit["entity"].get(self.text_key)
                except Exception:
                    raise ValueError("The passed in text_key value does not exist in the retrieved entity.") from None

                if hit["entity"].get("_node_content", None):
                    node = metadata_dict_to_node(hit["entity"], text)
                else:
                    metadata = {key: hit["entity"].get(key) for key in self.output_fields}
                    node = TextNode(text=text, metadata=metadata)

            nodes.append(node)
            similarities.append(hit["distance"])
            ids.append(hit["id"])
        return nodes, similarities, ids


class TritonRemoteM3SparseEmbeddingFunction(BaseSparseEmbeddingFunction):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        server_url: str,
        model_name: str,
    ):
        from tritonclient.grpc import InferenceServerClient

        self.tokenizer = tokenizer
        self.server_url = server_url
        self.model_name = model_name
        self.client = InferenceServerClient(self.server_url)

    def _embed_one_batch(self, text: str | list[str]) -> tuple[list[float], list[dict[int, float]]]:
        from tritonclient.grpc import InferInput

        inputs_arr = [text] if isinstance(text, str) else text

        tokenized = self.tokenizer(text, truncation=True, padding=True, return_tensors="np")
        input_ids, attention_mask = tokenized.input_ids, tokenized.attention_mask  # type: ignore
        triton_inputs = [
            InferInput("input_ids", [len(inputs_arr), len(input_ids[0])], "INT64"),
            InferInput("attention_mask", [len(inputs_arr), len(attention_mask[0])], "INT64"),
        ]
        triton_inputs[0].set_data_from_numpy(input_ids)
        triton_inputs[1].set_data_from_numpy(attention_mask)
        results = self.client.infer(self.model_name, triton_inputs)

        assert results is not None

        dense_vecs = cast(np.ndarray, results.as_numpy("dense_vecs"))
        sparse_vecs = cast(np.ndarray, results.as_numpy("sparse_vecs")).squeeze(-1)

        dense_ret = dense_vecs.tolist()
        sparse_ret = [
            {
                int(id_): w
                for id_, w in self.process_token_weights(
                    self.tokenizer,
                    sparse_vecs[i],
                    input_ids[i].tolist(),
                ).items()
            }
            for i in range(len(inputs_arr))
        ]
        return dense_ret, sparse_ret

    @dispatcher.span
    def encode_queries(self, queries: list[str]) -> list[dict[int, float]]:
        return self._embed_one_batch(queries)[1]

    @dispatcher.span
    def encode_documents(self, documents: list[str]) -> list[dict[int, float]]:
        return self._embed_one_batch(documents)[1]

    @staticmethod
    def process_token_weights(
        tokenizer: PreTrainedTokenizerBase,
        token_weights: np.ndarray,
        input_ids: list[int],
    ) -> defaultdict[str, float]:
        result = defaultdict[str, float](float)
        unused_tokens = set(
            [
                tokenizer.cls_token_id,
                tokenizer.eos_token_id,
                tokenizer.pad_token_id,
                tokenizer.unk_token_id,
            ]
        )
        for w, i in zip(token_weights, input_ids, strict=False):
            if i not in unused_tokens and w > 0:
                idx = str(i)
                if w > result[idx]:
                    result[idx] = w
        return result
