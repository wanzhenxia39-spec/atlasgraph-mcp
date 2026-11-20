from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore, QueryBundle
from typing import List, Optional, Set, Dict, Tuple
from llama_index.core import VectorStoreIndex
from llama_index.core.indices.keyword_table.utils import extract_keywords_given_response
from llama_index.core.prompts.default_prompts import (
    DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE,
)
from llama_index.core.prompts import BasePromptTemplate
from collections import Counter
from llama_index.core.vector_stores.types import VectorStoreQuery
from rag.store.document_store import AtlasDocumentStore
from rag.store.graph_store import AtlasGraphStore
from llama_index.core import Settings
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import LLM
from rag.utils.log import rag_logger as log
from llama_index.core.indices.base import BaseIndex
from llama_index.core.data_structs.data_structs import KG
from .graph_base_retriever import GraphBaseRetriever
from typing import cast
from rag.utils.constant import HyBridRetrieverMode


class HybridRetriever(GraphBaseRetriever):
    DEFAULT_NODE_SCORE = 1.0

    def __init__(
        self,
        vector_index: VectorStoreIndex,
        kg_index: BaseIndex[KG],
        hybrid_retriever_mode: HyBridRetrieverMode = HyBridRetrieverMode.FINE_GRAINED,
        similarity_top_k: int = 5,
        query_keyword_extract_template: Optional[BasePromptTemplate] = None,
        max_keywords_per_query: int = 10,
        graph_store_query_depth: int = 4,
        num_chunks_per_query: int = 20,
        llm: Optional[LLM] = None,
        embed_model: Optional[BaseEmbedding] = None,
    ):
        self._vector_index = vector_index
        self._kg_index = kg_index
        self._mode = hybrid_retriever_mode
        self._similarity_top_k = similarity_top_k
        self._query_keyword_extract_template = (
            query_keyword_extract_template or DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE
        )
        self._max_keywords_per_query = max_keywords_per_query
        self._graph_store_query_depth = graph_store_query_depth
        self._num_chunks_per_query = num_chunks_per_query
        self.llm = llm or Settings.llm
        self.embed_model = embed_model or Settings.embed_model
        self._graph_store = cast(AtlasGraphStore, self._kg_index._graph_store)
        super(HybridRetriever, self).__init__()

    def _get_keywords(self, query_str: str) -> Set[str]:
        """Extract keywords."""
        response = self.llm.predict(
            self._query_keyword_extract_template,
            max_keywords=self._max_keywords_per_query,
            question=query_str,
        )
        keywords = extract_keywords_given_response(
            response, start_token="KEYWORDS:", lowercase=False
        )
        return keywords

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return self.retrieve_with_graph(query_bundle)[0]

    def retrieve_with_graph(
        self, query_bundle: QueryBundle
    ) -> Tuple[List[NodeWithScore], Dict[str, List[Dict[str, str]]]]:
        """Get nodes for response."""
        from rag.utils.timer import Timer
        t = Timer()
        if self._vector_index._vector_store.is_embedding_query:
            if query_bundle.embedding is None and len(query_bundle.embedding_strs) > 0:
                query_bundle.embedding = (
                    self.embed_model.get_agg_embedding_from_queries(
                        query_bundle.embedding_strs
                    )
                )
        query_result = self._vector_index.vector_store.query(
            VectorStoreQuery(
                query_embedding=query_bundle.embedding,
                similarity_top_k=self._similarity_top_k,
            )
        )
        query_result.nodes = self._vector_index._docstore.get_nodes(query_result.ids)  # type: ignore[arg-type]
        log.debug("---------query result---------")
        log.debug(query_result.nodes)
        if self._mode == HyBridRetrieverMode.FINE_GRAINED:
            if isinstance(self._kg_index.docstore, AtlasDocumentStore):
                keywords = self._kg_index.docstore.get_keywords(query_result.ids)  # type: ignore[arg-type]
            else:
                keywords = {
                    keyword
                    for node in query_result.nodes
                    for keyword in self._get_keywords(node.get_content())
                }
        else:
            keywords = self._get_keywords(query_bundle.query_str)
        rel_map = self._kg_index._graph_store.get_rel_map(
            list(keywords), self._graph_store_query_depth
        )
        vertex_set = set()
        edge_set = set()
        for start_subj, rel_list in rel_map.items():
            vertex_set.add(start_subj)
            subj = start_subj
            for keywords_list in rel_list:
                for index, keyword in enumerate(keywords_list):
                    if index % 2 == 0:
                        rel = keyword
                    else:
                        vertex_set.add(keyword)
                        edge_set.add((subj, rel, keyword))
                        subj = keyword
        graph = {
            "vertices": [
                {"label": self._graph_store._node_label, "keyword": keyword}
                for keyword in vertex_set
            ],
            "edges": [
                {
                    "label": self._graph_store._rel_label,
                    "keyword": rel,
                    "from_key": self._graph_store._node_label,
                    "from_pk": subj,
                    "to_key": self._graph_store._node_label,
                    "to_pk": obj,
                }
                for subj, rel, obj in edge_set
            ],
        }

        keywords = {
            keyword
            for values in rel_map.values()
            for rel_list in values
            for keyword in rel_list
        }
        node_ids = self._graph_store.get_chunk_ids(list(keywords))
        chunk_indices_count = Counter(node_ids)
        sorted_chunk_indices = sorted(
            chunk_indices_count.keys(),
            key=lambda x: chunk_indices_count[x],
            reverse=True,
        )
        sorted_chunk_indices = sorted_chunk_indices[: self._num_chunks_per_query]
        sorted_nodes = self._kg_index._docstore.get_nodes(sorted_chunk_indices)
        log.debug("---------sorted nodes result---------")
        log.debug(sorted_nodes)
        log.debug("-------------------------------------")
        sorted_nodes_with_scores = [
            NodeWithScore(node=node, score=self.DEFAULT_NODE_SCORE)
            for node in sorted_nodes
        ]
        if len(sorted_nodes_with_scores) == 0:
            sorted_nodes_with_scores = [
                NodeWithScore(node=node, score=similarity)
                for node, similarity in zip(
                    query_result.nodes, query_result.similarities  # type: ignore[arg-type]
                )
            ]
        if self._mode == HyBridRetrieverMode.COARSE_GRAINED:
            include_set = set()
            nodes_with_scores = []
            sorted_nodes_with_scores.extend(
                [
                    NodeWithScore(node=node, score=similarity)
                    for node, similarity in zip(
                        query_result.nodes, query_result.similarities  # type: ignore[arg-type]
                    )
                ]
            )
            for node in sorted_nodes_with_scores:
                if node.node_id not in include_set:
                    nodes_with_scores.append(node)
                    include_set.add(node.node_id)
            return nodes_with_scores, graph
        return sorted_nodes_with_scores, graph
