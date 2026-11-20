from typing import Any, List, Sequence

from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryResult,
    VectorStore,
)
from driver3.service.AtlasGraph import AtlasGraph
from driver3.service.ConnectionConfiguration import ConnectionConfiguration
from rag.utils.log import rag_logger as log


class AtlasVectorStore(VectorStore):
    stores_text: bool = True
    flat_metadata: bool = True

    def __init__(
        self,
        username: str,
        password: str,
        url: str,
        embedding_dimension: int,
        database: str = "test",
        node_label: str = "Chunk",
        chunk_index_node_property: str = "chunkIndex",
        distance_strategy: str = "cosine",
        retrieval_query: str = "",
        init_database: bool = True,
        **kwargs: Any,
    ) -> None:
        if distance_strategy not in ["cosine", "euclidean"]:
            raise ValueError("distance_strategy must be either 'euclidean' or 'cosine'")
        self._driver = AtlasGraph(ConnectionConfiguration(url, username, password))
        self._database = database
        self.distance_strategy = distance_strategy
        self._node_label = node_label
        self._chunk_index_node_property = chunk_index_node_property
        self._retrieval_query = retrieval_query
        self._embedding_dimension = embedding_dimension
        self.stores_text = False
        if init_database:
            self.init_database()

    @property
    def client(self) -> AtlasGraph:
        return self._driver

    def init_database(self) -> None:
        """Initialization database, ensures that the attribute exists"""
        show_vertex = self._driver.use(self._database).exec(cypher=f"SHOW VERTEX;")
        has_label = False
        has_chunk_index = False
        for row in show_vertex.result_set["table"]:
            has_label |= row[1] == self._node_label
            has_chunk_index |= (
                row[1] == self._node_label and row[3] == self._chunk_index_node_property
            )
            if has_label and has_chunk_index:
                break
        # init graph schema
        if not has_label:
            cypher_query = f"CREATE VERTEX {self._node_label}({self._chunk_index_node_property} String PRIMARY KEY);"
            self._driver.use(self._database).exec(cypher_query)
        elif not has_chunk_index:
            raise ValueError(
                f"{self._node_label} has not {self._chunk_index_node_property} attribute"
            )
        # init es schema
        cypher_query = f'CALL db.listIndex() yield index_name, index_type where index_name="{self._node_label.lower()}" and index_type="vector" return index_name'
        result = self._driver.use(self._database).exec(cypher_query).result_set["table"]
        if len(result) == 0:
            cypher_query = f'CALL db.vectorIndex.createNodeIndex("{self._node_label.lower()}", "{self._node_label}", "{self._chunk_index_node_property}", {self._embedding_dimension}, "{self.distance_strategy}");'
            self._driver.use(self._database).exec(cypher_query)

    def add(self, nodes: Sequence[BaseNode], **add_kwargs: Any) -> List[str]:
        """add node to atlas graph database [node.embedding: embedding node.node_id: chunkIndex]"""
        database = self._driver.use(self._database)
        for node in nodes:
            # add graph node
            cypher_query = f'MATCH (n:{self._node_label}) WHERE n.{self._chunk_index_node_property} = "{node.node_id}" RETURN count(n);'
            exists_node = bool(database.exec(cypher_query).result_set["table"][0][0])
            if not exists_node:
                cypher_query = f'CREATE (n:{self._node_label}{{{self._chunk_index_node_property}: "{node.node_id}"}});'
                database.exec(cypher_query)
            # add es node
            try:
                cypher_query = f'CALL db.vectorIndex.putNodeVector("{self._node_label.lower()}", "{self._node_label}", "{node.node_id}", {node.embedding});'
                database.exec(cypher_query)
            except Exception as e:
                pass
                # log.error(f"put vertex index error: {str(e)}")
        return [node.node_id for node in nodes]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """delete chunk_index node from atlas graph database"""
        pass

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """return topk node [node._id: chunkIndex] and similarities"""
        cypher_query = f'CALL db.vectorIndex.queryNodes("{self._node_label.lower()}", {query.similarity_top_k}, {query.query_embedding}) YIELD node, score RETURN node.{self._chunk_index_node_property} as chunkIndex, score;'
        result = self._driver.use(self._database).exec(cypher_query).result_set["table"]
        return VectorStoreQueryResult(
            ids=[row[0] for row in result],
            similarities=[row[1] for row in result],
        )
