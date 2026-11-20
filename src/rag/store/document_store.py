from typing import List, Optional, Sequence
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.storage.docstore import DocumentStore
from rag.utils.utils import to_legal_cypher_string
from rag.store.index_store import AtlasKG
from typing import Any, Set
from driver3.service.AtlasGraph import AtlasGraph
from driver3.service.ConnectionConfiguration import ConnectionConfiguration


class AtlasDocumentStore(DocumentStore):
    def __init__(
        self,
        username: str,
        password: str,
        url: str,
        database: str = "test",
        node_label: str = "Chunk",
        chunk_index_node_property: str = "chunkIndex",
        text_node_property: str = "text",
        document_name_node_property: str = "documentName",
        start_index_node_property: str = "startIndex",
        extract_rel_label: str = "Extract",
        entity_keyword_property: str = "keyword",
        init_database: bool = True,
        **kwargs: Any,
    ) -> None:
        self._driver = AtlasGraph(ConnectionConfiguration(url, username, password))
        self._database = database
        self._node_label = node_label
        self._chunk_index_node_property = chunk_index_node_property
        self._text_node_property = text_node_property
        self._document_name_node_property = document_name_node_property
        self._start_index_node_property = start_index_node_property
        self._extract_rel_label = extract_rel_label
        self._entity_keyword_property = entity_keyword_property
        if init_database:
            self.init_database()

    def init_database(self) -> None:
        """Initialization database, ensures that the attribute exists"""
        show_vertex = self._driver.use(self._database).exec(cypher=f"SHOW VERTEX;")
        has_label = False
        has_chunk_index = False
        has_text = False
        has_document_name = False
        has_start_index = False
        for row in show_vertex.result_set["table"]:
            has_label |= row[1] == self._node_label
            has_chunk_index |= (
                row[1] == self._node_label and row[3] == self._chunk_index_node_property
            )
            has_text |= (
                row[1] == self._node_label and row[3] == self._text_node_property
            )
            has_document_name |= (
                row[1] == self._node_label
                and row[3] == self._document_name_node_property
            )
            has_start_index |= (
                row[1] == self._node_label and row[3] == self._start_index_node_property
            )
            if (
                has_label
                and has_chunk_index
                and has_text
                and has_document_name
                and has_start_index
            ):
                break
        # init graph schema
        if not has_label:
            cypher_query = f"CREATE VERTEX {self._node_label}({self._chunk_index_node_property} String PRIMARY KEY, {self._text_node_property} String, {self._document_name_node_property} String {self._start_index_node_property} int32);"
            self._driver.use(self._database).exec(cypher_query)
        else:
            if not (
                has_chunk_index and has_text and has_document_name and has_start_index
            ):
                cypher_query = (
                    (
                        f"CREATE ATTRIBUTE on VERTEX {self._node_label}("
                        + (
                            f"{self._chunk_index_node_property} String INDEXED, "
                            if not has_chunk_index
                            else ""
                        )
                        + (
                            f"{self._text_node_property} String, "
                            if not has_text
                            else ""
                        )
                        + (
                            f"{self._document_name_node_property} String, "
                            if not has_document_name
                            else ""
                        )
                        + (
                            f"{self._start_index_node_property} int32, "
                            if not has_start_index
                            else ""
                        )
                    )[:-2]
                ) + ");"
                self._driver.use(self._database).exec(cypher_query)

    def add_documents(
        self,
        docs: Sequence[TextNode],  # type: ignore[override]
        allow_update: bool = True,
        batch_size: Optional[int] = None,
        store_text: bool = True,
    ) -> None:
        """add node to atlas graph database [node.text: text node.node_id: chunkIndex]"""
        "Temporarily batch_size = 1 and store_text = True."
        database = self._driver.use(self._database)
        for doc in docs:
            cypher_query = f'MATCH (n:{self._node_label}) WHERE n.{self._chunk_index_node_property} = "{doc.node_id}" RETURN count(n);'
            exists_node = bool(database.exec(cypher_query).result_set["table"][0][0])
            text = to_legal_cypher_string(doc.get_content())
            document_name = to_legal_cypher_string(doc.metadata["file_name"])
            start_index = doc.start_char_idx or 0
            if exists_node:
                if allow_update:
                    cypher_query = f'MATCH (doc: {self._node_label}{{{self._chunk_index_node_property}: "{doc.node_id}"}}) SET doc.{self._text_node_property} = "{text}";'
                    database.exec(cypher_query)
                    cypher_query = f'MATCH (doc: {self._node_label}{{{self._chunk_index_node_property}: "{doc.node_id}"}}) SET doc.{self._document_name_node_property} = "{document_name}";'
                    database.exec(cypher_query)
                    cypher_query = f'MATCH (doc: {self._node_label}{{{self._chunk_index_node_property}: "{doc.node_id}"}}) SET doc.{self._start_index_node_property} = {start_index};'
                    database.exec(cypher_query)
            else:
                cypher_query = f'CREATE (n:{self._node_label}{{{self._chunk_index_node_property}: "{doc.node_id}", {self._text_node_property}: "{text}", {self._document_name_node_property}: "{document_name}", {self._start_index_node_property}: {start_index}}});'
                database.exec(cypher_query)

    def document_exists(self, doc_id: str) -> bool:
        cypher_query = f'MATCH (n:{self._node_label}) WHERE n.{self._chunk_index_node_property} = "{doc_id}" RETURN count(n);'
        result = self._driver.use(self._database).exec(cypher_query).result_set["table"]
        return bool(result[0][0])

    def get_document(self, doc_id: str, raise_error: bool = True) -> Optional[BaseNode]:
        cypher_query = f'MATCH (n:{self._node_label}) WHERE n.{self._chunk_index_node_property} = "{doc_id}" RETURN n.{self._text_node_property} as {self._text_node_property};'
        result = self._driver.use(self._database).exec(cypher_query).result_set["table"]
        if len(result) == 0:
            if raise_error:
                raise ValueError(
                    f"can not find node {self._chunk_index_node_property}: {doc_id}."
                )
        return TextNode(id_=doc_id, text=result[0][0])

    def get_nodes(
        self, node_ids: List[str], raise_error: bool = True
    ) -> List[BaseNode]:
        database = self._driver.use(self._database)
        results: List[BaseNode] = []
        for node_id in node_ids:
            cypher_query = f'MATCH (n:{self._node_label}) WHERE n.{self._chunk_index_node_property} = "{node_id}" RETURN n.{self._text_node_property} as {self._text_node_property};'
            result = database.exec(cypher_query).result_set["table"]
            if len(result) == 0:
                if raise_error:
                    raise ValueError(
                        f"can not find node {self._chunk_index_node_property}: {node_id}."
                    )
            else:
                results.append(TextNode(id_=node_id, text=result[0][0]))
        return results

    def add_keywords(self, index_struct: AtlasKG, allow_update: bool = True) -> None:
        pass

    def get_keywords(self, node_ids: List[str], raise_error: bool = True) -> Set[str]:
        database = self._driver.use(self._database)
        keywords: Set[str] = set()
        for node_id in node_ids:
            cypher_query = f'MATCH (n1:{self._node_label})-[r:{self._extract_rel_label}]->(n2) WHERE n1.{self._chunk_index_node_property} = "{node_id}" RETURN n2.{self._entity_keyword_property};'
            result = database.exec(cypher_query).result_set["table"]
            if len(result) != 0:
                keywords.update({keyword[0] for keyword in result})
        return keywords
