"""Atlas graph store index."""

from typing import Any, Dict, List, Optional, Tuple

from llama_index.core.graph_stores.types import GraphStore
from rag.utils.utils import to_legal_cypher_string
from driver3.service.AtlasGraph import AtlasGraph
from driver3.service.ConnectionConfiguration import ConnectionConfiguration
from llama_index.core.schema import TextNode
from rag.schema import Schema, DEFAULT_SCHEMA
from rag.store.vector_store import AtlasVectorStore
from rag.utils.config import rag_config as config
from llama_index.core import Settings


class AtlasGraphStore(GraphStore):

    def __init__(
        self,
        username: str,
        password: str,
        url: str,
        database: str = "test",
        schema: Optional[Schema] = DEFAULT_SCHEMA,
        keyword_property: str = "keyword",
        description_property: str = "description",
        chunk_node_label: str = "Chunk",
        chunk_index_node_property: str = "chunkIndex",
        extract_rel_label: str = "Extract",
        schema_label: str = "Schema",
        index_name: str = "keywordIndex",
        store_vector: bool = True,
        init_database: bool = True,
        **kwargs: Any,
    ) -> None:
        self._driver = AtlasGraph(ConnectionConfiguration(url, username, password))
        self._database = database
        self._schema_label = schema_label
        if schema is None:
            self._schema = Schema.from_str(self.get_schema())
        else:
            self._schema = schema
        self._node_label = list(self._schema.canonical_edge_labels)[0][0]
        self._rel_label = list(self._schema.canonical_edge_labels)[0][1]
        self._keyword_property = keyword_property
        self._description_property = description_property
        self._chunk_node_label = chunk_node_label
        self._chunk_index_node_property = chunk_index_node_property
        self._extract_rel_label = extract_rel_label
        self._index_name = index_name
        self._embedding_model = Settings.embed_model
        if init_database:
            self.init_database()
        if store_vector:
            self.keyword_vector_stores = {
                node_label: AtlasVectorStore(
                    username,
                    password,
                    url,
                    config.embedding_dimension,
                    database,
                    node_label,
                    keyword_property,
                )
                for node_label in self._schema.vertices.keys()
            }

    @property
    def client(self) -> AtlasGraph:
        return self._driver

    def init_database(self) -> None:
        """Initialization database, ensures that the attribute exists"""
        vertex_labels = list(self._schema.vertices.keys())
        show_vertex = self._driver.use(self._database).exec(cypher=f"SHOW VERTEX;")
        database = self._driver.use(self._database)
        if not database.vertex_exists(self._schema_label):
            database.exec(
                cypher=f"CREATE VERTEX {self._schema_label}(schema String PRIMARY KEY);"
            )
            database.exec(
                cypher=f'CREATE (n:{self._schema_label}{{schema:"{to_legal_cypher_string(str(self._schema))}"}});'
            )
        for vertex_label in vertex_labels:
            has_vertex_label = False
            has_vertex_keyword = False
            has_vertex_description = False
            for row in show_vertex.result_set["table"]:
                has_vertex_label |= row[1] == vertex_label
                has_vertex_keyword |= (
                    row[1] == vertex_label and row[3] == self._keyword_property
                )
                has_vertex_description |= (
                    row[1] == vertex_label and row[3] == self._description_property
                )
                if has_vertex_label and has_vertex_keyword and has_vertex_description:
                    break
            if not has_vertex_label:
                cypher_query = f"CREATE VERTEX {vertex_label}({self._keyword_property} String PRIMARY KEY, {self._description_property} String);"
                self._driver.use(self._database).exec(cypher_query)
            else:
                if not has_vertex_keyword:
                    cypher_query = f"CREATE ATTRIBUTE on VERTEX {vertex_label}({self._keyword_property} String INDEXED);"
                    self._driver.use(self._database).exec(cypher_query)
                if not has_vertex_description:
                    cypher_query = f"CREATE ATTRIBUTE on VERTEX {vertex_label}({self._description_property} String);"
                    self._driver.use(self._database).exec(cypher_query)
        extract_from_to_rel = [
            f"FROM {self._chunk_node_label} TO {vertex_label}"
            for vertex_label in vertex_labels
        ]
        has_extract_label = False
        has_extract_rel = False
        show_edge = self._driver.use(self._database).exec(cypher=f"SHOW EDGE;")
        for row in show_edge.result_set["table"]:
            has_extract_label |= row[1] == self._extract_rel_label
            has_extract_rel |= row[1] == self._extract_rel_label and all(
                [from_to in row[2] for from_to in extract_from_to_rel]
            )
            if has_extract_label:
                break
        if not has_extract_label:
            cypher_query = f"CREATE DIRECTED EDGE {self._extract_rel_label}() RELATIONS ({str(extract_from_to_rel)[1:-1].replace("'", "").replace('"',"")});"
            self._driver.use(self._database).exec(cypher_query)
        else:
            if not has_extract_rel:
                raise ValueError(
                    f"edge label: {self._extract_rel_label} is already exists,but it's RELATIONS does not contain '{extract_from_to_rel}'."
                )
        for edge_label, edge_value in self._schema.edges.items():
            has_rel_label = False
            has_rel = False
            has_rel_keyword = False
            from_to_rel = [
                f'FROM {from_to["from_key"]} TO {from_to["to_key"]}'
                for from_to in edge_value["relations"]
            ]
            for row in show_edge.result_set["table"]:
                has_rel_label |= row[1] == edge_label
                has_rel |= row[1] == edge_label and all(
                    [from_to in row[2] for from_to in from_to_rel]
                )
                has_rel_keyword |= (
                    row[1] == edge_label
                    and all([from_to in row[2] for from_to in from_to_rel])
                    and row[4] == self._keyword_property
                )
                if has_rel_label and has_rel_keyword and has_rel:
                    break

            # init graph schema
            if not has_rel_label:
                cypher_query = f"CREATE DIRECTED EDGE {edge_label}({self._keyword_property} String PRIMARY KEY) RELATIONS ({str(from_to_rel)[1:-1].replace("'", "").replace('"',"")});"
                self._driver.use(self._database).exec(cypher_query)
            else:
                if has_rel:
                    if not has_rel_keyword:
                        cypher_query = f"CREATE ATTRIBUTE on EDGE {edge_label}({self._keyword_property} String INDEXED);"
                        self._driver.use(self._database).exec(cypher_query)
                else:
                    raise ValueError(
                        f"edge label: {edge_label} is already exists,but it's RELATIONS does not contain '{from_to_rel}'."
                    )
        index_names = {
            row[1] for row in database.exec("SHOW VERTEX INDEX").result_set["table"]
        }
        if self._index_name not in index_names:
            database.exec(f"CREATE INDEX {self._index_name} ON VERTEX *({self._keyword_property})")

    def get(self, subj: str) -> List[List[str]]:
        """Get triplets."""
        query = """
            MATCH (n1:%s)-[r]->(n2:%s)
            WHERE n1.ID = "%s"
            RETURN type(r), n2.ID;
        """
        subj = subj.replace('"', "'")

        prepared_statement = query % (self._node_label, self._node_label, subj)
        session = self._driver.use(self._database)
        query_result = session.exec(prepared_statement).result_set["table"]
        return query_result

    def get_rel_map_faster(
        self,
        subjs_with_label: Optional[List[Tuple[str, str]]] = None, # 输入：起点的keyword与label  
        depth: int = 4, # 遍历的路径深度
        limit: int = 20,  # 限制结果数量
    ) -> Dict[str, List[List[str]]]:
        """Get flat rel map."""
        # The flat means for multi-hop relation path, we could get
        # knowledge like: subj -> rel -> obj -> rel -> obj -> rel -> obj.
        # This type of knowledge is useful for some tasks.
        # +-------------+------------------------------------+
        # | subj        | flattened_rels                     |
        # +-------------+------------------------------------+
        # | "player101" | [95, "player125", 2002, "team204"] |
        # | "player100" | [1997, "team204"]                  |
        # ...
        # +-------------+------------------------------------+
        rel_map: Dict[Any, List[Any]] = {}
        if subjs_with_label is None or len(subjs_with_label) == 0:
            # unlike simple graph_store, we don't do get_all here
            return rel_map
        sub_match = [
            f"match(n1: {to_legal_cypher_string(label)}{{{self._keyword_property}:{subj}}}) return n1"
            for subj, label in subjs_with_label
        ]
        cypher_query = (
            f"call{{{"union ".join(sub_match)}}}"
            f"with a"
            f"MATCH p=(n1)-[r*1..{depth}]->(n2)"
            f'WHERE all(x in relationships(p) WHERE label(x) <> "Extract") '
            f"AND n1.{self._keyword_property} IN {[to_legal_cypher_string(subj) for subj,_ in subjs_with_label]} "
            f"UNWIND relationships(p) AS rel "
            f"WITH n1.{self._keyword_property} AS subj, p, apoc.coll.flatten(apoc.coll.toSet("
            f"collect([rel.{self._keyword_property}, endNode(rel).{self._keyword_property}]))) AS flattened_rels "
            f"RETURN subj, collect(flattened_rels) AS flattened_rels LIMIT {limit}"
        )
        result = self._driver.use(self._database).exec(cypher_query)
        for record in result.result_set["table"]:
            rel_map[record[0]] = record[1]
        return rel_map 

    def get_rel_map(
        self, subjs: Optional[List[str]] = None, depth: int = 4, limit: int = 20
    ) -> Dict[str, List[List[str]]]:
        """Get flat rel map."""
        # The flat means for multi-hop relation path, we could get
        # knowledge like: subj -> rel -> obj -> rel -> obj -> rel -> obj.
        # This type of knowledge is useful for some tasks.
        # +-------------+------------------------------------+
        # | subj        | flattened_rels                     |
        # +-------------+------------------------------------+
        # | "player101" | [95, "player125", 2002, "team204"] |
        # | "player100" | [1997, "team204"]                  |
        # ...
        # +-------------+------------------------------------+
        rel_map: Dict[Any, List[Any]] = {}
        if subjs is None or len(subjs) == 0:
            # unlike simple graph_store, we don't do get_all here
            return rel_map
        cypher_query = (
            f"MATCH p=(n1)-[r*1..{depth}]->(n2)"
            f'WHERE all(x in relationships(p) WHERE label(x) <> "Extract") '
            f"AND n1.{self._keyword_property} IN {[to_legal_cypher_string(subj) for subj in subjs]} "
            f"UNWIND relationships(p) AS rel "
            f"WITH n1.{self._keyword_property} AS subj, p, apoc.coll.flatten(apoc.coll.toSet("
            f"collect([rel.{self._keyword_property}, endNode(rel).{self._keyword_property}]))) AS flattened_rels "
            f"RETURN subj, collect(flattened_rels) AS flattened_rels LIMIT {limit}"
        )
        result = self._driver.use(self._database).exec(cypher_query)
        for record in result.result_set["table"]:
            rel_map[record[0]] = record[1]
        return rel_map

    def upsert_triplet(self, subj: str, rel: str, obj: str) -> None:
        """Add triplet."""
        subj = to_legal_cypher_string(subj)
        rel = to_legal_cypher_string(rel)
        obj = to_legal_cypher_string(obj)
        if self.keyword_vector_stores:
            self.keyword_vector_stores[self._node_label].add(
                [
                    TextNode(
                        id_=subj,
                        embedding=self._embedding_model.get_text_embedding(subj),
                    ),
                    TextNode(
                        id_=obj,
                        embedding=self._embedding_model.get_text_embedding(obj),
                    ),
                ]
            )
        cypher_query = (
            f'MERGE (n1:{self._node_label}{{{self._keyword_property}: "{subj}"}})-'
            f'[r:{self._rel_label}{{{self._keyword_property}:"{rel}"}}]->'
            f'(n2:{self._node_label}{{{self._keyword_property}: "{obj}"}});'
        )
        self._driver.use(self._database).exec(cypher_query)

    def upsert_graph(self, graph_data: Dict[str, List[Dict]], node_id: str) -> None:
        for vertex in graph_data["vertices"]:
            label = to_legal_cypher_string(vertex["label"])
            keyword = to_legal_cypher_string(vertex[self._keyword_property])
            desc = to_legal_cypher_string(vertex[self._description_property])
            cypher_query = f'MERGE (n:{label}{{{self._keyword_property}:"{keyword}"}}) ON CREATE SET n.{self._description_property}="{desc}"'
            self._driver.use(self._database).exec(cypher_query)
            self.keyword_vector_stores[label].add(
                [
                    TextNode(
                        id_=keyword,
                        embedding=self._embedding_model.get_text_embedding(
                            f"keyword: {keyword} desc: {desc}"
                        ),
                    )
                ]
            )
            cypher_query = (
                f'MERGE (n1:{self._chunk_node_label}{{{self._chunk_index_node_property}: "{node_id}"}})-'
                f"[r:{self._extract_rel_label}]->"
                f'(n2:{label}{{{self._keyword_property}: "{keyword}"}});'
            )
            self._driver.use(self._database).exec(cypher_query)
        for edge in graph_data["edges"]:
            label = to_legal_cypher_string(edge["label"])
            keyword = to_legal_cypher_string(edge[self._keyword_property])
            from_key = to_legal_cypher_string(edge["from_key"])
            from_pk = to_legal_cypher_string(edge["from_pk"])
            to_key = to_legal_cypher_string(edge["to_key"])
            to_pk = to_legal_cypher_string(edge["to_pk"])
            cypher_query = (
                f'MERGE (n1:{from_key}{{{self._keyword_property}: "{from_pk}"}})-'
                f'[r:{label}{{{self._keyword_property}:"{keyword}"}}]->'
                f'(n2:{to_key}{{{self._keyword_property}: "{to_pk}"}});'
            )
            self._driver.use(self._database).exec(cypher_query)

    def delete(self, subj: str, rel: str, obj: str) -> None:
        """Delete triplet."""

        def delete_rel(subj: str, obj: str, rel: str) -> None:
            session = self._driver.use(self._database)
            session.exec(
                'MATCH (n1: %s)-[r:%s]->(n2:%s) WHERE n1.ID = "%s" AND n2.ID = "%s" AND r.predicate = "%s" DELETE r'
                % (self._node_label, self._rel_label, self._node_label, subj, obj, rel)
            )

        def delete_entity(entity: str) -> None:
            session = self._driver.use(self._database)
            session.exec(
                'MATCH (n:%s) WHERE n.ID = "%s" DELETE n' % (self._node_label, entity)
            )

        def check_edges(entity: str) -> bool:
            session = self._driver.use(self._database)
            is_exists_result = session.exec(
                'MATCH (n1:%s)-[]-(n2) WHERE n1.ID = "%s" RETURN count(n2)'
                % (self._node_label, entity)
            )
            return bool(is_exists_result.result_set["table"][0][0])

        subj = subj.replace('"', "'")
        obj = obj.replace('"', "'")

        delete_rel(subj, obj, rel)
        if not check_edges(subj):
            delete_entity(subj)
        if not check_edges(obj):
            delete_entity(obj)

    def get_schema(self, refresh: bool = False) -> str:
        return (
            self._driver.use(self._database)
            .exec(f"MATCH (n:{self._schema_label}) RETURN n.schema LIMIT 1;")
            .result_set["table"][0][0]
        )

    def query(self, query: str, param_map: Optional[Dict[str, Any]] = {}) -> Any:
        session = self._driver.use(self._database)
        result = session.exec(query)
        return result

    def get_chunk_ids(self, keywords: List[str]) -> List[str]:
        session = self._driver.use(self._database)
        result = session.exec(
            f'UNWIND {keywords} as kw MATCH (n) where n.{self._keyword_property} = kw with n match (n) <- [:Extract]-(chunk: {self._chunk_node_label}) RETURN chunk.{self._chunk_index_node_property}'
        )
        return [row[0] for row in result.result_set["table"]]
