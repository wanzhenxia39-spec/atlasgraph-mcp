from llama_index.core.schema import TextNode
from typing import List, Optional, Dict, cast, Set, Tuple, Generator, Any
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.core import Settings
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import LLM
from rag.utils.log import rag_logger as log
from rag.index import AtlasRAGIndex
from rag.prompt import (
    EntityExtractPrompt,
    BasePrompt,
    RAGCypherPrompt,
    ReasoningPrompt,
    SimplifyTextPrompt,
)
from rag.store import AtlasGraphStore
from driver3.service.AtlasGraph import AtlasGraph
from rag.utils.utils import parse_response_with_retry


class AtlasRAGAgent:
    DEFAULT_NODE_SCORE = 1.0

    def __init__(
        self,
        atlas_rag_index: AtlasRAGIndex,
        similarity_top_k: int = 1,
        entity_extract_prompt: BasePrompt = EntityExtractPrompt(),
        simplify_text_prompt: BasePrompt = SimplifyTextPrompt(),
        reasoning_prompt: BasePrompt = ReasoningPrompt(),
        rag_cypher_prompt: BasePrompt = RAGCypherPrompt(),
        max_entity_num: int = 10,
        min_depth: int = 1,
        max_depth: int = 6,
        num_chunks_per_query: int = 20,
        llm: Optional[LLM] = None,
        embed_model: Optional[BaseEmbedding] = None,
    ):
        self._atlas_rag_index = atlas_rag_index
        self._similarity_top_k = similarity_top_k
        self._entity_extract_prompt = entity_extract_prompt
        self._simplify_text_prompt = simplify_text_prompt
        self._reasoning_prompt = reasoning_prompt
        self._rag_cypher_prompt = rag_cypher_prompt
        self._max_entity_num = max_entity_num
        self._min_depth = min_depth
        self._max_depth = max_depth
        self._num_chunks_per_query = num_chunks_per_query
        self.llm = llm or Settings.llm
        self.embed_model = embed_model or Settings.embed_model
        self._graph_store = cast(
            AtlasGraphStore, self._atlas_rag_index.storage_context.graph_store
        )
        self._driver: AtlasGraph = self._graph_store.client
        self._database = self._graph_store._database

    def _match_start_entity(
        self, query_str: str
    ) -> Tuple[List[str], List[Dict[str, str]]]:
        """Extract keywords."""
        entity_list = self._entity_extract_prompt.parse_response(
            self.llm.complete(
                self._entity_extract_prompt.build_prompt(
                    entity_types=list(self._atlas_rag_index.schema.vertices.keys()),
                    max_entity_num=1,
                    input=query_str,
                )
            )
        )
        log.debug(
            f"[think] Keyword extracted from that question include: {entity_list}"
        )
        entity_label_set = set()
        similarity_entity_list = []
        for entity in entity_list:
            label = entity["entity_type"]
            keyword = entity["entity_name"]
            embedding = self.embed_model.get_text_embedding(keyword)
            if label in self._graph_store.keyword_vector_stores:
                similarity_result = self._graph_store.keyword_vector_stores[
                    label
                ].query(
                    VectorStoreQuery(
                        query_embedding=embedding,
                        similarity_top_k=self._similarity_top_k,
                    )
                )
                for keyword in cast(list, similarity_result.ids):
                    if (label, keyword) not in entity_label_set:
                        entity_label_set.add((label, keyword))
                        cypher_query = f'MATCH (n:{label}) WHERE n.keyword="{keyword}" RETURN n.description;'
                        description = (
                            self._driver.use(self._database)
                            .exec(cypher_query)
                            .result_set["table"][0][0]
                        )
                        similarity_entity_list.append(
                            {
                                "label": label,
                                "keyword": keyword,
                                "description": description,
                            }
                        )
            else:
                similarity_entity_list_with_score = []
                entity_label_set_with_score = set()
                for (
                    label,
                    vector_store,
                ) in self._graph_store.keyword_vector_stores.items():
                    similarity_result = vector_store.query(
                        VectorStoreQuery(
                            query_embedding=embedding,
                            similarity_top_k=self._similarity_top_k,
                        )
                    )
                    for keyword, score in zip(similarity_result.ids, similarity_result.similarities):  # type: ignore[arg-type]
                        if (label, keyword) not in entity_label_set_with_score:
                            entity_label_set_with_score.add((label, keyword))
                            similarity_entity_list_with_score.append(
                                {
                                    "label": label,
                                    "keyword": keyword,
                                    "score": score,
                                }
                            )
                for entity in sorted(
                    similarity_entity_list_with_score,
                    key=lambda x: x["score"],
                    reverse=True,
                )[0 : self._max_entity_num]:
                    if (entity["label"], entity["keyword"]) not in entity_label_set:
                        entity_label_set.add((entity["label"], entity["keyword"]))
                        cypher_query = f'MATCH (n:{entity['label']}) WHERE n.keyword="{entity["keyword"]}" RETURN n.description;'
                        description = (
                            self._driver.use(self._database)
                            .exec(cypher_query)
                            .result_set["table"][0][0]
                        )
                        similarity_entity_list.append(
                            {
                                "label": entity["label"],
                                "keyword": entity["keyword"],
                                "description": description,
                            }
                        )
        log.debug(
            f"[think] Matching the starting point of keywords from the graph includes: {similarity_entity_list}"
        )
        return entity_list, similarity_entity_list

    def query(self, query_str: str, instruct: Optional[str] = None) -> str:
        gen = self._query(query_str, streaming=False, instruct=instruct)
        try:
            while True:
                next(gen)
        except StopIteration as e:
            return e.value

    def stream_query(
        self, query_str: str, instruct: Optional[str] = None
    ) -> Generator[str, None, str]:
        answer = yield from self._query(query_str, streaming=True, instruct=instruct)
        return answer

    def _query(
        self, query_str: str, streaming: bool = False, instruct: Optional[str] = None
    ) -> Generator[str, None, str]:
        """Get nodes for response."""
        query_embedding = self.embed_model.get_text_embedding(query_str)
        similarity_result = self._atlas_rag_index.storage_context.vector_store.query(
            VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=self._similarity_top_k,
            )
        )
        doc_nodes = self._atlas_rag_index.storage_context.docstore.get_nodes(
            similarity_result.ids  # type: ignore[arg-type]
        )
        doc_set = set()
        unique_doc_nodes = []
        for doc in doc_nodes:
            if doc.node_id not in doc_set:
                doc_set.add(doc.node_id)
                unique_doc_nodes.append(doc)
        doc_nodes = unique_doc_nodes
        self.text_chunks = [cast(TextNode, node).text for node in doc_nodes]
        entity_list, similarity_entity_list = self._match_start_entity(query_str)
        if streaming:
            yield "<think>"
            yield f"根据问题提取到的关键词为: {entity_list}:\n"
            yield f"图库中检索到相似的关键词为: {similarity_entity_list}:\n"
        self.graph: Dict[str, List[Dict[str, str]]] = {
            "vertices": similarity_entity_list,
            "edges": [],
        }
        vertex_set: Set[Tuple[str, str]] = {
            (vertex["label"], vertex["keyword"]) for vertex in similarity_entity_list
        }
        doc_ids = self._graph_store.get_chunk_ids(
            list(set([entity["keyword"] for entity in self.graph["vertices"]]))
        )
        doc_ids = [doc_id for doc_id in doc_ids if doc_id not in similarity_result.ids]  # type: ignore[operator]
        doc_nodes = self._atlas_rag_index.storage_context.docstore.get_nodes(doc_ids)
        self.text_chunks.extend([cast(TextNode, node).text for node in doc_nodes])
        query = self._simplify_text_prompt.build_prompt(
            text_chunks=self.text_chunks, input=query_str
        )
        if len(self.text_chunks) != 0:

            def _answer() -> Generator[str, None, Any]:
                if streaming:
                    yield "思考提取并简化问题相关的文本片段:\n"
                    for ans in self.llm.stream_complete(query):
                        if ans.delta:
                            yield ans.delta
                    yield "\n"
                    return self._simplify_text_prompt.parse_response(ans)

                else:
                    return self._simplify_text_prompt.parse_response(
                        self.llm.complete(query)
                    )

            # self.text_chunks = yield from parse_response_with_retry(_answer)
        edge_set: Set[Tuple[str, str, str, str, str, str]] = set()
        cypher_set = set()
        for depth in range(self._max_depth):
            log.debug(
                f"[think] Current known text document:{self.text_chunks} Currently known graph:{self.graph} depth:{depth}"
            )
            if streaming:
                yield f"当前已知的文本数据为: {self.text_chunks} 已知的图谱数据为: {self.graph} 当前思考深度为: {depth}:\n"
            if depth >= self._min_depth:
                query = self._reasoning_prompt.build_prompt(
                    text_chunks=self.text_chunks,
                    graph=self.graph,
                    input=(
                        query_str
                        if instruct is None
                        else f"[extra struct: {instruct}] {query_str}"
                    ),
                )

                def _answer1() -> Generator[str, None, Any]:
                    if streaming:
                        yield "思考当前已知信息能否回答当前问题:\n"
                        for ans in self.llm.stream_complete(query):
                            if ans.delta:
                                yield ans.delta
                        yield "\n"
                        return self._reasoning_prompt.parse_response(ans)

                    else:
                        return self._reasoning_prompt.parse_response(
                            self.llm.complete(query)
                        )

                response = yield from parse_response_with_retry(_answer1)
                if response["can_answer"]:
                    if streaming:
                        yield "<think>"
                    yield response["answer"]
                    return response["answer"]
            query = self._rag_cypher_prompt.build_prompt(
                schema=self._atlas_rag_index.schema,
                graph=self.graph,
                input=query_str,
            )

            def _answer2() -> Generator[str, None, Any]:
                if streaming:
                    yield "思考查询使用的cypher语句:\n"
                    for ans in self.llm.stream_complete(query):
                        if ans.delta:
                            yield ans.delta
                    yield "\n"
                    return self._rag_cypher_prompt.parse_response(ans)
                else:
                    return self._rag_cypher_prompt.parse_response(
                        self.llm.complete(query)
                    )

            cypher_list = yield from parse_response_with_retry(_answer2)
            cypher_list = [
                (etype, cypher_query)
                for etype, cypher_query in cypher_list
                if cypher_query not in cypher_set
            ]
            log.debug(
                f"[think] Generate cypher statement to get extra data: {cypher_list} depth:{depth}"
            )
            for (
                src_node_label,
                rel_label,
                dst_node_label,
            ), cypher_query in cypher_list:
                cypher_set.add(cypher_query)
                if streaming:
                    yield f"使用cypher查询: {cypher_query}:\n"
                try:
                    result = (
                        self._driver.use(self._database)
                        .exec(cypher_query)
                        .result_set["table"]
                    )
                except Exception as e:
                    log.warning(f"cypher: {cypher_query} failed.")
                    self.text_chunks.append(
                        f"cypher: {cypher_query} failed and error: {e}"
                    )
                    if streaming:
                        yield f"该cypher语句支持失败，失败原因: {e}\n"
                    continue
                if streaming:
                    yield f"该cypher语句执行结果为: {result}\n"
                if src_node_label == None:
                    self.text_chunks.append(f"cypher: {cypher_query} result: {result}")
                    continue
                for (
                    src_node_keyword,
                    src_node_desc,
                    rel_keyword,
                    dst_node_keyword,
                    dst_node_desc,
                ) in result:
                    if (src_node_label, src_node_keyword) not in vertex_set:
                        vertex_set.add((src_node_label, src_node_keyword))
                        self.graph["vertices"].append(
                            {
                                "label": src_node_label,
                                "keyword": src_node_keyword,
                                "description": src_node_desc,
                            }
                        )
                    if (dst_node_label, dst_node_keyword) not in vertex_set:
                        vertex_set.add((dst_node_label, dst_node_keyword))
                        self.graph["vertices"].append(
                            {
                                "label": dst_node_label,
                                "keyword": dst_node_keyword,
                                "description": dst_node_desc,
                            }
                        )
                    if (
                        src_node_label,
                        src_node_keyword,
                        rel_label,
                        rel_keyword,
                        dst_node_label,
                        dst_node_keyword,
                    ) not in edge_set:
                        edge_set.add(
                            (
                                src_node_label,
                                src_node_keyword,
                                rel_label,
                                rel_keyword,
                                dst_node_label,
                                dst_node_keyword,
                            )
                        )
                        self.graph["edges"].append(
                            {
                                "label": rel_label,
                                "keyword": rel_keyword,
                                "from_key": src_node_label,
                                "from_pk": src_node_keyword,
                                "to_key": dst_node_label,
                                "to_pk": dst_node_keyword,
                            }
                        )
        doc_ids = self._graph_store.get_chunk_ids(
            list(set([entity["keyword"] for entity in self.graph["vertices"]]))
        )
        doc_nodes = []
        for doc in self._atlas_rag_index.storage_context.docstore.get_nodes(doc_ids):
            if doc.node_id not in doc_set:
                doc_set.add(doc.node_id)
                doc_nodes.append(doc)
        self.text_chunks.extend([cast(TextNode, node).text for node in doc_nodes])
        self.text_chunks = list(set(self.text_chunks))
        log.debug(
            f"[think] Current known text document:{self.text_chunks} Currently known graph:{self.graph} depth:{depth}"
        )
        if streaming:
            yield f"当前已知的文本数据为: {self.text_chunks} 已知的图谱数据为: {self.graph} 当前思考深度为: {depth}:\n"
        query = self._reasoning_prompt.build_prompt(
            text_chunks=self.text_chunks,
            graph=self.graph,
            input=(
                query_str
                if instruct is None
                else f"[extra struct: {instruct}] {query_str}"
            ),
            force_anwer=True,
        )

        def _answer3() -> Generator[str, None, Any]:
            if streaming:
                yield "思考当前已知信息能否回答当前问题:\n"
                for ans in self.llm.stream_complete(query):
                    if ans.delta:
                        yield ans.delta
                yield "\n"
                return self._reasoning_prompt.parse_response(ans)

            else:
                return self._reasoning_prompt.parse_response(self.llm.complete(query))

        response = yield from parse_response_with_retry(_answer3)
        if streaming:
            yield "<think>"
        if response["can_answer"]:
            yield response["answer"]
            return response["answer"]
        yield "根据已知信息，无法回答该问题。详情请查看上述思考过程。"
        return "根据已知信息，无法回答该问题。详情请查看上述思考过程。"
