from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.base.response.schema import (
    RESPONSE_TYPE,
    StreamingResponse,
    Response,
)
from llama_index.core.base.base_query_engine import dispatcher
from llama_index.core.schema import QueryBundle, NodeWithScore
from typing import Tuple, List, Union, Generator, Dict, Any, Optional
from llama_index.core.instrumentation.events.query import QueryEndEvent, QueryStartEvent
from rag.utils.log import rag_logger as log
from rag.utils.timer import Timer
from typing import Callable


# 对字符串进行转义处理，使其能安全地用于 Cypher 查询中
def to_legal_cypher_string(origin_str: str) -> str:
    new_str = ""
    for i, char in enumerate(origin_str):
        if char in ['"', "'", "\\"]:
            new_str += "\\"
        new_str += char
    return new_str


@dispatcher.span
def query(
    engine: RetrieverQueryEngine, question: str, raw: bool = False
) -> Tuple[RESPONSE_TYPE, Union[List[str], List[NodeWithScore]]]:
    dispatcher.event(QueryStartEvent(query=question))
    with engine.callback_manager.as_trace("query"):
        query_bundle = QueryBundle(question)
        timer = Timer()
        nodes = engine.retrieve(query_bundle)
        log.info(f"\nretrieve time cost: {timer.cost():.2f} s")
        timer.reset()
        response = engine._response_synthesizer.synthesize(
            query=query_bundle,
            nodes=nodes,
        )
        log.info(f"\nllm time cost: {timer.cost():.2f} s")
        dispatcher.event(QueryEndEvent(query=query_bundle, response=response))
    return response, [node.node_id for node in nodes] if not raw else nodes


@dispatcher.span
def stream_query(
    engine: RetrieverQueryEngine,
    question: str,
    instruct: Optional[str] = None,
    raw: bool = False,
    yield_node: bool = False,
) -> Generator[
    Union[
        str,
        Tuple[Union[List[str], List[NodeWithScore]], Dict[str, List[Dict[str, str]]]],
    ],
    None,
    Tuple[
        RESPONSE_TYPE,
        Union[List[str], List[NodeWithScore]],
        Dict[str, List[Dict[str, str]]],
    ],
]:
    dispatcher.event(QueryStartEvent(query=question))
    with engine.callback_manager.as_trace("query"):
        query_bundle = QueryBundle(question)

        timer = Timer()
        graph = {"vertices": [], "edges": []}
        from rag.retriever import GraphBaseRetriever

        if isinstance(engine.retriever, GraphBaseRetriever):
            nodes, graph = engine.retriever.retrieve_with_graph(query_bundle)
        else:
            nodes = engine.retrieve(query_bundle)
        if yield_node:
            yield [node.node_id for node in nodes] if not raw else nodes, graph
        log.info(f"retrieve time cost: {timer.cost():.2f} s")
        timer.reset()
        if instruct is not None:
            query_bundle = QueryBundle(f"instruct: {instruct}\n question: {question}")
        streaming_response: StreamingResponse = engine._response_synthesizer.synthesize(
            query=query_bundle,
            nodes=nodes,
        )
        response_txt = ""
        for text in streaming_response.response_gen:
            yield text
            response_txt += text
        response = Response(
            response_txt, streaming_response.source_nodes, streaming_response.metadata
        )
        log.info(f"llm time cost: {timer.cost():.2f} s")
        dispatcher.event(QueryEndEvent(query=query_bundle, response=response))
    return response, [node.node_id for node in nodes] if not raw else nodes, graph


def parse_response_with_retry(
    func: Callable[..., Generator[str, None, Any]], retry: int = 3
) -> Generator[str, None, Any]:
    for index in range(retry):
        try:
            response = yield from func()
            return response
        except:
            yield f"工具调用错误，开始重试......\n"
            if index == retry - 1:
                raise
