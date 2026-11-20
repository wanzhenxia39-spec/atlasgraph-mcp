from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolOutput
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.query_engine import BaseQueryEngine
from typing import Any, List, Dict
from rag.utils.utils import query


class RAGQueryEngineTool(QueryEngineTool):
    def call(self, *args: Any, **kwargs: Any) -> ToolOutput:
        query_str = self._get_query_str(*args, **kwargs)
        response, nodes = query(self._query_engine, query_str, raw=True)
        return ToolOutput(
            content=str(
                {
                    "context": [node.get_content() for node in nodes],
                    "answer": str(response),
                }
            ),
            tool_name=self.metadata.name,  # type: ignore[arg-type]
            raw_input={"input": query_str},
            raw_output=response,
        )


class GraphTool:
    def __init__(self, url: str, username: str, password: str):
        try:
            from driver3.service.AtlasGraph import AtlasGraph
            from driver3.service.ConnectionConfiguration import ConnectionConfiguration
        except ImportError:
            raise ImportError("Please install atlas: pip install atlas")
        self._driver = AtlasGraph(ConnectionConfiguration(url, username, password))

    def graph_schema_message(self, database: str) -> ChatMessage:
        result = {"vertices": [], "edges": []}
        show_vertex = self._driver.use(database).exec(cypher="SHOW VERTEX;")
        for table in show_vertex.result_set["table"]:
            result["vertices"].append(
                dict(zip(show_vertex.result_set["header"], table))
            )
        show_edge = self._driver.use(database).exec(cypher="SHOW EDGE;")
        for table in show_edge.result_set["table"]:
            result["edges"].append(dict(zip(show_vertex.result_set["header"], table)))
        return ChatMessage(
            role=MessageRole.SYSTEM,
            content=f"这是图库中存储的当前图的schema信息,回答问题时不要用图库中不存在的点边类型,不要自行构造,当前图的schema信息如下:\n\n {result}",
        )

    def gen_cypher_tool(self, database_name: str) -> FunctionTool:

        def cypher_executor(cypher_query: str) -> Dict[str, List[List[Any]]]:
            """Atlas Graph的cypher语句执行函数,通过这个工具函数可以在图库中执行查询语句并以表格的方式返回
            Args:
                cypher_query (str): 需要执行的cypher语句
            Returns:
                results (List[List[Any]]): 图库执行cypher语句返回的结果
            """
            database = self._driver.use(database_name)
            result_set = database.exec(cypher_query).result_set
            return {"header": result_set["header"], "table": result_set["table"]}

        return FunctionTool.from_defaults(cypher_executor, name="cypher_executor")

    def gen_vector_rag_tool(self, query_engine: BaseQueryEngine) -> QueryEngineTool:
        return RAGQueryEngineTool.from_defaults(
            query_engine,
            name="vector_rag",
            description="""信息查询工具，基于向量相似度的检索增强方法,通常如果问题不存在复杂的逻辑推理时使用该方法检索问题的回答,
                           该方法的优点是比较快速,缺点是对于复杂的问题回答可能不太准确
                        """,
        )

    def gen_hybrid_rag_tool(self, query_engine: BaseQueryEngine) -> QueryEngineTool:
        return RAGQueryEngineTool.from_defaults(
            query_engine,
            name="hybrid_rag",
            description="""信息查询工具，基于向量相似相似度和图库图谱推理混合的检索增强方法，由于可以同时利用上图库图谱中的游走链接
                           信息相比于纯的向量检索增强方法回答的效果会更加准确，通常如果问题中可能存在复杂的推理思考，可能涉及到图库
                           上推理的时候应该使用该方法，其缺点就是耗时会更长
                        """,
        )

    def gen_tog_rag_tool(self, query_engine: BaseQueryEngine) -> QueryEngineTool:
        return RAGQueryEngineTool.from_defaults(
            query_engine,
            name="tog_rag",
            description="""信息查询工具，基于向量相似度和图库推理的检索增强方法，通过细粒度的图上推理游走和点边实体的过滤的方式检索问题
                           的答案，通常可以解决需要涉及多层图谱查询的复杂问题，缺点是耗时比较长
            """,
        )
