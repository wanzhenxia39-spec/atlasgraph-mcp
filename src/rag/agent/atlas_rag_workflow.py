from rag.common.context import Context
from rag.agent.generator import LLMGenerator
from rag.common.task import Task
from rag.index import AtlasRAGIndex
from rag.prompt.planning_prompt import PlanningPrompt

# from rag.prompt.rewrite_prompt import RewriteSubQuery
from rag.utils.config import rag_config as config
from rag.model.atlas_model import AtlasLLM, AtlasEmbeddings
from rag.utils.constant import Tool
from rag.tool import BaseTool, DeduceTool, RetrieverTool
from rag.prompt import BasePrompt
from rag.utils.log import rag_logger as log
from typing import List, Set, AsyncGenerator, Optional, Generator
import asyncio


class AtlasRAGWorkflow:
    def __init__(
        self,
        atlas_rag_index: AtlasRAGIndex,
        executors: Set[Tool],
        plan_prompt: BasePrompt = PlanningPrompt(),
        answer_prompt: Optional[BasePrompt] = None,
        similarity_top_k: int = 5,
        max_depth: int = 4,
    ) -> None:
        self.atlas_rag_index = atlas_rag_index
        self.llm = AtlasLLM(
            model=config.chat_model,
            api_url=config.chat_url,
            api_base=config.chat_base,
            api_key=config.chat_key,
        )
        self.embeddings = AtlasEmbeddings(
            model=config.embedding_model,
            api_url=config.embedding_url,
            api_base=config.embedding_base,
            api_key=config.embedding_key,
        )
        self.plan_prompt = plan_prompt
        self.generator = LLMGenerator(self.llm, answer_prompt)
        self.executors = executors
        self.max_depth = max_depth
        self.similarity_top_k = similarity_top_k
        self.context: Context = Context()

    def select_executor(self, executor: Tool) -> BaseTool: # 对应server中实现的工具
        tool_mapping = {
            Tool.RETRIEVER: RetrieverTool(
                self.atlas_rag_index, similarity_top_k=self.similarity_top_k
            ),
            Tool.DEDUCE: DeduceTool(self.atlas_rag_index),
        }
        return tool_mapping[executor]

    def query(self, query: str) -> Generator[str, None, None]: # 对应client中的函数调用
        self.context = Context()
        yield "<think>"
        yield "思考问题执行链路......\n"
        tasks: List[Task] = self.plan(query)
        yield "执行链路如下:\n"
        for task in tasks:
            yield f"{task.name}: {task.arguments["query"]}\n"
        for task in tasks:
            self.context.add_task(task)

        for task_group in self.context.gen_task(group=True):
            assert isinstance(task_group, list)
            for task in task_group:
                executor: BaseTool = self.select_executor(task.executor)
                for answer in executor.execute(query, task, self.context):
                    yield answer
            # tasks = [
            #     asyncio.create_task(self.execute_task(query, task, context))
            #     for task in task_group
            # ]
            # for task in asyncio.as_completed(tasks):
            #     for answer in task:
            #         yield answer
        yield "<think>"
        for answer in self.generator.generate(query, self.context):
            yield answer

    def plan(self, query: str) -> List[Task]: # 对应mcp_prompt
        plan_prompt = PlanningPrompt()
        response = plan_prompt.get_result(self.llm, query=query)
        return response

    # async def query_rewrite(self, task: Task, query: str):
    #     """Performs asynchronous query rewriting using LLM and context.

    #     Args:
    #         task (Task): Task containing the query to rewrite

    #     Returns:
    #         str: Rewritten query with resolved dynamic references
    #     """
    #     query = task.arguments["query"]

    #     deps_context = format_task_dep_context(task.parents)
    #     context = {
    #         "target question": query,
    #         "history_qa": deps_context,
    #     }

    #     rewrite_prompt = RewriteSubQuery()
    #     new_query = rewrite_prompt.parse_response(self.llm.complete(
    #         rewrite_prompt.build_prompt(
    #         input = query,
    #         content = json.dumps(context, indent=2, ensure_ascii=False)
    #     )))
    #     logic_form_node = task.arguments.get("logic_form_node", None)
    #     if logic_form_node:
    #         logic_form_node.sub_query = new_query
    #         return {"rewrite_query": new_query, "logic_form_node": logic_form_node}
    #     # print(f"query rewrite context = {context}")
    #     # print(f"New query: {new_query}")
    #     return {"query": new_query}

    # async def execute_task(self, query: str, task: Task, context: Context) -> None:
    #     # if task.check_require_rewrite():
    #     #     task.update_memory("origin_arguments", task.arguments)
    #     #     updated_args = await self.query_rewrite(task, query=query)
    #     #     task.arguments.update(updated_args)
    #     executor: BaseTool = self.select_executor(task.executor)
    #     if executor:
    #         await executor.execute(query, task, context)
    #     else:
    #         rag_logger.warning(f"Executor not found for task {task}")

    # async def finish_check(self, query: str, answer: str) -> bool:
    #     finish_prompt = f"""
    #     # Task
    #     The answer is a response to a question. Please determine whether the content of this answer is invalid, such as  "UNKNOWN", "I don't know" or "Insufficient Information."  \n
    #     If the answer is invalid, return "Yes", otherwise, return "No".\n
    #     You output should only be "Yes" or "No".\n

    #     # Answer\n
    #     {answer}
    #     """
    #     try:
    #         response = await self.llm.complete(prompt=finish_prompt)
    #         if response.strip().lower() == "yes":
    #             return False
    #         return True
    #     except Exception as e:
    #         print(f"Failed to run finish_judger, info: {e}")
    #         import traceback

    #         traceback.print_exc()
    #         return True


# def format_task_dep_context(tasks: List[Task], is_recu=True):
#     """Formats parent task execution context into a structured dictionary.

#     Args:
#         task (Task): Current task whose parent context needs formatting

#     Returns:
#         dict: Mapping of parent task IDs to their execution details containing:
#             - action: Executor and arguments used
#             - result: Execution result of the parent task
#     """

#     def to_str(context):
#         if not context or "task" not in context:
#             return ""
#         return f"""{context['name']}:{context['task']}
# {context['result']}.{context.get('thought', '')}"""

#     if not tasks:
#         return []
#     formatted_context = []
#     if isinstance(tasks, Task):
#         tasks = [tasks]
#     for task in tasks:
#         # get all prvious tasks from context.
#         if is_recu:
#             parent_res = format_task_dep_context(task.parents, is_recu)
#             for p in parent_res:
#                 if p not in formatted_context:
#                     formatted_context.append(p)
#         res = to_str(task.get_task_context(with_all=True))

#         if res:
#             if res in formatted_context:
#                 continue
#             formatted_context.append(res)
#     return formatted_context
if __name__ == "__main__":
    from rag.store.vector_store import AtlasVectorStore
    from rag.store.graph_store import AtlasGraphStore
    from rag.store.document_store import AtlasDocumentStore
    import logging
    from driver3.atlas_logger.AtlasLogger import logger
    import argparse
    from llama_index.core.storage.storage_context import StorageContext

    parser = argparse.ArgumentParser(description="atlas rag")
    parser.add_argument(
        "--rag_config",
        default="/root/rag/config/config.yaml",
        type=str,
    )
    args = parser.parse_args()
    logger.setLevel(logging.CRITICAL)
    config.set_rag_server_path(args.rag_config)
    database = "project"
    vector_store = AtlasVectorStore(
        username=config.username,
        password=config.password,
        url=config.graph_url,
        embedding_dimension=config.embedding_dimension,
        database=database,
    )
    document_store = AtlasDocumentStore(
        username=config.username,
        password=config.password,
        url=config.graph_url,
        database=database,
    )
    graph_store = AtlasGraphStore(
        username=config.username,
        password=config.password,
        url=config.graph_url,
        database=database,
        schema=None,
    )
    storage_context = StorageContext.from_defaults(
        docstore=document_store,
        vector_store=vector_store,  # type: ignore[arg-type]
        graph_store=graph_store,
    )
    atlas_rag_index = AtlasRAGIndex([], storage_context, schema=graph_store._schema)
    flow = AtlasRAGWorkflow(atlas_rag_index, executors={Tool.RETRIEVER, Tool.DEDUCE},similarity_top_k=3)
    # question = "What county is Erik Hort's birthplace a part of?"
    # question = "What year did the publisher of Labyrinth end?"
    # question = "When was Lady Godiva's birthplace abolished?"
    # question = "When was the region immediately north of the region where Israel is located and the location of the Battle of Qurah and Umm al Maradim created?"
    # question = "Where does the body of water by the city where the Southeast Library designer died empty into the Gulf of Mexico?"
    question = "LDBC世界纪录被哪个公司打破了，该公司的法人代表是谁？"
    print(f"原始问题: {question}")
    for answer in flow.query(question):
        print(answer, end="")
