import json
from typing import Optional, Generator

from rag.prompt.base_prompt import BasePrompt
from rag.prompt.generator_prompt import GeneratorPrompt
from rag.model.atlas_model import AtlasLLM
from rag.common.context import Context
from rag.common.task import Task
from tenacity import retry, stop_after_attempt


class LLMGenerator:
    def __init__(
        self,
        llm_client: AtlasLLM,
        generated_prompt: Optional[BasePrompt] = None,
        # chunk_reranker: RerankByVector = None,
        enable_ref: bool = False,
    ):
        self.llm_client = llm_client
        self.generated_prompt = generated_prompt or GeneratorPrompt()
        # self.chunk_reranker = chunk_reranker or RerankByVector.from_config(
        #     {
        #         "type": "rerank_by_vector",
        #     }
        # )
        # if enable_ref:
        #     self.with_out_ref_prompt = init_prompt_with_fallback(
        #         "without_refer_generator_prompt", KAG_PROJECT_CONF.biz_scene
        #     )
        #     self.with_ref_prompt = init_prompt_with_fallback(
        #         "refer_generator_prompt", KAG_PROJECT_CONF.biz_scene
        #     )
        self.enable_ref = enable_ref

    # @retry(stop=stop_after_attempt(3))
    # def generate_answer(self, query, content, refer_data, **kwargs):
    #     pass
    # if not self.enable_ref:
    #     return self.llm_client.complete(
    #         {"query": query, "content": content},
    #         self.generated_prompt,
    #         segment_name="answer",
    #         tag_name="Final Answer",
    #         with_json_parse=self.generated_prompt.is_json_format(),
    #         **kwargs,
    #     )
    # if refer_data and len(refer_data):
    #     refer_data_str = json.dumps(refer_data, ensure_ascii=False, indent=2)
    #     return self.llm_client.invoke(
    #         {"query": query, "content": content, "ref": refer_data_str},
    #         self.with_ref_prompt,
    #         segment_name="answer",
    #         tag_name="Final Answer",
    #         with_json_parse=False,
    #         **kwargs,
    #     )
    # return self.llm_client.invoke(
    #     {"query": query, "content": content},
    #     self.with_out_ref_prompt,
    #     segment_name="answer",
    #     tag_name="Final Answer",
    #     with_json_parse=False,
    #     **kwargs,
    # )
    @retry(stop=stop_after_attempt(3))
    def generate(self, query: str, context: Context) -> Generator[str, None, None]:
        sub_question_answer = []
        for task in context.gen_task():
            if task is not None:
                assert isinstance(task, Task)
                if not task.result:
                    continue
                sub_question_answer.append(f"{task.name}: {task.result}")
        yield from self.generated_prompt.get_stream_result(
            self.llm_client,
            text_chunks=list(context.chunks),
            graph=context.graph,
            sub_question_answer=sub_question_answer,
            question=query,
        )


#         results = []
#         rerank_queries = []
#         chunks = []
#         graph_data = context.variables_graph
#         for task in context.gen_task(False):
#             if isinstance(task.result, KAGRetrievedResponse) and self.chunk_reranker:
#                 rerank_queries.append(
#                     task.arguments.get("rewrite_query", task.arguments["query"])
#                 )
#                 chunks.append(task.result.chunk_datas)
#             results.append(to_task_context_str(task.get_task_context()))

#         rerank_chunks = self.chunk_reranker.invoke(query, rerank_queries, chunks)
#         refer_data = to_reference_list(prefix_id=0, retrieved_datas=rerank_chunks)
#         content_json = {"step": results}

#         if len(refer_data) and (not self.enable_ref):
#             content_json["reference"] = refer_data

#         content = json.dumps(content_json, ensure_ascii=False, indent=2)
#         if not self.enable_ref:
#             refer_data = [
#                 f"Title:{x['document_name']}\n{x['content']}" for x in refer_data
#             ]
#             refer_data = "\n\n".join(refer_data)
#             thoughts = "\n\n".join(results)
#             content = f"""
# Docs:
# {refer_data}

# Step by Step Analysis:
# {thoughts}

#             """
#         return self.generate_answer(
#             query=query, content=content, refer_data=refer_data, **kwargs
#         )
