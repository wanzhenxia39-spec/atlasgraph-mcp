from .base_tool import BaseTool
from rag.common.task import Task
from rag.common.context import Context
from rag.common.logic_node_parser import MathNode
from typing import Generator, List, Tuple, Dict, Any, Optional, cast
from rag.prompt.rewrite_prompt import RewriteSubQuery
from rag.prompt.choice_label_prompt import ChoiceLabelPrompt
from rag.prompt.judge_equal_prompt import JudgeEqualPrompt
from rag.prompt.reasoning_prompt import ReasoningPrompt
from rag.utils.log import rag_logger as log
from llama_index.core.vector_stores.types import VectorStoreQuery
from rag.index.atlas_rag_index import AtlasRAGIndex
from rag.utils.utils import to_legal_cypher_string
from llama_index.core.schema import TextNode
import re


class DeduceTool(BaseTool):

    def execute(
        self, query: str, task: Task, context: Context
    ) -> Generator[str, None, None]:
        logic_form: MathNode = task.arguments["logic_form_node"]
        sub_query: str = task.arguments["query"]
        content = logic_form.content
        try:
            content_l = re.findall("`(.*?)`", content)
        except Exception as e:
            # breakpoint()
            content_l = []
        contents = []
        for c in content_l:
            if c in context.variables:
                values = context.variables[c]
                if values:
                    c = f"{c}={values}"
                else:
                    continue
            contents.append(c)
        reason_prompt = ReasoningPrompt()
        answer_dict: Dict[str, str] = reason_prompt.get_result(
            self.llm,
            input=f"问题: {sub_query} 目标: {logic_form.target}",
            text_chunks=contents,
        )
        if answer_dict["can_answer"]:
            yield f"问题: {sub_query} 答案: {answer_dict['answer']}"
            task.update_result(f"问题: {sub_query} 答案: {answer_dict['answer']}")
        else:
            task.update_result(f"问题: {sub_query} 答案: 无法回答")
