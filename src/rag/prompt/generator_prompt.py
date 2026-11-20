from rag.prompt.base_prompt import BasePrompt
from rag.model.atlas_model import CompletionResponse
from typing import List, Dict, Any
from rag.schema import DEFAULT_SCHEMA
from rag.utils.log import rag_logger as log
import json
import re


class GeneratorPrompt(BasePrompt):
    template = """
    请严格基于提供的检索文本信息和图谱信息和前置子问题的答案回答用户的问题，如果不能回答请解释原因
    ## 检索文本信息
    ---------------------
    {text_chunks}
    ---------------------
    ## 图谱信息
    ---------------------
    {graph}
    ---------------------
    ## 前置子问题和答案
    ---------------------
    {sub_question_answer}
    ---------------------
    ## 用户问题
    ---------------------
    {question}
    ---------------------
    """

    @property
    def template_variables(self) -> List[str]:
        return ["text_chunks", "graph", "sub_question_answer", "question"]

    def build_prompt(self, **kwargs: Any) -> str:
        variables: Dict[str, Any] = {}
        if "text_chunks" in kwargs:
            if isinstance(kwargs["text_chunks"], list) and all(
                [isinstance(x, str) for x in kwargs["text_chunks"]]
            ):
                variables["text_chunks"] = str(kwargs["text_chunks"])
            else:
                raise ValueError("text_chunks must be a List of String")
        else:
            variables["text_chunks"] = []
        if "graph" in kwargs:
            if isinstance(kwargs["graph"], dict):
                variables["graph"] = kwargs["graph"]
            else:
                raise ValueError("graph must be a dict")
        else:
            variables["graph"] = {}
        if "sub_question_answer" in kwargs:
            if isinstance(kwargs["sub_question_answer"], list) and all(
                map(lambda x: isinstance(x, str), kwargs["sub_question_answer"])
            ):
                variables["sub_question_answer"] = kwargs["sub_question_answer"]
            else:
                raise ValueError("sub_question_answe must be a list of string")
        else:
            variables["sub_question"] = []
        if "question" in kwargs:
            if isinstance(kwargs["question"], str):
                variables["question"] = kwargs["question"]
            else:
                raise ValueError("question must be a string")
        else:
            raise ValueError("question is required")
        return self.template.format(**variables)

    def parse_response(self, response: str | CompletionResponse) -> str:
        return str(response)
