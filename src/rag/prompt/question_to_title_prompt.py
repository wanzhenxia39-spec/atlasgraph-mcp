from rag.prompt.base_prompt import BasePrompt
from rag.model.atlas_model import CompletionResponse
from typing import List, Dict, Any
from rag.schema import DEFAULT_SCHEMA
from rag.utils.log import rag_logger as log
import json
import re


class QuestionToTitlePrompt(BasePrompt):
    template = """
    {{
        "instruction": "你是一个语义分析的专家，你能够精确且简要的概述用户的问题。请你根据用户提供的问题用不超过20个字符对其简要的概述，请保证语言的干练这个概述将作为标题，请不要使用富文本描述返回纯字符串的概述即可。你可以参照example进行提炼。",
        "examples": [
            {{
                "input": GET和POST的区别,
                "output": GET vs POST 区别
            }}
        ],
        "input": "{input}",
    }}
    """

    @property
    def template_variables(self) -> List[str]:
        return ["input"]

    def build_prompt(self, **kwargs: Any) -> str:
        variables: Dict[str, str] = {}
        if "input" in kwargs:
            if isinstance(kwargs["input"], str):
                variables["input"] = kwargs["input"]
            else:
                raise ValueError("input must be a string")
        else:
            raise ValueError("input is required")
        return self.template.format(**variables)

    def parse_response(self, response: str | CompletionResponse) -> str:
        response = str(response)
        return response
