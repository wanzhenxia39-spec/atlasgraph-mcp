from rag.prompt.base_prompt import BasePrompt
from rag.model.atlas_model import CompletionResponse
from typing import List, Dict, Any, Union, cast
from rag.utils.log import rag_logger as log
from rag.utils.config import rag_config as config
import json
import re
from rag.splitter import get_tokens


class SimplifyQuestionPrompt(BasePrompt):
    template = """
    {{
        "instruction": "你是一个问题分析简化的专家，你能够用不超过max_tokens的字符精确简要的概述用户的问题。结果以字符串的形式返回，你可以参照example进行抽取。",
        "examples": [
            {{
                "input": "海致星图成立于2015年，致力于通过自主可控的大数据、图计算、知识图谱等核心技术，服务于金融、能源、制造等政府机构与企业级用户的数字化转型，
                帮助用户构建数据驱动的智能管理运营体系，为用户的各项决策提供顶层视角。请问海致星图的图学习模块有哪些员工？",
                "max_tokens": 512,
                "output": "海致星图的图学习模块有哪些员工？",
            }}
        ],
        "input": "{input}",
        "max_tokens": {max_tokens}

    }}
    """

    @property
    def template_variables(self) -> List[str]:
        return ["input", "max_tokens"]

    def build_prompt(self, **kwargs: Any) -> str:
        variables: Dict[str, Union[int, str]] = {}
        if "max_tokens" in kwargs:
            if isinstance(kwargs["max_tokens"], int):
                variables["max_tokens"] = int(kwargs["max_tokens"])
            else:
                raise ValueError("max_tokens must be a int")
        else:
            variables["max_tokens"] = min(
                config.llm_max_context, config.embedding_max_context
            )
        if "input" in kwargs:
            if isinstance(kwargs["input"], str):
                variables["input"] = kwargs["input"]
            else:
                raise ValueError("input must be a string")
        else:
            raise ValueError("input is required")
        self.max_tokens: int = cast(int, variables["max_tokens"])
        return self.template.format(**variables)

    def parse_response(self, response: str | CompletionResponse) -> str:
        rsp = str(response)
        tokens = get_tokens(rsp)
        if len(tokens) >= self.max_tokens:
            return "".join(tokens[: self.max_tokens])
        else:
            return rsp
