from rag.prompt.base_prompt import BasePrompt
from rag.model.atlas_model import CompletionResponse
from typing import List, Dict, Any
from rag.utils.log import rag_logger as log
import json
import re


class SimplifyTextPrompt(BasePrompt):
    template = """
    {{
        "instruction": "你是一个文本相关性分析的专家，你能够精确的分析出给定的文本和问题之间的相关性，从文本中提取回答问题相关的信息片段并简要的概述。请你根据用户提供的文本片段和问题，简要的概述可能和回答问题相关的文本片段。结果以列表字符串的形式返回，如果没有和问题相关
        的可提取片段返回空列表。你可以参照example进行抽取。",
        "examples": [
            {{
                "text_chunks": [
                    "海致星图成立于2015年，致力于通过自主可控的大数据、图计算、知识图谱等核心技术，服务于金融、能源、制造等政府机构与企业级用户的数字化转型，帮助用户构建数据驱动的智能管理运营体系，为用户的各项决策提供顶层视角。",
                    "图数据库部门是其主要部门，其中包括图查询、图计算、图学习等模块",
                    "目前已知的公开的信息中，图查询模块有两名员工分别是李晨和张伟，图计算包含一名员工是David Chen，图学习包含一名员工是Olivia Tian。"
                ],
                "input": "海致星图的图学习模块有哪些员工？",
                "output": [
                    "图数据库部门包括图学习",
                    "图学习包含一名员工是Olivia Tian"
                ],
            }}
        ],
        "text_chunks": {text_chunks}
        "input": "{input}",
    }}
    """

    @property
    def template_variables(self) -> List[str]:
        return ["text_chunks", "input"]

    def build_prompt(self, **kwargs: Any) -> str:
        variables: Dict[str, str] = {}
        if "text_chunks" in kwargs:
            if isinstance(kwargs["text_chunks"], list) and all(
                [isinstance(text, str) for text in kwargs["text_chunks"]]
            ):
                variables["text_chunks"] = json.dumps(kwargs["text_chunks"],ensure_ascii=False)
            else:
                raise ValueError("text_chunks must be a list of string")
        else:
            raise ValueError("text_chunks is required")
        if "input" in kwargs:
            if isinstance(kwargs["input"], str):
                variables["input"] = kwargs["input"]
            else:
                raise ValueError("input must be a string")
        else:
            raise ValueError("input is required")
        return self.template.format(**variables)

    def parse_response(self, response: str | CompletionResponse) -> List[str]:
        re_match = re.search(r"\[.*\]", str(response), re.DOTALL)
        if not re_match:
            log.error(f"resonse: {response} failed to parse response")
            raise ValueError(f"resonse: {response} failed to parse response")
        rsp = json.loads(re_match.group(0))
        if "output" in rsp:
            rsp = rsp["output"]
        if isinstance(rsp, list) and all([isinstance(text, str) for text in rsp]):
            return rsp
        else:
            raise ValueError("response must be a list")
