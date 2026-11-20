from rag.prompt.base_prompt import BasePrompt
from rag.model.atlas_model import CompletionResponse
from typing import List, Dict, Any


class RewriteSubQuery(BasePrompt):
    template = """{{
    "instruction":"请根据上下文的信息，替换当前问题中涉及的指代实体，进而改写当前问题。",
    "cases": [
        {{
            "content": ["Step1:X先生的老婆是谁？\nanswer:Y女士"]
            "question": "Step1的答案中的人代表作品有哪些？"
            "rewrite question": "Y女士的代表作品有哪些？"
        }},
        {{
            "content": ["Step1:A先生的兄弟有哪些？\nanswer:A先生的兄弟有B先生、C先生、D先生",]
            "question": "在Step1中检索到的人在哪儿受教育"
            "rewrite question": "B先生、C先生、D先生分别在哪儿受教育"
        }},
    ],
    "content": {content},
    "question": {question},
    "notice": "不要输出json格式，只使用文本输出改写的问题，并且尽可能完整的输出改写问题，不要遗漏不要输出其他额外信息"
}}"""

    @property
    def template_variables(self) -> List[str]:
        return ["content", "question"]

    def build_prompt(self, **kwargs: Any) -> str:
        variables: Dict[str, Any] = {}
        if "content" in kwargs:
            if isinstance(kwargs["content"], list):
                variables["content"] = kwargs["content"]
            else:
                raise ValueError("content must be a list")
        else:
            variables["content"] = []
        if "question" in kwargs:
            if isinstance(kwargs["question"], str):
                variables["question"] = str(kwargs["question"])
            else:
                raise ValueError("question must be a str")
        else:
            variables["question"] = ""
        return self.template.format(**variables)

    def parse_response(self, response: str | CompletionResponse) -> str:
        return str(response)
