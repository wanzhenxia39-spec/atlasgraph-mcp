from rag.prompt.base_prompt import BasePrompt
from rag.model.atlas_model import CompletionResponse
from typing import List, Dict, Any


class QuestionTransformPrompt(BasePrompt):
    template = """
    {{
        "instruction": "你需要精确细致的拆解并按照编写cypher的步骤重写用户的问题以方便后续的cypher语句生成。直接返回问题的拆解步骤即可不用解释原因。请以字符串的形式返回。你可以参考examples中的示例。",
        "examples": [
            {{
                "input": "教授老师最多的学科的老师的平均年龄是多少？",
                "output": "Step 1：找出教授人数最多的学科\n  Step 2：计算该学科老师的平均年龄"
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
                self.question = kwargs["input"]
            else:
                raise ValueError("input must be a string")
        else:
            raise ValueError("input is required")
        return self.template.format(**variables)

    def parse_response(self, response: str | CompletionResponse) -> str:
        return f"原始问题: {self.question}\n拆解步骤如下:\n{str(response)}"
