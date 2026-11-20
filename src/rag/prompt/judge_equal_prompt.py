from rag.prompt.base_prompt import BasePrompt
from rag.model.atlas_model import CompletionResponse
from typing import List, Dict, Any


class JudgeEqualPrompt(BasePrompt):
    template = """
    {{
        "instruction":"你需要判断提供的两个实体或者关系是否是等价或者其中一个包含另外一个，你只需要回复YES或者NO，不需要输入其他额外信息。",
        "examples": [
            {{
                "input1": {{
                    "label": "Person",
                    "keyword": "鲁迅"
                }}
                "input2": {{
                    "label": "Entity",
                    "keyword": "周树人"
                }}
                "output": "YES"
            }},
            {{
                "input1": {{
                    "label": "Country",
                    "keyword": "中国"
                }}
                "input2": {{
                    "label": "Entity",
                    "keyword": "中华人民共和国"
                }}
                "output": "YES"
            }},
        ],
        "input1": {input1}
        "input2": {input2}
    }}
    """

    @property
    def template_variables(self) -> List[str]:
        return ["input1", "input2"]

    def build_prompt(self, **kwargs: Any) -> str:
        variables: Dict[str, Any] = {}
        if "input1" in kwargs:
            if (
                isinstance(kwargs["input1"], dict)
                and "label" in kwargs["input1"]
                and "keyword" in kwargs["input1"]
            ):
                variables["input1"] = {
                    "label": str(kwargs["input1"]["label"]),
                    "keyword": str(kwargs["input1"]["keyword"]),
                }
            else:
                raise ValueError(
                    "input1 must be a dict with 'label' and 'keyword' keys"
                )
        else:
            raise ValueError("input1 is required")
        if "input2" in kwargs:
            if (
                isinstance(kwargs["input2"], dict)
                and "label" in kwargs["input2"]
                and "keyword" in kwargs["input2"]
            ):
                variables["input2"] = {
                    "label": str(kwargs["input2"]["label"]),
                    "keyword": str(kwargs["input2"]["keyword"]),
                }
            else:
                raise ValueError(
                    "input2 must be a dict with 'label' and 'keyword' keys"
                )
        else:
            raise ValueError("input2 is required")
        return self.template.format(**variables)

    def parse_response(self, response: str | CompletionResponse) -> bool:
        if str(response).lower() == "yes":
            return True
        else:
            return False
