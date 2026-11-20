from rag.prompt.base_prompt import BasePrompt
from rag.model.atlas_model import CompletionResponse
from typing import List, Dict, Any


class ChoiceLabelPrompt(BasePrompt):
    template = """
    {{
        "instruction":"请根据提供的标签选项选择一个输入实体或者关系最贴近的标签,只要两者语意上有相似就认为是贴近的,labels和input一定都是同一类型的，比如都是点或者都是边,如果认为输入实体或者关系不符合任何标签选项，请选择Other标签。请注意不要输出其他额外信息,结果直接以字符串返回。",
        "examples": [
            {{
                "labels": ["Company", "Department", "Module", "Person"],
                "input": "海致星图",
                "output": "Company"
            }},
        ],
        "labels": {labels},
        "input": {input}
    }}
"""

    @property
    def template_variables(self) -> List[str]:
        return ["labels", "input"]

    def build_prompt(self, **kwargs: Any) -> str:
        variables: Dict[str, Any] = {}
        if "labels" in kwargs:
            if isinstance(kwargs["labels"], list) and all(
                isinstance(label, str) for label in kwargs["labels"]
            ):
                variables["labels"] = kwargs["labels"]
                self.labels = kwargs["labels"]
            else:
                raise ValueError("labels must be a list of str")
        else:
            raise ValueError("labels is required")
        if "input" in kwargs:
            if isinstance(kwargs["input"], str):
                variables["input"] = str(kwargs["input"])
            else:
                raise ValueError("question must be a str")
        else:
            raise ValueError("input is required")
        if "is_vertex" in kwargs:
            self.is_vertex = bool(kwargs["is_vertex"])
        else:
            self.is_vertex = True
        # print(self.template.format(**variables))
        return self.template.format(**variables)

    def parse_response(self, response: str | CompletionResponse) -> str:
        label = str(response)
        if label in self.labels:
            return label
        else:
            if self.is_vertex:
                return "Entity"
            else:
                return "Relation"
