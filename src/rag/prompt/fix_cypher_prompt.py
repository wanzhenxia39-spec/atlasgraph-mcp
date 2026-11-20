from rag.prompt.base_prompt import BasePrompt
from rag.model.atlas_model import CompletionResponse
from typing import List, Dict, Any


class FixCypherPrompt(BasePrompt):
    template = """
    {{
        "instruction":"你是一个Open Cypher的专家，你需要根据用户提供的原始问题、cypher语句以及报错信息，根据Open Cypher的语法规范将用户的cypher语句修改正确并返回，你不需要解释原因，直接返回修改正确的cypher语句即可。结果直接以字符串的方式返回。你可以参考examples里面的示例。",
        "examples": [
            {{
                "cypher": "MATCH (t:teacher) RETURN t.subject AS subject, COUNT(*) AS count ORDER BY count DESC LIMIT 1",
                "errors": [
                    "[Error: ParseFailed(4250)] UnrecognizedToken {{ token: (53, Star, 54), expected: ["\"%\"", "\"(\"", "\")\"", "\"+\"", "\"-\"", "\"ALL\"", "\"ANY\"", "\"APOC\"", "\"CASE\"", "\"DISTINCT\"", "\"FALSE\"", "\"NONE\"", "\"NOT\"", "\"NULL\"", "\"SINGLE\"", "\"TRUE\"", "\"[\"", "\"{{\"", "DoubleLiteral", "Identifier", "IntegerLiteral", "Parameter", "StringLiteral"] }}",
                ]
                "question": "教授哪个科目的老师最多？",
                "output": "MATCH (t:teacher) RETURN t.subject AS subject, COUNT(*) AS count ORDER BY count DESC LIMIT 1"
            }},
        ],
        "cypher": {cypher},
        "errors": {errors},
        "question": {question},
    }}
"""

    @property
    def template_variables(self) -> List[str]:
        return ["cypher", "errors", "question"]

    def build_prompt(self, **kwargs: Any) -> str:
        variables: Dict[str, Any] = {}
        if "cypher" in kwargs:
            if isinstance(kwargs["cypher"], str):
                variables["cypher"] = str(kwargs["cypher"])
            else:
                raise ValueError("cypher must be a str")
        else:
            raise ValueError("cypher is required")
        if "errors" in kwargs:
            if isinstance(kwargs["errors"], list):
                variables["errors"] = kwargs["errors"]
            else:
                raise ValueError("errors must be a list")
        else:
            raise ValueError("errors is required")
        if "question" in kwargs:
            if isinstance(kwargs["question"], str):
                variables["question"] = str(kwargs["question"])
            else:
                raise ValueError("question must be a str")
        else:
            raise ValueError("question is required")
        return self.template.format(**variables)

    def parse_response(self, response: str | CompletionResponse) -> str:
        return str(response)
