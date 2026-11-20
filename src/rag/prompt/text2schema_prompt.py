from rag.prompt.base_prompt import BasePrompt
from rag.model.atlas_model import CompletionResponse
from typing import List, Dict, Any
from rag.schema.graph_schema import GraphSchema, GraphSchemaModel
import re
from rag.utils.log import rag_logger as log
import json


class Text2SchemaPrompt(BasePrompt):
    template = """
    {{
        "instruction":"你是一个Open Cypher的专家，你需用根据用户提供的问题和已有的schema中分析出用户问题所涉及到的点边标签，并以JSON格式返回。
        请你保证返回的标签一定是用户提供的标签列表中存在的。你可以参考examples里面的示例",
        "examples": [
            {{
                "schema": {{
                    "vertices": [
                        {{
                            "label": "Company",
                            "explain": "公司",
                            "properties": [
                                {{
                                    "property": "name",
                                    "type": "string",
                                    "explain": "公司名称"
                                }}
                            ]
                        }},
                        {{
                            "label": "Module",
                            "explain": "产品模块",
                            "properties": [
                                {{
                                    "property": "name",
                                    "type": "string",
                                    "explain": "产品模块名称"
                                }}
                            ]
                        }},
                        {{
                            "label": "Person",
                            "explain": "员工",
                            "properties": [
                                {{
                                    "property": "person_id",
                                    "type": "string",
                                    "explain": "员工ID"
                                }},
                                {{
                                    "property": "name",
                                    "type": "string",
                                    "explain": "员工姓名"
                                }},
                                {{
                                    "property": "age",
                                    "type": "int",
                                    "explain": "员工年龄"
                                }}
                            ]
                        }}
                    ],
                    "edges": [
                        {{
                            "label": "Include",
                            "explain": "公司包含的模块"
                            "relations": [
                                {{
                                    "from_key": "Company",
                                    "to_key": "Module",
                                }}
                            ],
                            "properties": [],
                        }},
                        {{
                            "label": "WorkAt",
                            "explain": "员工就职的产品模块",
                            "relations": [
                                {{
                                    "from_key": "Person",
                                    "to_key": "Module",
                                }}
                            ],
                            "properties": [],
                        }},
                    ],
                }},
                "input": "员工的平均年龄是多少？",
                "output": {{
                    "vertices": [
                        {{
                            "label": "Person",
                            "explain": "员工",
                            "properties": [
                                {{
                                    "property": "age",
                                    "type": "int",
                                    "explain": "员工年龄"
                                }}
                            ]
                        }}
                    ],
                    "edges": [
                    ],
                }},
            }},
        ],
        "schema": {schema},
        "input": {input}
    }}
"""

    @property
    def template_variables(self) -> List[str]:
        return ["labels", "input"]

    def build_prompt(self, **kwargs: Any) -> str:
        variables: Dict[str, str] = {}
        if "schema" in kwargs:
            if isinstance(kwargs["schema"], GraphSchema):
                variables["schema"] = str(kwargs["schema"])
                self.schema = kwargs["schema"]
            else:
                raise ValueError("schema must be a GraphSchema instance")
        else:
            raise ValueError("schema is required")
        if "input" in kwargs:
            if isinstance(kwargs["input"], str):
                variables["input"] = str(kwargs["input"])
            else:
                raise ValueError("input must be a str")
        else:
            raise ValueError("input is required")
        return self.template.format(**variables)

    def parse_response(self, response: str | CompletionResponse) -> GraphSchema:
        re_match = re.search(r"\{.*\}", str(response), re.DOTALL)
        if not re_match:
            log.error(f"resonse: {response} failed to parse response")
            raise ValueError(f"resonse: {response} failed to parse response")
        rsp = json.loads(re_match.group(0))
        if "output" in rsp:
            rsp = rsp["output"]
        if isinstance(rsp, dict):
            return GraphSchema.from_json(rsp)
        else:
            raise ValueError("response must be a dict")


if __name__ == "__main__":
    from rag.model import AtlasLLM
    from rag.utils.config import rag_config as config

    prompt = Text2SchemaPrompt()
    schema = GraphSchema.from_json(
        {
            "vertices": [
                {
                    "label": "Company",
                    "explain": "公司",
                    "properties": [
                        {"property": "name", "type": "string", "explain": "公司名称"}
                    ],
                },
                {
                    "label": "Module",
                    "explain": "产品模块",
                    "properties": [
                        {
                            "property": "name",
                            "type": "string",
                            "explain": "产品模块名称",
                        }
                    ],
                },
                {
                    "label": "Person",
                    "explain": "员工",
                    "properties": [
                        {
                            "property": "person_id",
                            "type": "string",
                            "explain": "员工ID",
                        },
                        {"property": "name", "type": "string", "explain": "员工姓名"},
                        {"property": "age", "type": "int", "explain": "员工年龄"},
                    ],
                },
            ],
            "edges": [
                {
                    "label": "Include",
                    "explain": "公司包含的模块",
                    "relations": [{"from_key": "Company", "to_key": "Module"}],
                    "properties": [],
                },
                {
                    "label": "WorkAt",
                    "explain": "员工就职的产品模块",
                    "relations": [{"from_key": "Person", "to_key": "Module"}],
                    "properties": [],
                },
            ],
        }
    )
    config.llm_config["enable_thinking"] = False
    llm = AtlasLLM(
        model="Qwen/Qwen3-32B",
        api_base="https://api.siliconflow.cn/v1",
        api_key="sk-**",
    )
    input_text = "年龄在25岁到30岁的员工最多的前三个产品模块是什么？"
    print(prompt.get_result(llm=llm, schema=schema, input=input_text))
