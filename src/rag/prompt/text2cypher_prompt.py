from rag.prompt.base_prompt import BasePrompt
from rag.model.atlas_model import CompletionResponse
from typing import List, Dict, Any
from rag.schema.graph_schema import GraphSchema


class Text2CypherPrompt(BasePrompt):
    template = """
    {{
        "instruction":"你是一个Open Cypher的专家，你需用根据用户提供的问题和已有的schema中生成对应的一条cypher语句。请你保证你生成的cypher语句中使用的点边标签和属性字段是用户提供的schema中存在，注意label的大小写。你不需要输出任何解释信息，直接输出cypher即可。结果请直接以字符串的方式返回。你可以参考examples里面的示例",
        "tips": [
            '不要使用COUNT(*)的语法，不要使用COUNT(*)的语法，不要使用COUNT(*)的语法，而是使用COUNT(变量名)的语法',
            '如果涉及时间的字符串类型和date或者datetime类型变量比较,字符串类型需要使用date或者datetime函数进行转换。比如"2023-01-01" -> date("2023-01-01"),"2023-01-01 00:00:00" -> datetime("2023-01-01 00:00:00")，注意datetime("2023-01-01")是不合法的,必须写成datetime("2023-01-01 00:00:00")',
            '如果cypher的一个变量本身类型就是date或者datetime你就不需要重复调用date或者dattime进行转换',
        ]
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
                "input": "海致星图的产品模块中年龄在25岁到30岁的员工最多的前三个产品模块是什么？",
                "output": "MATCH (c:Company {{name: "海致星图"}})-[:Include]->(m:Module)<-[:WorkAt]-(p:Person) WHERE p.age >= 25 AND p.age <= 30 RETURN m.name, COUNT(p) AS count ORDER BY count DESC LIMIT 3"
            }},
        ],
        "schema": {schema},
        "input": {input}
    }}
"""

    @property
    def template_variables(self) -> List[str]:
        return ["schema", "input"]

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

    def parse_response(self, response: str | CompletionResponse) -> str:
        return str(response)


# if __name__ == "__main__":
#     from rag.model import AtlasLLM
#     from rag.utils.config import rag_config as config

#     prompt = Text2CypherPrompt()
#     schema = GraphSchema.from_json(
#         {
#             "vertices": [
#                 {
#                     "label": "Company",
#                     "explain": "公司",
#                     "properties": [
#                         {"property": "name", "type": "string", "explain": "公司名称"}
#                     ],
#                 },
#                 {
#                     "label": "Module",
#                     "explain": "产品模块",
#                     "properties": [
#                         {
#                             "property": "name",
#                             "type": "string",
#                             "explain": "产品模块名称",
#                         }
#                     ],
#                 },
#                 {
#                     "label": "Person",
#                     "explain": "员工",
#                     "properties": [
#                         {
#                             "property": "person_id",
#                             "type": "string",
#                             "explain": "员工ID",
#                         },
#                         {"property": "name", "type": "string", "explain": "员工姓名"},
#                         {"property": "age", "type": "int", "explain": "员工年龄"},
#                     ],
#                 },
#             ],
#             "edges": [
#                 {
#                     "label": "Include",
#                     "explain": "公司包含的模块",
#                     "relations": [{"from_key": "Company", "to_key": "Module"}],
#                     "properties": [],
#                 },
#                 {
#                     "label": "WorkAt",
#                     "explain": "员工就职的产品模块",
#                     "relations": [{"from_key": "Person", "to_key": "Module"}],
#                     "properties": [],
#                 },
#             ],
#         }
#     )
#     config.llm_config["enable_thinking"] = False
#     llm = AtlasLLM(
#         model="Qwen/Qwen3-32B",
#         api_base="https://api.siliconflow.cn/v1",
#         api_key="sk-**",
#     )
#     input_text = (
#         "海致星图的产品模块中年龄在25岁到30岁的员工最多的前三个产品模块是什么？"
#     )
#     print(prompt.get_result(llm=llm, schema=schema, input=input_text))
