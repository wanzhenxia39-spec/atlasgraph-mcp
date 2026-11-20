from rag.prompt.base_prompt import BasePrompt
from rag.model.atlas_model import CompletionResponse
from typing import List, Dict, Any, Set, Tuple
from rag.schema import Schema, DEFAULT_SCHEMA
from rag.utils.log import rag_logger as log
import json
import re


# 价值百万的prompt QAQ
class RAGCypherPrompt(BasePrompt):
    template = """
    {{
        "instruction": "你是一个open cypher语言的专家，能够根据用户的问题使用如下的cypher模版补充相关的参数编写一条或者多条的标准的open cypher语句查询相关的数据以便能够检索能回答用户问题的相关信息。请以JSON列表字符串格式回应,不需要返回任何其他信息。你可以参照examples进行回答。
        你有两种cypher生成方式，第一种是直接使用模版填充参数，第二种是自由根据当前已知信息和问题进行cypher生成,生成cypher语句的时候请注意方向以及严格参考schema结构否则可能出现无法匹配上点边的情况,如果采用第二用生成方式请保证生成cypher语句是可以直接执行，不需要任何额外的变量替换以
        及拒绝生成任何涉及删除图谱数据的cypher语句
        cypher模版如下:\
            MATCH p=(n1: $src_node_label)-[r:$rel_label]-(n2:$dst_node_label)
            WHERE n1.keyword in $keyword_list
            RETURN n1.keyword, n1.description, r.keyword, n2.keyword, n2.description
        生成cypher语句时请你注意必须参考schema中的信息，不要试图查询不存在的点边链接，同时生成keyword_list参数时，请你参考graph中的信息，不要生成不存在于graph中该label的keyword的值，对应同一个label下多个keyword尽可能的合成到一个keyword_list中。
        你仅需要生成那些在当前提供的graph中不存在需要额外通过cypher语句查询信息的填充参数信息，避免重复查询已知的信息。",
        "examples": [
            {{
                "schema": {{
                    "vertices": {{
                        "Company": ["keyword", "description"],
                        "Department": ["keyword", "description"],
                        "Module": ["keyword", "description"],
                        "Person": ["keyword", "description"],
                    }},
                    "edges": {{
                        "Include": {{
                            "relations": [
                                {{
                                    "from_key": "Company",
                                    "to_key": "Department",
                                }},
                                {{
                                    "from_key": "Department",
                                    "to_key": "Module",
                                }},
                            ],
                            "properties": ["keyword"],
                        }},
                        "Link": {{
                            "relations": [
                                {{
                                    "from_key": "Person",
                                    "to_key": "Module",
                                }}
                            ],
                            "properties": ["keywrod"],
                        }},
                    }},
                    "explain": "",
                }},
                "graph": {{
                    "vertices": [
                        {{
                            "label": "Module",
                            "keyword": "图学习",
                            "description": "图学习是海致星图图数据库的一个模块",
                        }},
                    ],
                    "edges": [],
                }},
                "input": "图学习的员工有谁以及总共有多少员工？",
                "output": [
                    {{
                        "src_node_label": "Module",
                        "rel_label": "Link",
                        "dst_node_label": "Person",
                        "keyword_list": ["图学习"]
                    }},
                    {{
                        "cypher": "MATCH (n1:Person)-[r:Link]->(n2:Module) WHERE n2.keyword="图学习" RETURN count(n1) AS count"
                    }}
                ]
            }}
        ],
        "schema": {schema},
        "graph": {graph},
        "input": "{input}"
    }}
    """

    @property
    def template_variables(self) -> List[str]:
        return ["schema", "graph", "input"]

    def build_prompt(self, **kwargs: Any) -> str:
        variables: Dict[str, Any] = {}
        if "schema" in kwargs:
            if isinstance(kwargs["schema"], Schema):
                self.schema: Schema = kwargs["schema"]
                variables["schema"] = str(kwargs["schema"])
            else:
                raise ValueError("schema must be a Schema object")
        else:
            self.schema = DEFAULT_SCHEMA
            variables["schema"] = str(DEFAULT_SCHEMA)
        if "graph" in kwargs:
            if isinstance(kwargs["graph"], dict):
                self.graph = kwargs["graph"]
                variables["graph"] = kwargs["graph"]
            else:
                raise ValueError("graph must be a dict")
        else:
            raise ValueError("graph must be provided")
        if "input" in kwargs:
            if isinstance(kwargs["input"], str):
                variables["input"] = kwargs["input"]
            else:
                raise ValueError("input must be a string")
        else:
            raise ValueError("input is required")
        return self.template.format(**variables)

    def parse_response(
        self, response: str | CompletionResponse
    ) -> List[Tuple[Tuple[str, str, str], str]]:
        re_match = re.search(r"\[.*\]", str(response), re.DOTALL)
        if not re_match:
            log.error(f"resonse: {response} failed to parse response")
            raise ValueError(f"resonse: {response} failed to parse response")
        try:
            rsp = json.loads(re_match.group(0))
        except:
            print(re_match.group(0))
            raise
        if "output" in rsp:
            rsp = rsp["output"]
        cypher_template = "MATCH p=(n1: {src_node_label})-[r:{rel_label}]-(n2:{dst_node_label}) WHERE n1.keyword in {keyword_list} RETURN n1.keyword, n1.description, r.keyword, n2.keyword, n2.description"
        cypher_list = []
        if isinstance(rsp, dict):
            rsp = [rsp]
        if isinstance(rsp, list):
            keyword_set = {
                (vertex["label"], vertex["keyword"])
                for vertex in self.graph["vertices"]
            }
            for kwargs in rsp:
                if not isinstance(kwargs, dict):
                    continue
                if (
                    all(
                        [
                            key in kwargs
                            for key in [
                                "src_node_label",
                                "rel_label",
                                "dst_node_label",
                                "keyword_list",
                            ]
                        ]
                    )
                    and all(
                        [
                            isinstance(kwargs[key], str)
                            for key in ["src_node_label", "rel_label", "dst_node_label"]
                        ]
                    )
                    and isinstance(kwargs["keyword_list"], list)
                    and all(
                        [isinstance(keyword, str) for keyword in kwargs["keyword_list"]]
                    )
                    and len(kwargs["keyword_list"]) != 0
                ):
                    src_node_label = kwargs["src_node_label"]
                    rel_label = kwargs["rel_label"]
                    dst_node_label = kwargs["dst_node_label"]
                    keyword_list = kwargs["keyword_list"]
                    if (
                        src_node_label in self.schema.vertices
                        and dst_node_label in self.schema.vertices
                        and (
                            (src_node_label, rel_label, dst_node_label)
                            in self.schema.canonical_edge_labels
                            or (dst_node_label, rel_label, src_node_label)
                            in self.schema.canonical_edge_labels
                        )
                        and all(
                            [
                                (src_node_label, keyword) in keyword_set
                                for keyword in keyword_list
                            ]
                        )
                    ):
                        cypher_list.append(
                            (
                                (src_node_label, rel_label, dst_node_label),
                                cypher_template.format(
                                    src_node_label=src_node_label,
                                    rel_label=rel_label,
                                    dst_node_label=dst_node_label,
                                    keyword_list=keyword_list,
                                ),
                            )
                        )
                    else:
                        continue
                elif "cypher" in kwargs and isinstance(kwargs["cypher"], str):
                    cypher_list.append(
                        (
                            (None, None, None),
                            kwargs["cypher"],
                        )
                    )
                else:
                    continue
        else:
            raise ValueError(f"response must be a list")
        return cypher_list


if __name__ == "__main__":
    from rag.utils.config import rag_config as config
    from rag.model.atlas_model import AtlasLLM

    llm = AtlasLLM(
        model=config.llm_model,
        api_url=config.llm_url,
        api_base=config.llm_base,
        api_key=config.llm_key,
    )
    prompt = RAGCypherPrompt()
    query = prompt.build_prompt(
        schema=Schema(
            {
                "vertices": {
                    "Company": ["keyword", "description"],
                    "Department": ["keyword", "description"],
                    "Module": ["keyword", "description"],
                    "Person": ["keyword", "description"],
                },
                "edges": {
                    "Include": {
                        "relations": [
                            {"from_key": "Company", "to_key": "Department"},
                            {"from_key": "Department", "to_key": "Module"},
                        ],
                        "properties": ["keyword"],
                    },
                    "Link": {
                        "relations": [{"from_key": "Person", "to_key": "Module"}],
                        "properties": ["keyword"],
                    },
                },
            }
        ),
        graph={
            "vertices": [
                {
                    "label": "Module",
                    "keyword": "自然语言处理",
                    "description": "自然语言处理是云智科技的一个模块",
                },
            ],
            "edges": [],
        },
        input="云智科技的自然语言处理有哪些人？",
    )
    print(prompt.parse_response(llm.complete(query)))
