from rag.prompt.base_prompt import BasePrompt
from rag.model.atlas_model import CompletionResponse
from typing import List, Dict, Any, Set, Tuple
from rag.schema import Schema, DEFAULT_SCHEMA
from rag.utils.log import rag_logger as log
import json
import re


# 价值百万的prompt QAQ
class GraphExtractPrompt(BasePrompt):
    template = """
    {{
        "instruction": "你是一个实体关系识别的专家，非常擅长开放与非开放信息提取并且能够简要准确的概述实体关系描述。请从输入中按照指定的schema格式检查graph中已提取出的点边关系是否还有缺漏，提取与模式定义匹配的缺漏部分的实体和关系（允许孤立点存在）。
        请注意schema中的某些点边类型在文本中可能不存在或者已经没有缺漏的点边实体了。这种情况下不用输出该类点边，此时请返回一个空的字典即可。在抽取边关系的时候请保证边关联的点实体也被抽取出来了，否则该条边会被忽略。对于边的label你需要按照起点标签、边标签、
        终点标签的列表格式返回。请以JSON字符串格式回应。你可以参照examples进行抽取。",
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
                        "Relation": {{
                            "relations": [
                                {{
                                    "from_key": "Company",
                                    "to_key": "Department",
                                }},
                                {{
                                    "from_key": "Department",
                                    "to_key": "Module",
                                }},
                                {{
                                    "from_key": "Module",
                                    "to_key": "Person",
                                }}
                            ],
                            "properties": ["keyword"],
                        }},
                    }},
                    "explain": ""
                }},
                "graph": {{
                    "vertices": [
                        {{
                            "label": "Company",
                            "keyword": "海致星图",
                            "description": "海致星图成立于2015年，致力于通过自主可控的大数据、图计算、知识图谱等核心技术，服务于金融、能源、制造等政府机构与企业级用户的数字化转型的公司",
                        }},
                    ],
                    "edges": []
                }}
                "input": "海致星图成立于2015年，致力于通过自主可控的大数据、图计算、知识图谱等核心技术，服务于金融、能源、制造等政府机构与企业级用户的数字化转型，帮助用户构建数据驱动的智能管理运营体系，为用户的各项决策提供顶层视角。其主要部门包括图数据库等。\
                              其中图数据库部门是其主要部门，其中包括图学习等模块，截止于2025年1月30日，目前已知的公开的信息中，图学习包含一名员工是Olivia Tian。",
                "output": {{
                    "vertices": [
                        {{
                            "label": "Department",
                            "keyword": "图数据库",
                            "description": "图数据库是海致星图公司的一个子部门",
                        }},
                        {{
                            "label": "Module",
                            "keyword": "图学习",
                            "description": "图学习是海致星图图数据库的一个模块",
                        }},
                        {{
                            "label": "Person",
                            "keyword": "Olivia Tian",
                            "description": "Olivia Tian是海致星图图数据库图学习模块的员工",
                        }},
                    ],
                    "edges": [
                        {{
                            "label": ["Company","Relation","Department"],
                            "keyword": "包含",
                            "from_pk": "海致星图",
                            "to_pk": "图数据库",
                        }},
                        {{
                            "label": ["Department","Relation","Module"],
                            "keyword": "包含",
                            "from_pk": "图数据库",
                            "to_pk": "图学习",
                        }},
                        {{
                            "label": ["Module","Relation","Person"],
                            "keyword": "就职于",
                            "from_pk": "图学习",
                            "to_pk": "Olivia Tian",
                        }},
                    ],
                }},
            }}
        ],
        "schema": {schema},
        "graph": {graph},
        "input": "{input}",
    }}
    """

    @property
    def template_variables(self) -> List[str]:
        return ["schema", "input"]

    def build_prompt(self, **kwargs: Any) -> str:
        variables: Dict[str, str] = {}
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
                variables["graph"] = str(kwargs["graph"])
            else:
                raise ValueError("graph must be a dict")
        else:
            self.graph = {"vertices": [], "edges": []}
            variables["graph"] = str(self.graph)
        if "input" in kwargs:
            if isinstance(kwargs["input"], str):
                variables["input"] = kwargs["input"]
            else:
                raise ValueError("input must be a string")
        else:
            raise ValueError("input is required")
        self.input = variables["input"]
        return self.template.format(**variables) # 将变量替换到模板中返回完整提示

    # 1. 正则提取JSON响应
    # 2. 解析JSON
    # 3. 处理顶点(vertices)
    # 4. 处理边(edges)
    # 5. 合并结果到原图
    def parse_response(self, response: str | CompletionResponse) -> Tuple[Dict, bool]:
        re_match = re.search(r"\{.*\}", str(response), re.DOTALL)
        if not re_match:
            log.error(f"resonse: {response} failed to parse response")
            raise ValueError(f"resonse: {response} failed to parse response")
        rsp = json.loads(re_match.group(0))
        if "output" in rsp:
            rsp = rsp["output"]
        if len(rsp) == 0:
            return self.graph, False
        vertex_set: Set[Tuple[(str, str)]] = {
            (vertex["label"], vertex[self.schema.vertices[vertex["label"]][0]])
            for vertex in self.graph["vertices"]
        }
        if "vertices" in rsp:
            vertices = rsp["vertices"]
            m_vertices = []
            if isinstance(vertices, list):
                for vertex in vertices:
                    if isinstance(vertex, dict):
                        if "label" in vertex:
                            if vertex["label"] in self.schema.vertices:
                                m_vertex = {"label": vertex["label"]}
                            else:
                                m_vertex = {"label": "Entity"}
                            for property in self.schema.vertices[vertex["label"]]:
                                if property in vertex and isinstance(
                                    vertex[property], str
                                ):
                                    m_vertex[property] = vertex[property]
                                else:
                                    m_vertex = {}
                            if len(m_vertex) != 0:
                                m_vertices.append(m_vertex)
                                vertex_set.add(
                                    (
                                        vertex["label"],
                                        vertex[
                                            self.schema.vertices[vertex["label"]][0]
                                        ],
                                    )
                                )
            else:
                raise ValueError("vertices must be list")
        if "edges" in rsp:
            edges = rsp["edges"]
            m_edges = []
            if isinstance(edges, list):
                for edge in edges:
                    if isinstance(edge, dict):
                        if all(
                            [
                                key in edge
                                for key in [
                                    "label",
                                    "from_pk",
                                    "to_pk",
                                ]
                            ]
                        ):
                            if (
                                tuple(
                                    edge["label"],
                                )
                                in self.schema.canonical_edge_labels
                            ):
                                m_edge = {
                                    "label": edge["label"][1],
                                    "from_key": edge["label"][0],
                                    "from_pk": edge["from_pk"],
                                    "to_key": edge["label"][2],
                                    "to_pk": edge["to_pk"],
                                }
                            else:
                                m_edge = {
                                    "label": "Relation",
                                    "from_key": "Entity",
                                    "from_pk": edge["from_pk"],
                                    "to_key": "Entity",
                                    "to_pk": edge["to_pk"],
                                }

                            for property in self.schema.edges[edge["label"][1]][
                                "properties"
                            ]:
                                if property in edge and isinstance(edge[property], str):
                                    m_edge[property] = edge[property]
                                else:
                                    m_edge = {}
                            if len(m_edge) != 0:
                                if (
                                    m_edge["from_key"],
                                    m_edge["from_pk"],
                                ) not in vertex_set:
                                    m_vertices.append(
                                        {
                                            "label": m_edge["from_key"],
                                            "keyword": m_edge["from_pk"],
                                            "description": m_edge["from_pk"],
                                        }
                                    )
                                if (
                                    m_edge["to_key"],
                                    m_edge["to_pk"],
                                ) not in vertex_set:
                                    m_vertices.append(
                                        {
                                            "label": m_edge["to_key"],
                                            "keyword": m_edge["to_pk"],
                                            "description": m_edge["to_pk"],
                                        }
                                    )
                                m_edges.append(m_edge)
                        else:
                            log.warning(f"unable to build edge relationship:{edge}")
        if len(m_vertices) == 0 and len(m_edges) == 0:
            return self.graph, False
        self.graph["vertices"].extend(m_vertices)
        self.graph["edges"].extend(m_edges)
        print(f"chunk: {self.input} graph: {self.graph}")
        return self.graph, True


if __name__ == "__main__":
    from rag.utils.config import rag_config as config
    from rag.model.atlas_model import AtlasLLM

    llm = AtlasLLM(
        model="deepseek-r1-distill-qwen-32b",
        api_url=config.llm_url,
        api_base=config.llm_base,
        api_key=config.llm_key,
    )
    prompt = GraphExtractPrompt()
    query = prompt.build_prompt(
        schema=Schema(
            {
                "vertices": {
                    "Company": [
                        "keyword",
                        "description",
                        "name",
                        "industry",
                        "location",
                    ],
                    "Department": [
                        "keyword",
                        "description",
                        "name",
                        "size",
                        "focus_area",
                    ],
                    "Module": ["keyword", "description", "name", "type", "purpose"],
                    "Person": ["keyword", "description", "name", "role", "experience"],
                },
                "edges": {
                    "Relation": {
                        "relations": [
                            {"from_key": "Company", "to_key": "Department"},
                            {"from_key": "Department", "to_key": "Module"},
                        ],
                        "properties": ["keyword"],
                    },
                    "Link": {
                        "relations": [{"from_key": "Person", "to_key": "Module"}],
                        "properties": ["keyword", "role"],
                    },
                },
            }
        ),
        input="创新科技公司成立于2010年，总部位于北京，致力于云计算、大数据、人工智能等领域的技术研发。其主要部门包括AI研发部、云计算部门、产品开发部等。\
                    AI研发部包含语音识别、计算机视觉、自然语言处理等模块。云计算部门主要负责数据存储与云服务平台的建设。产品开发部则侧重于智能硬件的研发和产品设计。\
                    截至2025年1月30日，已知的公开信息中，语音识别模块的员工包括张华和李娜，计算机视觉模块有三名员工，分别是王明、刘杰和陈涛，\
                    自然语言处理模块有两名员工，分别是赵云和周婷。",
    )
    print(prompt.parse_response(llm.complete(query)))
