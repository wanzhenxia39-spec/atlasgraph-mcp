from rag.prompt.base_prompt import BasePrompt
from rag.model.atlas_model import CompletionResponse
from typing import List, Dict, Any
from rag.schema import DEFAULT_SCHEMA
from rag.utils.log import rag_logger as log
import json
import re


# 价值百万的prompt QAQ
class ReasoningPrompt(BasePrompt):
    template = """
    {{
        "instruction": "你是一个具有深度思维能力的智能助手，能够从用户提供的文本数据与图谱数据证据以及cypher执行结果中分析出问题的回答，请你注意cypher执行的结果非常重要。
        如果可以回答的问题，你只用简洁的回答答案本身即可。如果不能回答或者只能部分回答，解释为什么不能回答。只要有部分问题没找到相应的证据就请倾向于回复No,这样后续可以通过查询语句检索更多的信息。
        请注意当没有边链接时点上的信息可能无法判断是否和问题本身是直接存在关系的。请以JSON字符串格式回应。你可以参照example进行回答。",
        "examples": [
            {{
                "text_chunks": ["海致星图有两个部门分别是图数据库和图平台"],
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
                "force_answer": false,
                "input": "图学习的员工有谁？",
                "output": {{
                    "can_answer": false,
                    "answer": "当前信息下无法推断图学习员工有谁，只知道他可能是海致星图图数据的员工。"
                }}
            }},
            {{
                "text_chunks": ["海致星图有两个部门分别是图数据库和图平台"],
                "graph": {{
                    "vertices": [
                        {{
                            "label": "Module",
                            "keyword": "图学习",
                            "description": "图学习是海致星图图数据库的一个模块",
                        }},
                        {{
                            "label": "Person",
                            "keyword": "Olivia Tian",
                            "description": "Olivia Tian是图学习模块的员工",
                        }},
                    ],
                    "edges": [
                        {{
                            "label": "Relation",
                            "keyword": "就职于",
                            "from_key": "Person",
                            "from_pk": "Olivia Tian",
                            "to_key": "Module",
                            "to_pk": "图学习",
                        }},
                    ],
                }},
                "input": "图学习的员工有谁？",
                "output": {{
                    "can_answer": true,
                    "answer": "Olivia Tian"
                }}
            }},
        ],
        "text_chunks": {text_chunks},
        "graph": {graph},
        "force_answer": {force_answer},
        "input": "{input}",
    }}
    """

    @property
    def template_variables(self) -> List[str]:
        return ["text_chunks", "graph", "input"]

    def build_prompt(self, **kwargs: Any) -> str:
        variables: Dict[str, Any] = {}
        if "text_chunks" in kwargs:
            if isinstance(kwargs["text_chunks"], list) and all(
                [isinstance(x, str) for x in kwargs["text_chunks"]]
            ):
                variables["text_chunks"] = str(kwargs["text_chunks"])
            else:
                raise ValueError("text_chunks must be a List of String")
        else:
            variables["text_chunks"] = []
        if "graph" in kwargs:
            if isinstance(kwargs["graph"], dict):
                variables["graph"] = kwargs["graph"]
            else:
                raise ValueError("graph must be a dict")
        else:
            variables["graph"] = {}
        if "force_answer" in kwargs:
            if isinstance(kwargs["force_answer"], bool):
                variables["force_answer"] = kwargs["force_answer"]
            else:
                raise ValueError("force_answer must be a bool")
        else:
            variables["force_answer"] = False
        if "input" in kwargs:
            if isinstance(kwargs["input"], str):
                variables["input"] = kwargs["input"]
            else:
                raise ValueError("input must be a string")
        else:
            raise ValueError("input is required")
        return self.template.format(**variables)

    def parse_response(self, response: str | CompletionResponse) -> Dict:
        re_match = re.search(r"\{.*\}", str(response), re.DOTALL)
        if not re_match:
            log.error(f"resonse: {response} failed to parse response")
            raise ValueError(f"resonse: {response} failed to parse response")
        rsp = json.loads(re_match.group(0))
        if "output" in rsp:
            rsp = rsp["output"]
        if isinstance(rsp, dict):
            if "can_answer" in rsp:
                if (
                    (
                        (isinstance(rsp["can_answer"], bool) and rsp["can_answer"])
                        or (
                            isinstance(rsp["can_answer"], str)
                            and rsp["can_answer"].lower() == "true"
                        )
                    )
                    and "answer" in rsp
                    and isinstance(rsp["answer"], str)
                ):
                    if rsp["answer"].lower() == "no":
                        return {
                            "can_answer": False,
                            "answer": "",
                        }
                    else:
                        return {
                            "can_answer": True,
                            "answer": rsp["answer"],
                        }
                else:
                    return {
                        "can_answer": False,
                        "answer": "",
                    }
            else:
                raise ValueError("response must has 'can_answer' field")
        else:
            raise ValueError("response must be a dict")


if __name__ == "__main__":
    from rag.utils.config import rag_config as config
    from rag.model.atlas_model import AtlasLLM

    llm = AtlasLLM(
        model=config.llm_model,
        api_url=config.llm_url,
        api_base=config.llm_base,
        api_key=config.llm_key,
    )
    prompt = ReasoningPrompt()
    query = prompt.build_prompt(
        text_chunks=["云智科技的自然语言处理模块有周明（首席算法工程师）"],
        graph={
            "vertices": [
                {
                    "label": "Module",
                    "keyword": "自然语言处理",
                    "description": "自然语言处理是云智科技的一个模块",
                },
                {
                    "label": "Person",
                    "keyword": "Emma Wang",
                    "description": "Emma Wang（语义分析专家）是自然语言处理模块的负责人",
                },
            ],
            "edges": [
                {
                    "label": "Link",
                    "keyword": "负责人",
                    "from_key": "Module",
                    "from_pk": "自然语言处理",
                    "to_key": "Person",
                    "to_pk": "Emma Wang",
                },
            ],
        },
        force_answer=True,
        input="we科技的自然语言处理模块有哪些人？",
    )
    print(query)
    print(prompt.parse_response(llm.complete(query)))
