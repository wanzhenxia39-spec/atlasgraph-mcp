from rag.prompt.base_prompt import BasePrompt
from rag.model.atlas_model import CompletionResponse
from typing import List, Dict, Any
from rag.schema import DEFAULT_SCHEMA
from rag.utils.log import rag_logger as log
import json
import re


# 价值百万的prompt QAQ
class EntityExtractPrompt(BasePrompt):
    template = """
    {{
        "instruction": "你是命名实体识别的专家。请从输入中提取与模式定义匹配的实体，并且根据提供的实体类型列表对其进行分类，用户提供的文本中可能存在无法按照entity_types提取到全部实体类型的情况存在，你不需要强行提取类似未知、未命名的实体名。提取出的实体名必须是在原始文本中出现过的
        如果不存在该类型的实体，可以将其归为Other类型。注意不能超过用户指定的最大的解析的实体数，如果存在多余max_entity_num的实体数时请根据你的经验提取返回更为重要的前max_entity_num个实体。请以JSON列表字符串格式回应,不用返回任何解析信息。你可以参照example进行抽取。",
        "examples": [
            {{
                "entity_types": ["Company", "Department", "Module", "Person"],
                "max_entity_num": 15,
                "input": "海致星图成立于2015年，致力于通过自主可控的大数据、图计算、知识图谱等核心技术，服务于金融、能源、制造等政府机构与企业级用户的数字化转型，帮助用户构建数据驱动的智能管理运营体系，为用户的各项决策提供顶层视角。其主要部门包括图平台、图数据库等。\
                              其中图数据库部门是其主要部门，其中包括图查询、图计算、图学习等模块，截止于2025年1月30日，目前已知的公开的信息中，图查询模块有两名员工分别是李晨和张伟，图计算包含一名员工是David Chen\
                              图学习包含一名员工是Olivia Tian。",
                "output": [
                    {{
                        "entity_type": "Company",
                        "entity_name": "海致星图"
                    }},
                    {{
                        "entity_type": "Other",
                        "entity_name": "2025年1月30日"
                    }},
                    {{
                        "entity_type": "Department",
                        "entity_name": "图平台"
                    }},
                    {{
                        "entity_type": "Department",
                        "entity_name": "图数据库"
                    }},
                    {{
                        "entity_type": "Module",
                        "entity_name": "图查询"
                    }},
                    {{
                        "entity_type": "Module",
                        "entity_name": "图计算"
                    }},
                    {{
                        "entity_type": "Module",
                        "entity_name": "图学习"
                    }},
                    {{
                        "entity_type": "Person",
                        "entity_name": "李晨"
                    }},
                    {{
                        "entity_type": "Person",
                        "entity_name": "张伟"
                    }},
                    {{
                        "entity_type": "Person",
                        "entity_name": "David Chen"
                    }},
                    {{
                        "entity_type": "Person",
                        "entity_name": "Olivia Tian"
                    }}
                ],
            }}
        ],
        "entity_types": {entity_types},
        "max_entity_num": {max_entity_num}
        "input": "{input}",
    }}
    """

    @property
    def template_variables(self) -> List[str]:
        return ["entity_types", "max_entity_num", "input"]

    def build_prompt(self, **kwargs: Any) -> str:
        variables: Dict[str, str] = {}
        if "entity_types" in kwargs:
            if isinstance(kwargs["entity_types"], list) and all(
                [isinstance(x, str) for x in kwargs["entity_types"]]
            ):
                self.entity_types = kwargs["entity_types"]
                variables["entity_types"] = str(kwargs["entity_types"])
            else:
                raise ValueError("entity_types must be a List of String")
        else:
            self.entity_types = list(DEFAULT_SCHEMA.vertices.keys())
            variables["entity_types"] = str(self.entity_types)
        if "max_entity_num" in kwargs:
            if isinstance(kwargs["max_entity_num"], int):
                variables["max_entity_num"] = str(kwargs["max_entity_num"])
            else:
                raise ValueError("max_entity_num must be a int")
        else:
            variables["max_entity_num"] = "5"
        if "input" in kwargs:
            if isinstance(kwargs["input"], str):
                variables["input"] = kwargs["input"]
            else:
                raise ValueError("input must be a string")
        else:
            raise ValueError("input is required")
        return self.template.format(**variables)

    def parse_response(self, response: str | CompletionResponse) -> List:
        re_match = re.search(r"\[.*\]", str(response), re.DOTALL)
        if not re_match:
            log.error(f"resonse: {response} failed to parse response")
            raise ValueError(f"resonse: {response} failed to parse response")
        rsp = json.loads(re_match.group(0))
        if "output" in rsp:
            rsp = rsp["output"]
        if isinstance(rsp, dict):
            rsp = [rsp]
        if isinstance(rsp, list):
            entity_list = []
            for entity in rsp:
                if (
                    "entity_type" in entity
                    and "entity_name" in entity
                    and (
                        entity["entity_type"] in self.entity_types
                        or entity["entity_type"] == "Other"
                    )
                ):
                    entity_list.append(
                        {
                            "entity_type": entity["entity_type"],
                            "entity_name": entity["entity_name"],
                        }
                    )
            return entity_list
        else:
            raise ValueError("response must be a list")


if __name__ == "__main__":
    from rag.utils.config import rag_config as config
    from rag.model.atlas_model import AtlasLLM

    llm = AtlasLLM(
        model=config.llm_model,
        api_url=config.llm_url,
        api_base=config.llm_base,
        api_key=config.llm_key,
    )
    prompt = EntityExtractPrompt()
    query = prompt.build_prompt(
        entity_types=["Company", "Department", "Module", "Person"],
        max_entity_num=5,
        input="创新科技公司成立于2010年，总部位于北京，致力于云计算、大数据、人工智能等领域的技术研发。其主要部门包括AI研发部、云计算部门、产品开发部等。\
                    AI研发部包含语音识别、计算机视觉、自然语言处理等模块。云计算部门主要负责数据存储与云服务平台的建设。产品开发部则侧重于智能硬件的研发和产品设计。\
                    截至2025年1月30日，已知的公开信息中，语音识别模块的员工包括张华和李娜，计算机视觉模块有三名员工，分别是王明、刘杰和陈涛，\
                    自然语言处理模块有两名员工，分别是赵云和周婷。",
    )
    print(prompt.parse_response(llm.complete(query)))
