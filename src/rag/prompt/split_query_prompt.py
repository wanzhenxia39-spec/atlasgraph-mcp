from rag.prompt.base_prompt import BasePrompt
from typing import List, Dict, Any, Tuple
import json
import re
from rag.utils.log import rag_logger as log
from rag.prompt.base_prompt import CompletionResponse  # 确保导入

class SplitQueryPrompt(BasePrompt):
    template = """
    {{
        "instruction": "你是一个问题拆解专家，需将用户的复杂查询拆解为多个子问题。要求：
            1. 每个子问题必须能通过调用available_tools中的工具解决(参考工具的description判断能力)。
            2. 子问题需明确、具体，避免模糊表述（例如需包含明确的图库名称、节点标签等信息）。
            3. 子问题按解决顺序排列。
            4. 若问题简单无需拆解, 直接返回包含原问题的单元素列表。
            请以JSON数组格式回应, 元素为子问题字符串。",
        "examples": [
            {{
                "available_tools": [
                    {{"name": "get_schema", "description": "获取指定图库的schema"}},
                    {{"name": "text2cypher", "description": "生成cypher语句"}},
                    {{"name": "execute_cypher", "description": "执行cypher语句"}}
                ],
                "query": "先获取project图库的schema，再生成查询所有Entity节点的cypher并执行",
                "output": [
                    "获取project图库的schema",
                    "根据project图库的schema，生成查询所有Entity节点的cypher",
                    "在project图库中执行生成的cypher语句"
                ]
            }},
            {{
                "available_tools": [{{"name": "get_multi_hop_paths", "description": "获取节点多跳路径"}}],
                "query": "查询project图库中标签为Entity、关键词为'张三'的节点的2跳路径，最多返回5条",
                "output": ["查询project图库中标签为Entity、关键词为'张三'的节点的2跳路径，最多返回5条"]
            }}
        ],
        "available_tools": {available_tools},
        "query": "{query}",
        "history": "{history}"
    }}
    """

    @property
    def template_variables(self) -> List[str]:
        return ["available_tools", "query", "history"]

    def build_prompt(self, **kwargs: Any) -> str:
        # 复用MCPToolPrompt的变量处理逻辑（转义、序列化）
        variables: Dict[str, str] = {}
        if "available_tools" in kwargs:
            variables["available_tools"] = json.dumps(
                kwargs["available_tools"], ensure_ascii=False, separators=(',', ':')
            )
        else:
            raise ValueError("available_tools is required")
        if "query" in kwargs:
            variables["query"] = kwargs["query"].replace('"', '\\"')
        else:
            raise ValueError("query is required")
        variables["history"] = json.dumps(
            kwargs.get("history", []), ensure_ascii=False, separators=(',', ':')
        ).replace('"', '\\"')
        return self.template.format(** variables)

    def parse_response(self, response: str | CompletionResponse) -> Tuple[List[str], bool]:
        # 解析大模型返回的子问题列表
        response_str = str(response)
        re_match = re.search(r"\[.*\]", response_str, re.DOTALL)  # 匹配JSON数组
        if not re_match:
            log.error(f"子问题解析失败: {response_str}")
            return [], False
        try:
            sub_problems = json.loads(re_match.group(0))
            if not isinstance(sub_problems, list) or not all(isinstance(p, str) for p in sub_problems):
                return [], False
            return sub_problems, True
        except json.JSONDecodeError:
            log.error(f"子问题JSON格式错误: {response_str}")
            return [], False