from rag.prompt.base_prompt import BasePrompt
from rag.model.atlas_model import CompletionResponse
from typing import List, Dict, Any, Tuple
from rag.utils.log import rag_logger as log
import json
import re


class MCPToolPrompt(BasePrompt):
    template = """
    {{
        "instruction": 
            "你是一个工具调用决策专家，能够根据用户查询、可用工具列表和历史上下文，选择最合适的工具并生成正确的调用参数。请仔细分析以下信息：
            1. 历史上下文：包含之前的查询、工具调用及返回结果，可能包含已获取的图库schema等关键信息。
            2. 当前用户查询：需要解决的新问题。
            3. 可用工具列表：可调用的工具及参数要求。
            请注意：
            1. 必须从available_tools中选择工具，优先利用历史上下文（如已获取的schema）生成参数，避免重复调用工具（如已获取schema则无需再次调用get_schema）。
            2. 参数必须严格匹配工具的input_schema，若需要历史信息（如schema），需从history中提取并填充。 
            3. 如果没有合适的工具可以使用，请返回空字典。
            4. 请以JSON字符串格式回应，必须包含'name'（需要调用的工具名称）和'arguments（工具所需的参数）'这两个字段"。
        "examples": 
        [
            # 示例1:history为空，获取schema操作应该执行get_schema工具
            {{
                "history": [],
                "available_tools": [
                    {{'name': 'get_schema', 'description': '获取指定图库的schema信息\n 参数: database(图库名称)\n 返回: 包含顶点和边信息的字典\n', 'input_schema': {{'properties': {{'database': {{'title': 'Database', 'type': 'string'}}}}, 'required': ['database'], 'title': 'get_schemaArguments', 'type': 'object'}}}},
                    {{"name": "text2cypher", "description": "生成cypher", "input_schema": {{"properties": {{"database": {{"type": "string"}}, "query": {{"type": "string"}}, "schema": {{"type": "string"}}}}, "required": ["database", "query", "schema"]}}}}
                ],
                "query": "获取project图库的schema",
                "output": {{
                    "name": "get_schema",
                    "arguments": {{"database": "project"}}
                }}
            }},
            # 示例2: 有历史记录时, 复用schema
            {{
                "history": [
                    {{"role": "user", "content": "获取project图库的schema"}},
                    {{"role": "assistant", "content": [{{"name": "get_schema", "arguments": {{"database": "project"}}]}},
                    {{"role": "user", "content": [{{"type": "tool_result", "content": "{{\"database\": \"project\", \"vertices\": [{{\"LABEL_NAME\": \"Entity\"}}, ...], \"edges\": [...]}}"}}]}}
                ],
                "available_tools": [
                    {{'name': 'get_schema', 'description': '获取指定图库的schema信息  \n    参数: database(图库名称)\n    返回: 包含顶点和边信息的字典\n    ', 'input_schema': {{'properties': {{'database': {{'title': 'Database', 'type': 'string'}}}}, 'required': ['database'], 'title': 'get_schemaArguments', 'type': 'object'}}}},
                    {{
                        "name": "text2cypher",
                        "description": "根据自然语言查询和图库schema生成对应的Cypher语句",
                        "input_schema": {{
                            "type": "object",
                            "properties": {{
                                "database": {{
                                    "title": "Database",
                                    "type": "string"
                                }},
                                "query": {{
                                    "title": "Cypher",
                                    "type": "string"
                                }}
                                "schema": {{
                                    "title": "Schema",
                                    "type": "string"
                                }}
                            }},
                            "title": "text2cypherAguments",
                            "required": ["database", "cypher"]
                        }}
                    }}
                ],
                "query": "生成查询所有类型为Entity节点的cypher",
                "output": {{
                    "name": "get_schema",
                    "arguments": {{
                        "database": "project"
                        "query": "查询所有Entity",
                        "schema": "{{\"database\": \"project\", \"vertices\": [{{\"LABEL_NAME\": \"Entity\"}}, ...], \"edges\": [...]}}"
                    }}
                }}
            }},
            # 示例3: 获取多跳路径
            {{
                "available_tools": [
                    {{'name': 'get_schema', 'description': '获取指定图库的schema信息  \n    参数: database(图库名称)\n    返回: 包含顶点和边信息的字典\n    ', 'input_schema': {{'properties': {{'database': {{'title': 'Database', 'type': 'string'}}}}, 'required': ['database'], 'title': 'get_schemaArguments', 'type': 'object'}}}},
                    {{
                        "name": "get_multi_hop_paths",
                        "description": "获取节点的多跳路径信息\n参数:\n    database: 图库名称\n    subjs_with_label: 起点的关键词与标签列表，格式为[(keyword1, label1), (keyword2, label2), ...]\n    depth: 遍历的路径深度，默认4\n    limit: 限制结果数量，默认20\n返回:\n    包含每个起点多跳路径信息的字符串",
                        "input_schema": {{
                            "type": "object",
                            "properties": {{
                                "database": {{
                                    "title": "Database",
                                    "type": "string"
                                }},
                                "subjs_with_label": {{
                                    "title": "Subjs With Label",
                                    "type": "array",
                                    "items": {{
                                        "type": "array",
                                        "minItems": 2,
                                        "maxItems": 2,
                                        "prefixItems": [
                                            {{"type": "string"}},
                                            {{"type": "string"}}
                                        ]
                                    }}
                                }},
                                "depth": {{
                                    "title": "Depth",
                                    "type": "integer",
                                    "default": 4
                                }},
                                "limit": {{
                                    "title": "Limit",
                                    "type": "integer",
                                    "default": 20
                                }}
                            }},
                            "title": "get_multi_hop_pathsArguments",
                            "required": ["database", "subjs_with_label"]
                        }}
                    }}
                ],
                "query": "请查询数据库project中，与标签为Entity、关键词为"维克托·迈尔·舍恩伯格"这个节点相关的多跳关系路径，深度为2，最多返回10条结果",
                "output": {{
                    "name": "get_multi_hop_paths",
                    "arguments": {{
                        "database": "project",
                        "subjs_with_label": [（"\"维克托·迈尔·舍恩伯格\"", "Entity"）],
                        "depth": 2
                        "limit": 10
                    }}
                }}
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
        variables: Dict[str, str] = {}
        # 处理可用工具
        if "available_tools" in kwargs:
            if isinstance(kwargs["available_tools"], list):
                self.available_tools = kwargs["available_tools"]
                # 序列化时移除多余空格和换行，确保结构化
                variables["available_tools"] = json.dumps(
                    kwargs["available_tools"],
                    ensure_ascii=False,
                    separators=(',', ':')  # 紧凑格式，无空格
                )
            else:
                raise ValueError("available_tools must be a list")
        else:
            raise ValueError("available_tools is required")
        
        # 处理查询
        if "query" in kwargs:
            if isinstance(kwargs["query"], str):
                # 转义query中的双引号，防止破坏外层JSON
                variables["query"] = kwargs["query"].replace('"', '\\"')
            else:
                raise ValueError("query must be a string")
        else:
            raise ValueError("query is required")

        # 处理历史上下文
        if "history" in kwargs:
            if isinstance(kwargs["history"], list):
                # 序列化历史消息，转义特殊字符
                variables["history"] = json.dumps(
                    kwargs["history"],
                    ensure_ascii=False,
                    separators=(',', ':')
                ).replace('"', '\\"')  # 转义双引号，避免破坏外层JSON
            else:
                raise ValueError("history must be a list")
        else:
            # 若无历史，传入空列表
            variables["history"] = "[]"
        
        return self.template.format(**variables) # 将变量替换到模板中，返回完整提示词
    

    def parse_response(self, response: str | CompletionResponse) -> Tuple[Dict, bool]:
        # 提取JSON部分
        response_str = str(response)
        re_match = re.search(r"\{.*\}", response_str, re.DOTALL)
        
        if not re_match: # 容错1
            log.error(f"Failed to parse response: {response_str}")
            raise ValueError(f"Invalid response format: {response_str}")
            
        try:
            rsp = json.loads(re_match.group(0))        
            if not rsp:  # 空字典表示没有合适的工具 # 容错2
                return {}, False
                
            if "name" not in rsp or "arguments" not in rsp: # 容错3
                log.error(f"Response missing required fields: {rsp}")
                return {}, False
                
            # 验证工具是否在可用列表中
            tool_names = [tool["name"] for tool in self.available_tools]
            if rsp["name"] not in tool_names:
                log.error(f"Tool {rsp['name']} not in available tools")
                return {}, False
                        
            return rsp, True
            
        except json.JSONDecodeError as e:
            log.error(f"JSON decode error: {e} in response {response_str}")
            raise ValueError(f"Invalid JSON in response: {response_str}")
        except Exception as e:
            log.error(f"Error parsing response: {e}")
            raise
