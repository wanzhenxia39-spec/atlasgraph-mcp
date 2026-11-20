import asyncio
from typing import Optional
from contextlib import AsyncExitStack
from mcp import ClientSession
from rag.model import AtlasLLM
from mcp.client.sse import sse_client  # 导入SSE客户端
from rag.prompt.mcp_prompt import MCPToolPrompt
from rag.prompt.split_query_prompt import SplitQueryPrompt
from rag.utils import rag_logger as log
from rag.common.context import Context
from rag.utils.config import rag_config as config
import argparse

class MCPClient:
    def __init__(self) -> None:
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.llm = AtlasLLM(config.chat_model)
        self.tool_prompt = MCPToolPrompt() 
        self.split_query_prompt = SplitQueryPrompt()
        self.context_messages = Context() # session上下文
    

    # 基于sse通讯的连接函数
    async def connect_to_server(self, endpoint: str) -> None:    
        """通过SSE连接到MCP server，将连接生命周期与客户端绑定"""
        # 将SSE连接加入exit_stack，而非使用局部async with
        self.sse_streams = await self.exit_stack.enter_async_context(
            sse_client(url = endpoint) 
        )
        # 创建客户端会话，同样托管到exit_stack
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(
                self.sse_streams[0],  # 输入流（reader）
                self.sse_streams[1]   # 输出流（writer）
            )
        )
        # 初始化会话
        await self.session.initialize()
        # 列出可用工具（验证连接有效性）
        response = await self.session.list_tools()
        tools = response.tools
        log.info(f"连接到SSE服务器 {endpoint}，可用工具: {[tool.name for tool in tools]}")
    
    
    # 拆解子问题
    def decompose_query(self, query: str, available_tools: list) -> list[str]:
        """将复杂问题拆解为子问题列表"""
        try:
            prompt_str = self.split_query_prompt.build_prompt(
                available_tools=available_tools,
                query=query,
                history=self.context_messages.get_messages() # 让llm根据上下文信息拆解子问题
            )
            llm_response = self.llm.complete(prompt_str)
            sub_problems, success = self.split_query_prompt.parse_response(llm_response)
            return sub_problems if success else [query]  # 拆解失败则返回原问题
        except Exception as e:
            log.warning(f"子问题拆解失败: {e}")
            return [query] # 原问题为默认子问题
        
        
    # 处理单个子问题
    async def process_sub_query(self, sub_query: str, available_tools: list) -> str:
        """处理单个子问题，返回结果字符串"""
        try:
            content, success = self.tool_prompt.get_result(
                llm=self.llm,  
                available_tools=available_tools,  
                query=sub_query,                  
                history=self.context_messages.get_messages()  
            )
            if not success:
                return f"子问题「{sub_query}」：未找到合适工具处理"
            if not content:
                return f"子问题「{sub_query}」：未提取出工具参数"
            
            print("content is: ", content)
            # 特殊处理
            if content["name"] == "get_multi_hop_paths":
                if "subjs_with_label" in content["arguments"]:
                    content["arguments"]["subjs_with_label"] = [
                        tuple(item) if isinstance(item, list) else item
                        for item in content["arguments"]["subjs_with_label"]
                    ]

            # 把工具调用决策结果(llm第一次返回结果)加入到上下文中
            assistant_message_content = [content]
            related_query = [sub_query] 
            self.context_messages.add_message(  # 使用Context添加消息
                role="assistant", 
                content=assistant_message_content,
                query=related_query
            )
            log.info(f"子问题\"{sub_query}\"的工具选择结果是: {content}")
            
            # 工具调用
            assert self.session is not None, "ClientSession not initialized"
            tool_result = await self.session.call_tool(content["name"], content["arguments"]) # 触发POST请求，目标端点为服务器的消息接收端点（如/messages）。
            
            # 把工具调用返回结果加入到上下文中
            tool_result_message = {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result", 
                        "tool_use_id": "0", 
                        "content": [c.model_dump() for c in tool_result.content]
                    }
                ]
            }
            self.context_messages.add_message(  # 使用Context添加消息
                role = str(tool_result_message["role"]), 
                content = tool_result_message["content"]
            )
            return f"子问题「{sub_query}」的返回结果：{str(tool_result.content)}"
        
        except Exception as e:
            return f"子问题「{sub_query}」处理失败：{str(e)}"


    # 处理query
    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        # 为大模型呈现有哪些工具
        assert self.session is not None, "ClientSession not initialized"
        response = await self.session.list_tools()
        available_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            for tool in response.tools
        ]
        # 1. 拆解子问题
        sub_problems = self.decompose_query(query, available_tools)
        if len(sub_problems) > 1:
            final_text = [f"已将问题拆解为{len(sub_problems)}个子问题：{sub_problems}"]
        else:
            final_text = ["无需拆解，直接处理问题"]

        # 2. 循环处理每个子问题
        sub_results = []
        for i, sub_q in enumerate(sub_problems, 1):
            final_text.append(f"\n处理子问题 {i}/{len(sub_problems)}: {sub_q}")
            sub_result = await self.process_sub_query(sub_q, available_tools)
            sub_results.append(sub_result)
            final_text.append(sub_result)

        # 3. 整合子问题结果，生成最终回答
        merge_prompt = f"""
            请根据以下子问题的处理结果，回答原始查询：{query}
            子问题结果汇总：
            {chr(10).join(sub_results)}
            要求: 1. 综合所有子结果,用自然语言总结.
                 2. 保留关键细节（如工具调用结果、数据）。
        """
        final_response = self.llm.complete(merge_prompt)
        final_text.append(f"\n最终回答：{final_response}")
        return "\n".join(final_text)
    

    async def chat_loop(self) -> None:
        """Run an interactive chat loop"""
        log.info("MCP Client Started!")
        log.info("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == "quit":
                    break
                response = await self.process_query(query)
                log.info(f"\nFinal response is: {response}")
            except Exception as e:
                log.info(f"\nError: {str(e)}")


    async def cleanup(self) -> None:
        """Clean up resources"""
        await self.exit_stack.aclose()
        log.info("SSE connection closed!")


async def main() -> None:
     # 创建参数解析器
    parser = argparse.ArgumentParser(description="Atlas RAG Client")
    parser.add_argument(
        "--rag_config",
        default="/root/rag/config/config.yaml",
        help="Path to the configuration file"
    )
    parser.add_argument(
        "server_url",
        help="The URL of the MCP server connection"
    )
    args = parser.parse_args()
    config.set_rag_server_path(args.rag_config)

    client = MCPClient()
    try:
        await client.connect_to_server(args.server_url)
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":    
    asyncio.run(main())