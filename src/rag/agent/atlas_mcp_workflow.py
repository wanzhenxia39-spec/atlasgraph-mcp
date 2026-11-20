from rag.mcp.client import MCPClient
from rag.prompt.mcp_prompt import MCPToolPrompt
from rag.prompt.split_query_prompt import SplitQueryPrompt
from rag.utils import rag_logger as log
from rag.common.context import Context
from rag.utils.config import rag_config as config
from rag.mcp.server import starlette_app
import asyncio
from uvicorn import Server, Config
import argparse


class AtlasMCPWorkflow:
    def __init__(
        self,
        client: MCPClient,
        split_prompt: SplitQueryPrompt,
        map_prompt: MCPToolPrompt,
        context: Context
    ):
        self.client = client
        self.context = context
        self.split_prompt = split_prompt
        self.map_prompt = map_prompt


    # 处理查询的主函数
    async def query(self, query: str) -> str:
        log.info(f"Processing query: {query}")
        
        # 初始化连接
        server_url = f"http://localhost:{config.mcp_port}/sse"
        await self.client.connect_to_server(server_url)
        
        # 列出所有可用工具
        assert self.client.session is not None, "ClientSession not initialized"
        response = await self.client.session.list_tools()
        available_tools = [
            {"name": tool.name, "description": tool.description}
            for tool in response.tools
        ]

        # 1. 问题拆解
        sub_problems = self.client.decompose_query(query, available_tools)
        log.info(f"拆解后的子问题: {sub_problems}")
        
        # 2. 处理子问题
        results = []
        for sub_q in sub_problems:
            result = await self.client.process_sub_query(sub_q, available_tools)
            results.append(result)
            
        # 3. 整合结果
        final_response = await self.generate_final_response(query, results)
        
        # 清理资源
        await self.client.cleanup()
        
        return final_response
        

    # 生成最终回答
    async def generate_final_response(self, query: str, sub_results: list) -> str:
        merge_prompt = f"""
            请根据以下子问题的处理结果，回答原始查询：{query}
            子问题结果汇总：
            {chr(10).join(sub_results)}
            要求: 
            1. 综合所有子结果，用自然语言总结
            2. 保留关键细节（如工具调用结果、数据）
        """
        return self.client.llm.complete(merge_prompt)


async def main() -> None:
     # 解析命令行参数
    parser = argparse.ArgumentParser(description="Atlas MCP Workflow")
    parser.add_argument(
        "--rag_config",
        default="/root/rag/config/config.yaml",
        help="Path to the configuration file"
    )
    args = parser.parse_args()
    config.set_rag_server_path(args.rag_config)

    # 创建依赖组件
    context = Context()
    client = MCPClient()
    split_prompt = SplitQueryPrompt()
    map_prompt = MCPToolPrompt()
    
    # 初始化
    flow = AtlasMCPWorkflow(
        client=client,
        split_prompt=split_prompt,
        map_prompt=map_prompt,
        context=context
    )
    
    # 配置UVicorn服务器
    server_config = Config(app=starlette_app, host="0.0.0.0", port=config.mcp_port) # 从配置读取
    server = Server(server_config)
    
    # 同时运行服务器和查询逻辑
    task = asyncio.create_task(server.serve())
    await asyncio.sleep(1)  # 等待服务器完全启动
    
    # 执行查询
    question = "获取project图库的schema,然后生成查询所有类型为Entity节点的cypher并执行"
    result = await flow.query(question)
    log.info(f"最终回答: {result}")
    
    # 关闭服务器
    await server.shutdown()


if __name__ == "__main__":
    asyncio.run(main())