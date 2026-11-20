import json
import asyncio
import websockets
import requests
from uuid import uuid4
from typing import Optional


# 获取服务端配置
WS_ENDPOINT = f"ws://0.0.0.0:21032/ws/complete"  # 问答接口
SESSION_ENDPOINT = f"http://0.0.0.0:21032/api/create_session"  # 创建会话

# 配置用户和数据库信息
USERNAME = "root"
DATABASE = "project"
QUESTION = "《大数据时代》的作者有几位？"  # 问题

async def create_session() -> Optional[str]:
    """创建新的会话"""
    response = requests.post(SESSION_ENDPOINT, params={
        "username": USERNAME,
        "question": QUESTION[:50]  # 使用问题前50字符作为会话描述
    })
    
    if response.status_code != 200:
        print(f"会话创建失败: {response.text}")
        return None
    
    return response.json()['session_id']

async def query_knowledge_graph() -> None:
    # 创建会话
    session_id = await create_session()
    if not session_id:
        return
    
    print(f"会话创建成功，ID: {session_id}")
    
    # 构造RAG配置
    rag_config = {
        "session_id": session_id,
        "question": QUESTION,
        "database": DATABASE,
        "username": USERNAME,
        "index_mode": "vector",  # 可切换为"tog"、"hybrid"等模式
        # "instruct": "问题中的《大数据时代》指维克托·迈尔-舍恩伯格的著作，请基于此回答",
        # "vector_config": {"similarity_top_k": 3},
        # "kg_config": {"similarity_top_k": 3},
        # "hybrid_config": {},
        # "tog_config": {},
        # "agent_config": {}
    }
    
    # 通过WebSocket进行问答
    async with websockets.connect(WS_ENDPOINT) as websocket:
        # 发送配置
        await websocket.send(json.dumps(rag_config))
        print("问题已发送，等待回答...")
        
        # 接收并处理响应
        full_answer = ""
        data_received = False  # 标记是否已收到DATA消息
        while True:
            try:
                response = await websocket.recv()
                print("收到服务端原始响应:", response)  # 确认服务端是否真的未返回任何消息
                data = json.loads(response)
                
                # 处理code为0的参考数据消息
                if data['code'] == 0:
                    print("\n\n收到参考数据:")
                    print(f"相关文本片段: {data['data']['contents']}")
                    print(f"抽取的实体关系: {data['data']['graph']}")
                    data_received = True
                
                # 处理code为1的答案片段消息
                elif data['code'] == 1:
                    answer_chunk = data.get('answer', '')
                    # print(answer_chunk, end='', flush=True)
                    full_answer += answer_chunk
                
                # 处理错误消息
                elif data['code'] == "error":  # 实际应该不是这个标识
                    error_msg = data.get('error_msg', '未知错误')
                    print(f"\n错误: {error_msg}")
                    print(f"完整错误响应: {data}")
                    break
                
            except websockets.exceptions.ConnectionClosed as e:
                # 连接关闭时检查是否已收到数据和答案
                if data_received or full_answer:
                    print(f"\n\n连接正常关闭: 状态码={e.code}")
                    print(f"\n最终的答案是：\n{full_answer}")
                else:
                    print(f"\n\n连接异常关闭: 状态码={e.code}, 原因={e.reason}")
                break

if __name__ == "__main__":
    asyncio.run(query_knowledge_graph())