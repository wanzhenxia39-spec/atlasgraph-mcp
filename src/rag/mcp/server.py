from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport 
from driver3.service.AtlasGraph import AtlasGraph
from driver3.service.ConnectionConfiguration import ConnectionConfiguration
import argparse
from starlette.applications import Starlette
from starlette.routing import Mount, Route
import uvicorn
from starlette.requests import Request
from rag.utils.utils import to_legal_cypher_string
from typing import List, Tuple
from rag.schema.graph_schema import GraphSchema
from rag.model import AtlasLLM
from rag.prompt.text2cypher_prompt import Text2CypherPrompt
from rag.prompt.text2schema_prompt import Text2SchemaPrompt
from rag.prompt.question_transform_prompt import QuestionTransformPrompt
from rag.prompt.fix_cypher_prompt import FixCypherPrompt
from rag.utils import rag_logger as log
from rag.utils.config import rag_config as config
from rag.schema.graph_schema import (
    GraphSchema,
    GraphSchemaModel,
    VertexModel,
    EdgeModel,
    PropertyModel,
    RelationModel,
)

# Initialize FastMCP server
mcp = FastMCP("rag")
sse_transport = SseServerTransport("/messages")


# 1.get_schema
@mcp.tool()
async def get_schema(database: str) -> dict:
    """获取指定图库的schema信息  
    参数: database(图库名称)
    返回: 包含顶点和边信息的字典
    """
    try:
        driver = AtlasGraph(
            ConnectionConfiguration(config.graph_url, config.username, config.password)
        )
        # 获取顶点信息
        show_vertex = driver.use(database).exec(cypher="SHOW VERTEX;")
        vertex_header = show_vertex.result_set["header"]
        vertices_data = [dict(zip(vertex_header, table)) for table in show_vertex.result_set["table"]]
        
        # 获取边信息
        show_edge = driver.use(database).exec(cypher="SHOW EDGE;")
        edge_header = show_edge.result_set["header"]
        edges_data = [dict(zip(edge_header, table)) for table in show_edge.result_set["table"]]
        
        # 转换顶点数据为VertexModel列表
        vertex_models = []
        # 按顶点标签分组
        vertex_groups = {}
        for v in vertices_data:
            label = v["LABEL_NAME"]
            if label not in vertex_groups:
                vertex_groups[label] = []
            vertex_groups[label].append(v)
        
        # 构建VertexModel
        for label, props in vertex_groups.items():
            properties = []
            for prop in props:
                properties.append(PropertyModel(
                    property=prop["PROPERTY_NAME"],
                    explain="",
                    type=prop["TYPE"]
                ))
            vertex_models.append(VertexModel(
                label=label,
                explain="",
                properties=properties
            ))
        
        # 转换边数据为EdgeModel列表
        edge_models = []
        # 按边标签分组
        edge_groups = {}
        for e in edges_data:
            label = e["LABEL_NAME"]
            if label not in edge_groups:
                edge_groups[label] = []
            edge_groups[label].append(e)
        
        # 构建EdgeModel
        for label, edges in edge_groups.items():
            relations = []
            # 解析关系字符串 (格式: "FROM <from_label> TO <to_label>")
            for rel_str in edges[0]["RELATIONS"]:
                parts = rel_str.split()
                if len(parts) >= 4 and parts[0] == "FROM" and parts[2] == "TO":
                    relations.append(RelationModel(
                        from_key=parts[1],
                        to_key=parts[3]
                    ))
            properties = []
            for edge in edges:
                # 只添加有属性的边
                if edge["PROPERTY_NAME"]:
                    properties.append(PropertyModel(
                        property=edge["PROPERTY_NAME"],
                        explain="",
                        type=edge["TYPE"]
                    ))
            edge_models.append(EdgeModel(
                label=label,
                explain="",
                relations=relations,
                properties=properties
            ))
        
        # 构建完整的图模型并返回字典
        graph_schema_model = GraphSchemaModel(
            vertices=vertex_models,
            edges=edge_models
        )
        return graph_schema_model.model_dump()
    
    except Exception as e:
        # 错误时返回空结构
        return GraphSchemaModel().model_dump()
    

# 2.根据query生成cypher
@mcp.tool()
async def text2cypher(query: str, database: str, schema: dict) -> str:
    """根据自然语言查询和图库schema生成对应的Cypher语句
    参数:
        query: 自然语言查询文本
        database: 目标图库名称
        schema: 图库的schema信息(GraphSchema类型对象)
    返回:
        生成的合法Cypher语句或错误信息
    """
    try:
        # 将传入的schema字典转换为GraphSchema对象
        graph_schema = GraphSchema(GraphSchemaModel(**schema))

        # 初始化LLM
        llm = AtlasLLM(config.llm_model)

        # 问题重写（可选）:优化自然语言查询
        rewrite_question = True  # 可配置
        if rewrite_question:
            question_prompt = QuestionTransformPrompt() 
            try:
                query = question_prompt.get_result(llm, input=query)
                log.info(f"重写后的查询: {query}")
            except Exception as e:
                log.warning(f"问题重写失败，使用原始查询: {str(e)}")

        # 生成子schema（可选）: 从完整schema中提取与当前查询相关的部分
        gen_sub_schema = True  # 可配置
        if gen_sub_schema:
            schema_prompt = Text2SchemaPrompt()
            try:
                current_schema = schema_prompt.get_result(
                    llm, input=query, schema=graph_schema
                )
                log.info(f"生成子schema: {current_schema.model_dump()}")
            except Exception as e:
                log.warning(f"子schema生成失败，使用完整schema: {str(e)}")
                current_schema = graph_schema
        else:
            current_schema = graph_schema

        # 生成初始Cypher
        cypher_prompt = Text2CypherPrompt()
        try:
            cypher = cypher_prompt.get_result(
                llm, input=query, schema=current_schema
            )
            log.info(f"初始生成Cypher: {cypher}")
        except Exception as e:
            return f"Cypher生成失败: {str(e)}"

        # 验证并修复Cypher
        fix_num = 3  # 最大修复次数
        fix_prompt = FixCypherPrompt()
        errors = []
        driver = AtlasGraph(
            ConnectionConfiguration(config.graph_url, config.username, config.password)
        )

        for _ in range(fix_num):
            try:
                # 执行EXPLAIN验证Cypher合法性
                driver.use(database).exec(f"EXPLAIN {cypher}")
                return f"成功生成Cypher:\n{cypher}"
            except Exception as e:
                error_msg = str(e)
                errors.append(error_msg)
                log.warning(f"Cypher验证失败: {error_msg}")
                
                # 修复Cypher
                try:
                    cypher = fix_prompt.get_result(
                        llm, question=query, errors=errors, cypher=cypher, schema=current_schema
                    )
                    log.info(f"修复后Cypher: {cypher}")
                except Exception as fix_err:
                    log.warning(f"Cypher修复失败: {str(fix_err)}")
                    continue

        # 达到最大修复次数仍失败
        return f"经过{fix_num}次修复仍无法生成合法Cypher。错误信息:\n{'; '.join(errors)}\n最后生成的Cypher:\n{cypher}"

    except Exception as e:
        return f"工具执行失败: {str(e)}"


# 3.执行cypher语句
@mcp.tool()
async def execute_cypher(database: str, cypher: str) -> str:
    """执行cypher查询并以表格的方式返回  
    参数: 
        database: 图库名称
        cypher: Cypher查询语句
    返回: 
        cypher的执行结果，包含表头和数据
    """
    try:
        # 初始化数据库连接
        driver = AtlasGraph(
            ConnectionConfiguration(config.graph_url, config.username, config.password)
        )
        
        # 在指定图库中执行Cypher查询
        result = driver.use(database).exec(cypher=cypher)
                
        # 处理查询结果
        if "header" in result.result_set and "table" in result.result_set: # 确定是这个字段吗
            header = result.result_set["header"]
            table_data = result.result_set["table"]
            
            # 将结果转换为字典列表
            formatted_results = [dict(zip(header, row)) for row in table_data]
            
            # 格式化输出结果
            if formatted_results:
                result_str = f"对于cypher:{cypher}, 查询结果为: \n"
                for i, item in enumerate(formatted_results, 1):
                    result_str += f"记录 {i}:\n"
                    for key, value in item.items():
                        result_str += f"  {key}: {value}\n"
                return result_str
            else:
                return "查询成功，但未返回任何结果。"
        else:
            return f"查询执行成功，但结果格式不符合预期: {str(result.result_set)}"
    
    except Exception as e:
        return f"执行Cypher失败: {str(e)}"
    

# 4.获取节点多跳关系路径
@mcp.tool()
async def get_multi_hop_paths(
    database: str,
    subjs_with_label: List[Tuple[str, str]], # [(keyword1, label1), (keyword2, label2), ...]
    depth: int = 4,
    limit: int = 20
) -> str:
    """获取节点的多跳路径信息
    参数:
        database: 图库名称
        subjs_with_label: 起点的关键词与标签列表，格式为[(keyword1, label1), (keyword2, label2), ...](请注意此参数是一个由元组构成的列表)
        depth: 遍历的路径深度，默认4
        limit: 限制结果数量，默认20
    返回: 
        包含每个起点多跳路径信息的字符串
    """
    try:
        # 初始化数据库连接
        driver = AtlasGraph(
            ConnectionConfiguration(config.graph_url, config.username, config.password)
        )
        if not subjs_with_label:
            return "起点信息列表不能为空"
        
        # 生成起始节点匹配语句(eg: 若输入为[("meixi", "Player")]，则生成片段：match(n1: 'Player'{keyword:'meixi'}) return n1，用于查找label为Player且keyword属性值为meixi的节点，并返回它的全部信息。
        sub_match = []
        for subj, label in subjs_with_label:
            query = f'match(n1: {label}{{keyword:"{to_legal_cypher_string(subj)}"}}) return n1'
            sub_match.append(query)
        # 构建Cypher查询
        cypher_query = (
            f"call{{{"union ".join(sub_match)}}} " # 匹配所有起点节点,n1
            f"with n1 "
            f"MATCH p=(n1)-[r*1..{depth}]->(n2) " # 匹配多跳路径
            f'WHERE all(x in relationships(p) WHERE type(x) <> "Extract") ' # 过滤路径条件
            # f"AND n1.keyword IN {[subj for subj, _ in subjs_with_label]} "
            f"UNWIND relationships(p) AS rel " # 将路径p中的关系列表拆分为单个关系处理
            # 处理路径数据并格式化
            f"WITH n1.keyword AS subj, p, apoc.coll.flatten(apoc.coll.toSet("
            f"collect([rel.keyword, endNode(rel).keyword]))) AS flattened_rels "
            f"RETURN subj, collect(flattened_rels) AS flattened_rels LIMIT {limit}" # 返回结果（按subj分组汇总）
        )
        log.info(f"\ncypher_query is: {cypher_query}")
        # 执行查询
        result = driver.use(database).exec(cypher=cypher_query)
        # 处理查询结果
        if "header" in result.result_set and "table" in result.result_set:
            rel_map = {}
            for record in result.result_set["table"]:
                rel_map[record[0]] = record[1]
            # 格式化输出
            if rel_map:
                result_str = "多跳路径查询结果:\n"
                for subj, paths in rel_map.items():
                    result_str += f"起点 '{subj}' 的路径:\n"
                    for i, path in enumerate(paths, 1):
                        result_str += f"  路径 {i}: {path}\n"
                return result_str
            else:
                return "查询成功，但未找到任何多跳路径。"
        else:
            return f"查询执行成功，但结果格式不符合预期: {str(result.result_set)}"
    
    except Exception as e:
        return f"获取多跳路径失败: {str(e)}"
            

async def sse_handler(request:Request) -> None:
    async with sse_transport.connect_sse(request.scope, request._receive, request._send) as streams:
        await mcp._mcp_server.run(streams[0], streams[1], mcp._mcp_server.create_initialization_options()) # keep sse session


# 定义处理客户端向服务器发送的消息
starlette_app = Starlette(
    debug = True,
    routes=[
        Route('/sse', endpoint=sse_handler), # 端口1:处理client端第一次发来的get连接请求
        Mount('/messages', app=sse_transport.handle_post_message) # 端口2:处理后续client端post请求
    ]
)

# 启动服务
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="atlas rag")
    parser.add_argument(
        "--rag_config",
        default="/root/rag/config/config.yaml",
        type=str,
    )
    args = parser.parse_args()
    config.set_rag_server_path(args.rag_config)

    # 用uvicorn运行Starlette应用，启动SSE服务器，绑定到所有网络接口（0.0.0.0）的8000端口
    uvicorn.run(starlette_app, host="0.0.0.0", port=8000)