# AtlasGraph MCP server and client

这是 GraphRAG 项目的个人实习开发归档，重点展示 AtlasGraph 图数据库的 MCP 服务端、客户端和连续查询流程。GraphRAG 基础代码来自原项目团队；本仓库不是海致的官方产品发布，也不表示底层 GraphRAG 或 Text2Cypher 能力由仓库维护者独立开发。

## 已有能力

- 使用 FastMCP 注册 4 类工具：`get_schema`、`text2cypher`、`execute_cypher`、`get_multi_hop_paths`。
- 通过 SSE 建立 MCP 客户端会话，发现工具、发起调用并读取结果。
- 结合模型输出完成工具选择、参数解析、子问题拆解与结果汇总。
- 使用 `Context` 维护会话内工具调用历史。
- 在 FastAPI 主服务中提供 `/sse` 与 `/messages` 通信入口。
- 保留原项目的文档处理、存储适配和多种 RAG 查询模式。

## 代码位置

| 位置 | 作用 |
| --- | --- |
| `src/rag/mcp/server.py` | MCP 工具与 SSE 服务 |
| `src/rag/mcp/client.py` | 连接、工具发现、调用和查询流程 |
| `src/rag/agent/atlas_mcp_workflow.py` | 单次请求驱动的多步骤查询示例 |
| `src/rag/prompt/mcp_prompt.py` | 工具选择与参数生成提示词 |
| `src/rag/prompt/split_query_prompt.py` | 子问题拆解提示词 |
| `src/rag/common/context.py` | 上下文和工具调用历史 |
| `src/rag/service/service.py` | FastAPI 主服务及 MCP 通道挂载 |
| `src/rag/test/` | 上传、问答和建库联调示例 |

## 运行前置条件

该代码归档依赖外部服务，不是独立数据库发行包。运行前需要自行准备：

1. 与 `driver3` 接口兼容的 AtlasGraph 实例和 Python SDK。
2. 与代码请求格式兼容的大模型、Embedding 服务；图片处理还需多模态服务。
3. 项目所需 Python 依赖。源码包含 Python 3.12 的 f-string 写法；历史依赖清单位于 `docker/tmpfiles_root/requirements*.txt`，MCP SDK 需要另外安装并锁定兼容版本。

AtlasGraph SDK wheel、字体和公司内部基础镜像不随本仓库分发。Docker 文件保留了原有构建方式，需要自行准备兼容基础镜像以及 SDK，并传入 `BASE_IMAGE` 构建参数。旧版依赖和示例仍需在实际运行环境验证。

## 配置与启动

```sh
cp config/config.example.yaml config/config.yaml
# 编辑 config/config.yaml，填写自己的数据库与模型服务配置。
export PYTHONPATH="$PWD/src"
python src/rag/service/service.py --rag_config config/config.yaml
```

主服务启动后，可以使用 SSE 客户端连接：

```sh
python src/rag/mcp/client.py --rag_config config/config.yaml http://127.0.0.1:21032/sse
```

示例问题：获取指定图库的 Schema，然后生成查询 Entity 节点的 Cypher 并执行。

真实服务配置应只保存到被 Git 忽略的 `config/config.yaml`。旧的实体对齐索引若调用 ZhipuAI，需要在本机另行设置 `ZHIPUAI_API_KEY`。

## 归档说明

此公开版本从原工作目录生成新的源码快照，未携带旧 Git 历史、内部 CI 配置、固定凭据、内部服务地址、个人周报、运行日志、缓存、SDK wheel 和字体。连接地址已改成本机示例，数据库默认密码已清空。核心 Python 代码没有在本次归档中进行功能重构。

本次归档只验证代码语法和发布文件；未连接真实 AtlasGraph 或模型服务，也未宣称全量测试、类型检查或生产验证已经通过。现有 MCP 工作流为串行示例，Resources、持久化执行、审批和恢复机制均不在当前实现范围内。历史代码中的多模态配置键判断等兼容问题尚未在本次归档中修复，启用相关能力前应单独验证。
