from llama_index.core import (
    VectorStoreIndex,
    KnowledgeGraphIndex,
    get_response_synthesizer,
    Settings,
)
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.schema import NodeWithScore
from rag.store.vector_store import AtlasVectorStore
from rag.store.graph_store import AtlasGraphStore
from rag.store.document_store import AtlasDocumentStore
from rag.store.index_store import AtlasIndexDict, AtlasKGWrapper, AtlasIndexStore
from rag.retriever.hybrid_retriever import HybridRetriever
from rag.retriever.tog_retriever import ToGRetriever
from fastapi import FastAPI, UploadFile, status, WebSocket
from fastapi.responses import JSONResponse, Response
from fastapi.exceptions import HTTPException, WebSocketException
from rag.prompt import SimplifyQuestionPrompt
import os
import uuid
import tarfile
import zipfile
import json
from rag.utils.constant import (
    UploadFileType,
    IndexMode,
    UploadConfig,
    RAGConfig,
    RAGResponse,
    RAGCode,
    Tool,
)
from rag.utils.config import rag_config as config
from rag.utils.log import rag_logger as log
from rag.index.atlas_rag_index import AtlasRAGIndex
from rag.model.atlas_model import AtlasLLM, AtlasEmbeddings
from rag.utils.utils import stream_query
from typing import cast, Optional, List, Optional
import rag
import shutil
import traceback
from rag.service.global_database import global_database
from rag.agent.atlas_rag_workflow import AtlasRAGWorkflow
from rag.splitter import count_tokens
import logging
import uvicorn
import os
from starlette.requests import Request
from rag.mcp.server import mcp, sse_transport


app = FastAPI(debug=True)


@app.get("/api/version")
def version() -> JSONResponse:
    return JSONResponse(content={"version": rag.get_version()})


@app.get("/api/list_task")
def list_task_ids(
    username: str,
    database: Optional[str] = None,
    task_status: Optional[str] = None,
    limit: int = 30,
) -> Response:
    try:
        return JSONResponse(
            content=global_database.list_task(username, database, task_status, limit)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@app.post("/api/delete_task/{task_id}")
def delete_task(task_id: str) -> Response:
    try:
        global_database.delete_task(task_id)
        return Response(content="OK")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@app.post("/api/cancel_task/{task_id}")
def cancel_task(task_id: str) -> Response:
    try:
        global_database.cancel_task(task_id)
        return Response(content="OK")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@app.post("/api/create_session")
def create_session(username: str, question: str) -> JSONResponse:
    try:
        return JSONResponse(content=global_database.create_session(username, question))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@app.post("/api/delete_session/{session_id}")
def delete_session(session_id: str) -> Response:
    try:
        global_database.delete_session(session_id)
        return Response(content="OK")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@app.get("/api/list_session")
def list_session(username: str, limit: int = 30) -> JSONResponse:
    try:
        return JSONResponse(content=global_database.list_session(username, limit))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@app.get("/api/get_session_description/{session_id}")
def get_session_description(session_id: str) -> Response:
    try:
        return Response(content=global_database.get_session_description(session_id))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@app.websocket("/ws/get_session/{session_id}")
async def get_session(websocket: WebSocket, session_id: str) -> None:
    try:
        await websocket.accept()
        for conversation in global_database.get_session(session_id):
            await websocket.send_json(conversation)
        await websocket.close()
    except Exception as e:
        log.error(traceback.format_exc())
        raise WebSocketException(code=status.WS_1011_INTERNAL_ERROR, reason=str(e))


@app.websocket("/ws/complete")
async def complete(websocket: WebSocket) -> None:
    try:
        await websocket.accept()
        rag_config_json = await websocket.receive_json()
        log.info(f"receive rag_config_json: {rag_config_json}")
        rag_config = RAGConfig(**rag_config_json)
        if not global_database.is_valid_session_id(rag_config.session_id):
            raise ValueError(f"session_id: {rag_config.session_id} does not exist.")
        tokens_len = count_tokens(rag_config.question)
        if tokens_len > config.llm_max_context * 0.9:
            raise ValueError(
                f"question must have less than max_llm_context: {config.llm_max_context} tokens"
            )
        log.info("QA requested")
        log.info(f"question: {rag_config.question}")
        try:
            llm = AtlasLLM(
                model=config.chat_model,
                api_url=config.chat_url,
                api_base=config.chat_base,
                api_key=config.chat_key,
            )
            embeddings = AtlasEmbeddings(
                model=config.embedding_model,
                api_url=config.embedding_url,
                api_base=config.embedding_base,
                api_key=config.embedding_key,
            )

        except Exception as e:
            log.error("Failed loading LLM: {e}")
            raise WebSocketException(
                code=status.WS_1003_UNSUPPORTED_DATA, reason=str(e)
            )
        if tokens_len > config.embedding_max_context * 0.9:
            prompt = SimplifyQuestionPrompt()
            rag_config.question = prompt.parse_response(
                llm.complete(
                    prompt.build_prompt(
                        max_tokens=config.embedding_max_context,
                        input=rag_config.question,
                    )
                )
            )
        try:
            Settings.llm = llm
            Settings.embed_model = embeddings
        except Exception as e:
            log.error("Failed initializing service settings")
            log.error(str(e))
            raise WebSocketException(
                code=status.WS_1003_UNSUPPORTED_DATA, reason=str(e)
            )
        total_answer = ""
        try:
            index_store = AtlasIndexStore()
            log.info("index_store initialized")
            vector_store = AtlasVectorStore(
                username=config.username,
                password=config.password,
                url=config.graph_url,
                embedding_dimension=config.embedding_dimension,
                database=rag_config.database,
            )
            log.info("vector_store initialized")
            document_store = AtlasDocumentStore(
                username=config.username,
                password=config.password,
                url=config.graph_url,
                database=rag_config.database,
            )
            log.info("document_store initialized")
            graph_store = AtlasGraphStore(
                username=config.username,
                password=config.password,
                url=config.graph_url,
                database=rag_config.database,
                schema=None,
            )
            log.info("graph_store initialized")
            storage_context = StorageContext.from_defaults(
                docstore=document_store,
                index_store=index_store,
                vector_store=vector_store,  # type: ignore[arg-type]
                graph_store=graph_store,
            )
            log.info("storage_context initialized")
        except Exception as e:
            log.error(f"Failed loading stores: {e}")
            raise WebSocketException(
                code=status.WS_1003_UNSUPPORTED_DATA, reason=str(e)
            )
        nodes = []
        if rag_config.index_mode == IndexMode.VECTOR:
            log.info("Using vector mode")
            VectorStoreIndex.index_struct_cls = AtlasIndexDict  # type: ignore[assignment]
            vector_index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
            )
            stream_query_gen = stream_query(
                vector_index.as_query_engine(
                    similarity_top_k=rag_config.vector_config.similarity_top_k,
                    streaming=True,
                ),
                question=rag_config.question,
                instruct=rag_config.instruct,
                raw=True,
                yield_node=True,
            )
            nodes, graph = next(stream_query_gen)
            data = {
                "contents": [
                    cast(NodeWithScore, node).node.get_content() for node in nodes
                ],
                "graph": graph,
            }

        elif rag_config.index_mode == IndexMode.KG:
            log.info("Using kg mode")
            KnowledgeGraphIndex.index_struct_cls = AtlasKGWrapper(
                graph_store=graph_store
            )  # type: ignore[assignment]
            kg_index = KnowledgeGraphIndex(nodes, storage_context=storage_context)
            document_store.add_keywords(kg_index.index_struct)  # type: ignore[arg-type]
            stream_query_gen = stream_query(
                kg_index.as_query_engine(
                    similarity_top_k=rag_config.kg_config.similarity_top_k,
                    streaming=True,
                ),
                question=rag_config.question,
                instruct=rag_config.instruct,
                raw=True,
                yield_node=True,
            )
            nodes, graph = next(stream_query_gen)
            data = {
                "contents": [
                    cast(NodeWithScore, node).node.get_content() for node in nodes
                ],
                "graph": graph,
            }

        elif rag_config.index_mode == IndexMode.TOG:
            log.info("Using tog mode")

            VectorStoreIndex.index_struct_cls = AtlasIndexDict  # type: ignore[assignment]
            vector_index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
            )
            KnowledgeGraphIndex.index_struct_cls = AtlasKGWrapper(
                graph_store=graph_store
            )  # type: ignore[assignment]
            kg_index = KnowledgeGraphIndex(nodes, storage_context=storage_context)

            tog_retriever = ToGRetriever(
                vector_index=vector_index,
                kg_index=kg_index,
                temperature_exploration=rag_config.tog_config.temperature_exploration,
                temperature_reasoning=rag_config.tog_config.temperature_reasoning,
                depth=rag_config.tog_config.depth,
                max_width=rag_config.tog_config.max_width,
                min_score=rag_config.tog_config.min_score,
                vector_prune=rag_config.tog_config.vector_prune,
                username=config.username,
                password=config.password,
                url=config.graph_url,
                database=rag_config.database,
            )

            response_synthesizer = get_response_synthesizer(
                response_mode=ResponseMode.SIMPLE_SUMMARIZE,
                streaming=True,
            )
            tog_query_engine = RetrieverQueryEngine(
                retriever=tog_retriever, response_synthesizer=response_synthesizer
            )
            stream_query_gen = stream_query(
                tog_query_engine,
                question=rag_config.question,
                instruct=rag_config.instruct,
                raw=True,
                yield_node=True,
            )
            nodes, graph = next(stream_query_gen)
            data = {
                "contents": [
                    cast(NodeWithScore, node).node.get_content() for node in nodes
                ],
                "graph": graph,
            }

        elif rag_config.index_mode == IndexMode.AGENT:
            log.info("Using agent mode")
            atlas_rag_index = AtlasRAGIndex(
                nodes, storage_context, schema=graph_store._schema
            )
            work_flow = AtlasRAGWorkflow(
                atlas_rag_index,
                executors={Tool.RETRIEVER, Tool.DEDUCE},
                similarity_top_k=rag_config.agent_config.similarity_top_k,
                max_depth=rag_config.agent_config.max_depth,
            )
            for answer in work_flow.query(rag_config.question):
                await websocket.send_json(
                    RAGResponse(code=RAGCode.ANSWER, answer=answer).model_dump()
                )
                total_answer += answer
            data = {
                "contents": list(work_flow.context.chunks),
                "graph": work_flow.context.graph,
            }
            # atlas_rag_agent = AtlasRAGAgent(
            #     atlas_rag_index,
            #     max_entity_num=rag_config.agent_config.max_entity_num,
            #     similarity_top_k=rag_config.agent_config.similarity_top_k,
            #     min_depth=rag_config.agent_config.min_depth,
            #     max_depth=rag_config.agent_config.max_depth,
            # )
            # for answer in atlas_rag_agent.stream_query(
            #     rag_config.question, instruct=rag_config.instruct
            # ):
            #     await websocket.send_json(
            #         RAGResponse(code=RAGCode.ANSWER, answer=answer).model_dump()
            #     )
            #     total_answer += answer
            # data = {
            #     "contents": atlas_rag_agent.text_chunks,
            #     "graph": atlas_rag_agent.graph,
            # }
            await websocket.send_json(
                RAGResponse(code=RAGCode.DATA, data=data).model_dump()
            )
            global_database.add_conversation(
                rag_config.session_id, rag_config.question, total_answer, data
            )
            await websocket.close()
            return

        elif rag_config.index_mode == IndexMode.HYBRID:
            log.info("Using hybrid mode")
            VectorStoreIndex.index_struct_cls = AtlasIndexDict  # type: ignore[assignment]
            vector_index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
            )
            KnowledgeGraphIndex.index_struct_cls = AtlasKGWrapper(
                graph_store=graph_store
            )  # type: ignore[assignment]
            kg_index = KnowledgeGraphIndex(nodes, storage_context=storage_context)
            hybrid_retriever = HybridRetriever(
                vector_index=vector_index,
                kg_index=kg_index,
                similarity_top_k=rag_config.hybrid_config.similarity_top_k,
                hybrid_retriever_mode=rag_config.hybrid_config.retriever_mode,
                max_keywords_per_query=rag_config.hybrid_config.max_entity_num,
                graph_store_query_depth=rag_config.hybrid_config.max_depth,
            )
            from llama_index.core.prompts.default_prompts import (
                DEFAULT_TEXT_QA_PROMPT_TMPL,
                PromptTemplate,
                PromptType,
            )

            qa_prompt_template = PromptTemplate(
                DEFAULT_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.TREE_SELECT
            )
            response_synthesizer = get_response_synthesizer(
                response_mode=ResponseMode.SIMPLE_SUMMARIZE,
                streaming=True,
                text_qa_template=qa_prompt_template,
            )
            hybrid_query_engine = RetrieverQueryEngine(
                retriever=hybrid_retriever, response_synthesizer=response_synthesizer
            )
            stream_query_gen = stream_query(
                hybrid_query_engine,
                question=rag_config.question,
                instruct=rag_config.instruct,
                raw=True,
                yield_node=True,
            )
            nodes, graph = next(stream_query_gen)
            data = {
                "contents": [
                    cast(NodeWithScore, node).node.get_content() for node in nodes
                ],
                "graph": graph,
            }
        await websocket.send_json(
            RAGResponse(code=RAGCode.DATA, data=data).model_dump()
        )
        for answer in stream_query_gen:
            await websocket.send_json(
                RAGResponse(code=RAGCode.ANSWER, answer=answer).model_dump()
            )
            total_answer += answer
        global_database.add_conversation(
            rag_config.session_id, rag_config.question, total_answer, data
        )
        await websocket.close()

    except Exception as e:
        await websocket.send_json(
            RAGResponse(code=RAGCode.ERROR, error_msg=str(e)).model_dump()
        )
        await websocket.close()
        log.error(traceback.format_exc())
        raise WebSocketException(code=status.WS_1011_INTERNAL_ERROR, reason=str(e))


# upload files
@app.post("/api/upload")
def upload_files(upload_file: UploadFile, upload_config_form: str) -> JSONResponse:
    log.info("File upload requested")
    upload_config: UploadConfig = UploadConfig(**json.loads(upload_config_form))
    # Generate id
    transmission_id = str(uuid.uuid4())
    log.info("transmission_id: " + transmission_id)
    if upload_file.content_type not in [
        UploadFileType.UPLOAD_TAR_MIMETYPE,
        UploadFileType.UPLOAD_ZIP_MIMETYPE,
    ]:
        log.error("error: Only tar or zip file is supported")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Only tar or zip file is supported",
        )

    # unique folder for each transmission
    file_base_path = config.upload_file_dir_path.joinpath(transmission_id)
    os.makedirs(file_base_path)

    # Save file
    if not bool(upload_file.filename):
        log.error("Empty filename")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Empty filename",
        )
    file_path = file_base_path.joinpath(upload_file.filename)  # type: ignore[arg-type]
    with file_path.open("wb") as buffer:
        log.info(f"downloading file {upload_file.filename}")
        buffer.write(upload_file.file.read())
    log.info(f"extracting {upload_file.filename}")
    if upload_file.content_type == UploadFileType.UPLOAD_TAR_MIMETYPE:
        tar = tarfile.open(file_path)
        tar.extractall(path=file_base_path)
    else:
        zip_ref = zipfile.ZipFile(file_path, "r")
        zip_ref.extractall(file_base_path)
    os.remove(file_path)
    # judge whether the file type supports it
    for m_file in file_base_path.rglob("*"):
        if m_file.is_file():
            if m_file.suffix[1:] not in [
                "doc",
                "docx",
                "md",
                "ppt",
                "pptx",
                "pdf",
                "txt",
            ]:
                shutil.rmtree(file_base_path)
                log.error(
                    f'file types: {m_file.suffix} are not supported and supported file types include: ["doc","docx","md","ppt","pptx","pdf","txt"]'
                )
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f'file types: {m_file.suffix} are not supported and supported file types include: ["doc","docx","md","ppt","pptx","pdf","txt"]',
                )
    # multiprocessing for file process
    try:
        global_database.add_task(
            transmission_id,
            upload_config.username,
            upload_config.database,
            upload_config,
        )
    except Exception as e:
        log.error(f"Error processing files: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)
        )

    return JSONResponse(
        content={
            "message": "Files uploaded",
            "transmission_id": transmission_id,
        }
    )


# Check status by ID
@app.get("/api/status/{transmission_id}")
def check_status(transmission_id: str) -> JSONResponse:
    log.info("Upload status requested")
    try:
        return JSONResponse(
            content={
                "id": transmission_id,
                "status": global_database.get_task_status(transmission_id),
                "rate": global_database.get_task_processing_rate(transmission_id),
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )

async def sse_handler(request: Request) -> None:
    async with sse_transport.connect_sse(request.scope, request._receive, request._send) as streams:
        await mcp._mcp_server.run(streams[0], streams[1], mcp._mcp_server.create_initialization_options())

@app.get("/sse")
async def sse_endpoint(request : Request) -> None:
    return await sse_handler(request)

app.mount("/messages", app=sse_transport.handle_post_message)

if __name__ == "__main__":
    import logging
    from driver3.atlas_logger.AtlasLogger import logger
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="atlas rag")
    parser.add_argument(
        "--rag_config",
        default="/root/rag/config/config.yaml",
        type=str,
    )
    args = parser.parse_args()
    logger.setLevel(logging.CRITICAL)
    config.set_rag_server_path(args.rag_config)

    log.info(f"Atlas Rag Version: {rag.get_version()}")
    uvicorn.run(app, host="0.0.0.0", port=config.app_port)
    log.info("App Closed!")
    
