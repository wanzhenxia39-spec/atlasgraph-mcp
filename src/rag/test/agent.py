from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core.storage.storage_context import StorageContext
from rag.store.vector_store import AtlasVectorStore
from rag.store.graph_store import AtlasGraphStore
from rag.store.document_store import AtlasDocumentStore
from rag.store.index_store import AtlasIndexStore
import argparse
import logging
from driver3.atlas_logger.AtlasLogger import logger
from rag.model.atlas_model import AtlasLLM, AtlasEmbeddings
from rag.utils.config import rag_config as config
from rag.utils.log import rag_logger as log
from rag.utils.timer import Timer
from llama_index.core import SimpleDirectoryReader
from rag.index import AtlasRAGIndex
from rag.schema import Schema
from rag.reader import DocReader, PptReader
from rag.service.pdf_parser import PdfParser
from rag.agent.atlas_rag_agent import AtlasRAGAgent
from rag.splitter import LengthSplitter, PatternSplitter
from rag.utils.constant import SplitterMode
import os

if __name__ == "__main__":

    logger.setLevel(logging.CRITICAL)
    import logging
    from driver3.atlas_logger.AtlasLogger import logger

    logger.setLevel(logging.CRITICAL)
    parser = argparse.ArgumentParser(description="atlas rag")
    parser.add_argument("--documents", default="/tmp/rag-data/fund", type=str)
    parser.add_argument("--splitter_mode", default=SplitterMode.PATTERN, type=str)
    parser.add_argument("--chunk_size", default=1024, type=int)
    parser.add_argument("--min_chunk_size", default=1024, type=int)
    parser.add_argument("--max_chunk_size", default=8192, type=int)
    parser.add_argument("--chunk_overlap", default=256, type=int)
    parser.add_argument("--database", default="test", type=str)
    parser.add_argument("--vector_node_label", default="Chunk", type=str)
    parser.add_argument("--schema", default="./schema.json", type=str)
    parser.add_argument("--num_extract", default=1, type=int)
    parser.add_argument("--show_progress", default=False, type=bool)
    parser.add_argument(
        "--response_mode", default=ResponseMode.TREE_SUMMARIZE.value, type=str
    )
    parser.add_argument("--question", default="What happened at interleaf?", type=str)
    parser.add_argument("--reload_flag", default=False, type=bool)
    parser.add_argument("--parse_pdf", default=False, type=bool)
    parser.add_argument("--entity_align", default=False, type=bool)
    parser.add_argument("--min_depth", default=0, type=int)
    parser.add_argument("--max_depth", default=6, type=int)
    parser.add_argument(
        "--rag_config",
        default="./config.yaml",
        type=str,
    )
    args = parser.parse_args()
    config.set_rag_server_path(args.rag_config)
    log.info(f"question: {args.question}")
    doc_reader = DocReader()
    ppt_reader = PptReader()
    file_extractor = {".doc": doc_reader, ".ppt": ppt_reader, ".pptx": ppt_reader}
    if args.parse_pdf:
        work_path = os.path.abspath(os.path.join(args.documents, "parse_pdf"))
        if args.reload_flag:
            if not os.path.exists(work_path):
                os.mkdir(work_path)
            PdfParser(args.documents, work_path).parse()
        documents = SimpleDirectoryReader(
            args.documents, recursive=True, file_extractor=file_extractor
        ).load_data()
        sentence_splitter = (
            LengthSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
            if args.splitter_mode == SplitterMode.LENGTH
            else PatternSplitter(
                min_chunk_size=args.min_chunk_size,
                max_chunk_size=args.max_chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
        )
    else:
        documents = SimpleDirectoryReader(
            args.documents, recursive=True, file_extractor=file_extractor
        ).load_data()
        sentence_splitter = (
            LengthSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
            if args.splitter_mode == SplitterMode.LENGTH
            else PatternSplitter(
                min_chunk_size=args.min_chunk_size,
                max_chunk_size=args.max_chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
        )

    llm = AtlasLLM(
        model=config.llm_model,
        api_url=config.llm_url,
        api_base=config.llm_base,
        api_key=config.llm_key,
    )
    embeddings = AtlasEmbeddings(
        model=config.embedding_model,
        api_url=config.embedding_url,
        api_base=config.embedding_base,
        api_key=config.embedding_key,
    )

    Settings.llm = llm
    Settings.embed_model = embeddings
    schema = Schema.from_json_file(args.schema)
    index_store = AtlasIndexStore()
    vector_store = AtlasVectorStore(
        username=config.username,
        password=config.password,
        url=config.graph_url,
        embedding_dimension=config.embedding_dimension,
        database=args.database,
        node_label=args.vector_node_label,
        init_database=args.reload_flag or True,
    )
    document_store = AtlasDocumentStore(
        username=config.username,
        password=config.password,
        url=config.graph_url,
        database=args.database,
        node_label=args.vector_node_label,
        init_database=args.reload_flag or True,
    )
    graph_store = AtlasGraphStore(
        username=config.username,
        password=config.password,
        url=config.graph_url,
        database=args.database,
        schema=schema,
        init_database=args.reload_flag or True,
    )
    storage_context = StorageContext.from_defaults(
        docstore=document_store,
        index_store=index_store,
        vector_store=vector_store,  # type: ignore[arg-type]
        graph_store=graph_store,
    )
    nodes = sentence_splitter.split(documents) if args.reload_flag else []
    atlas_rag_index = AtlasRAGIndex(
        nodes, storage_context, schema=schema, num_extract=args.num_extract
    )
    chat = AtlasLLM(
        model=config.chat_model,
        api_url=config.chat_url,
        api_base=config.chat_base,
        api_key=config.chat_key,
    )
    atlas_rag_agent = AtlasRAGAgent(atlas_rag_index, min_depth=args.min_depth, max_depth=args.max_depth, llm=chat)
    timer = Timer()
    log.info(f"answer: {atlas_rag_agent.query(args.question)}")
    log.info(f"time cost: {timer.cost():.2f} sec.")
