from llama_index.core import (
    VectorStoreIndex,
    KnowledgeGraphIndex,
    SimpleDirectoryReader,
    get_response_synthesizer,
    Settings,
)
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.storage.storage_context import StorageContext
from rag.store.vector_store import AtlasVectorStore
from rag.store.graph_store import AtlasGraphStore
from rag.store.document_store import AtlasDocumentStore
from rag.store.index_store import AtlasIndexDict, AtlasIndexStore
from llama_index.core.ingestion import run_transformations
import argparse
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from rag.retriever.hybrid_retriever import HybridRetriever, HyBridRetrieverMode
from rag.retriever.tog_retriever import ToGRetriever
from rag.service.pdf_parser import PdfParser
from rag.index.aligned_knowledge_graph_index import AlignedKnowledgeGraphIndex
from rag.utils.utils import stream_query
import os
from typing import List, Union
from rag.utils.config import rag_config as config
from rag.utils.log import rag_logger as log
from rag.reader import DocReader, PptReader
from rag.model.atlas_model import AtlasLLM, AtlasEmbeddings
from rag.index.atlas_rag_index import AtlasRAGIndex
from rag.schema import Schema

if __name__ == "__main__":
    import logging
    from driver3.atlas_logger.AtlasLogger import logger

    logger.setLevel(logging.CRITICAL)
    parser = argparse.ArgumentParser(description="atlas rag")
    parser.add_argument("--documents", default="/root/text", type=str)
    parser.add_argument("--chunk_size", default=1024, type=int)
    parser.add_argument("--chunk_overlap", default=256, type=int)
    parser.add_argument("--database", default="rag", type=str)
    parser.add_argument("--vector_node_label", default="Chunk", type=str)
    parser.add_argument("--schema", default="./schema.json", type=str)
    parser.add_argument("--show_progress", default=False, type=bool)
    parser.add_argument(
        "--response_mode", default=ResponseMode.TREE_SUMMARIZE.value, type=str
    )
    parser.add_argument("--question", default="What happened at interleaf?", type=str)
    parser.add_argument("--reload_flag", default=False, type=bool)
    parser.add_argument("--parse_pdf", default=False, type=bool)
    parser.add_argument("--entity_align", default=False, type=bool)
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
        sentence_splitter = SentenceSplitter(
            chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
        )
    else:
        documents = SimpleDirectoryReader(
            args.documents, recursive=True, file_extractor=file_extractor
        ).load_data()
        sentence_splitter = SentenceSplitter(
            chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
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
    Settings.node_parser = sentence_splitter
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
    nodes = (
        run_transformations(
            documents,  # type: ignore
            [sentence_splitter],
            show_progress=args.show_progress,
        )
        if args.reload_flag
        else []
    )
    atlas_rag_index = AtlasRAGIndex(nodes, storage_context, schema=schema)

    def return_page_number(
        node_ids: List[str],
        chunk_index_node_property: str = "chunkIndex",
        document_name_node_property: str = "documentName",
    ) -> List[int]:
        if len(node_ids) == 0:
            return []
        try:
            from driver3.service.AtlasGraph import AtlasGraph
            from driver3.service.ConnectionConfiguration import ConnectionConfiguration
        except ImportError:
            raise ImportError("Please install atlas: pip install atlas")
        database = AtlasGraph(
            ConnectionConfiguration(config.graph_url, config.username, config.password)
        ).use(args.database)
        cypher_query = f"match(n:{args.vector_node_label}) where n.{chunk_index_node_property} in {node_ids}  return collect(distinct n.{document_name_node_property});"
        page_numbers: List[str] = database.exec(cypher_query).result_set["table"][0][0]
        return sorted([int(file.split(".")[0]) for file in page_numbers])

    VectorStoreIndex.index_struct_cls = AtlasIndexDict  # type: ignore[assignment]
    vector_index = VectorStoreIndex(nodes=[], storage_context=storage_context)
    streaming_answer_gen = stream_query(
        vector_index.as_query_engine(similarity_top_k=3, streaming=True), args.question
    )
    log.info(f"only use vector index's answer:")
    while True:
        try:
            print(next(streaming_answer_gen), end="", flush=True)
        except StopIteration as e:
            print("")
            answer, node_ids, _ = e.value
            log.info(f"node_ids: {node_ids}\n")
            break
    # log.info(
    #     f"only use vector index's answer: {answer}\n node_ids: {node_ids}\n"
    # )
    if not args.entity_align:
        kg_index: Union[KnowledgeGraphIndex, AlignedKnowledgeGraphIndex] = (
            KnowledgeGraphIndex(
                nodes=[],
                storage_context=storage_context,
                max_triplets_per_chunk=30,
            )
        )
    else:
        kg_index = AlignedKnowledgeGraphIndex(nodes=[], storage_context=storage_context)
    document_store.add_keywords(kg_index.index_struct)  # type: ignore[arg-type]
    # log.info(
    #     f"only use kg index's answer: {kg_index.as_query_engine().query(args.question)}\n\n"
    # )
    hybrid_retriever = HybridRetriever(
        vector_index=vector_index,
        kg_index=kg_index,
        similarity_top_k=3,
        hybrid_retriever_mode=HyBridRetrieverMode.FINE_GRAINED,
    )
    response_synthesizer = get_response_synthesizer(
        response_mode=args.response_mode, streaming=True
    )

    hybrid_query_engine = RetrieverQueryEngine(
        retriever=hybrid_retriever, response_synthesizer=response_synthesizer
    )
    streaming_answer_gen = stream_query(hybrid_query_engine, args.question)
    log.info("use hybrid index's answer: ")
    while True:
        try:
            print(next(streaming_answer_gen), end="", flush=True)
        except StopIteration as e:
            print("")
            answer, node_ids, _ = e.value
            print(f"node_ids: {node_ids}\n\n")
            break
    # log.info(
    #     f"use hybrid index's answer: {answer}\n node_ids: {node_ids}\n\n"
    # )

    tog_retriever = ToGRetriever(
        vector_index=vector_index,
        kg_index=kg_index,
        username=config.username,
        password=config.password,
        url=config.graph_url,
        node_label=list(schema.vertices.keys())[0],
        database=args.database,
    )
    tog_query_engine = RetrieverQueryEngine(
        retriever=tog_retriever, response_synthesizer=response_synthesizer
    )
    streaming_answer_gen = stream_query(tog_query_engine, args.question)
    log.info("use tog index's answer: ")
    while True:
        try:
            print(next(streaming_answer_gen), end="", flush=True)
        except StopIteration as e:
            print("")
            answer, node_ids, _ = e.value
            print(f"\n node_ids: {node_ids}\n\n")
            break
    # log.info(f"user tog index's answer: {tog_query_engine.query(args.question)}\n\n")
