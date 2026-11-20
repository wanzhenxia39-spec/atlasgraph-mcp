from llama_index.core.embeddings import BaseEmbedding
from rag.store.graph_store import AtlasGraphStore
from rag.store.vector_store import AtlasVectorStore
from rag.store.document_store import AtlasDocumentStore
from llama_index.core.schema import TextNode
from rag.store.index_store import AtlasKG, AtlasIndexStore
from pathlib import Path
import csv
from typing import Generator, Tuple, List, Iterator
from rag.utils.config import rag_config as config
from rag.utils.log import rag_logger as log


def batch_csv_reader(
    csv_reader: Iterator[Tuple[str, str, str]], batch_size: int
) -> Generator[List[Tuple[str, str, str]], None, None]:
    batch = []
    for row in csv_reader:
        batch.append(row)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def create_index_from_custom_data(
    triple_file_path: Path,
    embedding_model: BaseEmbedding,
    document_store: AtlasDocumentStore,
    index_store: AtlasIndexStore,
    graph_store: AtlasGraphStore,
    vector_store: AtlasVectorStore,
    index_kg: AtlasKG,
    batch_size: int = 1,
) -> None:
    embedding_model.get_text_embedding_batch
    with triple_file_path.open("r") as file:
        csv_reader: Iterator[Tuple[str, str, str]] = csv.reader(file)  # type: ignore[assignment]
        for batch_triple in batch_csv_reader(csv_reader, batch_size):
            embeddings = embedding_model.get_text_embedding_batch(
                [str(triple) for triple in batch_triple]
            )
            for (subj, rel, obj), embedding in zip(batch_triple, embeddings):
                text_node = TextNode(embedding=embedding, text=str((subj, rel, obj)))
                index_kg.add_node([subj, obj], text_node)
                vector_store.add([text_node])
                document_store.add_documents([text_node])
                graph_store.upsert_triplet(subj, rel, obj)
        document_store.add_keywords(index_kg)
        index_store.add_index_struct(index_kg)


if __name__ == "__main__":
    import logging
    import argparse
    from llama_index.core.response_synthesizers.type import ResponseMode
    from rag.model.atlas_model import AtlasLLM, AtlasEmbeddings
    from llama_index.core import (
        Settings,
        StorageContext,
        VectorStoreIndex,
        KnowledgeGraphIndex,
        get_response_synthesizer,
    )
    from llama_index.core.query_engine import RetrieverQueryEngine
    from rag.retriever.hybrid_retriever import HybridRetriever, HyBridRetrieverMode
    from rag.store.index_store import AtlasKGWrapper, AtlasIndexDict

    logging.disable(logging.CRITICAL)
    parser = argparse.ArgumentParser(description="atlas rag")
    parser.add_argument("--csv", default="./test.csv", type=str)
    parser.add_argument("--embedding_dimension", default=1024, type=int)  # 1536
    parser.add_argument("--database", default="test", type=str)
    parser.add_argument("--vector_node_label", default="custom_vector_entity", type=str)
    parser.add_argument(
        "--keyword_node_label", default="custom_keyword_entity", type=str
    )
    parser.add_argument("--rel_label", default="custom_link", type=str)
    parser.add_argument("--show_progress", default=False, type=bool)
    parser.add_argument(
        "--response_mode", default=ResponseMode.TREE_SUMMARIZE.value, type=str
    )
    parser.add_argument("--question", default="制度A的上游制度有哪些?", type=str)
    parser.add_argument("--reload_flag", default=False, action="store_true")
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument(
        "--rag_config",
        default="./config.yaml",
        type=str,
    )
    args = parser.parse_args()
    config.set_rag_server_path(args.rag_config)
    llm = AtlasLLM(
        model=config.llm_model,
        api_base=config.llm_base,
        api_key=config.llm_key,
        api_url=config.llm_url,
    )
    embeddings = AtlasEmbeddings(
        model=config.embedding_model,
        api_base=config.embedding_base,
        api_key=config.embedding_key,
        api_url=config.embedding_url,
    )
    Settings.llm = llm
    Settings.embed_model = embeddings
    vector_store = AtlasVectorStore(
        username=config.username,
        password=config.password,
        url=config.graph_url,
        embedding_dimension=config.embedding_dimension,
        database=args.database,
        node_label=args.vector_node_label,
    )
    index_store = AtlasIndexStore()
    document_store = AtlasDocumentStore(
        username=config.username,
        password=config.password,
        url=config.graph_url,
        database=args.database,
        node_label=args.vector_node_label,
    )
    graph_store = AtlasGraphStore(
        username=config.username,
        password=config.password,
        url=config.graph_url,
        database=args.database,
        node_label=args.keyword_node_label,
        rel_label=args.rel_label,
    )
    index_kg = AtlasKGWrapper(
        username=config.username,
        password=config.password,
        url=config.graph_url,
        database=args.database,
        node_label=args.keyword_node_label,
    )()
    storage_context = StorageContext.from_defaults(
        docstore=document_store,
        index_store=index_store,
        vector_store=vector_store,  # type: ignore[arg-type]
        graph_store=graph_store,
    )
    if args.reload_flag:
        create_index_from_custom_data(
            Path(args.csv),
            embeddings,
            document_store,
            index_store,
            graph_store,
            vector_store,
            index_kg,
            args.batch_size,
        )
    log.info(f"question: {args.question}\n\n")
    VectorStoreIndex.index_struct_cls = AtlasIndexDict  # type: ignore[assignment]
    vector_index = VectorStoreIndex(nodes=[], storage_context=storage_context)
    log.info(
        f"only use vector index's answer: {vector_index.as_query_engine(similarity_top_k=1).query(args.question)}\n\n"
    )
    kg_index = KnowledgeGraphIndex(nodes=[], storage_context=storage_context)
    log.info(
        f"only use kg index's answer: {kg_index.as_query_engine().query(args.question)}\n\n"
    )
    hybrid_retriever = HybridRetriever(
        vector_index=vector_index,
        kg_index=kg_index,
        hybrid_retriever_mode=HyBridRetrieverMode.FINE_GRAINED,
    )
    response_synthesizer = get_response_synthesizer(response_mode=args.response_mode)

    hybrid_query_engine = RetrieverQueryEngine(
        retriever=hybrid_retriever, response_synthesizer=response_synthesizer
    )
    log.info(
        f"use hybrid index's answer: {hybrid_query_engine.query(args.question)}\n\n"
    )
