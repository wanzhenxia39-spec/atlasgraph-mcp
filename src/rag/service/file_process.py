from llama_index.core import (
    SimpleDirectoryReader,
    Settings,
)
from llama_index.core.storage.storage_context import StorageContext
from rag.store.vector_store import AtlasVectorStore
from rag.store.graph_store import AtlasGraphStore
from rag.store.document_store import AtlasDocumentStore
from rag.store.index_store import AtlasIndexStore
from rag.service.pdf_parser import PdfParser
import shutil
from rag.utils.timer import Timer
from rag.schema import Schema
from rag.model.atlas_model import AtlasLLM, AtlasEmbeddings
from rag.utils.log import rag_logger as log

from rag.utils.config import rag_config as config
from rag.utils.timer import Timer
from rag.reader import DocReader, PptReader
from rag.utils.constant import SplitterMode, UploadConfig
from rag.splitter import LengthSplitter, PatternSplitter
from rag.index import AtlasRAGIndex
import logging
from driver3.atlas_logger.AtlasLogger import logger
from driver3.service.AtlasGraph import AtlasGraph
from driver3.service.ConnectionConfiguration import ConnectionConfiguration
from pathlib import Path
import traceback


def process_files(
    upload_config: UploadConfig,
    transmission_id: str,
    config_path: Path,
) -> bool:
    logger.setLevel(logging.CRITICAL)
    config.set_rag_server_path(config_path, init_global_base=False)
    from rag.service.global_database import GlobalDatabase

    global_database = AtlasGraph(
        ConnectionConfiguration(config.graph_url, config.username, config.password)
    ).use(config.global_database)
    try:
        log.info("Start file processing")
        log.info("transmission_id: " + transmission_id)
        timer = Timer()

        documents_path = config.upload_file_dir_path.joinpath(transmission_id)
        splitter_mode = upload_config.splitter_mode
        pattern = upload_config.pattern
        chunk_size = upload_config.chunk_size
        min_chunk_size = upload_config.min_chunk_size
        max_chunk_size = upload_config.max_chunk_size
        chunk_overlap = upload_config.chunk_overlap
        database = upload_config.database
        schema = (
            Schema(upload_config.graph_schema)
            if upload_config.graph_schema is not None
            else None
        )
        max_workers = upload_config.max_workers
        num_extract = upload_config.num_extract
        parse_pdf = upload_config.parse_pdf
        entity_align = upload_config.entity_align
        threashold = upload_config.threashold
        doc_reader = DocReader()
        ppt_reader = PptReader()
        file_extractor = {".doc": doc_reader, ".ppt": ppt_reader, ".pptx": ppt_reader}
        # file processing and storing
        if parse_pdf:
            work_path = documents_path.joinpath("parse_pdf").absolute()
            if not work_path.exists():
                work_path.mkdir(parents=True, exist_ok=True)
                PdfParser(str(documents_path), str(work_path)).parse()
            documents = SimpleDirectoryReader(
                work_path, recursive=True, file_extractor=file_extractor
            ).load_data()
            sentence_splitter = (
                LengthSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                if splitter_mode == SplitterMode.LENGTH
                else PatternSplitter(
                    min_chunk_size=min_chunk_size,
                    max_chunk_size=max_chunk_size,
                    chunk_overlap=chunk_overlap,
                    pattern=pattern,
                )
            )
        else:
            documents = SimpleDirectoryReader(
                documents_path, recursive=True, file_extractor=file_extractor
            ).load_data()
            sentence_splitter = (
                LengthSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                if splitter_mode == SplitterMode.LENGTH
                else PatternSplitter(
                    min_chunk_size=min_chunk_size,
                    max_chunk_size=max_chunk_size,
                    chunk_overlap=chunk_overlap,
                    pattern=pattern,
                )
            )

        try:
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

        except Exception as e:
            log.error("Failed loading LLM")
            raise e

        try:
            Settings.llm = llm
            Settings.embed_model = embeddings
        except Exception as e:
            log.error("Failed initializing service settings")
            raise e

        try:
            index_store = AtlasIndexStore()
            log.info("index_store initialized")
            vector_store = AtlasVectorStore(
                username=config.username,
                password=config.password,
                url=config.graph_url,
                embedding_dimension=config.embedding_dimension,
                database=database,
            )
            log.info("vector_store initialized")
            document_store = AtlasDocumentStore(
                username=config.username,
                password=config.password,
                url=config.graph_url,
                database=database,
            )
            log.info("document_store initialized")
            graph_store = AtlasGraphStore(
                username=config.username,
                password=config.password,
                url=config.graph_url,
                database=database,
                schema=schema,
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
            log.error("Failed loading stores")
            raise e

        nodes = sentence_splitter.split(documents)
        AtlasRAGIndex(
            nodes,
            storage_context,
            schema=(
                Schema.from_str(graph_store.get_schema()) if schema is None else schema
            ),
            num_extract=num_extract,
            entity_align=entity_align,
            threashold=threashold,
            max_workers=max_workers,
            task_id=transmission_id,
        )
        shutil.rmtree(documents_path)
        log.info(
            f"Done processing transmission: {transmission_id} and Time taken: {timer.cost():.2f} s"
        )
        global_database.exec(
            f'MATCH (t: Task{{id: "{transmission_id}"}}) SET t.status = "{GlobalDatabase.COMPLETE}"'
        )
        return True
    except BaseException as e:
        log.error(f"Error upload files:{repr(e)}")
        log.error(traceback.format_exc())
        global_database.exec(
            f'MATCH (t: Task{{id: "{transmission_id}"}}) SET t.status = "{GlobalDatabase.FAULTED}"'
        )
        return False
