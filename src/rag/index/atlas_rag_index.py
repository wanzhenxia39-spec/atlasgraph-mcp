from rag.utils.config import rag_config as config
from rag.prompt.graph_extract_prompt import GraphExtractPrompt
from rag.schema.schema import Schema, DEFAULT_SCHEMA
from rag.store.graph_store import AtlasGraphStore
from rag.model import AtlasEmbeddings, AtlasLLM
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.core.schema import BaseNode, MetadataMode
from typing import Iterable, cast, List, Optional
from rag.utils.timer import Timer
from rag.utils.log import rag_logger as log
from concurrent.futures import ThreadPoolExecutor, Future, wait, as_completed
import copy
import os
from tqdm import tqdm


class AtlasRAGIndex:
    NCOLS = 160

    def __init__(
        self,
        nodes: Iterable[BaseNode],
        storage_context: StorageContext,
        schema: Schema = DEFAULT_SCHEMA,
        entity_align: bool = False,
        threashold: float = 0.95,
        num_extract: int = 1,
        max_workers: int = 64,
        task_id: Optional[str] = None,
    ) -> None:
        self.storage_context = storage_context
        self.schema = schema
        self.entity_align = entity_align
        self.threashold = threashold
        self.num_extract = num_extract
        self.llm = AtlasLLM(
            model=config.llm_model,
            api_url=config.llm_url,
            api_base=config.llm_base,
            api_key=config.llm_key,
        )
        self.embeddings = AtlasEmbeddings(
            model=config.embedding_model,
            api_url=config.embedding_url,
            api_base=config.embedding_base,
            api_key=config.embedding_key,
        )
        timer = Timer()

        def task(node: BaseNode, dependencis: List[Future]) -> None:
            graph_store = cast(AtlasGraphStore, self.storage_context.graph_store)
            for i in range(3):
                if self.storage_context.docstore.document_exists(node.node_id):
                    break
                try:
                    prompt = GraphExtractPrompt()
                    text = f"切片来源文档: {node.metadata["file_name"]} 内容: {node.get_content(metadata_mode=MetadataMode.NONE)}"
                    node.embedding = self.embeddings.get_text_embedding(text)
                    graph_data = {"vertices": [], "edges": []}
                    for _ in range(self.num_extract):
                        graph_data, no_new_addition = prompt.parse_response(
                            self.llm.complete(
                                prompt.build_prompt(
                                    schema=self.schema, graph=graph_data, input=text
                                )
                            )
                        )
                        if no_new_addition:
                            break
                    if entity_align:
                        wait(dependencis)
                        vertex_entity_align_map = {}
                        for vertex in graph_data["vertices"]:
                            vector_result = graph_store.keyword_vector_stores[
                                vertex["label"]
                            ].query(
                                VectorStoreQuery(
                                    query_embedding=self.embeddings.get_text_embedding(
                                        f"keyword: {vertex["keyword"]} desc: {vertex["description"]}"
                                    ),
                                    similarity_top_k=3,
                                )
                            )
                            if (
                                vector_result.similarities
                                and len(vector_result.similarities) >= 1
                                and vector_result.similarities[0] >= self.threashold
                                and vector_result.ids
                                and len(vector_result.ids) >= 1
                                and vertex["keyword"] != vector_result.ids[0]
                            ):
                                vertex_entity_align_map[
                                    (vertex["label"], vertex["keyword"])
                                ] = vector_result.ids[0]
                                vertex["keyword"] = vector_result.ids[0]
                        for edge in graph_data["edges"]:
                            if (
                                edge["from_key"],
                                edge["from_pk"],
                            ) in vertex_entity_align_map:
                                edge["from_pk"] = vertex_entity_align_map[
                                    (edge["from_key"], edge["from_pk"])
                                ]
                            if (
                                edge["to_key"],
                                edge["to_pk"],
                            ) in vertex_entity_align_map:
                                edge["to_pk"] = vertex_entity_align_map[
                                    (edge["to_key"], edge["to_pk"])
                                ]
                    self.storage_context.docstore.add_documents([node])
                    self.storage_context.vector_store.add([node])
                    graph_store.upsert_graph(graph_data, node.node_id)
                    break
                except Exception as e:
                    log.warning(f"create atlas rag index error:{e} try again: {i}")
                    continue

        flag = False
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            dependencis: List[Future] = []
            for node in nodes:
                flag = True
                dependencis.append(executor.submit(task, node, copy.copy(dependencis)))
            with log.terminator_manager(), tqdm(
                total=len(dependencis),
                file=open(os.devnull, "w"),
                ncols=AtlasRAGIndex.NCOLS,
            ) as pbar:
                pbar.set_description(f"Chunk Extract Processing")
                for index, _ in enumerate(as_completed(dependencis)):
                    pbar.update()
                    if index != len(dependencis) - 1:
                        log.set_terminator(end="\r")
                    else:
                        log.set_terminator(end="\n")
                    log.info(str(pbar))
                    if task_id is not None:
                        from rag.service.global_database import global_database

                        global_database.update_task_processing_rate(
                            task_id, round((index + 1) / len(dependencis), 2)
                        )
        if flag:
            log.info(
                f"create atlas rag index finsh and time cost:{timer.cost():.2f} sec"
            )
