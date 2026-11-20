from rag.common.task import Task
from rag.common.context import Context
from abc import ABC, abstractmethod
from typing import Generator, Any
from rag.index import AtlasRAGIndex
from rag.store import AtlasGraphStore, AtlasVectorStore, AtlasDocumentStore
from rag.utils.config import rag_config as config
from rag.model import AtlasLLM, AtlasEmbeddings


class BaseTool(ABC):
    def __init__(self, rag_index: AtlasRAGIndex, **kwargs: Any) -> None:
        self.llm = AtlasLLM(
            model=config.chat_model,
            api_url=config.chat_url,
            api_base=config.chat_base,
            api_key=config.chat_key,
        )
        self.embedding_model = AtlasEmbeddings(
            model=config.embedding_model,
            api_url=config.embedding_url,
            api_base=config.embedding_base,
            api_key=config.embedding_key,
        )
        self.graph_store: AtlasGraphStore = rag_index.storage_context.graph_store  # type: ignore[assignment]
        self.vector_store: AtlasVectorStore = rag_index.storage_context.vector_store  # type: ignore[assignment]
        self.document_store: AtlasDocumentStore = rag_index.storage_context.docstore  # type: ignore[assignment]
        self.schema = rag_index.schema
        self.database = self.graph_store.client.use(self.graph_store._database)

    @abstractmethod
    def execute(
        self, query: str, task: Task, context: Context
    ) -> Generator[str, None, None]:
        raise NotImplementedError
