from typing import Any, Callable, TypeVar, Optional, Dict
from pydantic import BaseModel
from enum import Enum, IntEnum
from llama_index.core.response_synthesizers.type import ResponseMode

F = TypeVar("F", bound=Callable[..., Any])


class StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class UploadFileType(StrEnum):
    UPLOAD_TAR_MIMETYPE = "application/x-tar"
    UPLOAD_ZIP_MIMETYPE = "application/zip"


class IndexMode(StrEnum):
    VECTOR = "vector"
    KG = "kg"
    TOG = "tog"
    HYBRID = "hybrid"
    AGENT = "agent"


class HyBridRetrieverMode(StrEnum):
    FINE_GRAINED = "fine-grained"
    COARSE_GRAINED = "coarse-grained"


class SplitterMode(StrEnum):
    PATTERN = "pattern"
    LENGTH = "length"


class RAGCode(IntEnum):
    DATA = 0
    ANSWER = 1
    ERROR = -1


class Tool(StrEnum):
    DEDUCE = "deduce"
    RETRIEVER = "retriever"
    MATH = "math"


class UploadConfig(BaseModel):
    database: str
    username: str
    graph_schema: Optional[Dict[str, Any]] = None
    splitter_mode: SplitterMode = SplitterMode.LENGTH
    pattern: Optional[str] = None
    chunk_size: int = 512
    min_chunk_size: int = 1024
    max_chunk_size: int = 8192
    chunk_overlap: int = 100
    max_workers: int = 1
    num_extract: int = 1
    parse_pdf: bool = False
    entity_align: bool = False
    threashold: float = 0.95


class VectorConfig(BaseModel):
    similarity_top_k: int = 3


class KGConfig(BaseModel):
    similarity_top_k: int = 3
    max_entity_num: int = 10
    depth: int = 4


class TogConfig(BaseModel):
    temperature_exploration: float = 0.4
    temperature_reasoning: float = 0.0
    depth: int = 3
    max_width: int = 10
    min_score: float = 0.35
    vector_prune: bool = True


class HybridConfig(BaseModel):
    similarity_top_k: int = 3
    retriever_mode: HyBridRetrieverMode = HyBridRetrieverMode.FINE_GRAINED
    max_entity_num: int = 10
    max_depth: int = 4


class AgentConfig(BaseModel):
    max_entity_num: int = 10
    similarity_top_k: int = 3
    min_depth: int = 0
    max_depth: int = 6


class RAGConfig(BaseModel):
    database: str
    question: str
    username: str
    session_id: str
    instruct: Optional[str] = None
    index_mode: IndexMode = IndexMode.VECTOR
    vector_config: VectorConfig = VectorConfig()
    kg_config: KGConfig = KGConfig()
    hybrid_config: HybridConfig = HybridConfig()
    tog_config: TogConfig = TogConfig()
    agent_config: AgentConfig = AgentConfig()


class RAGResponse(BaseModel):
    code: RAGCode
    data: Dict[str, Any] = {}
    answer: str = ""
    error_msg: str = ""
