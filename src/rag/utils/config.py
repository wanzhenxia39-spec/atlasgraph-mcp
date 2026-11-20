from pathlib import Path
from typing import Optional, Dict
from rag.utils.log import rag_logger as log
from rag.utils.log import transform_log_level
import yaml
from functools import cached_property
import logging


class RAGConfig:
    def __init__(self) -> None:
        self.server_id: int = 0
        self.app_port: int = 21032
        self.num_workers: int = 4
        self.app_debug: bool = False
        self.graph_url: str = "127.0.0.1:21021"
        self.username: str = "root"
        self.password: str = ""
        self.log_level: int = logging.INFO
        self.max_num_task: int = 100
        self.llm_max_context: int = 32768
        self.embedding_max_context: int = 8192
        self.log_dir_path: Path = Path("./logs")
        self.upload_file_dir_path: Path = Path("./upload")
        self.global_database: str = "GraphRAG"
        # llm config
        self.llm_config: Dict = {"max_tokens": 4096}
        # llm
        self.llm_url: Optional[str] = None
        self.llm_base: str = "http://127.0.0.1:8000/v1"
        self.llm_model: str = "Qwen2.5-72B-Instruct-AWQ"
        self.llm_key: Optional[str] = None
        # embedding
        self.embedding_url: Optional[str] = None
        self.embedding_base: str = "http://127.0.0.1:8000/v1"
        self.embedding_model: str = "acge_text_embedding"
        self.embedding_key: Optional[str] = None
        # mllm
        self.mllm_url: Optional[str] = None
        self.mllm_base: Optional[str] = "http://127.0.0.1:8000/v1"
        self.mllm_model: Optional[str] = "Qwen2-VL-2B-Instruct-AWQ"
        self.mllm_key: Optional[str] = None
        # chat
        self.chat_url: Optional[str] = None
        self.chat_base: str = "http://127.0.0.1:8000/v1"
        self.chat_model: str = "Qwen2.5-72B-Instruct-AWQ"
        self.chat_key: Optional[str] = None
        # mcp
        self.mcp_port: int = 8000

    @cached_property
    def embedding_dimension(self) -> int:
        from rag.model.atlas_model import AtlasEmbeddings

        embeddings = AtlasEmbeddings(
            model=self.embedding_model,
            api_base=self.embedding_base,
            api_key=self.embedding_key,
            api_url=self.embedding_url,
        )
        return len(embeddings.get_text_embedding("embeddings dimension"))

    def set_rag_server_path(
        self, rag_server_path: Path, init_global_base: bool = True
    ) -> None:
        if type(rag_server_path) == str:
            rag_server_path = Path(rag_server_path)
        self.rag_server_path = rag_server_path
        with rag_server_path.open(mode="r") as rag_server_file:
            configs = yaml.safe_load(rag_server_file)
            rag_server = configs["RAGServer"]
            if "server_id" in rag_server:
                self.server_id = int(rag_server["server_id"])
            if "app_port" in rag_server:
                self.app_port = int(rag_server["app_port"])
            if "app_debug" in rag_server:
                self.app_debug = rag_server["app_debug"]
            if "graph_url" in rag_server:
                self.graph_url = rag_server["graph_url"]
            if "username" in rag_server:
                self.username = rag_server["username"]
            if "password" in rag_server:
                self.password = rag_server["password"]
            if "num_workers" in rag_server:
                self.file_service_pool_processes = rag_server["num_workers"]
            if "max_num_task" in rag_server:
                self.max_num_task = rag_server["max_num_task"]
            if "llm_max_context" in rag_server:
                self.llm_max_context = rag_server["llm_max_context"]
            if "embedding_max_context" in rag_server:
                self.embedding_max_context = rag_server["embedding_max_context"]
            if "global_database" in rag_server:
                self.global_database = rag_server["global_database"]
            if "log_level" in rag_server:
                self.log_level = transform_log_level(rag_server["log_level"])
            if "output_dir" in rag_server:
                self.log_dir_path = Path(rag_server["output_dir"]).joinpath("logs")
                self.upload_file_dir_path = Path(rag_server["output_dir"]).joinpath(
                    "upload"
                )
            if "mcp_port" in rag_server:
                self.mcp_port = int(rag_server["mcp_port"])  # 确保端口为整数

            self.log_dir_path.mkdir(parents=True, exist_ok=True)
            self.upload_file_dir_path.mkdir(parents=True, exist_ok=True)
            self.llm_config = configs["LLMConfig"]
            llm = configs["LLM"]
            if "llm_url" in llm:
                self.llm_url = llm["llm_url"]
            if "llm_base" in llm:
                self.llm_base = llm["llm_base"]
            if "llm_model" in llm:
                self.llm_model = llm["llm_model"]
            if "llm_key" in llm:
                self.llm_key = llm["llm_key"]

            embedding = configs["Embedding"]
            if "embedding_url" in embedding:
                self.embedding_url = embedding["embedding_url"]
            if "embedding_base" in embedding:
                self.embedding_base = embedding["embedding_base"]
            if "embedding_model" in embedding:
                self.embedding_model = embedding["embedding_model"]
            if "embedding_key" in embedding:
                self.embedding_key = embedding["embedding_key"]
            mllm = configs["MLLM"]
            if "mllm_url" in mllm:
                self.mllm_url = mllm["mllm_url"]
            if "mllm_base" in mllm:
                self.mllm_base = mllm["mllm_base"]
            if "llm_model" in mllm:
                self.mllm_model = mllm["mllm_model"]
            if "mllm_key" in mllm:
                self.mllm_key = mllm["mllm_key"]
            log.set_log_path(self.log_dir_path, log_level=self.log_level)
            chat = configs["CHAT"]
            if "chat_url" in chat:
                self.chat_url = chat["chat_url"]
            if "chat_base" in chat:
                self.chat_base = chat["chat_base"]
            if "chat_model" in chat:
                self.chat_model = chat["chat_model"]
            if "chat_key" in chat:
                self.chat_key = chat["chat_key"]
                    
        if init_global_base:
            from rag.service.global_database import global_database

            global_database.init()
    

def load_config(config_path : str) -> Optional[dict]:
    """加载YAML配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) # 解析yaml的内容
        return config
    except FileNotFoundError:
        print(f"错误：找不到配置文件 {config_path}")
        return None
    except yaml.YAMLError as e:
        print(f"解析YAML文件出错：{e}")
        return None


rag_config: RAGConfig = RAGConfig()
