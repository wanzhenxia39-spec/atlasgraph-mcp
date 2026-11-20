from typing import List
from rag.common.base_model import SPOEntity
from rag.utils.config import rag_config as config
from rag.model import AtlasEmbeddings

class SchemaSearchUtil():
    def __init__(
        self,
        **kwargs
    ):
        self.embeddings = AtlasEmbeddings(
            model=config.embedding_model,
            api_url=config.embedding_url,
            api_base=config.embedding_base,
            api_key=config.embedding_key,
        )
        

    def retrieval_entity(self, mention_entity: SPOEntity, **kwargs) -> List[EntityData]:
        pass