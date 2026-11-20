from llama_index.core.data_structs.data_structs import IndexStruct, IndexDict, KG
from llama_index.core.storage.index_store.types import BaseIndexStore
from llama_index.core.schema import BaseNode
from typing import Optional, List, Dict, Set, Any, Iterable
from dataclasses import dataclass, field
from rag.utils.utils import to_legal_cypher_string
from .graph_store import AtlasGraphStore

class FooDict:
    def __getitem__(self, key: Any) -> Any:
        return key

    def __setitem__(self, key: Any, value: Any) -> Any:
        pass

    def __delitem__(self, key: Any) -> None:
        pass

    def __iter__(self) -> Iterable:
        return iter([])

    def values(self) -> Iterable:
        return self.__iter__()

    def keys(self) -> Iterable:
        return self.__iter__()


@dataclass
class AtlasIndexDict(IndexDict):
    nodes_dict: FooDict = field(default_factory=FooDict)  # type: ignore[assignment]

    def add_node(
        self,
        node: BaseNode,
        text_id: Optional[str] = None,
    ) -> str:
        return ""


@dataclass
class AtlasKG(KG):

    graph_store: Optional[AtlasGraphStore] = None
    keyword_table: Dict[str, Set[str]] = field(default_factory=dict)
    text_table: Dict[str, Set[str]] = field(default_factory=dict)

    def add_node(self, keywords: List[str], node: BaseNode) -> None:
        """Add text to table."""
        node_id = node.node_id
        for keyword in keywords:
            # keyword = to_legal_cypher_string(keyword)
            if keyword not in self.keyword_table:
                self.keyword_table[keyword] = set()
            self.keyword_table[keyword].add(node_id)
        if node_id not in self.text_table:
            self.text_table[node_id] = set()
        self.text_table[node_id].update(keywords)

    def search_node_by_keyword(self, keyword: str) -> List[str]:
        """Search for nodes by keyword."""
        keyword = to_legal_cypher_string(keyword)
        if keyword in self.keyword_table:
            return list(self.keyword_table[keyword])
        else:
            return self.graph_store.get_chunk_ids([keyword])  # type: ignore[union-attr]


class AtlasKGWrapper:

    def __init__(self, graph_store: AtlasGraphStore):
        self.graph_store = graph_store

    def __call__(self, *args: Any, **kwds: Any) -> AtlasKG:
        return AtlasKG(graph_store=self.graph_store)


class AtlasIndexStore(BaseIndexStore):

    def __init__(
        self,
        allow_update: bool = True,
    ) -> None:
        self._allow_update = allow_update

    def add_index_struct(self, index_struct: IndexStruct) -> None:
        pass

    def index_structs(self) -> List[IndexStruct]:
        return []

    def delete_index_struct(self, key: str) -> None:
        pass

    def get_index_struct(
        self, struct_id: Optional[str] = None
    ) -> Optional[IndexStruct]:
        return None
