from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.retrievers import BaseRetriever
from typing import List, Tuple, Dict


class GraphBaseRetriever(BaseRetriever):
    def retrieve_with_graph(
        self, query_bundle: QueryBundle
    ) -> Tuple[List[NodeWithScore], Dict[str, List[Dict[str, str]]]]:
        raise NotImplementedError
