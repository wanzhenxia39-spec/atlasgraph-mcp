"""Knowledge Graph Index.

Build a KG by extracting triplets, and leveraging the KG during query-time.

"""

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from llama_index.core.constants import GRAPH_STORE_KEY
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.data_structs.data_structs import KG
from llama_index.core.graph_stores.simple import SimpleGraphStore
from llama_index.core.graph_stores.types import GraphStore
from llama_index.core.indices.base import BaseIndex
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_KG_TRIPLET_EXTRACT_PROMPT
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.service_context import ServiceContext
from llama_index.core.storage.docstore.types import RefDocInfo
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.utils import get_tqdm_iterable
from llama_index.core import Settings
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from zhipuai import ZhipuAI
from rag.model.atlas_model import AtlasEmbeddings
from rag.utils.config import rag_config as config

# from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class AlignedKnowledgeGraphIndex(BaseIndex[KG]):
    """Knowledge Graph Index with Entity Alignment.

    Build a KG by extracting triplets, and leveraging the KG during query-time.

    Args:
        kg_triple_extract_template (BasePromptTemplate): The prompt to use for
            extracting triplets.
        max_triplets_per_chunk (int): The maximum number of triplets to extract.
        service_context (Optional[ServiceContext]): The service context to use.
        storage_context (Optional[StorageContext]): The storage context to use.
        graph_store (Optional[GraphStore]): The graph store to use.
        show_progress (bool): Whether to show tqdm progress bars. Defaults to False.
        include_embeddings (bool): Whether to include embeddings in the index.
            Defaults to False.
        max_object_length (int): The maximum length of the object in a triplet.
            Defaults to 128.
        kg_triplet_extract_fn (Optional[Callable]): The function to use for
            extracting triplets. Defaults to None.

    """

    index_struct_cls = KG

    def __init__(
        self,
        nodes: Optional[Sequence[BaseNode]] = None,
        index_struct: Optional[KG] = None,
        service_context: Optional[ServiceContext] = None,
        storage_context: Optional[StorageContext] = None,
        kg_triple_extract_template: Optional[BasePromptTemplate] = None,
        max_triplets_per_chunk: int = 10,
        include_embeddings: bool = False,
        show_progress: bool = False,
        max_object_length: int = 128,
        kg_triplet_extract_fn: Optional[Callable] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        # need to set parameters before building index in base class.
        self.include_embeddings = include_embeddings
        self.max_triplets_per_chunk = max_triplets_per_chunk
        self.kg_triple_extract_template = (
            kg_triple_extract_template or DEFAULT_KG_TRIPLET_EXTRACT_PROMPT
        )
        # NOTE: Partially format keyword extract template here.
        self.kg_triple_extract_template = (
            self.kg_triple_extract_template.partial_format(
                max_knowledge_triplets=self.max_triplets_per_chunk
            )
        )
        self._max_object_length = max_object_length
        self._kg_triplet_extract_fn = kg_triplet_extract_fn

        self.client = ZhipuAI(
            api_key=os.environ.get("ZHIPUAI_API_KEY")
        )

        super().__init__(
            nodes=nodes,
            index_struct=index_struct,
            service_context=service_context,
            storage_context=storage_context,
            show_progress=show_progress,
            **kwargs,
        )

        # TODO: legacy conversion - remove in next release
        if (
            len(self.index_struct.table) > 0
            and isinstance(self.graph_store, SimpleGraphStore)
            and len(self.graph_store._data.graph_dict) == 0
        ):
            logger.warning("Upgrading previously saved KG index to new storage format.")
            self.graph_store._data.graph_dict = self.index_struct.rel_map

    @property
    def graph_store(self) -> GraphStore:
        return self._graph_store

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        from llama_index.core.indices.knowledge_graph.retrievers import (
            KGRetrieverMode,
            KGTableRetriever,
        )

        if len(self.index_struct.embedding_dict) > 0 and "retriever_mode" not in kwargs:
            kwargs["retriever_mode"] = KGRetrieverMode.HYBRID

        return KGTableRetriever(self, **kwargs)  # type: ignore

    def _extract_triplets(self, text: str) -> List[Tuple[str, str, str]]:
        if self._kg_triplet_extract_fn is not None:
            return self._kg_triplet_extract_fn(text)
        else:
            return self._llm_extract_triplets(text)

    def _llm_extract_triplets(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract keywords from text."""
        response = Settings.llm.predict(
            self.kg_triple_extract_template,
            text=text,
        )
        return self._parse_triplet_response(
            response, max_length=self._max_object_length
        )

    @staticmethod
    def _parse_triplet_response(
        response: str, max_length: int = 128
    ) -> List[Tuple[str, str, str]]:
        knowledge_strs = response.strip().split("\n")
        results = []
        for text in knowledge_strs:
            if "(" not in text or ")" not in text or text.index(")") < text.index("("):
                # skip empty lines and non-triplets
                continue
            triplet_part = text[text.index("(") + 1 : text.index(")")]
            tokens = triplet_part.split(",")
            if len(tokens) != 3:
                continue

            if any(len(s.encode("utf-8")) > max_length for s in tokens):
                # We count byte-length instead of len() for UTF-8 chars,
                # will skip if any of the tokens are too long.
                # This is normally due to a poorly formatted triplet
                # extraction, in more serious KG building cases
                # we'll need NLP models to better extract triplets.
                continue

            subj, pred, obj = map(str.strip, tokens)
            if not subj or not pred or not obj:
                # skip partial triplets
                continue
            results.append((subj, pred, obj))
        return results

    def _similarity(self, vec1: List[float], vec2: List[float]) -> np.ndarray:
        return cosine_similarity([vec1], [vec2])[0][0]

    def _is_most_digits(self, s: str) -> bool:
        cnt = 0
        for ch in s:
            if ch.isdigit() or (
                ch
                in [
                    "一",
                    "二",
                    "三",
                    "四",
                    "五",
                    "六",
                    "七",
                    "八",
                    "九",
                    "十",
                    "百",
                    "千",
                    "万",
                    "亿",
                ]
            ):
                cnt += 1

        if cnt >= int(0.5 * len(s)):
            return True
        else:
            words = s.split(" ")
            for word in words:
                if word.lower() in [
                    "january",
                    "february",
                    "march",
                    "april",
                    "may",
                    "june",
                    "july",
                    "august",
                    "september",
                    "october",
                    "november",
                    "december",
                ]:
                    return True
            return False

    def _is_most_letters(self, s: str) -> bool:
        cnt = 0
        for ch in s:
            if "a" <= ch <= "z" or "A" <= ch <= "Z":
                cnt += 1

        if cnt >= int(0.5 * len(s)):
            return True
        else:
            return False

    def _is_too_long(self, s: str) -> bool:
        if self._is_most_letters(s):
            return len(s.split(" ")) >= 8
        else:
            return len(s) >= 10

    def _get_mapping(self, x: str) -> str:
        if x not in self.mapping:
            self.mapping[x] = x
        if self.mapping[x] == x:
            return x
        self.mapping[x] = self._get_mapping(self.mapping[x])
        return self.mapping[x]

    def _merge_mapping(self, a: str, b: str) -> None:
        f1 = self._get_mapping(a)
        f2 = self._get_mapping(b)
        if f1 != f2:
            self.mapping[f2] = f1

    def _llm_align(self, entity_list: str) -> List[str]:
        """
        Entity alignment using LLM
        Args:
            entity_list (str): a string with format '[entity1, entity2, entity3], [entity4, entity5], ...'
        """
        response = self.client.chat.completions.create(
            model="glm-4",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""你是一名数据处理助理。您的任务是识别列表中的重复实体，并决定应合并哪些实体。\n这些实体在格式或内容上可能略有不同，但本质上指的是同一件事。运用你的分析技能来确定重复项。\n以下是识别重复项的规则：\n1.明确指代同一现实世界对象的具体实体，应被视为重复。\n2.如果它指的是不同的数字、日期或产品，请不要合并结果。\n3.如果指的是同一事物的不同方面或者与同一实体有关的不同事件和概念，请不要合并。\n4. 较为抽象或广泛的概念，例如“国家”、“歌曲”、“书籍”、“公司”等，请不要合并。\n以下是要处理的实体列表：\n{entity_list}\n请首先给出推理过程，最后给出合并后的结果，以格式如下：```\nFinal Result: \n[['实体1', '实体2'], ['实体3', '实体4'], ...]\n```，每个小列表中包含的是重复实体，不重复的实体无需指出，不明确的实体请保持独立。""",
                        }
                    ],
                }
            ],
        )
        try:
            pattern = r"Final Result:(.*?)\]\]"
            result = (
                re.findall(
                    pattern, response.choices[0].message.content, flags=re.DOTALL
                )[0]
                + "]]"
            )
            result = (
                result.replace("['", '["')
                .replace("',", '",')
                .replace(", '", ', "')
                .replace("']", '"]')
            )
            result = eval(result)
            # print(result)
            return result
        except:
            return []

    def _deduplicate(self, all_entity: List[str], importance: Dict[str, int]) -> None:
        """
        Entity Alignment
        """
        for item in all_entity:
            self.mapping[item] = item
        entity = []
        for item in all_entity:
            if not self._is_most_digits(item):
                entity.append(item)
        # embed_model = self._service_context.embed_model
        # embed_outputs = embed_model.get_text_embedding_batch(entity)
        model = AtlasEmbeddings(
            model=config.embedding_model,
            api_base=config.embedding_base,
            api_url=config.embedding_url,
            api_key=config.embedding_key,
        )
        embed_outputs = model.get_text_embedding_batch(entity)
        embed_outputs = [np.array(embed) for embed in embed_outputs]
        important_entries = {}
        sim_score = {}
        for i in range(len(entity)):
            for j in range(i + 1, len(entity)):
                # overlap = pylcs.lcs_sequence_length(entity[i], entity[j])
                score = self._similarity(embed_outputs[i], embed_outputs[j])
                # if sim > 0.86 or (sim > 0.6 and 1.0 * overlap >= 0.6 * max(len(entity[i]), len(entity[j]))):
                #     self._merge_mapping(entity[i], entity[j])
                if 0.75 <= score < 0.83:
                    if self._is_too_long(entity[i]) or self._is_too_long(entity[j]):
                        continue
                    if entity[i] not in important_entries:
                        important_entries[entity[i]] = 0
                    if entity[j] not in important_entries:
                        important_entries[entity[j]] = 0
                    important_entries[entity[i]] += 1
                    important_entries[entity[j]] += 1
                    sim_score[entity[i] + "<SPLIT>" + entity[j]] = score
                elif score >= 0.83:
                    self._merge_mapping(entity[i], entity[j])

        union_list = {}
        for entry, score in sim_score.items():
            entities = entry.split("<SPLIT>")
            entity0, entity1 = entities[0], entities[1]
            if important_entries[entity0] > important_entries[entity1] or (
                important_entries[entity0] == important_entries[entity1]
                and entity0 < entity1
            ):
                union_name = entity0
                if union_name not in union_list:
                    union_list[union_name] = [union_name]
                union_list[union_name].append(entity1)
            else:
                union_name = entity1
                if union_name not in union_list:
                    union_list[union_name] = [union_name]
                union_list[union_name].append(entity0)

        send_list = []
        for _, list in union_list.items():
            send_list += list
            if len(send_list) >= 30:
                result = self._llm_align(str(send_list))
                send_list = []
                for res in result:
                    for i in range(1, len(res)):
                        self._merge_mapping(res[i], res[0])

        merged_result = {}
        for entry, _ in self.mapping.items():
            merged = self._get_mapping(entry)
            if merged not in merged_result:
                merged_result[merged] = []
            merged_result[merged].append(entry)

        refined_merged_result = {}
        for _, merged_list in merged_result.items():
            repr = ""
            max_importance = 0
            for item in merged_list:
                if item in importance and importance[item] > max_importance:
                    repr = item
                    max_importance = importance[item]
            for item in merged_list:
                self.mapping[item] = repr
            refined_merged_result[repr] = merged_list

        # with open('merge_result.json', 'w') as f:
        #     json.dump(refined_merged_result, f, ensure_ascii=False, indent=4)

    def _build_index_from_nodes(
        self, nodes: Sequence[BaseNode], **build_kwargs: Any
    ) -> KG:
        """Build the index from nodes."""
        # do simple concatenation
        index_struct = self.index_struct_cls()
        nodes_with_progress = get_tqdm_iterable(
            nodes, self._show_progress, "Processing nodes"
        )
        all_entities = {}
        all_triplets = []
        importance = {}
        for n in nodes_with_progress:
            triplets = self._extract_triplets(
                n.get_content(metadata_mode=MetadataMode.LLM)
            )
            all_triplets.append(triplets)
            logger.debug(f"> Extracted triplets: {triplets}")
            for triplet in triplets:
                if triplet[0] not in all_entities:
                    all_entities[triplet[0]] = triplet[0]
                if triplet[2] not in all_entities:
                    all_entities[triplet[2]] = triplet[2]
                if triplet[0] not in importance:
                    importance[triplet[0]] = 0
                if triplet[2] not in importance:
                    importance[triplet[2]] = 0
                importance[triplet[0]] += 1
                importance[triplet[2]] += 1

        self.mapping = {}  # entity mapping after alignment
        self._deduplicate(list(all_entities.keys()), importance)

        for n, triplets in zip(nodes_with_progress, all_triplets):
            for i in range(len(triplets)):
                triplet = triplets[i]
                subj, rel, obj = triplet
                subj = self._get_mapping(subj)
                obj = self._get_mapping(obj)
                triplet = (subj, rel, obj)
                triplets[i] = triplet
                self.upsert_triplet(triplet)
                index_struct.add_node([subj, obj], n)

            if self.include_embeddings:
                triplet_texts = [str(t) for t in triplets]

                embed_model = Settings.embed_model
                embed_outputs = embed_model.get_text_embedding_batch(
                    triplet_texts, show_progress=self._show_progress
                )
                for rel_text, rel_embed in zip(triplet_texts, embed_outputs):
                    index_struct.add_to_embedding_dict(rel_text, rel_embed)

        return index_struct

    def _insert(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        """Insert a document."""
        for n in nodes:
            triplets = self._extract_triplets(
                n.get_content(metadata_mode=MetadataMode.LLM)
            )
            logger.debug(f"Extracted triplets: {triplets}")
            for triplet in triplets:
                subj, _, obj = triplet
                triplet_str = str(triplet)
                self.upsert_triplet(triplet)
                self._index_struct.add_node([subj, obj], n)
                if (
                    self.include_embeddings
                    and triplet_str not in self._index_struct.embedding_dict
                ):
                    rel_embedding = Settings.embed_model.get_text_embedding(triplet_str)
                    self._index_struct.add_to_embedding_dict(triplet_str, rel_embedding)

    def upsert_triplet(self, triplet: Tuple[str, str, str]) -> None:
        """Insert triplets.

        Used for manual insertion of KG triplets (in the form
        of (subject, relationship, object)).

        Args:
            triplet (str): Knowledge triplet

        """
        self._graph_store.upsert_triplet(*triplet)

    def add_node(self, keywords: List[str], node: BaseNode) -> None:
        """Add node.

        Used for manual insertion of nodes (keyed by keywords).

        Args:
            keywords (List[str]): Keywords to index the node.
            node (Node): Node to be indexed.

        """
        self._index_struct.add_node(keywords, node)
        self._docstore.add_documents([node], allow_update=True)

    def upsert_triplet_and_node(
        self, triplet: Tuple[str, str, str], node: BaseNode
    ) -> None:
        """Upsert KG triplet and node.

        Calls both upsert_triplet and add_node.
        Behavior is idempotent; if Node already exists,
        only triplet will be added.

        Args:
            keywords (List[str]): Keywords to index the node.
            node (Node): Node to be indexed.

        """
        subj, _, obj = triplet
        self.upsert_triplet(triplet)
        self.add_node([subj, obj], node)

    def _delete_node(self, node_id: str, **delete_kwargs: Any) -> None:
        """Delete a node."""
        raise NotImplementedError("Delete is not supported for KG index yet.")

    @property
    def ref_doc_info(self) -> Dict[str, RefDocInfo]:
        """Retrieve a dict mapping of ingested documents and their nodes+metadata."""
        node_doc_ids_sets = list(self._index_struct.table.values())
        node_doc_ids = list(set().union(*node_doc_ids_sets))
        nodes = self.docstore.get_nodes(node_doc_ids)

        all_ref_doc_info = {}
        for node in nodes:
            ref_node = node.source_node
            if not ref_node:
                continue

            ref_doc_info = self.docstore.get_ref_doc_info(ref_node.node_id)
            if not ref_doc_info:
                continue

            all_ref_doc_info[ref_node.node_id] = ref_doc_info
        return all_ref_doc_info

    def get_networkx_graph(self, limit: int = 100) -> Any:
        """Get networkx representation of the graph structure.

        Args:
            limit (int): Number of starting nodes to be included in the graph.

        NOTE: This function requires networkx to be installed.
        NOTE: This is a beta feature.

        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "Please install networkx to visualize the graph: `pip install networkx`"
            )

        g = nx.Graph()
        subjs = list(self.index_struct.table.keys())

        # add edges
        rel_map = self._graph_store.get_rel_map(subjs=subjs, depth=1, limit=limit)

        added_nodes = set()
        for keyword in rel_map:
            for path in rel_map[keyword]:
                subj = keyword
                for i in range(0, len(path), 2):
                    if i + 2 >= len(path):
                        break

                    if subj not in added_nodes:
                        g.add_node(subj)
                        added_nodes.add(subj)

                    rel = path[i + 1]
                    obj = path[i + 2]

                    g.add_edge(subj, obj, label=rel, title=rel)
                    subj = obj
        return g

    @property
    def query_context(self) -> Dict[str, Any]:
        return {GRAPH_STORE_KEY: self._graph_store}


# legacy
GPTKnowledgeGraphIndex = AlignedKnowledgeGraphIndex
