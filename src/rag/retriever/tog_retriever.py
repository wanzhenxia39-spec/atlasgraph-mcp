import re
import random
from typing import List, Tuple, Optional
from collections import Counter, defaultdict


from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import VectorStoreIndex
from llama_index.core.indices.keyword_table.utils import extract_keywords_given_response
from llama_index.core.prompts.default_prompts import (
    DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE,
)
from llama_index.core.vector_stores.types import VectorStoreQuery
from driver3.service.AtlasGraph import AtlasGraph
from driver3.service.ConnectionConfiguration import ConnectionConfiguration
from rag.utils.log import rag_logger
from rag.model.atlas_model import BaseEmbedding
from llama_index.core import Settings
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import LLM
from typing import Dict, Union, Any
from rag.utils.tog_prompts import (
    extract_relation_prompt_history,
    score_entity_candidates_prompt_wiki,
    prompt_evaluate_wiki,
)
from driver3.model.CypherResponse import CypherResponse
from llama_index.core.indices.base import BaseIndex
from llama_index.core.data_structs.data_structs import KG


class ToGRetriever(BaseRetriever):
    DEFAULT_NODE_SCORE = 1.0

    def __init__(
        self,
        vector_index: VectorStoreIndex,
        kg_index: BaseIndex[KG],
        max_length: int = 256,
        temperature_exploration: float = 0.4,
        temperature_reasoning: float = 0,
        depth: int = 3,
        max_width: int = 10,
        min_score: float = 0.35,
        # remove_unnecessary_rel:bool=True,
        num_retain_entity: int = 5,
        prune_tools: str = "llm",
        vector_prune: bool = True,
        num_chunks_per_query: int = 20,
        username: str = "root",
        password: str = "",
        url: str = "127.0.0.1",
        database: str = "test",
        keyword: str = "keyword",
        llm: Optional[LLM] = None,
        embed_model: Optional[BaseEmbedding] = None,
    ) -> None:
        self.vector_index = vector_index
        self.kg_index = kg_index
        self.max_length = max_length
        self.temperature_exploration = temperature_exploration
        self.temperature_reasoning = temperature_reasoning
        self.depth = depth
        self.max_width = max_width
        self.min_score = min_score
        self.num_retain_entity = num_retain_entity
        self.prune_tools = prune_tools
        self.vector_prune = vector_prune
        self._num_chunks_per_query = num_chunks_per_query
        self.username = username
        self.password = password
        self.url = url
        self.databse = database
        self.keyword = keyword

        self._query_keyword_extract_template = DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE

        self.graph_client = GraphClient(
            url=url,
            username=username,
            password=password,
            database=database,
            keyword_property=keyword,
        )
        self.llm = llm or Settings.llm
        self.embed_model = embed_model or Settings.embed_model
        super(ToGRetriever, self).__init__()

    def _get_keywords(self, query_str: str) -> List[str]:
        """Extract keywords."""
        response = self.llm.predict(
            self._query_keyword_extract_template,
            max_keywords=self.max_width,
            question=query_str,
        )
        keywords = extract_keywords_given_response(
            response, start_token="KEYWORDS:", lowercase=False
        )
        return list(keywords)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        if self.vector_index._vector_store.is_embedding_query:
            if query_bundle.embedding is None and len(query_bundle.embedding_strs) > 0:
                query_bundle.embedding = (
                    self.embed_model.get_agg_embedding_from_queries(
                        query_bundle.embedding_strs
                    )
                )
        query_result = self.vector_index.vector_store.query(
            VectorStoreQuery(
                query_embedding=query_bundle.embedding,
                similarity_top_k=self.max_width,
            )
        )
        query_result.nodes = self.vector_index._docstore.get_nodes(query_result.ids)  # type: ignore[arg-type]

        question = query_bundle.query_str
        rag_logger.info("Question: " + question)

        cluster_chain_of_entities = []
        keywords = self._get_keywords(question)
        # TODO: add query keywords after vector prune
        # keywords.extend(self.kg_index.docstore.get_keywords(query_result.ids))
        entity_keywords = keywords

        pre_relations = []

        for depth in range(1, self.depth + 1):
            total_candidates = []
            total_scores = []
            total_relations = []
            total_entities = []
            total_head = []

            current_entity_relations_list = self.relation_search_node_prune(
                entity_keywords,
                pre_relations,
                question,
                cluster_chain_of_entities,
            )

            for entity in current_entity_relations_list:
                value_flag = False

                if len(entity["targets"]) == 0:
                    continue
                    # TODO: use vector similarity instead of random
                elif len(entity["targets"]) >= 50:
                    # TODO: use vector similarity instead of random
                    indices = random.sample(range(len(entity["targets"])), 30)
                    entity["targets"] = [entity["targets"][i] for i in indices]

                scores, entity_candidates_keyword = self.entity_score(
                    question,
                    cluster_chain_of_entities,
                    entity["entity"],
                    entity["targets"],
                    entity["score"],
                    entity["relation"],
                )

                (
                    total_candidates,
                    total_scores,
                    total_relations,
                    total_entities,
                    total_head,
                ) = self.update_history(
                    entity_candidates_keyword,
                    entity,
                    scores,
                    total_candidates,
                    total_scores,
                    total_relations,
                    total_entities,
                    total_head,
                    value_flag,
                )

            if len(total_candidates) == 0:
                rag_logger.info("out of candidates")
                sorted_nodes_with_scores = self.cluster_chain_to_node_with_score(
                    cluster_chain_of_entities
                )
                if len(sorted_nodes_with_scores) == 0:
                    sorted_nodes_with_scores = [
                        NodeWithScore(node=node, score=similarity)
                        for node, similarity in zip(
                            query_result.nodes, query_result.similarities  # type: ignore[arg-type]
                        )
                    ]
                return sorted_nodes_with_scores

            (
                flag,
                chain_of_entities,
                entity_keywords_pruned,
                pre_relations,
                pre_heads,
                candidates,
            ) = self.entity_prune(
                total_relations,
                total_candidates,
                total_entities,
                total_head,
                total_scores,
            )

            cluster_chain_of_entities.append(chain_of_entities)
            if flag:
                stop, results = self.reasoning(question, cluster_chain_of_entities)
                if stop:
                    rag_logger.info("ToG stoped at depth %d." % depth)
                    sorted_nodes_with_scores = self.cluster_chain_to_node_with_score(
                        cluster_chain_of_entities
                    )
                    return sorted_nodes_with_scores
                else:
                    entity_keywords = candidates

            else:
                sorted_nodes_with_scores = self.cluster_chain_to_node_with_score(
                    cluster_chain_of_entities
                )
                return sorted_nodes_with_scores

        rag_logger.info("Out of depth")
        sorted_nodes_with_scores = self.cluster_chain_to_node_with_score(
            cluster_chain_of_entities
        )
        if len(sorted_nodes_with_scores) == 0:
            sorted_nodes_with_scores = [
                NodeWithScore(node=node, score=similarity)
                for node, similarity in zip(
                    query_result.nodes, query_result.similarities  # type: ignore[arg-type]
                )
            ]
        return sorted_nodes_with_scores

    def clean_relations(
        self, string: str, entity_id: str, head_relations: list
    ) -> Tuple[bool, Union[str, List[Dict[str, Any]]]]:
        pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
        relations = []
        for match in re.finditer(pattern, string):
            relation = match.group("relation").strip("\" '")
            if ";" in relation:
                continue
            score = match.group("score")
            if not relation or not score:
                return False, "output uncompleted.."
            try:
                score = float(score)
            except ValueError:
                return False, "Invalid score"
            if relation in head_relations:
                relations.append(
                    {
                        "entity": entity_id,
                        "relation": relation,
                        "score": score,
                        "head": True,
                    }
                )
            else:
                relations.append(
                    {
                        "entity": entity_id,
                        "relation": relation,
                        "score": score,
                        "head": False,
                    }
                )
        if not relations:
            return False, "No relations found"
        return True, relations

    def clean_relations_prune(
        self, string: str, entity_id: str, head_relations: dict, tail_relations: dict
    ) -> Tuple[bool, Union[str, List[Dict[str, Any]]]]:
        pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
        relations = []
        for match in re.finditer(pattern, string):
            relation = match.group("relation").strip("\" '")
            if ";" in relation:
                continue
            score = match.group("score")
            if not relation or not score:
                return False, "output uncompleted.."
            try:
                score = float(score)
            except ValueError:
                return False, "Invalid score"
            if score > self.min_score:
                if relation in head_relations.keys():
                    relations.append(
                        {
                            "entity": entity_id,
                            "relation": relation,
                            "targets": head_relations[relation],
                            "score": score,
                            "head": True,
                        }
                    )
                elif relation in tail_relations.keys():
                    relations.append(
                        {
                            "entity": entity_id,
                            "relation": relation,
                            "targets": tail_relations[relation],
                            "score": score,
                            "head": False,
                        }
                    )
        if not relations:
            return False, "No relations found"

        if len(relations) > self.max_width:
            sorted_relations = sorted(
                [y for y in relations], key=lambda x: x["score"], reverse=True
            )
            rag_logger.info("reducing collected relations to " + str(self.max_width))
            return True, sorted_relations[: self.max_width]
        return True, relations

    def cosSimilarity(self, v1: list, v2: list) -> float:
        if len(v1) != len(v2):
            raise Exception("Vector length mismatch", (len(v1), len(v2)))
        result: float = 0
        n = 0
        d1: float = 0
        d2: float = 0
        for i in range(len(v1)):
            n += v1[i] * v2[i]
            d1 += v1[i] ** 2
            d2 += v2[i] ** 2
        d1 = d1**0.5
        d2 = d2**0.5
        result = n / (d1 * d2)
        return result

    def relation_embedding_prune(
        self, question: str, total_relations: list, embeddings: BaseEmbedding
    ) -> List[Tuple[str, float]]:
        if len(total_relations) > self.max_width:
            question_embedding = embeddings._get_query_embedding(question)
            relations_embedding = embeddings._get_text_embeddings(total_relations)
            relation_score_pair = []
            for i in range(len(total_relations)):
                relation_score_pair.append(
                    (
                        total_relations[i],
                        self.cosSimilarity(question_embedding, relations_embedding[i]),
                    )
                )

            def second(l: Tuple[str, float]) -> float:
                return l[1]

            relation_score_pair.sort(key=second)
            relation_score_pair = [x[0] for x in relation_score_pair[-self.max_width :]]
            return relation_score_pair
        else:
            return total_relations

    def relation_search_node_prune(
        self,
        entity_keywords: List[str],
        pre_relations: list,
        question: str,
        cluster_chain_of_entities: list,
    ) -> List[Dict]:
        current_entity_relations_list = []

        for i, entity in enumerate(entity_keywords):
            cypher_query_head = f'match (n1)-[r]->(n2) where n1.keyword="{entity}" return n1.keyword, r.keyword,n2.keyword'
            cypher_query_tail = f'match (n1)<-[r]-(n2) where n1.keyword="{entity}" return n1.keyword, r.keyword,n2.keyword'
            head_relations_node = self.graph_client.cypher_exec(
                cypher_query_head
            ).result_set["table"]
            tail_relations_node = self.graph_client.cypher_exec(
                cypher_query_tail
            ).result_set["table"]
            head_relations_node = [
                row
                for row in head_relations_node
                if all(map(lambda item: item is not None, row))
            ]
            tail_relations_node = [
                row
                for row in tail_relations_node
                if all(map(lambda item: item is not None, row))
            ]
            head_relations = defaultdict(list)
            tail_relations = defaultdict(list)
            for r in head_relations_node:
                if r[1] not in pre_relations:
                    head_relations[r[1]].append(r[2])
            for r in tail_relations_node:
                if r[1] not in pre_relations:
                    tail_relations[r[1]].append(r[2])

            total_relations = list(head_relations.keys()) + list(tail_relations.keys())
            if total_relations:
                tmp_cluster = "\n".join(
                    [
                        "\n".join(["\n".join([",".join(l3) for l3 in l2]) for l2 in l1])
                        for l1 in cluster_chain_of_entities
                    ]
                )
                if not cluster_chain_of_entities:
                    tmp_cluster = "None"

                if self.vector_prune:
                    total_relations = self.relation_embedding_prune(
                        question,
                        total_relations,
                        self.embed_model,
                    )

                result = self.llm.complete(
                    extract_relation_prompt_history
                    + f"{question}\nKnown information: {tmp_cluster}\nTopic Entity: {entity}\nRelations: {'; '.join(total_relations)}\nA: "
                ).text

                flag, retrieve_relations_with_scores = self.clean_relations_prune(
                    result, entity, head_relations, tail_relations
                )

                if flag:
                    current_entity_relations_list.extend(retrieve_relations_with_scores)

                else:
                    pass
            else:
                pass
        return current_entity_relations_list

    def clean_scores(self, string: str, entity_candidates: list) -> List[float]:
        scores = re.findall(r"\d+\.\d+", string)
        scores = [float(number) for number in scores]
        if len(scores) == len(entity_candidates):
            return scores
        elif len(scores) > len(entity_candidates):
            rag_logger.info(
                "Number of scores returned greater than number of entity candidates\nRetrieving first k scores"
            )
            return scores[: len(entity_candidates)]
        else:
            rag_logger.info("All entities are created equal.")
            return [1 / len(entity_candidates)] * len(entity_candidates)

    def all_zero(self, topn_scores: list) -> bool:
        return all(score == 0 for score in topn_scores)

    def entity_score(
        self,
        question: str,
        cluster_chain_of_entities: list,
        current_entity: str,
        candidates_keyword: list,
        score: float,
        relation: str,
    ) -> Tuple[List[float], List[str]]:
        if len(candidates_keyword) == 1:
            return [score], candidates_keyword
        if len(candidates_keyword) == 0:
            return [0.0], candidates_keyword

        tmp_cluster = "\n".join(
            [
                "\n".join(["\n".join([",".join(l3) for l3 in l2]) for l2 in l1])
                for l1 in cluster_chain_of_entities
            ]
        )
        if not cluster_chain_of_entities:
            tmp_cluster = "None"

        result = self.llm.complete(
            score_entity_candidates_prompt_wiki.format(
                question, tmp_cluster, current_entity, relation
            )
            + "; ".join(candidates_keyword)
            + "\nScore: "
        ).text

        entity_scores = self.clean_scores(result, candidates_keyword)
        if self.all_zero(entity_scores):
            return [1 / len(candidates_keyword) * score] * len(
                candidates_keyword
            ), candidates_keyword
        else:
            return [float(x) * score for x in entity_scores], candidates_keyword

    def update_history(
        self,
        entity_candidates_keyword: list,
        entity: dict,
        scores: list,
        total_candidates: list,
        total_scores: list,
        total_relations: list,
        total_entities: list,
        total_head: list,
        value_flag: bool,
    ) -> Tuple[List[str], List[float], List[str], List[str], List[bool]]:
        if value_flag:
            scores = [1 / len(entity_candidates_keyword) * entity["score"]]
        candidates_relation = [entity["relation"]] * len(entity_candidates_keyword)
        topic_entities = [entity["entity"]] * len(entity_candidates_keyword)
        head_num = [entity["head"]] * len(entity_candidates_keyword)
        total_candidates.extend(entity_candidates_keyword)
        total_scores.extend(scores)
        total_relations.extend(candidates_relation)
        total_entities.extend(topic_entities)
        total_head.extend(head_num)
        return (
            total_candidates,
            total_scores,
            total_relations,
            total_entities,
            total_head,
        )  # type: ignore[return-value]

    def entity_prune(
        self,
        total_relations: list,
        total_candidates: list,
        total_entities: list,
        total_head: list,
        total_scores: list,
    ) -> Tuple[bool, list, List[str], List[str], List[bool], List[str]]:
        zipped = list(
            zip(
                total_relations,
                total_candidates,
                total_entities,
                total_head,
                total_scores,
            )
        )
        sorted_zipped = sorted(
            [y for y in zipped if y[4] >= self.min_score],
            key=lambda x: x[4],
            reverse=True,
        )
        (
            sorted_relations,
            sorted_candidates,
            sorted_topic_entities,
            sorted_head,
            sorted_scores,
        ) = (
            [x[0] for x in sorted_zipped],
            [x[1] for x in sorted_zipped],
            [x[2] for x in sorted_zipped],
            [x[3] for x in sorted_zipped],
            [x[4] for x in sorted_zipped],
        )

        if len(sorted_zipped) > self.max_width:
            relations, candidates, topics, heads, scores = (
                sorted_relations[: self.max_width],
                sorted_candidates[: self.max_width],
                sorted_topic_entities[: self.max_width],
                sorted_head[: self.max_width],
                sorted_scores[: self.max_width],
            )
        else:
            relations, candidates, topics, heads, scores = (
                sorted_relations,
                sorted_candidates,
                sorted_topic_entities,
                sorted_head,
                sorted_scores,
            )
        merged_list = list(zip(relations, candidates, topics, heads, scores))
        filtered_list = [
            (rel, ent, top, hea, score)
            for rel, ent, top, hea, score in merged_list
            if score != 0
        ]
        if len(filtered_list) == 0:
            return False, [], [], [], [], []
        relations, candidates, tops, heads, scores = map(list, zip(*filtered_list))

        cluster_chain_of_entities = [[]]
        for i, is_head in enumerate(heads):
            if is_head:
                cluster_chain_of_entities[0].append(
                    (tops[i], relations[i], candidates[i])
                )
            else:
                cluster_chain_of_entities[0].append(
                    (candidates[i], relations[i], tops[i])
                )

        return True, cluster_chain_of_entities, tops, relations, heads, candidates

    def reasoning(
        self, question: str, cluster_chain_of_entities: list
    ) -> Tuple[bool, str]:
        prompt = prompt_evaluate_wiki + question
        chain_prompt = "\n".join(
            [
                ", ".join([str(x) for x in chain])
                for sublist in cluster_chain_of_entities
                for chain in sublist
            ]
        )
        prompt += "\nKnowledge Triplets: " + chain_prompt + "A: "

        response = self.llm.complete(prompt).text

        result = self.extract_answer(response)
        if self.if_true(result):
            return True, response
        else:
            return False, response

    def extract_answer(self, text: str) -> str:
        start_index = text.find("{")
        end_index = text.find("}")
        if start_index != -1 and end_index != -1:
            return text[start_index + 1 : end_index].strip()
        else:
            return ""

    def if_true(self, prompt: str) -> bool:
        if prompt.lower().strip().replace(" ", "") == "yes":
            return True
        return False

    def cluster_chain_to_node_with_score(self, chain: list) -> List[NodeWithScore]:
        flat_chain = [l4 for l1 in chain for l2 in l1 for l3 in l2 for l4 in l3]
        nested_node_ids = [
            self.kg_index._index_struct.search_node_by_keyword(keyword)
            for keyword in flat_chain
        ]
        chunk_indices_count = Counter([_id for ids in nested_node_ids for _id in ids])
        sorted_chunk_indices = sorted(
            chunk_indices_count.keys(),
            key=lambda x: chunk_indices_count[x],
            reverse=True,
        )
        sorted_chunk_indices = sorted_chunk_indices[: self._num_chunks_per_query]
        sorted_nodes = self.kg_index._docstore.get_nodes(sorted_chunk_indices)
        sorted_nodes_with_scores = [
            NodeWithScore(node=node, score=self.DEFAULT_NODE_SCORE)
            for node in sorted_nodes
        ]

        return sorted_nodes_with_scores


class GraphClient:
    def __init__(
        self,
        url: str,
        username: str = "root",
        password: str = "",
        depth: int = 4,
        limit: int = 20,
        database: str = "test",
        keyword_property: str = "keyword",
    ) -> None:

        self.url = url
        self.username = username
        self.password = password
        self.depth = depth
        self.limit = limit
        self.driver = AtlasGraph(ConnectionConfiguration(url, username, password))
        self.database = database
        self.keyword_property = keyword_property

    def cypher_exec(self, command: str = "") -> CypherResponse:
        tmp = command.split()
        if tmp[-1].isdigit():
            result = self.driver.use(self.database).exec(command)
        else:
            result = self.driver.use(self.database).exec(command + " limit 5000")
        return result
