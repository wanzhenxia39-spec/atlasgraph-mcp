from .base_tool import BaseTool
from rag.common.task import Task
from rag.common.context import Context
from rag.common.logic_node_parser import GetSPONode, SPOEntity, SPORelation
from typing import Generator, List, Tuple, Dict, Any, Optional, cast
from rag.prompt.rewrite_prompt import RewriteSubQuery
from rag.prompt.choice_label_prompt import ChoiceLabelPrompt
from rag.prompt.judge_equal_prompt import JudgeEqualPrompt
from rag.prompt.reasoning_prompt import ReasoningPrompt
from rag.utils.log import rag_logger as log
from llama_index.core.vector_stores.types import VectorStoreQuery
from rag.index.atlas_rag_index import AtlasRAGIndex
from rag.utils.utils import to_legal_cypher_string
from llama_index.core.schema import TextNode


class RetrieverTool(BaseTool):
    def __init__(
        self, rag_index: AtlasRAGIndex, similarity_top_k: int, **kwargs: Any
    ) -> None:
        super(RetrieverTool, self).__init__(rag_index, **kwargs)
        self.similarity_top_k = similarity_top_k

    def match_entity(
        self,
        label: str,
        keyword: str,
    ) -> Optional[Dict[str, str]]:
        embedding = self.embedding_model.get_text_embedding(keyword or "None")
        flag = True
        if label in self.graph_store.keyword_vector_stores:
            std_label = label
            similarity_result = self.graph_store.keyword_vector_stores[label].query(
                VectorStoreQuery(
                    query_embedding=embedding,
                    similarity_top_k=self.similarity_top_k * 50,
                )
            )
            ids: List[str] = similarity_result.ids  # type: ignore[assignment]
            std_keyword = ids[0] if len(ids) != 0 else None
            flag = std_keyword is None
        if flag:
            similarity_entity_list_with_score = []
            for (
                label,
                vector_store,
            ) in self.graph_store.keyword_vector_stores.items():
                similarity_result = vector_store.query(
                    VectorStoreQuery(
                        query_embedding=embedding,
                        similarity_top_k=self.similarity_top_k * 50,
                    )
                )
                for keyword, score in zip(similarity_result.ids, similarity_result.similarities):  # type: ignore[arg-type]
                    similarity_entity_list_with_score.append(
                        {
                            "label": label,
                            "keyword": keyword,
                            "score": score,
                        }
                    )
            similarity_entity_list_with_score = list(
                sorted(
                    similarity_entity_list_with_score,
                    key=lambda x: x["score"],  # type: ignore
                    reverse=True,
                )
            )
            if len(similarity_entity_list_with_score) != 0:
                std_label = similarity_entity_list_with_score[0]["label"]  # type: ignore
                std_keyword = similarity_entity_list_with_score[0]["keyword"]  # type: ignore
            else:
                std_keyword = None
        if std_keyword is None:
            return None
        # yield f'从图库中匹配到问题关联起点: {{"label": {std_label}, "keyword": {std_keyword}}}\n'
        cypher = f'MATCH (n: {std_label}) WHERE n.keyword = "{to_legal_cypher_string(std_keyword)}" RETURN n.description;'
        description = self.database.exec(cypher).result_set["table"][0][0]
        judge_euqual_prompt = JudgeEqualPrompt()
        if judge_euqual_prompt.get_result(
            self.llm,
            input1={"label": label, "keyword": keyword},
            input2={"label": std_label, "keyword": f"keyword: {std_keyword} description: {description}"},
        ):
            return {
                "label": std_label,
                "keyword": std_keyword,
                "description": description,
            }
        return None

    def match_relation(
        self, src_label: str, src_keyword: str, label: str, keyword: str, dst_label: str
    ) -> Dict[str, Optional[str]]:
        if label in self.schema.edges:
            std_label = label
            cypher = f'MATCH (n1: {src_label})-[r: {label}]-(n2: {dst_label}) WHERE n1.keyword = "{to_legal_cypher_string(src_keyword)}" RETURN r.keyword'
            keywords = [
                row[0] for row in self.database.exec(cypher).result_set["table"]
            ]
            if len(keywords) == 0:
                return {"label": label, "keyword": None}
            choice_label_prompt = ChoiceLabelPrompt()
            std_keyword: str = choice_label_prompt.get_result(
                self.llm, input=keyword, labels=keywords, is_vertex=False
            )
            if std_keyword == "Relation":
                return {"label": std_label, "keyword": None}
            else:
                return {"label": std_label, "keyword": std_keyword}
        else:
            return {"label": None, "keyword": None}

    def execute(
        self, query: str, task: Task, context: Context
    ) -> Generator[str, None, None]:
        logic_form: GetSPONode = task.arguments["logic_form_node"]
        sub_query: str = task.arguments["query"]
        dep_tasks = task.parents
        history_context: List[str] = []
        yield f"原始{task.name}: {sub_query}\n"
        for dep_task in dep_tasks:
            if dep_task is not None:
                if not dep_task.result:
                    continue
                history_context.append(f"{dep_task.name}: {dep_task.result}")
        if task.id != "0":
            rewrite_prompt = RewriteSubQuery()
            rewrite_sub_query: str = rewrite_prompt.get_result(
                self.llm, question=sub_query, content=history_context
            )
            yield f"改写后{task.name}: {rewrite_sub_query}\n"
        else:
            rewrite_sub_query = sub_query
        yield f"思考当前信息能否回答该问题......\n"
        reason_prompt = ReasoningPrompt()
        answer_dict: Dict[str, str] = reason_prompt.get_result(
            self.llm,
            input=rewrite_sub_query,
            text_chunks=history_context,
        )
        if answer_dict["can_answer"]:
            yield f"问题： {rewrite_sub_query} \n答案： {answer_dict['answer']}\n"
            return
        else:
            yield f"当前问题无法回答，开始从图库检索问题相关数据\n"
        pre_entity = logic_form.s
        assert isinstance(pre_entity, SPOEntity)
        last_entity = logic_form.o
        assert isinstance(last_entity, SPOEntity)
        relation = logic_form.p
        assert isinstance(relation, SPORelation)
        if pre_entity.alias_name in context.variables:
            pre_entity: SPOEntity = context.variables[pre_entity.alias_name]  # type: ignore
        else:
            context.update_variables(pre_entity)
        if relation.alias_name in context.variables:
            relation: SPORelation = context.variables[relation.alias_name]  # type: ignore
        else:
            context.update_variables(relation)
        if last_entity.alias_name in context.variables:
            last_entity: SPORelation = context.variables[last_entity.alias_name]  # type: ignore
        else:
            context.update_variables(last_entity)

        choice_label_prompt = ChoiceLabelPrompt()
        pre_std_entity_label: str = choice_label_prompt.get_result(
            self.llm,
            input=f"[{pre_entity.get_un_std_entity_first_type_or_std()}]:{pre_entity.entity_name}",
            labels=list(self.schema.vertices.keys()),
        )
        pre_entity_dict = self.match_entity(
            pre_std_entity_label,
            pre_entity.entity_name if pre_entity.entity_name else "",
        )
        if pre_entity_dict:
            yield f"图库中匹配到问题关联起点: {pre_entity_dict}\n"
        else:
            yield f"无法从图库中匹配上问题的关联起点\n"
        can_answer = False
        text_chunks = []
        if pre_entity_dict is not None:
            last_std_entity_label: str = choice_label_prompt.get_result(
                self.llm,
                input=last_entity.get_un_std_entity_first_type_or_std(),
                labels=list(self.schema.vertices.keys()),
            )
            edge_labels = []
            for label, value in self.schema.edges.items():
                for src_dst in value["relations"]:
                    if (
                        src_dst["from_key"] == pre_std_entity_label
                        and src_dst["to_key"] == last_std_entity_label
                    ):
                        edge_labels.append(label)
                        break
            if len(edge_labels) != 0:
                std_relation_label: str = choice_label_prompt.get_result(
                    self.llm,
                    input=relation.get_un_std_entity_first_type_or_std(),
                    labels=edge_labels,
                    is_vertex=False,
                )
            else:
                std_relation_label = "Relation"
            relation_dict = self.match_relation(
                pre_entity_dict["label"],
                pre_entity_dict["keyword"],
                std_relation_label,
                relation.get_un_std_entity_first_type_or_std(),
                last_std_entity_label,
            )
            if relation_dict is not None:
                if (
                    relation_dict["label"] is not None
                    and relation_dict["keyword"] is not None
                ):
                    cypher = f"MATCH p=(n:{pre_entity_dict["label"]})-[r:{relation_dict['label']}]-(m) WHERE n.keyword='{pre_entity_dict["keyword"]}' AND r.keyword='{relation_dict['keyword']}' RETURN r.keyword, m.keyword, m.description;"
                elif relation_dict["label"] is None:
                    cypher = f"MATCH p=(n:{pre_entity_dict["label"]})-[r]-(m) WHERE n.keyword='{pre_entity_dict["keyword"]}' RETURN r.keyword, m.keyword, m.description;"
                else:
                    cypher = f"MATCH p=(n:{pre_entity_dict["label"]})-[r:{relation_dict['label']}]-(m) WHERE n.keyword='{pre_entity_dict["keyword"]}' RETURN r.keyword, m.keyword, m.description;"
                yield f"思考使用的cypher查询语句: {cypher}\n"
                result = self.database.exec(cypher).result_set["table"]
                yield f"cypher查询结果: {result}\n"
                text_chunks.append(
                    f"cypher查询语句: {cypher}\ncypher查询结果: {result}"
                )
                reason_prompt = ReasoningPrompt()
                answer_dict = reason_prompt.get_result(
                    self.llm,
                    text_chunks=text_chunks,
                    input=rewrite_sub_query,
                )
                if answer_dict["can_answer"]:
                    can_answer = True
                    yield f"问题： {rewrite_sub_query} \n答案： {answer_dict['answer']}\n"
                    last_entity.entity_name = answer_dict["answer"]
                    task.update_result(
                        f"问题： {rewrite_sub_query} \n答案： {answer_dict['answer']}"
                    )
                else:
                    yield f"当前信息不足够回答问题：'{rewrite_sub_query}'\n尝试检索更多信息......\n"
                vertices = [pre_entity_dict]
                edges = []
                for edge_keyword,dst_keyword, dst_description in result:
                    vertices.append(
                        {
                            "label": last_std_entity_label,
                            "keyword": dst_keyword,
                            "description": dst_description,
                        }
                    )
                    edges.append(
                        {
                            "label": relation_dict["label"],
                            "keyword": edge_keyword,
                            "from_key": pre_entity_dict["label"],
                            "from_pk": pre_entity_dict["keyword"],
                            "to_key": last_std_entity_label,
                            "to_pk": dst_keyword,
                        }
                    )
                context.update_graph(vertices, edges)

        if not can_answer:
            yield "检索实体和问题相关的文档\n"
            embedding = self.embedding_model.get_text_embedding(rewrite_sub_query)
            similarity_result = self.vector_store.query(
                VectorStoreQuery(
                    query_embedding=embedding,
                    similarity_top_k=self.similarity_top_k * 50,
                )
            )
            ids: List[str] = similarity_result.ids[0 : self.similarity_top_k]  # type: ignore
            if pre_entity_dict is not None:
                ids = [self.graph_store.get_chunk_ids([pre_entity_dict["keyword"]])[0]] + ids
            ids = list(set(ids))
            docs = self.document_store.get_nodes(ids)
            text_chunks.extend([cast(TextNode, doc).text for doc in docs])
            reason_prompt = ReasoningPrompt()
            answer_dict = reason_prompt.get_result(
                self.llm,
                text_chunks=text_chunks,
                input=rewrite_sub_query,
            )
            if answer_dict["can_answer"]:
                can_answer = True
                yield f"问题： {rewrite_sub_query} \n答案： {answer_dict['answer']}\n"
                last_entity.entity_name = answer_dict["answer"]
                task.update_result(
                    f"问题： {rewrite_sub_query} \n答案： {answer_dict['answer']}"
                )
            else:
                yield f"当前信息无法回答问题：{rewrite_sub_query}\n"
                last_entity.entity_name = f"当前信息无法回答问题：{rewrite_sub_query}"
                task.update_result(f"当前信息无法回答问题：{rewrite_sub_query}\n")

        context.update_chunks(text_chunks)
