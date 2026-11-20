from driver3.service.AtlasGraph import (
    AtlasGraph,
    GraphDatabaseDefinition,
)
from driver3.grpc_build.graph_system_pb2 import GetAllGraphInfoRequest
from driver3.grpc_build.meta_pb2 import GetAllGraphInfoResp
from driver3.service.ConnectionConfiguration import ConnectionConfiguration
from rag.utils.utils import to_legal_cypher_string
from datetime import datetime, timezone
from rag.utils.config import rag_config as config
from rag.utils.log import rag_logger as log
from rag.model import AtlasLLM
from rag.prompt import QuestionToTitlePrompt
from typing import Dict, List, Any, Union, Optional, Tuple
import sys
import uuid
from multiprocessing import get_context
from multiprocessing.context import SpawnProcess
from threading import Thread
from rag.service.file_process import process_files
from rag.utils.constant import UploadConfig
import time
import multiprocessing as mp


class GlobalDatabase:
    COMPLETE = "Complete"
    FAULTED = "Faulted"
    PROCESSING = "Processing"
    UNKNOWNID = "UnknownID"

    def __init__(self) -> None:
        self.max_num_task = sys.maxsize
        self.num_workers = sys.maxsize
        self.process_map: Dict[str, SpawnProcess] = {}
        self.task_queue: mp.Queue = mp.Queue()

    def _process_worker(self) -> None:
        while True:
            (upload_config, task_id) = self.task_queue.get()
            while len(self.process_map) >= self.num_workers:
                del_task_ids = []
                for cur_task_id, process in self.process_map.items():
                    assert isinstance(process, SpawnProcess)
                    if not process.is_alive():
                        del_task_ids.append(cur_task_id)
                for cur_task_id in del_task_ids:
                    del self.process_map[cur_task_id]
                time.sleep(1)
            context = get_context("spawn")
            process = context.Process(
                target=process_files,
                args=(upload_config, task_id, config.rag_server_path),
                daemon=True,
            )
            process.start()
            self.process_map[task_id] = process

    def get_task_count(self, username: str, database_id: int) -> int:
        return self.database.exec(
            f'MATCH (u: User{{username:"{username}"}})-[r: Link]->(t: Task) WHERE t.database_id={database_id} return COUNT(t)'
        ).result_set["table"][0][0]

    def new_database(self) -> None:
        self.driver = AtlasGraph(
            ConnectionConfiguration(config.graph_url, config.username, config.password)
        )
        if not self.driver.graph_exists(config.global_database):
            self.driver.create_graph_database(
                GraphDatabaseDefinition(
                    config.global_database, 3, 1, "GraphRAG Database"
                )
            )
        self.database = self.driver.use(config.global_database)

    def init(self) -> None:
        self.max_num_task = (
            config.max_num_task if config.max_num_task >= 1 else sys.maxsize
        )
        self.num_workers = (
            config.num_workers if config.num_workers >= 1 else sys.maxsize
        )
        self.new_database()
        cypher_query_list = [
            (
                f"CREATE VERTEX User(username STRING PRIMARY KEY)"
                if not self.database.vertex_exists("User")
                else None
            ),
            (
                f"CREATE VERTEX Task(id STRING PRIMARY KEY, server_id INT32, database_id INT32, time DATETIME, status STRING, rate FLOAT)"
                if not self.database.vertex_exists("Task")
                else None
            ),
            (
                f"CREATE VERTEX Session(id STRING PRIMARY KEY, description STRING, time DATETIME)"
                if not self.database.vertex_exists("Session")
                else None
            ),
            (
                f"CREATE VERTEX Conversation(id STRING PRIMARY KEY, order_num INT32, question STRING, answer STRING, data MAP)"
                if not self.database.vertex_exists("Conversation")
                else None
            ),
            (
                f"CREATE DIRECTED EDGE Link() RELATIONS (FROM User TO Task, FROM User TO Session, FROM Session TO Conversation)"
                if not self.database.edge_exists("Link")
                else None
            ),
        ]
        for cypher_query in cypher_query_list:
            if cypher_query is None:
                continue
            self.database.exec(cypher_query)
        cypher_query = f'MATCH (n: Task{{server_id: {config.server_id}, status: "{GlobalDatabase.PROCESSING}"}}) SET n.status = "{GlobalDatabase.FAULTED}"'
        self.database.exec(cypher_query)
        self.llm = AtlasLLM(
            model=config.chat_model,
            api_base=config.chat_base,
            api_url=config.chat_url,
            api_key=config.chat_key,
        )
        self.task_thread = Thread(target=self._process_worker, daemon=True)
        self.task_thread.start()

    def get_graph_info_mapping(self) -> Tuple[Dict[int, str], Dict[str, int]]:
        graph_infos: GetAllGraphInfoResp = self.driver.graph_sys_stub.GetAllGraphInfo(
            GetAllGraphInfoRequest(token=self.database.token())
        )
        id2name = {}
        name2id = {}
        for graph_info in graph_infos.graph_infos:
            id2name[graph_info.graph_id] = graph_info.name
            name2id[graph_info.name] = graph_info.graph_id
        return id2name, name2id

    def add_task(
        self, task_id: str, username: str, database: str, upload_config: UploadConfig
    ) -> None:
        self.new_database()
        _, name2id = self.get_graph_info_mapping()
        if database not in name2id:
            raise ValueError(f"database {database} not found")
        while self.get_task_count(username, name2id[database]) >= self.max_num_task:
            cypher_query = f'CALL {{MATCH (u: User{{username:"{username}"}})-[]->(t: Task) WHERE t.database_id={name2id[database]} RETURN t ORDER BY t.time ASC LIMIT 1}} DETACH DELETE t;'
            self.database.exec(cypher_query)
        cypher_query = f'MERGE (u: User{{username:"{username}"}})-[r: Link]->(t: Task{{id: "{task_id}", server_id: {config.server_id}, database_id: {name2id[database]}, time: datetime("{datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")}"), status: "{GlobalDatabase.PROCESSING}", rate: 0.0}})'
        self.database.exec(cypher_query)
        self.task_queue.put((upload_config, task_id))

    def list_task(
        self,
        username: str,
        database: Optional[str],
        task_status: Optional[str] = None,
        limit: int = 30,
    ) -> List[Dict[str, Union[str, int]]]:
        if task_status is not None and task_status not in [
            self.COMPLETE,
            self.PROCESSING,
            self.FAULTED,
        ]:
            raise ValueError("status must be one of [Complete, Processing, Faulted]")
        self.new_database()
        id2name, name2id = self.get_graph_info_mapping()
        if database is not None and database not in name2id:
            raise ValueError(f"database {database} not found")
        if database is None:
            if task_status is None:
                cypher_query = f'MATCH (u: User{{username:"{username}"}})-[r: Link]->(t: Task) RETURN t.id, t.time, t.database_id, t.server_id, t.status, t.rate ORDER BY t.time '
            else:
                cypher_query = f'MATCH (u: User{{username:"{username}"}})-[r: Link]->(t: Task) WHERE t.status="{task_status}" RETURN t.id, t.time, t.database_id, t.server_id, t.status, t.rate ORDER BY t.time '
            if limit >= 0:
                cypher_query += f"DESC LIMIT {limit}"

        else:
            if task_status is None:
                cypher_query = f'MATCH (u: User{{username:"{username}"}})-[r: Link]->(t: Task) WHERE t.database_id={name2id[database]} RETURN t.id, t.time, t.database_id, t.server_id, t.status, t.rate ORDER BY t.time '
            else:
                cypher_query = f'MATCH (u: User{{username:"{username}"}})-[r: Link]->(t: Task) WHERE t.database_id={name2id[database]} AND t.status="{task_status}" RETURN t.id, t.time, t.database_id, t.server_id, t.status, t.rate ORDER BY t.time '
            if limit >= 0:
                cypher_query += f"DESC LIMIT {limit}"
        result = self.database.exec(cypher_query).result_set["table"]
        return [
            {
                "taks_id": row[0],
                "datetime": row[1],
                "database": id2name[row[2]],
                "server_id": row[3],
                "status": row[4],
                "rate": row[5],
            }
            for row in result
            if row[2] in id2name
        ]

    def delete_task(self, task_id: str) -> None:
        self.new_database()
        cypher_query = (
            f'MATCH (t: Task{{id: "{task_id}"}}) DETACH DELETE t RETURN count(t)'
        )
        if self.database.exec(cypher_query).result_set["table"][0][0] == 0:
            log.error(f"[Delete Task] task: {task_id} not found")
            raise ValueError(f"[Delete Task] task: {task_id} not found")

    def cancel_task(self, task_id: str) -> None:
        self.new_database()
        if task_id in self.process_map:
            self.process_map[task_id].terminate()  # type: ignore[union-attr]
            del self.process_map[task_id]
        cypher_query = f'MATCH (t: Task{{id: "{task_id}"}}) SET t.status = "{GlobalDatabase.FAULTED}"'
        if len(self.database.exec(cypher_query).result_set["table"]) == 0:
            log.error(f"[Cancel Task] task: {task_id} not found")
            raise ValueError(f"[Cancel Task] task: {task_id} not found")

    def update_task_processing_rate(self, task_id: str, rate: float) -> None:
        self.new_database()
        cypher_query = f'MATCH (t: Task{{id: "{task_id}"}}) SET t.rate = {rate}'
        self.database.exec(cypher_query)

    def get_task_processing_rate(self, task_id: str) -> float:
        self.new_database()
        cypher_query = f'MATCH (t: Task{{id: "{task_id}"}}) RETURN t.rate'
        result = self.database.exec(cypher_query).result_set["table"]
        if len(result) == 0:
            return 0.0
        else:
            return result[0][0]

    def update_task_status(self, task_id: str, status: str) -> None:
        self.new_database()
        cypher_query = f'MATCH (t: Task{{id: "{task_id}"}}) SET t.status = "{status}"'
        self.database.exec(cypher_query)

    def get_task_status(self, task_id: str) -> str:
        self.new_database()
        cypher_query = f'MATCH (t: Task{{id: "{task_id}"}}) RETURN t.status'
        result = self.database.exec(cypher_query).result_set["table"]
        if len(result) == 0:
            return GlobalDatabase.UNKNOWNID
        else:
            return result[0][0]

    def create_session(self, username: str, question: str) -> Dict[str, str]:
        self.new_database()
        session_id = str(uuid.uuid4())
        prompt = QuestionToTitlePrompt()
        description = prompt.parse_response(
            self.llm.complete(prompt.build_prompt(input=question))
        )
        cypher_query = (
            f'MERGE (u: User{{username:"{username}"}})-[r: Link]->(s: Session{{id: "{session_id}", '
            f'description: "{to_legal_cypher_string(description)}", time: datetime("{datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")}")}})'
        )
        self.database.exec(cypher_query)
        return {"session_id": session_id, "description": description}

    def get_session_description(self, session_id: str) -> str:
        self.new_database()
        cypher_query = f'MATCH (s: Session{{id: "{session_id}"}}) RETURN s.description'
        result = self.database.exec(cypher_query).result_set["table"]
        if len(result) == 0:
            log.error(f"[Get Session Description] session: {session_id} not found")
            raise ValueError(
                "[Get Session Description] session: {session_id} not found"
            )
        else:
            return result[0][0]

    def list_session(self, username: str, limit: int) -> List[Dict[str, str]]:
        self.new_database()
        cypher_query = f'MATCH (u: User{{username:"{username}"}})-[r: Link]->(s: Session) RETURN s.id, s.description, s.time ORDER BY s.time DESC LIMIT {limit}'
        result = self.database.exec(cypher_query).result_set["table"]
        return [
            {"session_id": row[0], "description": row[1], "datetime": row[2]}
            for row in result
        ]

    def convert_to_atlas_map(self, data: Any) -> str:
        if isinstance(data, dict):
            return (
                "{"
                + ",".join(
                    [
                        f"`{key}`:{self.convert_to_atlas_map(value)}"
                        for key, value in data.items()
                    ]
                )
                + "}"
            )
        elif isinstance(data, list):
            return (
                "["
                + ",".join([self.convert_to_atlas_map(value) for value in data])
                + "]"
            )
        else:
            return '"' + to_legal_cypher_string(str(data)) + '"'

    def is_valid_session_id(self, session_id: str) -> bool:
        self.new_database()
        cypher_query = f'MATCH (s: Session{{id: "{session_id}"}}) RETURN COUNT(s)'
        return bool(self.database.exec(cypher_query).result_set["table"][0][0])

    def add_conversation(
        self, session_id: str, question: str, answer: str, data: Dict[str, Any]
    ) -> None:
        self.new_database()
        cypher_query = f'MATCH (s: Session{{id: "{session_id}"}})-[r: Link]->(c: Conversation) RETURN COUNT(c)'
        conversation_id = str(uuid.uuid4())
        conversation_order_num = self.database.exec(cypher_query).result_set["table"][
            0
        ][0]
        cypher_query = (
            f'MERGE (s: Session{{id: "{session_id}"}})-[r: Link]->(c: Conversation{{id: "{conversation_id}", order_num: {conversation_order_num}, '
            f'question: "{to_legal_cypher_string(question)}", answer: "{to_legal_cypher_string(answer)}", data: {self.convert_to_atlas_map(data)}}})'
        )
        self.database.exec(cypher_query)

    def convert_to_dict(self, data: Any) -> Any:
        if isinstance(data, dict):
            if (
                "type" in data
                and "entries" in data
                and isinstance(data["entries"], dict)
            ):
                return {
                    key: self.convert_to_dict(value)
                    for key, value in data["entries"].items()
                }
        if isinstance(data, list):
            return [self.convert_to_dict(value) for value in data]
        return data

    def get_session(self, session_id: str) -> List[Dict[str, Any]]:
        self.new_database()
        cypher_query = f'MATCH (s: Session{{id: "{session_id}"}})-[r: Link]->(c: Conversation) RETURN c.question, c.answer, c.data ORDER BY c.order_num ASC'
        result_list: List[Dict[str, Any]] = []
        for result in self.database.exec(cypher_query).result_set["table"]:
            result_list.append(
                {
                    "question": result[0],
                    "answer": result[1],
                    "data": self.convert_to_dict(result[2]),
                }
            )
        return result_list

    def delete_session(self, session_id: str) -> None:
        self.new_database()
        cypher_query = f'MATCH (s: Session{{id: "{session_id}"}})-[r: Link]->(c: Conversation) DETACH DELETE c'
        self.database.exec(cypher_query)
        cypher_query = (
            f'MATCH (s: Session{{id: "{session_id}"}}) DETACH DELETE s RETURN count(s)'
        )
        if self.database.exec(cypher_query).result_set["table"][0][0] == 0:
            log.error(f"[Delete Session] session: {session_id} not found")
            raise ValueError(f"[Delete Session] session: {session_id} not found")


global_database = GlobalDatabase()
