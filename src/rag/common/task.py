import uuid
from enum import Enum
from typing import Dict, List, Any, Optional, cast
from rag.utils.constant import Tool


class TaskStatus(Enum):
    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class Task:
    """Represents an executable task that can be processed by an executor.

    Attributes:
        executor (str): Name of the executor responsible for running this task.
        arguments (dict): Parameters required for task execution.
        status (TaskStatus): Current execution status of the task (PENDING by default).
        id (str): Unique identifier generated using UUID.
        parents (List[Task]): List of predecessor tasks that must complete before execution.
        children (List[Task]): List of successor tasks that depend on this task's completion.
        memory (dict): Storage for intermediate execution data.
        result (Any): Final output of the task execution.
    """

    def __init__(
        self,
        executor: Tool,
        arguments: dict,
        parents: List = None,
        children: List = None,
        id: str = None,
        name: str = None,
        thought: str = None,
    ):
        """Represents an executable task that can be processed by an executor.

        Attributes:
            executor (str): Name of the executor responsible for running this task.
            arguments (dict): Parameters required for task execution.
            status (TaskStatus): Current execution status of the task (PENDING by default).
            id (str): Unique identifier.
            parents (List[Task]): List of predecessor tasks that must complete before execution.
            children (List[Task]): List of successor tasks that depend on this task's completion.
            memory (dict): Storage for intermediate execution data.
            result (Any): Final output of the task execution.
        """
        self.executor = executor
        self.arguments = arguments
        self.thought = thought
        self.status = TaskStatus.PENDING
        if id is None:
            self.id = str(uuid.uuid4())
        else:
            self.id = id
        if parents is None:
            self.parents: List[Optional[Task]] = []
        else:
            self.parents = parents
        if children is None:
            self.children = []
        else:
            self.children = children

        self.memory = {}
        self.result: Optional[str] = None

        self.name = name if name else str(self.id)

    def add_parent(self, parent: "Task") -> None:
        """Adds a predecessor task to the dependency list.

        Args:
            parent: Task object that must complete before this task can execute.
        """
        self.parents.append(parent)

    def add_child(self, child: "Task") -> None:
        """Adds a successor task to the dependency list.

        Args:
            child: Task object that depends on this task's completion.
        """
        self.children.append(child)

    def update_memory(self, key: str, value: str) -> None:
        """Stores intermediate data during task execution.

        Args:
            key: Identifier for the stored data.
            value: Data value to be stored in task memory.
        """
        self.memory[key] = value

    def update_result(self, result: str | None) -> None:
        """Sets the final result of the task execution.

        Args:
            result: Output value produced by the task.
        """
        self.result = result

    def check_require_rewrite(self, task: "Task") -> bool:
        """Determines if query rewriting is needed based on parameter patterns.

        Args:
            task (Task): Task to check for rewrite requirements

        Returns:
            bool: True if query contains dynamic parameter references (e.g., {{1.output}})
        """
        return self.arguments.get("is_need_rewrite", False)

    def get_task_context(self, with_all: bool = False) -> Dict[str, str]:
        """Generates a dictionary representation of the task's context."""
        result = {}
        if self.thought:
            result["thought"] = self.thought
        if self.result:
            if isinstance(self.result, str):
                result["result"] = self.result
            elif hasattr(self.result, "summary"):
                summary = getattr(self.result, "summary", "")
                if summary and (with_all or "i don't know" not in summary.lower()):
                    result["result"] = summary
            else:
                result["result"] = str(self.result)

        if result:
            result["task"] = self.arguments.get("query", "")

        if self.name:
            result["name"] = self.name
        return result

    def __str__(self) -> str:
        """Generates a string representation of the task.

        Returns:
            Formatted string containing task ID, executor name, and arguments.
        """
        msg = f"Task<{self.id}>\n\texecutor: {self.executor}\n\targuments {self.arguments}\n"
        return msg

    __repr__ = __str__

    @staticmethod
    def create_tasks_from_dag(task_dag: Dict[int, Dict[str, Any]]) -> List["Task"]:
        """Constructs task objects from a task dependency graph definition.

        Example DAG format:
        {
            "0": {
                "executor": "call_kg_retriever",
                "dependent_task_ids": [],
                "arguments": {"query": "张学友出演过的电影列表"},
            },
            "1": {
                "executor": "call_kg_retriever",
                "dependent_task_ids": ["0"],
                "arguments": {"query": "刘德华出演过的电影列表"},
            },
        }

        Args:
            task_dag: Dictionary where keys are task IDs and values contain:
                - executor: Name of the task executor
                - dependent_task_ids: List of prerequisite task IDs
                - arguments: Execution parameters for the task

        Returns:
            List of Task objects with established parent/child relationships.
        """

        task_map = {}
        for task_order, task_info in task_dag.items():
            task = Task(
                Tool(cast(str, task_info["executor"]).lower()),
                task_info["arguments"],
                id=str(task_order),
                thought=task_info.get("thought", ""),
                name=task_info.get("name", None),
            )
            task_map[task_order] = task
        for task_order, task_info in task_dag.items():
            deps = task_info["dependent_task_ids"]
            for dep in deps:
                task_map[task_order].add_parent(task_map[dep])
                task_map[dep].add_child(task_map[task_order])

        return list(task_map.values())
