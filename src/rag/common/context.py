import networkx as nx
from typing import Optional, OrderedDict, Generator, List, Dict, Union
from rag.common.task import Task
from rag.common.base_model import SPOBase
from typing import Any


class Context:
    """Pipeline execution context that tracks tasks and their dependencies."""

    def __init__(self) -> None:
        """Initializes an ordered dictionary to store tasks in insertion order."""

        self._tasks = OrderedDict()
        self.chunks = set()
        self.graph = {"vertices": [], "edges": []}
        self.vertex_set = set()
        self.edge_set = set()
        self.variables = {}
        self.message_history = [] # 新建一个history以及对应的函数记录上下文信息

    # 添加消息到上下文历史
    def add_message(self, role: str, content: Any, query: Optional[list] = None) -> None:
        message = {"role": role, "content": content}
        if query is not None: # 用于关联消息与特定问题
            message["query"] = query
        self.message_history.append(message)
        
    # 获取完整的消息历史
    def get_messages(self) -> list:
        return self.message_history
    
    def update_chunks(self, chunks: List[str]) -> None:
        self.chunks.update(chunks)

    def update_graph(
        self, vertices: List[Dict[str, str]], edges: List[Dict[str, str]]
    ) -> None:
        for vertex in vertices:
            if (vertex["label"], vertex["keyword"]) not in self.vertex_set:
                self.graph["vertices"].append(vertex)
                self.vertex_set.add((vertex["label"], vertex["keyword"]))
        for edge in edges:
            if (
                edge["from_key"],
                edge["label"],
                edge["to_key"],
                edge["keyword"],
            ) not in self.edge_set:
                self.graph["edges"].append(edge)
                self.edge_set.add(
                    (edge["from_key"], edge["label"], edge["to_key"], edge["keyword"])
                )

    def update_variables(self, logic_node: SPOBase) -> None:
        self.variables[logic_node.alias_name] = logic_node

    def __str__(self) -> str:
        return f"检索信息：{self.chunks}\n 图谱信息：{self.graph}"

    def add_task(self, task: Task) -> None:
        """Adds a task to the context.

        Args:
            task: Task instance to be added to the execution context.
        """
        self._tasks[task.id] = task

    def last_task(self) -> Optional[Task]:
        """Retrieves the most recently added task.

        Returns:
            The last task in the execution order, or None if no tasks exist.
        """
        if len(self._tasks) == 0:
            return None
        last_id = list(self._tasks.keys())[-1]
        return self._tasks[last_id]

    def append_task(self, task: Task) -> None:
        """Appends a new task and automatically sets its parent to the last task.

        Used for iterative planning where new tasks depend on the latest execution state.
        Automatically adds parent dependency only if the task has no existing parents.

        Args:
            task: Task to append to the execution sequence.
        """
        if len(task.parents) == 0 and len(self._tasks) > 0:
            task.parents = [self.last_task()]
        self.add_task(task)

    def get_task(self, task_id: str) -> Optional[Task]:
        """Retrieves a task by its unique identifier.

        Args:
            task_id: Unique identifier of the task to retrieve.

        Returns:
            The requested task if found, otherwise None.
        """
        return self._tasks.get(task_id)

    def topological_sort(self, dag: nx.DiGraph) -> Generator[int, None, None]:
        """Performs topological sort on a directed acyclic graph (DAG).

        Args:
            dag: NetworkX graph representing task dependencies.

        Returns:
            Generator yielding node IDs in topological order.
        """
        return nx.topological_sort(dag)

    def topological_generations(
        self, dag: nx.DiGraph
    ) -> List[Generator[int, None, None]]:
        """Generates topological generations (parallelizable task groups).

        Args:
            dag: NetworkX graph representing task dependencies.

        Returns:
            Generator yielding lists of node IDs that can be executed in parallel.
        """
        return nx.topological_generations(dag)

    def get_dag(self) -> nx.DiGraph:
        """Constructs a directed acyclic graph (DAG) of task dependencies.

        Verifies all referenced tasks exist in the context. Raises ValueError if any
        dependency references a non-existent task.

        Returns:
            A NetworkX DiGraph representing the task dependency structure.

        Raises:
            ValueError: If a task referenced in dependencies does not exist.
        """
        dag = nx.DiGraph()
        nodes = set()
        for task_id, task in self._tasks.items():
            nodes.add(task_id)
            for dep in task.parents:
                nodes.add(dep.id)
        for node in nodes:
            if node not in self._tasks:
                raise ValueError(f"task {node} not found.")
        dag.add_nodes_from(nodes)
        for task_id, task in self._tasks.items():
            for dep in task.parents:
                dag.add_edge(dep.id, task_id)
        self.topological_sort(dag)
        return dag

    def gen_task(
        self, group: bool = False
    ) -> Generator[Union[List[Task], Task], None, None]:
        """Generates tasks in execution order, optionally grouped by parallelizable stages.

        Args:
            group: If True, yields lists of tasks that can be executed in parallel.
                   If False (default), yields individual tasks sequentially.

        Returns:
            Generator yielding either individual Task objects or lists of Tasks,
            depending on the 'group' parameter.

        Raises:
            NetworkXUnfeasible: If the task dependency graph contains cycles.
        """
        dag = self.get_dag()
        if not group:
            nodes = self.topological_sort(dag)
            for node in nodes:
                yield self._tasks[node]
        else:
            node_groups = self.topological_generations(dag)
            for node_group in node_groups:
                yield [self._tasks[node] for node in node_group]
