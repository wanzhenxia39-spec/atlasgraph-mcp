from typing import Any, List, Tuple, Dict

from rag.common.task import Task
from rag.common.logic_node_parser import (
    DeduceNode,
    GetNode,
    GetSPONode,
    MathNode,
    ParseLogicForm,
)
from rag.prompt.base_prompt import BasePrompt
from rag.common.base_model import LogicNode
from rag.schema import Schema
from rag.model.atlas_model import CompletionResponse
import re


class PlanningPrompt(BasePrompt):
    template: str = """
    {{
        "instruction": "你是一个规划专家，你的任务是根据function中的算子来规划问题",
        "function": [
            {{
              "function_declaration": "Retrieval(s=s_alias:entity_type[`entity_name`], p=p_alias:edge_type, o=o_alias:entity_type[`entity_name`], p.prop=`value`, s.prop=`value`, o.prop=`value`)",
              "description": "用于文本信息检索，根据spo检索信息，s、p、o不能在同一表达式中反复多次出现，可对s、p、o进行带约束查询；多跳则进行多次检索。当前变量引用前文变量时，变量名必须和指代的变量名一致，且只需给出变量名，实体类型及名称仅在首次引用时给定。prop为被约束的属性名,属性约束的值`value`可以是文本、常数，也可以引用前面函数中的变量名"
            }},
            {{
              "function_declaration": "Deduce(content=[`XXX` or `o_alias/s_alias`], target=`XXX`)->deduce_alias",
              "description": "推断，对检索或计算结果进行推断，以回答问题，content为问题、历史对话或检索结果，可以是文本片段或以变量名指代。target为推理的目标。"
            }},
        ],
        "cases": [
            {{
                "query": "吴京是谁",
                "answer": "首先需要检索公众人物'吴京'的基本信息\n```\nStep1:查询吴京\nAction1:Retrieval(s=s1:公众人物[`吴京`], p=p1, o=o1)\n```\n",
            }},
            {{
                "query": "C罗在2011年效力的运动队中，哪一支成立时间最晚？",
                "answer": "首先需要查询C罗在2011年的效力球队\n```\nStep1:C罗在2011年效力于哪些运动队？\nAction1:Retrieval(s=s1:球员[`C罗`], p=p1:2011年效力于, o=o1:运动队)\n```\n获取这些球队的成立年份\n```\nStep2:这些队伍的成立年份分别是？\nAction2:Retrieval(s=o1, p=p2:成立年份, o=o2:年份)\n```\n比较成立时间并选择最晚的\n```\nStep3:哪支队伍成立最晚？\nAction3:Deduce(content=[`o2`], target=`成立时间最晚的球队`)->deduce3\n```",
            }},
            {{
                "query": "发表《心理治疗整合期刊》的协会首任主席是谁？",
                "answer": "首先需要确定期刊的出版机构\n```\nStep1:哪家协会出版《心理治疗整合期刊》？\nAction1:Retrieval(s=s1:出版物[`心理治疗整合期刊`], p=p1:出版机构, o=o1:协会)\n```\n查询该协会的首任主席\n```\nStep2:该协会的首任主席是谁？\nAction2:Retrieval(s=o1, p=p2:首任主席, o=o2:人物)\n```",
            }},
            {{
                "query": "波卡洪塔斯丘所在州何时加入美国？",
                "answer": "首先需要定位波卡洪塔斯丘的地理位置\n```\nStep1:波卡洪塔斯丘位于哪个州？\nAction1:Retrieval(s=s1:历史遗址[`波卡洪塔斯丘`], p=p1:所在地, o=o1:州)\n```\n查询该州加入美国的时间\n```\nStep2:该州何时成为美国的一部分？\nAction2:Retrieval(s=o1, p=p2:加入美国年份, o=o2:日期)\n```",
            }},
            {{
                "query": "两次龙卷风爆发中哪次致死人数更多？",
                "answer": "首先需要确定两次龙卷风事件\n```\nStep1:第一次龙卷风爆发是哪个事件？\nAction1:Retrieval(s=s1:事件[`龙卷风爆发`], p=p1:第一次事件, o=o1:事件)\n```\n获取第二次事件信息\n```\nStep2:第二次龙卷风爆发是哪个事件？\nAction2:Retrieval(s=s2:事件[`龙卷风爆发`], p=p2:第二次事件, o=o2:事件)\n```\n分别查询死亡人数\n```\nStep3:第一次事件的死亡人数？\nAction3:Retrieval(s=s1, p=p3:致死人数, o=o3:数字)\n```\n```\nStep4:第二次事件的死亡人数？\nAction4:Retrieval(s=s2, p=p4:致死人数, o=o4:数字)\n```\n比较死亡人数并确定更多的一方\n```\nStep5:比较两次事件的死亡人数\nAction5:Deduce(content=[`o3`,`o4`], target=`致死人数更多的事件`)->deduce5\n```",
            }},
            {{
                "query": "电影《Aas Ka Panchhi》和《Phoolwari》哪部更早上映？",
                "answer": "首先需要查询《Aas Ka Panchhi》的上映时间\n```\nStep1:《Aas Ka Panchhi》的上映时间是？\nAction1:Retrieval(s=s1:作品[`Aas Ka Panchhi`], p=p1:上映时间, o=o1:日期)\n```\n查询《Phoolwari》的上映时间\n```\nStep2:《Phoolwari》的上映时间是？\nAction2:Retrieval(s=s2:作品[`Phoolwari`], p=p2:上映时间, o=o2:日期)\n```\n比较两部电影的上映时间\n```\nStep3:比较两部电影的上映时间\nAction3:Deduce(content=[`o1`,`o2`], target=`更早上映的电影`)->deduce5\n```",
            }},
        ],
        "output_format": "使用文本格式输出，只输出case中的answer内容，不要在开始输出'```markdown'",
        "tips": [
                    "输出每一步的Step和Action前可以增加一些思路输出，类似’首先...其次...最后...‘",
                    "Each `Step` must contain exactly one `Action`",
                    "Each step is an indivisible atomic question, please re-split accordingly.",
                    "Output also needs to be a separate step."
                    "Step和Action内容需要使用代码风格输出"
        ],
        "query": {query}
    }}
    """

    @property
    def template_variables(self) -> List[str]:
        return ["query"]

    def build_prompt(self, **kwargs: Any) -> str:
        if "query" not in kwargs:
            raise ValueError("query is required")
        elif isinstance(kwargs["query"], str):
            query = kwargs["query"]
        else:
            raise ValueError("query must be a string")
        if "schema" not in kwargs:
            self.schema = None
        elif isinstance(kwargs["schema"], Schema):
            self.schema = kwargs["schema"]
        else:
            self.schema = None
        self.logic_node_parser = ParseLogicForm(schema=self.schema)
        return self.template.format(query=query)

    def parse_steps(self, response: str) -> Tuple[List[str], List[str]]:
        _output_string = response.replace("：", ":")
        _output_string = _output_string.strip()
        sub_querys = []
        logic_forms = []
        current_sub_query = ""
        for line in _output_string.split("\n"):
            if line.startswith("Step"):
                sub_querys_regex = re.search(r"Step\d+:(.*)", line)
                if sub_querys_regex is not None:
                    sub_querys.append(sub_querys_regex.group(1))
                    current_sub_query = sub_querys_regex.group(1)
                    current_sub_query = current_sub_query.strip()
                    if current_sub_query == "":
                        raise RuntimeError(f"{line} is not step query")
            elif line.startswith("Action"):
                logic_forms_regex = re.search(r"Action\d+:(.*)", line)
                if logic_forms_regex:
                    logic_forms.append(logic_forms_regex.group(1))
                    if len(logic_forms) - len(sub_querys) == 1:
                        sub_querys.append(current_sub_query)
        if len(sub_querys) != len(logic_forms):
            raise RuntimeError(
                f"sub query not equal logic form num {len(sub_querys)} != {len(logic_forms)}"
            )
        return sub_querys, logic_forms

    def _parse_lf(
        self, sub_queries: List[str], logic_forms: List[str]
    ) -> List[LogicNode]:
        return self.logic_node_parser.parse_logic_form_set(logic_forms, sub_queries, "")

    def _get_dep_task_id(self, logic_forms: List[LogicNode]) -> Dict[str, int]:
        all_alias_index_map = {}

        def add_index_map(alias: str, index: int) -> None:
            if alias in all_alias_index_map.keys():
                return
            all_alias_index_map[alias] = index

        for i, logic_form in enumerate(logic_forms):
            if isinstance(logic_form, GetSPONode):
                add_index_map(logic_form.s.alias_name.alias_name, i)  # type: ignore[union-attr]
                add_index_map(logic_form.p.alias_name.alias_name, i)  # type: ignore[union-attr]
                add_index_map(logic_form.o.alias_name.alias_name, i)  # type: ignore[union-attr]
            elif isinstance(logic_form, MathNode) or isinstance(logic_form, DeduceNode):
                add_index_map(logic_form.alias_name, i)

        return all_alias_index_map

    def _get_task_dep(
        self, index: int, logic_node: LogicNode, dep_task: Dict[str, int]
    ) -> List[int]:
        ret = []

        def add_dep_by_index(alias_name: str) -> None:
            alias_index = dep_task[alias_name]
            if alias_index < index:
                ret.append(alias_index)

        if isinstance(logic_node, GetSPONode):
            add_dep_by_index(logic_node.s.alias_name.alias_name)  # type: ignore[union-attr]
            add_dep_by_index(logic_node.p.alias_name.alias_name)  # type: ignore[union-attr]
            add_dep_by_index(logic_node.o.alias_name.alias_name)  # type: ignore[union-attr]
        if isinstance(logic_node, GetNode):
            return []

        if isinstance(logic_node, MathNode) or isinstance(logic_node, DeduceNode):
            for alias in dep_task.keys():
                if alias in logic_node.content:
                    ret.append(dep_task[alias])
        return ret

    def parse_response(self, response: str | CompletionResponse) -> List[Task]:
        response = str(response)
        print(response)
        sub_queries, m_logic_forms = self.parse_steps(response)
        logic_forms = self._parse_lf(sub_queries, m_logic_forms)
        tasks_dep: Dict[int, Dict[str, Any]] = {}
        alias_dep = self._get_dep_task_id(logic_forms)
        for i, logic_form in enumerate(logic_forms):
            deps = self._get_task_dep(i, logic_form, alias_dep)
            task_deps = [] if i == 0 else [i - 1]
            is_need_rewrite = (
                True if (deps or isinstance(logic_form, GetSPONode)) else False
            )
            if isinstance(logic_form, GetNode) or len(tasks_dep) == 0:
                is_need_rewrite = False
            tasks_dep[i] = {
                "name": f"Step{i+1}",
                "executor": logic_form.operator,
                "dependent_task_ids": task_deps,
                "arguments": {
                    "query": logic_form.sub_query,
                    "logic_form_node": logic_form,
                    "is_need_rewrite": is_need_rewrite,
                },
            }
        return Task.create_tasks_from_dag(tasks_dep)


if __name__ == "__main__":
    from rag.model import AtlasLLM

    plan = PlanningPrompt()
    llm = AtlasLLM(model="qwen2.5-72b-instruct")
    print(plan.get_result(llm, query="两次龙卷风爆发中哪次致死人数更多？"))
