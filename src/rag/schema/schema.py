from typing import Dict, Set, Tuple
import json
from pathlib import Path


class Schema:
    def __init__(self, schema_dict: Dict) -> None:
        if "vertices" in schema_dict:
            vertices = schema_dict["vertices"]
            if isinstance(vertices, dict):
                for vertex_label, properties in vertices.items():
                    if (
                        isinstance(vertex_label, str)
                        and isinstance(properties, list)
                        and all([isinstance(x, str) for x in properties])
                    ):
                        if not (len(properties)>=2 and properties[0]=="keyword"  and properties[1]== "description" ):
                            raise ValueError(
                                "vertex label's properties[0] and properties[1] must be keyword and description"
                            )
                    else:
                        raise ValueError(
                            "vertex label must be string and properties must be a list of strings"
                        )
                else:
                    self.vertices = vertices
            else:
                raise ValueError("vertices must be a dict ")

        else:
            self.vertices = {}
        if "edges" in schema_dict:
            edges = schema_dict["edges"]
            m_edges = {}
            if isinstance(edges, dict):
                for edge_label, edge_value in edges.items():
                    if isinstance(edge_label, str) and isinstance(edge_value, dict):
                        if "relations" in edge_value:
                            relations = edge_value["relations"]
                            m_relations = []
                            if isinstance(relations, list):
                                for relation in relations:
                                    if isinstance(relation, dict):
                                        if (
                                            "from_key" in relation
                                            and "to_key" in relation
                                            and isinstance(relation["from_key"], str)
                                            and isinstance(relation["to_key"], str)
                                        ):
                                            if (
                                                relation["from_key"] in self.vertices
                                                and relation["to_key"] in self.vertices
                                            ):
                                                m_relation = {
                                                    "from_key": relation["from_key"],
                                                    "to_key": relation["to_key"],
                                                }
                                                m_relations.append(m_relation)
                                            else:
                                                raise ValueError(
                                                    f"Relation {relation} from_key or to_key is invalid"
                                                )
                                        else:
                                            raise ValueError(
                                                "Invalid relation format: {}".format(
                                                    relation
                                                )
                                            )
                                    else:
                                        raise ValueError("relation must be ad dict")
                            else:
                                raise ValueError("relations must be list")
                        if "properties" in edge_value:
                            properties = edge_value["properties"]
                            if isinstance(properties, list) and all(
                                isinstance(x, str) for x in properties
                            ):
                                if len(properties)>=1 and properties[0] == "keyword":
                                    m_properties = properties
                                else:
                                    raise ValueError(
                                        "edge label properties[0] must be keyword"
                                    )
                            else:
                                raise ValueError("properties must be list of strings")
                        m_edges[edge_label] = {
                            "relations": m_relations,
                            "properties": m_properties,
                        }
                    else:
                        raise ValueError(
                            "edges label must be a string and edges value must be a dict"
                        )
                self.edges = m_edges
            else:
                raise ValueError("edges must be a dict ")
        else:
            self.edges = {}
        if "explain" in schema_dict and isinstance(schema_dict["explain"], str):
            self.explain = schema_dict["explain"]
        else:
            self.explain = ""
        self.canonical_edge_labels: Set[Tuple[str, str, str]] = {
            (relation["from_key"], edge_label, relation["to_key"])
            for edge_label, value in self.edges.items()
            for relation in value["relations"]
        }
        self.schema = {
            "vertices": self.vertices,
            "edges": self.edges,
            "explain": self.explain,
        }

    @staticmethod
    def from_json_file(file_path: Path | str) -> "Schema":
        if isinstance(file_path, str):
            file_path = Path(file_path)
        with open(file_path, "r") as f:
            schema_dict = json.load(f)
        return Schema(schema_dict)

    @staticmethod
    def from_str(schema_str: str) -> "Schema":
        schema_dict = json.loads(schema_str)
        return Schema(schema_dict)
    def __str__(self) -> str:
        return json.dumps(self.schema,ensure_ascii=False)

    @property
    def schema_str(self) -> str:
        return str(self)


DEFAULT_SCHEMA = Schema(
    {
        "vertices": {
            "Entity": ["keyword", "description"],
        },
        "edges": {
            "Relation": {
                "relations": [{"from_key": "Entity", "to_key": "Entity"}],
                "properties": ["keyword"],
            }
        },
    }
)
