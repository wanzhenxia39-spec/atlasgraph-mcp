from typing import Dict, Set, Tuple, List, Optional
import json
from pathlib import Path
from pydantic import BaseModel
from rag.schema.schema import Schema


class PropertyModel(BaseModel):
    property: str
    explain: str = ""
    type: str


class VertexModel(BaseModel):
    label: str
    explain: str = ""
    properties: List[PropertyModel] = []


class RelationModel(BaseModel):
    from_key: str
    to_key: str


class EdgeModel(BaseModel):
    label: str
    explain: str = ""
    relations: List[RelationModel]
    properties: List[PropertyModel] = []


class GraphSchemaModel(BaseModel):
    vertices: List[VertexModel] = []
    edges: List[EdgeModel] = []


class GraphSchema(): 
    def __init__(self, graph_schema: GraphSchemaModel) -> None:
        self.vertices: Dict[str, VertexModel] = {}
        for vertex in graph_schema.vertices:
            self.vertices[vertex.label] = vertex
        self.graph_schema = GraphSchemaModel(vertices=graph_schema.vertices)
        self.edges: Dict[str, EdgeModel] = {}
        self.canonical_edge_labels: Set[Tuple[str, str, str]] = set()
        for edge in graph_schema.edges:
            for relation in edge.relations:
                if (
                    relation.from_key in self.vertices
                    and relation.to_key in self.vertices
                ):
                    self.graph_schema.edges.append(edge)
                    self.canonical_edge_labels.add(
                        (relation.from_key, edge.label, relation.to_key)
                    )

    @staticmethod
    def from_json(schema_dict: Dict) -> "GraphSchema":
        return GraphSchema(GraphSchemaModel(**schema_dict))

    @staticmethod
    def from_json_file(file_path: Path | str) -> "GraphSchema":
        if isinstance(file_path, str):
            file_path = Path(file_path)
        with open(file_path, "r") as f:
            schema_dict = json.load(f)
        return GraphSchema(GraphSchemaModel(**schema_dict))

    @staticmethod
    def from_str(schema_str: str) -> "GraphSchema":
        schema_dict = json.loads(schema_str)
        return GraphSchema(GraphSchemaModel(**schema_dict))

    def to_schema(self) -> Schema:
        return Schema(
            {
                "vertices": {
                    vertex.label: [property.property for property in vertex.properties]
                    for vertex in self.graph_schema.vertices
                },
                "edges": {
                    edge.label: {
                        "relations": [
                            {"from_key": relation.from_key, "to_key": relation.to_key}
                            for relation in edge.relations
                        ],
                        "properties": [
                            property.property for property in edge.properties
                        ],
                    }
                    for edge in self.graph_schema.edges
                },
            }
        )

    def __str__(self) -> str:
        return self.graph_schema.model_dump_json()

    @property
    def schema_str(self) -> str:
        return str(self)


DEFAULT_GRAPH_SCHEMA = GraphSchema(
    GraphSchemaModel(
        vertices=[
            VertexModel(
                label="Entity",
                explain="点实体标签",
                properties=[
                    PropertyModel(
                        property="keyword", explain="点实体名", type="string"
                    ),
                    PropertyModel(
                        property="description", explain="点实体的描述", type="string"
                    ),
                ],
            )
        ],
        edges=[
            EdgeModel(
                label="Relation",
                explain="边实体标签",
                relations=[RelationModel(from_key="Entity", to_key="Entity")],
                properties=[
                    PropertyModel(
                        property="keyword", explain="边实体名", type="string"
                    ),
                    PropertyModel(
                        property="description", explain="边实体的描述", type="string"
                    ),
                ],
            )
        ],
    )
)
