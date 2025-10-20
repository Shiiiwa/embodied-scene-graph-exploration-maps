import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any


@dataclass
class Edge:
    source: str
    target: str
    relation: str

    def to_dict(self):
        return {"source": self.source, "target": self.target, "relation": self.relation}

    @staticmethod
    def from_dict(data):
        return Edge(**data)


@dataclass
class Node:
    object_id: str
    name: str
    position: Tuple[float, float, float]
    visibility: float = 0.0
    properties: Dict[str, Any] = None

    def to_dict(self, full=False):
        if full:
            return {
                "object_id": self.object_id,
                "name": self.name,
                "position": self.position,
                "visibility": self.visibility,
                "properties": self.properties,
            }
        else:
            return {"object_id": self.object_id, "name": self.name, "position": self.position}

    @staticmethod
    def from_dict(data):
        return Node(**data)


class SceneGraph:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []

    def add_node(self, node: Node):
        self.nodes[node.object_id] = node

    def add_edge(self, edge: Edge):
        self.edges.append(edge)

    def to_dict(self):
        return {"nodes": {k: v.to_dict() for k, v in self.nodes.items()}, "edges": [e.to_dict() for e in self.edges]}

    @staticmethod
    def from_dict(data):
        sg = SceneGraph()
        sg.nodes = {k: Node.from_dict(v) for k, v in data["nodes"].items()}
        sg.edges = [Edge.from_dict(e) for e in data["edges"]]
        return sg
