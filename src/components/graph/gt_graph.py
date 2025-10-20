import json
from typing import Dict, List

from src.components.graph.global_graph import GlobalSceneGraph
from src.components.graph.scene_graph import Edge, Node


class GTGraph(GlobalSceneGraph):
    def __init__(self):
        super().__init__()
        self.viewpoint_to_objects: Dict[str, List[Dict[str, float]]] = {}

    def add_viewpoint(self, viewpoint, nodes):
        if viewpoint in self.viewpoint_to_objects:
            raise ValueError(f"Viewpoint {viewpoint} already exists in the graph.")
        objects = []
        for key, node in nodes.items():
            objects.append({key: node.visibility})
        self.viewpoint_to_objects[viewpoint] = objects

    @classmethod
    def load_from_file(cls, path: str):
        with open(path, "r") as f:
            data = json.load(f)
        graph = cls()
        for node in data["nodes"]:
            graph.add_node(
                Node(
                    object_id=(node["object_id"].replace(",", ".") if "," in node["object_id"] else node["object_id"]),
                    name=node["name"],
                    position=tuple(node["position"]),
                    visibility=1,
                    properties=node["properties"],
                )
            )
        for edge in data["edges"]:
            graph.add_edge(
                Edge(
                    source=(edge["source"].replace(",", ".") if "," in edge["source"] else edge["source"]),
                    target=(edge["target"].replace(",", ".") if "," in edge["target"] else edge["target"]),
                    relation=edge["relation"],
                )
            )

        # load viewpoints
        graph.viewpoint_to_objects = data["viewpoint_to_objects"]

        return graph

    def save_to_file(self, path: str):
        with open(path, "w") as f:
            json.dump(
                {
                    "nodes": [
                        {
                            "object_id": str(node.object_id),
                            "name": node.name,
                            "position": node.position,
                            "visibility": 1,
                            "properties": node.properties,
                        }
                        for node in self.nodes.values()
                    ],
                    "edges": [{"source": str(edge.source), "target": str(edge.target), "relation": edge.relation} for edge in self.edges],
                    "viewpoint_to_objects": self.viewpoint_to_objects,
                },
                f,
                indent=2,
            )
