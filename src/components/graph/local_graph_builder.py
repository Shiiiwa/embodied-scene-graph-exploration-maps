import numpy as np

from src.components.graph.RelationExtractor import RelationExtractor
from src.components.graph.scene_graph import Node, SceneGraph


class LocalSceneGraphBuilder:
    def __init__(self):
        self.relation_extractor = RelationExtractor()

    def build_from_metadata(self, metadata: dict) -> SceneGraph:
        sg = SceneGraph()
        objects = metadata.get("objects", [])

        for obj in objects:
            if obj["visible"]:
                object_id = obj["objectId"]
                # Fix locale issue: convert comma to dot in numeric parts of objectId if needed
                if "," in object_id and "." not in object_id:
                    object_id = object_id.replace(",", ".")
                node = Node(
                    object_id=object_id,
                    name=obj["objectType"],
                    position=tuple(obj["position"].values()),
                    visibility=self.compute_soft_visibility(obj),
                    properties={k: v for k, v in obj.items() if k not in ("objectId", "objectType", "position", "visible")},
                )

                sg.add_node(node)

        edges = self.relation_extractor.extract_relations(objects)
        for edge in edges:
            sg.add_edge(edge)

        return sg

    @staticmethod
    def compute_soft_visibility(obj):
        """
        Calculates a soft visibility score for an object, ranging from 0 (not visible) to 1 (fully visible).

        The score is computed using a sigmoid function based on the distance between the agent and the object.
        To account for the fact that larger objects can typically be seen from further away, the center of the
        sigmoid is dynamically adjusted based on the maximum dimension of the object (`max_dim`):

            - `base_center` sets the reference distance for objects of size `ref_size`.
            - For objects larger than `ref_size`, the sigmoid center is shifted farther away, making them easier
              to see from a distance.
            - The sharpness of the sigmoid controls how quickly the visibility score drops off as the distance increases.

        Args:
            obj (dict): Object metadata, expected to have 'distance' (float) and 'size' (dict with 'x', 'y', 'z').

        Returns:
            float: Soft visibility score between 0 and 1.
        """
        vis_dist = obj.get("distance", None)
        if vis_dist is None:
            raise ValueError("Distance of object not available")

        size_dict = obj["axisAlignedBoundingBox"].get("size", {"x": 1.0, "y": 1.0, "z": 1.0})
        max_dim = max(size_dict["x"], size_dict["y"], size_dict["z"])

        base_center = 2.5
        size_scale = 3.0
        ref_size = 0.5
        sharpness = 1.5

        center = base_center + size_scale * (max_dim - ref_size)
        soft_score = 1 / (1 + np.exp(sharpness * (vis_dist - center)))
        return float(soft_score)
