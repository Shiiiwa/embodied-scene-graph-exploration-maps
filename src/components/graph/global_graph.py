from src.components.graph.scene_graph import SceneGraph


class GlobalSceneGraph(SceneGraph):
    def __init__(self):
        super().__init__()

    def add_local_sg(self, local_sg: SceneGraph, alpha=0.8):
        for obj_id, node in local_sg.nodes.items():
            local_vis = node.visibility
            if obj_id not in self.nodes:
                self.add_node(node)
            else:
                global_vis = self.nodes[obj_id].visibility
                self.nodes[obj_id].visibility = 1 - (1 - global_vis) * (1 - alpha * local_vis)

        for edge in local_sg.edges:
            if edge not in self.edges:
                self.add_edge(edge)
