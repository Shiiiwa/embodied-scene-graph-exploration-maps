from typing import List, Tuple

import numpy as np

from src.components.graph.scene_graph import Edge


class RelationExtractor:

    GROUP_A = {
        "Floor",
        "Wall",
        "CounterTop",
        "Cabinet",
        "Drawer",
        "Fridge",
        "Sink",
        "Shelf",
        "SideTable",
        "DiningTable",
        "Chair",
        "ShelvingUnit",
        "StoveBurner",
        "Stool",
    }

    GROUP_B = {
        "Apple",
        "Banana",
        "Bread",
        "Potato",
        "Tomato",
        "Lettuce",
        "Knife",
        "Fork",
        "Spoon",
        "Pan",
        "Plate",
        "Cup",
        "Mug",
        "Bowl",
        "DishSponge",
        "SoapBottle",
        "Spatula",
        "WineBottle",
        "CoffeeMachine",
        "Toaster",
        "CreditCard",
        "CellPhone",
        "Kettle",
        "Book",
        "Pencil",
        "Pen",
        "AluminumFoil",
        "Vase",
        "Statue",
        "SaltShaker",
        "PepperShaker",
        "ButterKnife",
        "Egg",
        "PaperTowelRoll",
        "SprayBottle",
        "Mirror",
        "Bottle",
    }

    HANGABLES = {"Mirror", "Window", "LightSwitch", "Painting", "Picture", "Poster"}

    HANGING_RELATION_MAP = {
        "Mirror": "hanging_on",
        "Painting": "hanging_on",
        "Picture": "pasting_on",
        "Poster": "pasting_on",
        "Window": "embed_on",
        "LightSwitch": "fixed_on",
    }

    CONNECTABLES = {"Cabinet", "Shelf", "Drawer", "SideTable", "ShelvingUnit"}

    ATTACHMENTS = {"Faucet": ["Sink"], "StoveKnob": ["StoveBurner"], "Button": ["CoffeeMachine"], "LightSwitch": ["Wall"]}

    COMPONENTS = {"Drawer": ["CounterTop", "Cabinet", "SideTable", "Desk"], "StoveKnob": ["StoveBurner"], "Knob": ["StoveBurner"]}

    def __init__(self):
        pass

    def is_group_a(self, obj: dict) -> bool:
        return obj.get("objectType") in self.GROUP_A

    def is_group_b(self, obj: dict) -> bool:
        return obj.get("objectType") in self.GROUP_B

    def extract_relations(self, objects: List[dict]) -> List[Edge]:
        edges: List[Edge] = []
        edges.extend(self.extract_support_relationships(objects))
        edges.extend(self.extract_placement_relationships(objects))
        edges.extend(self.extract_hanging_relationships(objects))
        edges.extend(self.extract_position_relationships(objects))
        edges.extend(self.extract_connecting_relationships(objects))
        edges.extend(self.extract_attachment_relationships(objects))
        edges.extend(self.extract_component_relationships(objects))

        return edges

    def extract_support_relationships(self, objects: List[dict]) -> List[Edge]:
        edges: List[Edge] = []

        group_b_objects = [obj for obj in objects if obj.get("visible", False) and self.is_group_b(obj)]
        group_a_objects = [obj for obj in objects if obj.get("visible", False) and self.is_group_a(obj)]
        for a in group_b_objects:
            for b in group_a_objects:
                if a is b:
                    continue
                if self.support_condition(a, b, max_distance=0.05):
                    a_id = a["objectId"].replace(",", ".")
                    b_id = b["objectId"].replace(",", ".")
                    edges.append(Edge(a_id, b_id, "supported_by"))
                    edges.append(Edge(b_id, a_id, "supports"))

        return edges

    def extract_placement_relationships(self, objects: List[dict]) -> List[Edge]:
        edges: List[Edge] = []

        group_a_objects = [obj for obj in objects if obj.get("visible", False) and self.is_group_a(obj)]
        group_b_objects = [obj for obj in objects if obj.get("visible", False) and self.is_group_b(obj)]
        for a in group_a_objects:
            for b in group_b_objects:
                if a is b:
                    continue
                if self.support_condition(b, a, max_distance=0.3):
                    subtype = self.get_placement_subtype(b, a)
                    a_id = a["objectId"].replace(",", ".")
                    b_id = b["objectId"].replace(",", ".")
                    edges.append(Edge(b_id, a_id, subtype))
                    edges.append(Edge(a_id, b_id, "has_on_top"))
        return edges

    def extract_hanging_relationships(self, objects: List[dict]) -> List[Edge]:
        edges: List[Edge] = []
        hangables = [o for o in objects if o.get("visible", False) and o.get("objectType") in self.HANGABLES]

        for obj in hangables:
            # In ai2thor there is no object "wall" -> assume its always hanging on a wall
            # if ...
            relation = self.get_hanging_subtype(obj)
            obj_id = obj["objectId"].replace(",", ".")
            edges.append(Edge(obj_id, "Wall", relation))
            edges.append(Edge("Wall", obj_id, f"supports_{relation}"))

        return edges

    def extract_position_relationships(self, objects: List[dict]) -> List[Edge]:
        edges: List[Edge] = []
        visible_objects = [o for o in objects if o.get("visible", False)]

        parent_groups = {}
        for obj in visible_objects:
            parent_list = obj.get("parentReceptacles", [])
            if not parent_list:
                continue
            parent = parent_list[0]
            parent_groups.setdefault(parent, []).append(obj)

        for group in parent_groups.values():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a = group[i]
                    b = group[j]

                    a_id = a["objectId"].replace(",", ".")
                    b_id = b["objectId"].replace(",", ".")

                    pa = self.get_position(a)
                    pb = self.get_position(b)
                    dist = np.linalg.norm(np.array(pa) - np.array(pb))

                    if dist < 0.5:  #
                        edges.append(Edge(a_id, b_id, "close_by"))
                        edges.append(Edge(b_id, a_id, "close_by"))

                    xy_dist = np.linalg.norm(np.array(pa)[[0, 2]] - np.array(pb)[[0, 2]])
                    if xy_dist < 0.1:
                        if pa[1] > pb[1] + 0.05:
                            edges.append(Edge(a_id, b_id, "above"))
                            edges.append(Edge(b_id, a_id, "below"))
                        elif pb[1] > pa[1] + 0.05:
                            edges.append(Edge(b_id, a_id, "above"))
                            edges.append(Edge(a_id, b_id, "below"))

        return edges

    def extract_connecting_relationships(self, objects: List[dict]) -> List[Edge]:
        edges: List[Edge] = []

        connectables = [o for o in objects if o.get("visible", False) and o.get("objectType") in self.CONNECTABLES]

        for i in range(len(connectables)):
            for j in range(i + 1, len(connectables)):
                a = connectables[i]
                b = connectables[j]

                if a["objectType"] != b["objectType"]:
                    continue

                pa = self.get_position(a)
                pb = self.get_position(b)
                dist = np.linalg.norm(np.array(pa) - np.array(pb))

                if dist < 0.2:
                    a_id = a["objectId"].replace(",", ".")
                    b_id = b["objectId"].replace(",", ".")
                    edges.append(Edge(a_id, b_id, "connects_to"))
                    edges.append(Edge(b_id, a_id, "connects_to"))

        return edges

    def extract_attachment_relationships(self, objects: List[dict]) -> List[Edge]:
        edges: List[Edge] = []

        attachments = [o for o in objects if o.get("visible", False) and o.get("objectType") in self.ATTACHMENTS]
        candidates_by_type = {}

        # filter objecttypes
        for obj in objects:
            if not obj.get("visible", False):
                continue
            obj_type = obj.get("objectType")
            if obj_type not in candidates_by_type:
                candidates_by_type[obj_type] = []
            candidates_by_type[obj_type].append(obj)

        for a in attachments:
            a_type = a["objectType"]
            pa = a["axisAlignedBoundingBox"]["center"]
            a_id = a["objectId"].replace(",", ".")

            for main_type in self.ATTACHMENTS[a_type]:
                # Edge case -> Wall doesnt exist
                if main_type == "Wall":
                    # harcode object
                    b_id = "Wall"
                    edges.append(Edge(a_id, b_id, "attach_on"))
                    edges.append(Edge(b_id, a_id, "has_attachment"))
                    continue

                for b in candidates_by_type.get(main_type, []):
                    if a is b:
                        continue

                    pb = b["axisAlignedBoundingBox"]["center"]
                    distance = ((pa["x"] - pb["x"]) ** 2 + (pa["y"] - pb["y"]) ** 2 + (pa["z"] - pb["z"]) ** 2) ** 0.5
                    if distance < 0.15:
                        b_id = b["objectId"].replace(",", ".")
                        edges.append(Edge(a_id, b_id, "attach_on"))
                        edges.append(Edge(b_id, a_id, "has_attachment"))
                        break

        return edges

    def extract_component_relationships(self, objects: List[dict]) -> List[Edge]:
        edges: List[Edge] = []

        components = [o for o in objects if o.get("visible", False) and o.get("objectType") in self.COMPONENTS]
        candidates_by_type = {}

        for obj in objects:
            if not obj.get("visible", False):
                continue
            obj_type = obj.get("objectType")
            if obj_type not in candidates_by_type:
                candidates_by_type[obj_type] = []
            candidates_by_type[obj_type].append(obj)

        for comp in components:
            comp_type = comp["objectType"]
            pc = comp["axisAlignedBoundingBox"]["center"]
            comp_id = comp["objectId"].replace(",", ".")

            for main_type in self.COMPONENTS[comp_type]:
                for main in candidates_by_type.get(main_type, []):
                    if comp is main:
                        continue

                    pm = main["axisAlignedBoundingBox"]["center"]
                    distance = ((pc["x"] - pm["x"]) ** 2 + (pc["y"] - pm["y"]) ** 2 + (pc["z"] - pm["z"]) ** 2) ** 0.5
                    if distance < 0.3:  # 30cm max
                        main_id = main["objectId"].replace(",", ".")
                        edges.append(Edge(comp_id, main_id, "part_of"))
                        edges.append(Edge(main_id, comp_id, "has_part"))
                        break

        return edges

    def support_condition(self, a: dict, b: dict, max_distance) -> bool:
        try:
            pa = a["axisAlignedBoundingBox"]["center"]
            pb = b["axisAlignedBoundingBox"]["center"]
            sa = a["axisAlignedBoundingBox"]["size"]
            sb = b["axisAlignedBoundingBox"]["size"]
        except KeyError:
            return False

        vertical_distance = pa["y"] - pb["y"]
        if not (-0.03 < vertical_distance < max_distance):  # 30cm max.
            return False

        def xy_box(pos, size):
            return (pos["x"] - size["x"] / 2, pos["x"] + size["x"] / 2, pos["z"] - size["z"] / 2, pos["z"] + size["z"] / 2)

        ax1, ax2, az1, az2 = xy_box(pa, sa)
        bx1, bx2, bz1, bz2 = xy_box(pb, sb)

        inside = ax1 >= bx1 and ax2 <= bx2 and az1 >= bz1 and az2 <= bz2
        return inside

    def get_placement_subtype(self, obj_b: dict, obj_a: dict) -> str:
        """
        Subdivides 'on' relation into:
        - standing_on
        - sitting_on
        - lying_on
        """
        material = obj_b.get("salientMaterials", [])
        b_type = obj_b.get("objectType", "")
        a_type = obj_a.get("objectType", "")

        if "Metal" in material or "Ceramic" in material:
            if b_type in {"Statue", "Vase"} and a_type in {"Desk", "Shelf", "CounterTop", "Cabinet"}:
                return "standing_on"

        if "Sponge" in material or "Fabric" in material:
            if b_type in {"Pillow", "TeddyBear"} and a_type in {"Sofa", "Bed"}:
                return "lying_on"

        if b_type in {"Mug", "Cup", "Pan", "Bottle"}:
            return "sitting_on"

        # default fallback
        return "on"

    def get_hanging_subtype(self, obj: dict) -> str:
        return self.HANGING_RELATION_MAP.get(obj.get("objectType", ""), "hanging_on")

    def is_against_wall(self, obj: dict) -> bool:
        pos = obj["axisAlignedBoundingBox"]["center"]
        return abs(pos["z"]) > 2.4 or abs(pos["x"]) > 2.4

    def get_position(self, obj: dict) -> Tuple[float, float, float]:
        return tuple(obj.get("axisAlignedBoundingBox", {}).get("center", {}).values())

    def get_all_relation_types_with_index(self) -> dict:
        """
        Returns a dictionary mapping unique relation types to integer indices.
        """
        relation_set = set()

        # Manuelle Extraktion der Relationsnamen aus allen Extraktoren
        relation_set.update(["supported_by", "supports"])
        relation_set.update(["standing_on", "sitting_on", "lying_on", "on", "has_on_top"])
        relation_set.update(set(self.HANGING_RELATION_MAP.values()))
        relation_set.update([f"supports_{rel}" for rel in self.HANGING_RELATION_MAP.values()])
        relation_set.update(["close_by", "above", "below"])
        relation_set.update(["connects_to"])
        relation_set.update(["attach_on", "has_attachment"])
        relation_set.update(["part_of", "has_part"])

        # Sort for reproducibility
        sorted_relations = sorted(relation_set)

        # Assign an index to each relation
        relation_dict = {relation: idx for idx, relation in enumerate(sorted_relations)}

        return relation_dict


if __name__ == "__main__":
    extractor = RelationExtractor()
    relations = extractor.get_all_relation_types_with_index()
    print(relations)
