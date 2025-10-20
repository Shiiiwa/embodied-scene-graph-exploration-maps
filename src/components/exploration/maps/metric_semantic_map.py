class MetricSemanticMap:

    def __init__(self, map_shape):
        self.map_shape = map_shape
        self.cells = {}
        self.reset()

    def reset(self):
        self.cells.clear()


    def update(self, cell_coords: tuple[int, int], event, local_sg, walkable=None):
        i, j = cell_coords
        H, W = self.map_shape
        if not (0 <= i < H and 0 <= j < W):
            return

        cell = self.cells.get((i, j))
        if cell is None:
            cell = {
                "visited": 1,
                "walkable": bool(walkable) if walkable is not None else None,
                "objects": {
                    "types": {},
                    "instances": {}
                }
            }
            self.cells[(i, j)] = cell
        else:
            cell["visited"] = 1
            if walkable is not None:
                cell["walkable"] = bool(walkable)

        for _, node in local_sg.nodes.items():
            obj_id = getattr(node, "object_id", None) or getattr(node, "id", None)
            obj_type = getattr(node, "name", None) or getattr(node, "object_type", None)
            if obj_type is None and isinstance(obj_id, str) and "|" in obj_id:
                obj_type = obj_id.split("|", 1)[0]

            vis = float(getattr(node, "visibility", 0.0) or 0.0)
            vis = max(0.0, min(1.0, vis))

            if obj_id:
                prev = cell["objects"]["instances"].get(obj_id, {})
                prev_vis = float(prev.get("vis", 0.0))
                if vis > prev_vis:
                    cell["objects"]["instances"][obj_id] = {"type": obj_type or "Unknown", "vis": vis}

            if obj_type:
                prev_type_vis = float(cell["objects"]["types"].get(obj_type, 0.0))
                if vis > prev_type_vis:
                    cell["objects"]["types"][obj_type] = vis



    def get(self):
        return self.cells

    def _parse_cell_key(self, key):
        """Return (i, j) indices for a cell key if possible."""

        if isinstance(key, tuple):
            if len(key) == 2:
                return key
            if len(key) > 2:
                return key[0], key[1]

        if isinstance(key, list) and len(key) >= 2:
            return key[0], key[1]

        if isinstance(key, str):
            cleaned = key.strip().lstrip("(").rstrip(")")
            parts = [p.strip() for p in cleaned.split(",") if p.strip()]
            if len(parts) >= 2:
                try:
                    return int(parts[0]), int(parts[1])
                except ValueError:
                    pass

        return None

    def to_dict(self):
        visited_list = []
        for key, cell in self.cells.items():
            parsed = self._parse_cell_key(key)
            if parsed is None:
                continue

            i, j = parsed
            visited_list.append({
                "i": i,
                "j": j,
                "visited": cell["visited"],
                "walkable": cell["walkable"],
                "objects": {
                    "types": cell["objects"]["types"],
                    "instances": cell["objects"]["instances"]
                }
            })
        return {"map_shape": self.map_shape, "visited": visited_list}

    def render_matplotlib(self, show=True, save_path=None):
        pass

    def render_ascii(self):
        H, W = self.map_shape
        for i in range(H):
            line = []
            for j in range(W):
                cell = self.cells.get((i, j))
                if cell and cell["visited"]:
                    line.append("⬛" if cell["walkable"] else "🟥")
                else:
                    line.append("⬜")
            print("".join(line))

    def print_visited_with_objects(self, visibility_threshold: float = 0.0):
        for (i, j), cell in self.cells.items():
            if cell["visited"]:
                filtered = {k: round(v, 2) for k, v in cell["objects"].items() if v >= visibility_threshold}
                print(f"Zelle ({i},{j}): Objects = {filtered}")

    @classmethod
    def from_dict(cls, data):
        obj = cls(map_shape=tuple(data["map_shape"]))

        if "visited" in data:
            for entry in data["visited"]:
                i, j = entry["i"], entry["j"]
                obj.cells[(i, j)] = {
                    "visited": entry.get("visited", 1),
                    "walkable": entry.get("walkable", None),
                    "objects": entry.get("objects", {})
                }
        else:
            legacy_map = data["map"]
            H = len(legacy_map)
            W = len(legacy_map[0]) if H > 0 else 0
            for i in range(H):
                for j in range(W):
                    cell = legacy_map[i][j]
                    if cell.get("visited", 0):
                        obj.cells[(i, j)] = {
                            "visited": 1,
                            "walkable": cell.get("walkable", None),
                            "objects": cell.get("objects", {})
                        }
        return obj
