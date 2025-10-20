import numpy as np
from collections import defaultdict

class MetricSemanticMapV2:
    """
    Dichtere Karte mit:
      - p_free (0..1) + confidence (unsicherheitssensitiv)
      - visit_count + time_since_visit (Zerfall)
      - frontier mask
      - verdichtete Objekt-Semantik (Top-K Klassen-Heatmaps)
      - globale Stats für die Policy
    API-kompatibel zu v1 (update/get/to_dict/from_dict/render_ascii/reset).
    """

    _ALPHA_FREE = 0.35      # EMA für p_free bei Beobachtung "walkable"
    _BETA_CONF  = 0.20      # Anhebung der confidence pro Beobachtung
    _DECAY_VISIT = 0.001    # Zerfall für time_since_visit
    _CONF_MIN_FOR_FRONTIER = 0.1
    _FREE_MIN_FOR_FRONTIER = 0.5
    _TOPK_CLASSES = 16

    def __init__(self, map_shape):
        self.map_shape = map_shape  # (H, W)
        H, W = map_shape

        # dichte Raster
        self.p_free     = np.full((H, W), 0.5, dtype=np.float32)   # 0.5 = unbekannt
        self.confidence = np.zeros((H, W), dtype=np.float32)       # 0..1
        self.visited    = np.zeros((H, W), dtype=np.uint8)         # 0/1
        self.visit_cnt  = np.zeros((H, W), dtype=np.float32)
        self.last_visit = np.full((H, W), -1, dtype=np.int32)
        self.frontier   = np.zeros((H, W), dtype=np.uint8)         # 0/1

        # semantische Verdichtung (pro Zelle max-Vis. pro Klasse)
        self.class_index = {}          # dynamische Zuordnung: classname -> channel idx (0..K-1)
        self.class_heat  = np.zeros((self._TOPK_CLASSES, H, W), dtype=np.float32)

        # Instanz-/Detail-Infos wie in v1 (kompatible Repräsentation)
        self.cells = {}  # {(i,j): {"visited":1,"walkable":bool,"objects":{"types":{cls->vis},"instances":{id->...}}}}

        # globaler Step-Zähler (für time_since_visit)
        self._t = 0

    def reset(self):
        H, W = self.map_shape
        self.p_free.fill(0.5)
        self.confidence.fill(0.0)
        self.visited.fill(0)
        self.visit_cnt.fill(0.0)
        self.last_visit.fill(-1)
        self.frontier.fill(0)
        self.class_index.clear()
        self.class_heat.fill(0.0)
        self.cells.clear()
        self._t = 0

    def update(self, cell_coords: tuple[int, int], event, local_sg, walkable=None):
        """
        cell_coords: (i,j) aktuelle Agenten-Zelle
        walkable: bool oder None (from env.try_action aggregation)
        """
        self._t += 1
        i, j = cell_coords
        H, W = self.map_shape
        if not (0 <= i < H and 0 <= j < W):
            return

        # --- dichte Raster-Updates ---
        self.visited[i, j] = 1
        self.visit_cnt[i, j] += 1.0
        self.last_visit[i, j] = self._t

        # time_since_visit natürlicher Zerfall (wird als Feature aus t-last verwendet)

        # --- semantische Verdichtung aus local_sg ---
        cell = self.cells.get((i, j))
        stored_walkable = None
        if cell is not None:
            stored_walkable = cell.get("walkable", None)

        if walkable is not None:
            walkable_flag = bool(walkable)
        elif stored_walkable is not None:
            walkable_flag = bool(stored_walkable)
        else:
            walkable_flag = True

        # Begehbarkeit / Unsicherheit updaten (einfaches EMA + Confidence)
        target = 1.0 if walkable_flag else 0.0
        self.p_free[i, j] = (1.0 - self._ALPHA_FREE) * self.p_free[i, j] + self._ALPHA_FREE * target
        # Confidence steigt bei jeder Beobachtung; clamp
        self.confidence[i, j] = min(1.0, self.confidence[i, j] + self._BETA_CONF)

        if cell is None:
            cell = {
                "visited": 1,
                "walkable": bool(walkable_flag),
                "objects": {"types": {}, "instances": {}},
            }
            self.cells[(i, j)] = cell
        else:
            cell["visited"] = 1
            cell["walkable"] = bool(walkable_flag)

        for _, node in local_sg.nodes.items():
            obj_id   = getattr(node, "object_id", None) or getattr(node, "id", None)
            obj_type = getattr(node, "name", None) or getattr(node, "object_type", None)
            if obj_type is None and isinstance(obj_id, str) and "|" in obj_id:
                obj_type = obj_id.split("|", 1)[0]
            vis = float(getattr(node, "visibility", 0.0) or 0.0)
            vis = 0.0 if vis < 0 else (1.0 if vis > 1.0 else vis)

            # v1-kompatible Strukturen auffüllen
            if obj_id:
                prev = cell["objects"]["instances"].get(obj_id, {})
                if vis > float(prev.get("vis", 0.0)):
                    cell["objects"]["instances"][obj_id] = {"type": obj_type or "Unknown", "vis": vis}
            if obj_type:
                prev_type_vis = float(cell["objects"]["types"].get(obj_type, 0.0))
                if vis > prev_type_vis:
                    cell["objects"]["types"][obj_type] = vis

                # Dense Top-K Heatmaps:
                ch = self._class_channel(obj_type)
                # Max-Pooling über Zeit (alternativ EMA, hier max = robust gegenüber Wiederholung)
                self.class_heat[ch, i, j] = max(self.class_heat[ch, i, j], vis)

        # --- Frontier nachführen (lokal + Nachbarschaft) ---
        self._recompute_frontier()

    def _recompute_frontier(self):
        """Global, vektorisierte Frontier-Berechnung:
        Frontier = bekannte freie Zelle (p_free, confidence über Schwellwert)
                   mit mindestens einem unbekannten Nachbarn (confidence unter Schwellwert).
        """
        H, W = self.map_shape

        # "Bekannt-frei": genug Confidence & ausreichend frei
        known_free = (self.confidence >= self._CONF_MIN_FOR_FRONTIER) & \
                     (self.p_free >= self._FREE_MIN_FOR_FRONTIER)

        # "Unbekannt": zu geringe Confidence
        unknown = (self.confidence < self._CONF_MIN_FOR_FRONTIER)

        # Nachbarschafts-Maps (4er-Nachbarschaft), ränder-safe
        up = np.zeros_like(unknown, dtype=bool);
        up[1:, :] = unknown[:-1, :]
        down = np.zeros_like(unknown, dtype=bool);
        down[:-1, :] = unknown[1:, :]
        left = np.zeros_like(unknown, dtype=bool);
        left[:, 1:] = unknown[:, :-1]
        right = np.zeros_like(unknown, dtype=bool);
        right[:, :-1] = unknown[:, 1:]

        unknown_nbr = up | down | left | right

        frontier = known_free & unknown_nbr

        # Binärmaske schreiben
        self.frontier[...] = frontier.astype(np.uint8)

    def get(self):
        """Kompatibler Rückgabewert wie v1 (für Logs/Serialisierung)."""
        return self.cells

    def to_dict(self):
        """Serialisierung – kombiniert dichte Raster + v1-Cells."""
        H, W = self.map_shape
        visited_list = []
        for (i, j), cell in self.cells.items():
            visited_list.append({
                "i": i, "j": j,
                "visited": cell["visited"],
                "walkable": cell.get("walkable", None),
                "objects": {
                    "types": cell["objects"]["types"],
                    "instances": cell["objects"]["instances"]
                }
            })
        return {
            "map_shape": self.map_shape,
            "visited": visited_list,
            "dense": {
                "p_free": self.p_free.tolist(),
                "confidence": self.confidence.tolist(),
                "visited_mask": self.visited.tolist(),
                "visit_cnt": self.visit_cnt.tolist(),
                "last_visit": self.last_visit.tolist(),
                "frontier": self.frontier.tolist(),
                "class_index": self.class_index,
                "class_heat": self.class_heat.tolist(),
                "t": int(self._t),
            }
        }

    @classmethod
    def from_dict(cls, data):
        obj = cls(map_shape=tuple(data["map_shape"]))
        # v1-Teil
        if "visited" in data:
            for entry in data["visited"]:
                i, j = entry["i"], entry["j"]
                obj.cells[(i, j)] = {
                    "visited": entry.get("visited", 1),
                    "walkable": entry.get("walkable", None),
                    "objects": entry.get("objects", {})
                }
                obj.visited[i, j] = 1
        # dichte Raster (falls vorhanden)
        dense = data.get("dense", None)
        if dense:
            obj.p_free     = np.array(dense["p_free"], dtype=np.float32)
            obj.confidence = np.array(dense["confidence"], dtype=np.float32)
            obj.visited    = np.array(dense["visited_mask"], dtype=np.uint8)
            obj.visit_cnt  = np.array(dense["visit_cnt"], dtype=np.float32)
            obj.last_visit = np.array(dense["last_visit"], dtype=np.int32)
            obj.frontier   = np.array(dense["frontier"], dtype=np.uint8)
            obj.class_index = dict(dense.get("class_index", {}))
            obj.class_heat  = np.array(dense["class_heat"], dtype=np.float32)
            obj._t = int(dense.get("t", 0))
        return obj

    # --- Zusätzliche Schnittstellen für die Policy ---

    def get_map_for_policy(self):
        """
        Liefert (features, meta):
          features: np.ndarray (C, H, W) – dichte, normierte Kanäle
          meta: dict mit globalen Stats und class_index
        """
        H, W = self.map_shape

        # time_since_visit aus t-last_visit (normiert ~[0,1])
        tsv = np.where(self.last_visit >= 0, (self._t - self.last_visit).astype(np.float32), 1e6)
        # weiches Clipping (z.B. 500 Steps -> 1.0)
        tsv = np.clip(tsv / 500.0, 0.0, 1.0)

        # Normierungen
        pf   = self.p_free
        conf = self.confidence
        vism = self.visited.astype(np.float32)
        frc  = self.frontier.astype(np.float32)
        vcnt = np.log1p(self.visit_cnt) / np.log(1.0 + 10.0)  # auf ~[0,1] bei 10+ Besuchen

        # Feature-Stack (Basiskanäle)
        feats = [
            pf, conf, vism, tsv, frc, vcnt
        ]

        # Top-K Klassenheatmaps (bereits [0..1])
        feats.append(self.class_heat)

        feat_stack = np.concatenate(
            [f[None, ...] if f.ndim == 2 else f for f in feats], axis=0
        )  # (C,H,W)

        # Globale Stats (nützlich als zusätzlicher Vektor)
        coverage = vism.mean()
        mean_conf_visited = conf[vism > 0.5].mean() if (vism > 0.5).any() else 0.0
        num_frontiers = float(frc.sum())

        meta = {
            "class_index": dict(self.class_index),
            "coverage": float(coverage),
            "mean_confidence_visited": float(mean_conf_visited),
            "num_frontiers": num_frontiers,
        }
        return feat_stack.astype(np.float32), meta

    def render_matplotlib(self, show=True, save_path=None):
        # Platzhalter – optional
        pass

    def render_ascii(self):
        H, W = self.map_shape
        for i in range(H):
            line = []
            for j in range(W):
                if self.frontier[i, j]:
                    line.append("🟩")
                else:
                    if self.visited[i, j]:
                        line.append("⬛" if self.p_free[i, j] >= 0.5 else "🟥")
                    else:
                        line.append("⬜")
            print("".join(line))

    # ------------- Hilfsfunktionen ----------------

    def _class_channel(self, name: str) -> int:
        """Dynamisch Top-K: seltene Klassen werden auf den letzten Kanal 'other' gemappt."""
        if name in self.class_index:
            return self.class_index[name]
        # Wenn noch Platz, neue Klasse anlegen
        if len(self.class_index) < self._TOPK_CLASSES - 1:
            ch = len(self.class_index)
            self.class_index[name] = ch
            return ch
        # Sonst 'other' (letzter Kanal)
        return self._TOPK_CLASSES - 1

    def _update_frontiers_around(self, i, j):
        """FIXED: Frontier nur für besuchte Zellen setzen, Rest auf 0."""
        H, W = self.map_shape

        # Reset frontier für ALLE Zellen in der Nähe (nicht nur die 5!)
        radius = 2  # Größerer Radius
        for ii in range(max(0, i - radius), min(H, i + radius + 1)):
            for jj in range(max(0, j - radius), min(W, j + radius + 1)):
                if self.visited[ii, jj]:  # Nur besuchte prüfen
                    if self.p_free[ii, jj] >= self._FREE_MIN_FOR_FRONTIER and \
                            self.confidence[ii, jj] >= self._CONF_MIN_FOR_FRONTIER:
                        # Check unbekannte Nachbarn
                        unknown_nb = False
                        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ni, nj = ii + di, jj + dj
                            if 0 <= ni < H and 0 <= nj < W and self.visited[ni, nj] == 0:
                                unknown_nb = True
                                break
                        self.frontier[ii, jj] = 1 if unknown_nb else 0
                    else:
                        self.frontier[ii, jj] = 0
                else:

                    self.frontier[ii, jj] = 0
