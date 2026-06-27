"""
point_data.py
Data model: stores and manages point markings across multiple images.
"""
from __future__ import annotations
import json
import os


class PointData:
    """Holds the correspondence table: {point_id -> {image_name -> (x, y)}}."""

    def __init__(self) -> None:
        self.point_ids: list[str] = []          # ordered list, e.g. ["P0","P1",...]
        self.marks: dict[str, dict[str, list]] = {}   # {pid: {img: [x, y]}}

    # ── mutation ──────────────────────────────────────────────────────────────

    def add_point(self) -> str:
        """Auto-create the next point ID (P0, P1, …) and return it."""
        new_id = f"P{len(self.point_ids)}"
        self.point_ids.append(new_id)
        self.marks[new_id] = {}
        return new_id

    def mark(self, point_id: str, image_name: str, x: int, y: int) -> None:
        """Record or overwrite the pixel position of point_id in image_name."""
        if point_id not in self.marks:
            self.marks[point_id] = {}
        self.marks[point_id][image_name] = [int(x), int(y)]

    def unmark(self, point_id: str, image_name: str) -> None:
        """Remove the mark for point_id in image_name (keeps the point ID)."""
        if point_id in self.marks:
            self.marks[point_id].pop(image_name, None)

    def delete_point(self, point_id: str) -> None:
        """Permanently remove a point ID and ALL its marks across every image."""
        if point_id in self.point_ids:
            self.point_ids.remove(point_id)
        self.marks.pop(point_id, None)

    def cleanup_empty(self) -> list[str]:
        """Remove and return IDs of points that have no marks in any image."""
        removed = []
        for pid in list(self.point_ids):
            if not self.marks.get(pid):
                self.point_ids.remove(pid)
                self.marks.pop(pid, None)
                removed.append(pid)
        return removed

    # ── queries ───────────────────────────────────────────────────────────────

    def get_mark(self, point_id: str, image_name: str) -> tuple[int, int] | None:
        """Return (x, y) if marked, else None."""
        val = self.marks.get(point_id, {}).get(image_name)
        return tuple(val) if val is not None else None

    def get_all_for_image(self, image_name: str) -> dict[str, tuple | None]:
        """Return {point_id: (x,y) | None} for every known point."""
        return {pid: self.get_mark(pid, image_name) for pid in self.point_ids}

    def first_unmarked(self, image_name: str) -> str | None:
        """First point_id not yet marked on this image, or None if all marked."""
        for pid in self.point_ids:
            if self.get_mark(pid, image_name) is None:
                return pid
        return None

    def next_unmarked_after(self, current_id: str, image_name: str) -> str | None:
        """Next unmarked point after current_id in order, or None."""
        try:
            start = self.point_ids.index(current_id) + 1
        except ValueError:
            start = 0
        for pid in self.point_ids[start:]:
            if self.get_mark(pid, image_name) is None:
                return pid
        return None

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"point_ids": self.point_ids, "marks": self.marks}, fh, indent=2)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.point_ids = data.get("point_ids", [])
        self.marks = data.get("marks", {})
