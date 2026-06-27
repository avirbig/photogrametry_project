"""
export_results.py
=================
Exports the marked correspondence data in two ways:

  1.  data/correspondences.csv   — one row per point, one column per image
  2.  output_marked/             — copy of every image with all its marked
                                   points drawn on it (circles + labels)

Run:
    python export_results.py
"""
from __future__ import annotations

import csv
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

# ── paths ─────────────────────────────────────────────────────────────────────

ROOT          = os.path.dirname(os.path.abspath(__file__))
JSON_PATH     = os.path.join(ROOT, "data", "correspondences.json")
IMAGE_FOLDER  = os.path.join(ROOT, "building_pictures")
OUTPUT_FOLDER = os.path.join(ROOT, "output_marked")
CSV_PATH      = os.path.join(ROOT, "data", "correspondences.csv")

# ── visual settings ───────────────────────────────────────────────────────────

COLORS = [
    "#FF4444", "#44AAFF", "#44FF88", "#FFAA00", "#FF44FF",
    "#00FFFF", "#FFD700", "#FF8C00", "#ADFF2F", "#FF69B4",
    "#7FFFD4", "#FF6347", "#9370DB", "#20B2AA", "#F0E68C",
    "#87CEEB", "#DDA0DD", "#98FB98", "#F4A460", "#6495ED",
]

CIRCLE_RADIUS = 5    # pixels (at original image resolution)
FONT_SIZE     = 13


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i: i + 2], 16) for i in (0, 2, 4))  # type: ignore


def load_data(path: str) -> tuple[list[str], dict]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return raw["point_ids"], raw["marks"]


def export_csv(point_ids: list[str], marks: dict, image_names: list[str],
               out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # header row: point_id, then one column per image
        writer.writerow(["point_id"] + image_names)
        for pid in point_ids:
            row = [pid]
            for img in image_names:
                xy = marks.get(pid, {}).get(img)
                row.append(f"{xy[0]},{xy[1]}" if xy else "")
            writer.writerow(row)
    print(f"  CSV saved → {out_path}")


def draw_points_on_image(
    img_path: str,
    point_ids: list[str],
    marks: dict,
    out_path: str,
) -> int:
    """Draw all marks that exist for this image; returns count of points drawn."""
    img_name = os.path.basename(img_path)
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # try to get a decent font; fall back to default if not available
    font = label_font = None
    try:
        font       = ImageFont.truetype("arial.ttf", FONT_SIZE)
        label_font = ImageFont.truetype("arial.ttf", FONT_SIZE - 4)
    except (OSError, IOError):
        try:
            font       = ImageFont.load_default(size=FONT_SIZE)
            label_font = ImageFont.load_default(size=FONT_SIZE - 4)
        except TypeError:
            font = label_font = ImageFont.load_default()

    drawn = 0
    for i, pid in enumerate(point_ids):
        xy = marks.get(pid, {}).get(img_name)
        if xy is None:
            continue
        x, y = int(xy[0]), int(xy[1])
        color = _hex_to_rgb(COLORS[i % len(COLORS)])
        r = CIRCLE_RADIUS

        # thin black border then filled dot
        draw.ellipse(
            [(x - r - 1, y - r - 1), (x + r + 1, y + r + 1)],
            fill=(0, 0, 0),
        )
        draw.ellipse(
            [(x - r, y - r), (x + r, y + r)],
            fill=color,
        )

        # small semi-transparent label
        label = pid
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        lx, ly = x + r + 3, y - th // 2
        draw.rectangle(
            [(lx - 1, ly - 1), (lx + tw + 2, ly + th + 1)],
            fill=(0, 0, 0),
        )
        draw.text((lx, ly), label, fill=color, font=font)

        drawn += 1

    img.save(out_path)
    return drawn


def main() -> None:
    if not os.path.exists(JSON_PATH):
        sys.exit(f"ERROR: {JSON_PATH} not found. Run the marker tool first.")

    point_ids, marks = load_data(JSON_PATH)

    # collect image names from the building_pictures folder (same order as marker)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    image_names = sorted(
        f for f in os.listdir(IMAGE_FOLDER)
        if os.path.splitext(f)[1].lower() in exts
    )

    if not image_names:
        sys.exit(f"ERROR: no images found in {IMAGE_FOLDER}")

    print(f"Found {len(point_ids)} points across {len(image_names)} images.\n")

    # ── 1. CSV export ──────────────────────────────────────────────────────────
    print("Exporting CSV…")
    export_csv(point_ids, marks, image_names, CSV_PATH)

    # ── 2. Annotated images ────────────────────────────────────────────────────
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"\nSaving annotated images to: {OUTPUT_FOLDER}")

    total_drawn = 0
    for img_name in image_names:
        img_path = os.path.join(IMAGE_FOLDER, img_name)
        out_path = os.path.join(OUTPUT_FOLDER, img_name)
        n = draw_points_on_image(img_path, point_ids, marks, out_path)
        total_drawn += n
        status = f"{n} point(s)" if n else "no marks"
        print(f"  {img_name:<40}  {status}")

    print(f"\nDone. {total_drawn} total marks drawn across {len(image_names)} images.")


if __name__ == "__main__":
    main()
