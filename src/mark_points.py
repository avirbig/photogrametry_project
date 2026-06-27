"""
mark_points.py
Interactive matplotlib GUI for manually marking correspondence points
across a set of photographs.

Layout
------
  [◄ Prev]  [image_name (N/total)]  [Next ►]  |  [+ Point]  [Del Mark]  [Save]  [Table]
  ┌─────────────────────────────────────┬─────────────────────────────────────┐
  │                                     │  Points:                            │
  │         MAIN IMAGE                  │  (●) P0    x: 234    y: 456        │
  │    click → mark active point        │  (○) P1    x: 100    y: 200        │
  │    colored dot + label per point    │  (○) P2    x:  —     y:  —         │
  │    active point = larger + ring     │  (○) P3    x:  —     y:  —         │
  │                                     │                                     │
  └─────────────────────────────────────┴─────────────────────────────────────┘
  [◄]  [thumb0]  [thumb1]  [thumb2]  [thumb3]  [thumb4]  [thumb5]  [►]

Interaction
-----------
  - Click main image         → mark active point (auto-advance to next unmarked)
  - Click panel row          → select that point as active (manual override / skip)
  - Click thumbnail          → navigate to that image
  - [+ Point]                → create P0, P1, … in order, auto-select new point
  - [Del Mark]               → remove active point's mark from current image
  - [Save]                   → write data/correspondences.json
  - [Table]                  → open correspondence table in a new window
  - [◄ Prev / Next ►]        → navigate images
  - [◄ / ►] (thumbnails)     → scroll thumbnail strip
"""

from __future__ import annotations
import os
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.widgets import Button
from PIL import Image

try:
    from .point_data import PointData   # when imported as part of the src package
except ImportError:
    from point_data import PointData    # when src/ is directly on sys.path

# ── constants ─────────────────────────────────────────────────────────────────

THUMB_W, THUMB_H = 130, 98          # max thumbnail pixel dimensions
THUMBS_VISIBLE   = 6                # thumbnails shown at once in the strip
ZOOM_FACTOR      = 0.80             # scroll-wheel zoom: 0.80 = 20% zoom-in per step

POINT_COLORS = [
    "#FF4444", "#44BB44", "#FFD700", "#4488FF", "#FF8C00",
    "#AA44FF", "#00CED1", "#FF69B4", "#7CFC00", "#FFA07A",
    "#DC143C", "#32CD32", "#1E90FF", "#FF6347", "#9400D3",
]

_BG        = "#2b2b2b"   # figure background
_PANEL_BG  = "#1e1e1e"   # main image / panel background
_HDR_CLR   = "#aaaaaa"   # column header text
_ACTIVE_A  = 0.18        # alpha for active-row highlight


# ── helper ────────────────────────────────────────────────────────────────────

def _make_btn(ax, label: str, color="#444444", hover="#666666") -> Button:
    btn = Button(ax, label, color=color, hovercolor=hover)
    btn.label.set_color("white")
    btn.label.set_fontsize(9)
    return btn


# ── app ───────────────────────────────────────────────────────────────────────

class PointMarkerApp:
    def __init__(self, image_folder: str, json_path: str) -> None:
        self.image_folder = image_folder
        self.json_path    = json_path
        self.data         = PointData()

        # discover images
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif",
                "*.JPG", "*.JPEG", "*.PNG", "*.BMP"]
        paths: list[str] = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(image_folder, ext)))
        self.image_paths = sorted(set(paths))
        self.image_names = [os.path.basename(p) for p in self.image_paths]

        if not self.image_paths:
            raise FileNotFoundError(f"No images found in '{image_folder}'")

        # state
        self.current_idx:      int       = 0
        self.active_point_id:  str | None = None
        self.thumb_offset:     int       = 0
        self._dirty:           bool      = False
        self._panel_rows:      list      = []   # [(y_bot, y_top, pid), ...]
        self._selected:        set       = set()  # point IDs checked for bulk-delete
        self._panel_offset:    int       = 0      # first visible row index (scroll)
        self._last_panel_idx:  int | None = None  # global idx of last clicked row (Shift)
        self._shift_held:      bool      = False
        self._ctrl_held:       bool      = False
        self._main_xlim:       list | None = None  # saved zoom state (None = full view)
        self._main_ylim:       list | None = None

        # image caches
        self._img_cache:   dict[str, np.ndarray] = {}
        self._thumb_cache: dict[str, dict]       = {}

        # load existing data
        if os.path.exists(json_path):
            self.data.load(json_path)
            self._sync_active()

        self._build_ui()
        self._redraw_all()

    # ── properties ───────────────────────────────────────────────────────────

    @property
    def current_image_name(self) -> str:
        return self.image_names[self.current_idx]

    # ── helpers ───────────────────────────────────────────────────────────────

    def _point_color(self, point_id: str) -> str:
        if point_id not in self.data.point_ids:
            return "#888888"
        return POINT_COLORS[self.data.point_ids.index(point_id) % len(POINT_COLORS)]

    def _load_image(self, path: str) -> np.ndarray:
        if path not in self._img_cache:
            self._img_cache[path] = np.array(Image.open(path).convert("RGB"))
        return self._img_cache[path]

    def _load_thumb(self, path: str) -> dict:
        """Returns {'arr': np.ndarray, 'orig_w': int, 'orig_h': int}."""
        if path not in self._thumb_cache:
            img = Image.open(path).convert("RGB")
            orig_w, orig_h = img.size
            img.thumbnail((THUMB_W, THUMB_H), Image.LANCZOS)
            self._thumb_cache[path] = {
                "arr":    np.array(img),
                "orig_w": orig_w,
                "orig_h": orig_h,
            }
        return self._thumb_cache[path]

    def _sync_active(self) -> None:
        """Set active point to first unmarked on current image, or last point."""
        first = self.data.first_unmarked(self.current_image_name)
        if first:
            self.active_point_id = first
        elif self.data.point_ids:
            self.active_point_id = self.data.point_ids[-1]
        else:
            self.active_point_id = None

    def _ensure_thumb_visible(self, idx: int) -> None:
        """Scroll thumbnail strip so image at idx is visible."""
        if idx < self.thumb_offset:
            self.thumb_offset = idx
        elif idx >= self.thumb_offset + THUMBS_VISIBLE:
            self.thumb_offset = max(0, idx - THUMBS_VISIBLE + 1)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.fig = plt.figure(figsize=(18, 10))
        self.fig.patch.set_facecolor(_BG)

        # ── main image axes ──────────────────────────────────────────────────
        self.ax_main = self.fig.add_axes([0.01, 0.21, 0.62, 0.71])
        self.ax_main.set_facecolor(_PANEL_BG)

        # ── panel axes (right side) ──────────────────────────────────────────
        self.ax_panel = self.fig.add_axes([0.65, 0.21, 0.34, 0.71])
        self.ax_panel.set_facecolor(_PANEL_BG)

        # ── thumbnail axes ───────────────────────────────────────────────────
        # 6 thumbs between x=0.06 and x=0.93, with equal spacing
        thumb_ax_w  = 0.13
        thumb_ax_h  = 0.17
        thumb_y     = 0.01
        available   = 0.93 - 0.06
        thumb_gap   = (available - THUMBS_VISIBLE * thumb_ax_w) / (THUMBS_VISIBLE - 1)
        self.ax_thumbs: list[plt.Axes] = []
        for i in range(THUMBS_VISIBLE):
            x = 0.06 + i * (thumb_ax_w + thumb_gap)
            ax = self.fig.add_axes([x, thumb_y, thumb_ax_w, thumb_ax_h])
            ax.set_facecolor("#333333")
            self.ax_thumbs.append(ax)

        # ── toolbar ──────────────────────────────────────────────────────────
        btn_y = 0.94
        btn_h = 0.05

        ax_prev  = self.fig.add_axes([0.01, btn_y, 0.05, btn_h])
        ax_next  = self.fig.add_axes([0.07, btn_y, 0.05, btn_h])
        self.ax_title = self.fig.add_axes([0.13, btn_y, 0.16, btn_h])
        for sp in self.ax_title.spines.values():
            sp.set_visible(False)
        self.ax_title.set_xticks([])
        self.ax_title.set_yticks([])
        self.ax_title.set_facecolor(_BG)

        ax_add   = self.fig.add_axes([0.30, btn_y, 0.055, btn_h])   # + Point
        ax_clr   = self.fig.add_axes([0.36, btn_y, 0.055, btn_h])   # Clear  (active, cur img)
        ax_delpt = self.fig.add_axes([0.42, btn_y, 0.055, btn_h])   # Del Pt (active, all imgs)
        ax_clrs  = self.fig.add_axes([0.48, btn_y, 0.055, btn_h])   # Clr Sel (selected, cur img)
        ax_delp  = self.fig.add_axes([0.54, btn_y, 0.055, btn_h])   # Del Pts (selected, all imgs)
        ax_clra  = self.fig.add_axes([0.60, btn_y, 0.055, btn_h])   # Clr All (confirm)
        ax_save  = self.fig.add_axes([0.66, btn_y, 0.055, btn_h])
        ax_table = self.fig.add_axes([0.72, btn_y, 0.055, btn_h])
        ax_reset = self.fig.add_axes([0.78, btn_y, 0.12,  btn_h])

        ax_tprev = self.fig.add_axes([0.01, 0.01, 0.04, thumb_ax_h])
        ax_tnext = self.fig.add_axes([0.94, 0.01, 0.04, thumb_ax_h])

        self.btn_prev  = _make_btn(ax_prev,  "◄ Prev")
        self.btn_next  = _make_btn(ax_next,  "Next ►")
        self.btn_add   = _make_btn(ax_add,   "+ Point",  "#1a7a3a", "#2a9a4a")
        self.btn_clr   = _make_btn(ax_clr,   "Clear",    "#8a2222", "#aa3333")
        self.btn_delpt = _make_btn(ax_delpt, "Del Pt",   "#8a3a00", "#b05000")
        self.btn_clrs  = _make_btn(ax_clrs,  "Clr Sel",  "#6a5500", "#8a7000")
        self.btn_delp  = _make_btn(ax_delp,  "Del Pts",  "#7a2200", "#aa3300")
        self.btn_clra  = _make_btn(ax_clra,  "Clr All",  "#550055", "#770077")
        self.btn_save  = _make_btn(ax_save,  "Save",     "#224488", "#335599")
        self.btn_table = _make_btn(ax_table, "Table")
        self.btn_reset = _make_btn(ax_reset, "⌂ Reset Zoom", "#555555", "#777777")
        self.btn_tprev = _make_btn(ax_tprev, "◄")
        self.btn_tnext = _make_btn(ax_tnext, "►")

        self.btn_prev .on_clicked(self._on_prev)
        self.btn_next .on_clicked(self._on_next)
        self.btn_add  .on_clicked(self._on_add_point)
        self.btn_clr  .on_clicked(self._on_del_mark)
        self.btn_delpt.on_clicked(self._on_del_point)
        self.btn_clrs .on_clicked(self._on_clr_selected)
        self.btn_delp .on_clicked(self._on_del_selected)
        self.btn_clra .on_clicked(self._on_clr_all)
        self.btn_save .on_clicked(self._on_save)
        self.btn_table.on_clicked(self._on_table)
        self.btn_reset.on_clicked(self._on_reset_zoom)
        self.btn_tprev.on_clicked(self._on_thumb_prev)
        self.btn_tnext.on_clicked(self._on_thumb_next)

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("scroll_event",       self._on_scroll)
        self.fig.canvas.mpl_connect("close_event",        self._on_close)
        self.fig.canvas.mpl_connect("key_press_event",    self._on_key_press)
        self.fig.canvas.mpl_connect("key_release_event",  self._on_key_release)

        try:
            self.fig.canvas.manager.set_window_title("Point Marker — Photogrammetry")
        except Exception:
            pass

    # ── redraw ────────────────────────────────────────────────────────────────

    def _redraw_all(self) -> None:
        self._redraw_title()
        self._redraw_main()
        self._redraw_panel()
        self._redraw_thumbnails()
        self.fig.canvas.draw_idle()

    def _redraw_title(self) -> None:
        self.ax_title.clear()
        for sp in self.ax_title.spines.values():
            sp.set_visible(False)
        self.ax_title.set_xticks([])
        self.ax_title.set_yticks([])
        self.ax_title.set_facecolor(_BG)

        dirty_flag = "  *" if self._dirty else ""
        label = (
            f"{self.current_image_name}"
            f"  ({self.current_idx + 1} / {len(self.image_names)})"
            f"{dirty_flag}"
        )
        self.ax_title.text(
            0.5, 0.5, label,
            ha="center", va="center",
            fontsize=10, color="#FFD700" if self._dirty else "white",
            transform=self.ax_title.transAxes,
        )

    def _redraw_main(self) -> None:
        self.ax_main.clear()
        img = self._load_image(self.image_paths[self.current_idx])
        self.ax_main.imshow(img)
        self.ax_main.axis("off")

        marks = self.data.get_all_for_image(self.current_image_name)
        for pid, xy in marks.items():
            if xy is None:
                continue
            x, y     = xy
            color    = self._point_color(pid)
            is_act   = pid == self.active_point_id
            ms       = 10 if is_act else 7
            zorder   = 10 if is_act else 5

            if is_act:
                # white ring behind the dot
                self.ax_main.plot(x, y, "o", ms=ms + 6, color="white",
                                  zorder=zorder - 1, markeredgewidth=0)
            self.ax_main.plot(x, y, "o", ms=ms, color=color,
                              zorder=zorder,
                              markeredgecolor="black", markeredgewidth=0.5)
            self.ax_main.annotate(
                pid, (x, y),
                xytext=(8, -8), textcoords="offset pixels",
                fontsize=8, color="white", fontweight="bold",
                zorder=zorder + 1,
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor=color, alpha=0.85, edgecolor="none"),
            )

        # restore zoom state after redraw (imshow always resets the view limits)
        if self._main_xlim is not None:
            self.ax_main.set_xlim(self._main_xlim)
            self.ax_main.set_ylim(self._main_ylim)

    def _redraw_panel(self) -> None:
        self.ax_panel.clear()
        self.ax_panel.set_facecolor(_PANEL_BG)
        self.ax_panel.set_xlim(0, 1)
        self.ax_panel.set_ylim(0, 1)
        for sp in self.ax_panel.spines.values():
            sp.set_visible(False)
        self.ax_panel.set_xticks([])
        self.ax_panel.set_yticks([])

        n     = len(self.data.point_ids)
        marks = self.data.get_all_for_image(self.current_image_name)

        # title — show selection count when items are checked
        sel_count = len(self._selected)
        title_txt = f"Points  ({sel_count} sel.)" if sel_count else "Points"
        self.ax_panel.text(
            0.5, 0.975, title_txt,
            ha="center", va="top", fontsize=11, color="white", fontweight="bold",
        )

        if n == 0:
            self.ax_panel.text(
                0.5, 0.88,
                "Click '+ Point' to add\na new point",
                ha="center", va="top", fontsize=10,
                color="#666666", style="italic", linespacing=1.6,
            )
            self._panel_rows = []
            return

        MAX_ROWS  = 16
        row_h     = 0.056   # fixed row height — fits 16 rows in ~0.90 of panel height
        top_start = 0.918

        # clamp scroll offset
        max_offset = max(0, n - MAX_ROWS)
        self._panel_offset = max(0, min(self._panel_offset, max_offset))

        # scroll indicators
        if self._panel_offset > 0:
            self.ax_panel.text(
                0.5, 0.955, f"▲  {self._panel_offset} more above",
                ha="center", va="bottom", fontsize=7, color="#888888",
            )
        if self._panel_offset + MAX_ROWS < n:
            below = n - self._panel_offset - MAX_ROWS
            self.ax_panel.text(
                0.5, 0.005, f"▼  {below} more below  (scroll)",
                ha="center", va="bottom", fontsize=7, color="#888888",
            )

        # column headers
        hy = 0.932
        self.ax_panel.text(0.23, hy, "ID", ha="left",   va="top", fontsize=8,
                           color=_HDR_CLR, fontweight="bold")
        self.ax_panel.text(0.56, hy, "x",  ha="center", va="top", fontsize=8,
                           color=_HDR_CLR, fontweight="bold")
        self.ax_panel.text(0.81, hy, "y",  ha="center", va="top", fontsize=8,
                           color=_HDR_CLR, fontweight="bold")
        self.ax_panel.plot([0.02, 0.98], [0.921, 0.921], color="#444444", lw=0.8)

        # visible slice
        visible = self.data.point_ids[self._panel_offset: self._panel_offset + MAX_ROWS]

        self._panel_rows = []
        for i, pid in enumerate(visible):
            row_top = top_start - i * row_h
            row_bot = row_top - row_h
            row_cy  = (row_top + row_bot) / 2
            self._panel_rows.append((row_bot, row_top, pid))

            color     = self._point_color(pid)
            is_act    = pid == self.active_point_id
            is_sel    = pid in self._selected
            xy        = marks.get(pid)
            has_marks = bool(self.data.marks.get(pid))

            # row background: selection (red) takes priority over active highlight
            if is_sel:
                self.ax_panel.add_patch(mpatches.Rectangle(
                    (0.01, row_bot + row_h * 0.05), 0.97, row_h * 0.90,
                    facecolor=mcolors.to_rgba("#cc2222", alpha=0.20),
                    edgecolor="#cc3333", linewidth=1.4, zorder=1,
                ))
            elif is_act:
                self.ax_panel.add_patch(mpatches.Rectangle(
                    (0.01, row_bot + row_h * 0.05), 0.97, row_h * 0.90,
                    facecolor=mcolors.to_rgba(color, alpha=_ACTIVE_A),
                    edgecolor=color, linewidth=1.2, zorder=1,
                ))

            # checkbox (☐ unchecked / ✕ checked-for-deletion)
            cb = row_h * 0.50
            self.ax_panel.add_patch(mpatches.Rectangle(
                (0.055 - cb / 2, row_cy - cb / 2), cb, cb,
                facecolor="#cc3333" if is_sel else "#2e2e2e",
                edgecolor="#cc3333"  if is_sel else "#666666",
                linewidth=1.0, zorder=3,
            ))
            if is_sel:
                self.ax_panel.text(
                    0.055, row_cy, "✕",
                    ha="center", va="center", fontsize=6, color="white",
                    fontweight="bold", zorder=4,
                )

            # radio circle
            r_outer = row_h * 0.24
            r_inner = row_h * 0.11
            self.ax_panel.add_patch(mpatches.Circle(
                (0.155, row_cy), r_outer,
                color=color if is_act else "#444444", zorder=3,
            ))
            if is_act:
                self.ax_panel.add_patch(mpatches.Circle(
                    (0.155, row_cy), r_inner, color="white", zorder=4,
                ))

            # point ID (dim if never marked anywhere)
            self.ax_panel.text(
                0.23, row_cy, pid,
                ha="left", va="center",
                fontsize=9, color=color if has_marks else "#555555",
                fontweight="bold", zorder=5,
            )

            # pixel values for the current image
            if xy is not None:
                xs, ys, vc = str(xy[0]), str(xy[1]), "white"
            else:
                xs = ys = "—"
                vc = "#444444"
            self.ax_panel.text(0.56, row_cy, xs, ha="center", va="center",
                               fontsize=8, color=vc, zorder=5)
            self.ax_panel.text(0.81, row_cy, ys, ha="center", va="center",
                               fontsize=8, color=vc, zorder=5)

    def _redraw_thumbnails(self) -> None:
        for i, ax in enumerate(self.ax_thumbs):
            ax.clear()
            ax.set_facecolor("#333333")
            ax.axis("off")

            img_idx = self.thumb_offset + i
            if img_idx >= len(self.image_paths):
                ax.set_title("", fontsize=6)
                continue

            td = self._load_thumb(self.image_paths[img_idx])
            ax.imshow(td["arr"])
            ax.axis("off")

            # border: gold = current, grey = others
            is_cur = img_idx == self.current_idx
            bc, blw = ("#FFD700", 3) if is_cur else ("#555555", 1)
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_color(bc)
                sp.set_linewidth(blw)

            # dots for marked points
            img_name = self.image_names[img_idx]
            marks    = self.data.get_all_for_image(img_name)
            th, tw   = td["arr"].shape[:2]
            sx = tw / td["orig_w"]
            sy = th / td["orig_h"]
            for pid, xy in marks.items():
                if xy is None:
                    continue
                ax.plot(xy[0] * sx, xy[1] * sy, "o",
                        ms=4, color=self._point_color(pid),
                        zorder=5, markeredgewidth=0)

            # image name label (shortened)
            name = self.image_names[img_idx]
            short = name if len(name) <= 14 else name[:12] + "…"
            ax.set_title(short, fontsize=6, color="#cccccc", pad=2)

            # mark-count badge
            count = sum(1 for xy in marks.values() if xy is not None)
            ax.text(0.97, 0.03, str(count),
                    ha="right", va="bottom", fontsize=7, color="#FFD700",
                    transform=ax.transAxes,
                    bbox=dict(facecolor="#00000088", edgecolor="none", pad=1.5))

    # ── event handlers ────────────────────────────────────────────────────────

    def _on_click(self, event) -> None:
        if event.button != 1:
            return
        if event.inaxes is self.ax_main:
            if event.xdata is not None and event.ydata is not None:
                self._handle_main_click(event.xdata, event.ydata)
        elif event.inaxes is self.ax_panel:
            if event.xdata is not None and event.ydata is not None:
                self._handle_panel_click(event.xdata, event.ydata)
        else:
            for i, ax in enumerate(self.ax_thumbs):
                if event.inaxes is ax:
                    self._navigate_to(self.thumb_offset + i)
                    break

    def _handle_main_click(self, x: float, y: float) -> None:
        if not self.data.point_ids or self.active_point_id is None:
            return
        was_unmarked = (
            self.data.get_mark(self.active_point_id, self.current_image_name) is None
        )
        self.data.mark(self.active_point_id, self.current_image_name, int(x), int(y))
        self._dirty = True
        # auto-advance only when marking a previously empty slot
        if was_unmarked:
            nxt = self.data.next_unmarked_after(
                self.active_point_id, self.current_image_name
            )
            if nxt:
                self.active_point_id = nxt
        self._redraw_all()

    def _handle_panel_click(self, x_data: float, y_data: float) -> None:
        """Checkbox/Ctrl: toggle. Shift: range-select. Plain click: set active."""
        for local_idx, (y_bot, y_top, pid) in enumerate(self._panel_rows):
            if y_bot <= y_data <= y_top:
                global_idx = self._panel_offset + local_idx
                if x_data <= 0.10 or self._ctrl_held:
                    # checkbox column OR Ctrl — toggle single item
                    if pid in self._selected:
                        self._selected.discard(pid)
                    else:
                        self._selected.add(pid)
                    self._last_panel_idx = global_idx
                    self._redraw_panel()
                    self.fig.canvas.draw_idle()
                elif self._shift_held and self._last_panel_idx is not None:
                    # Shift — select range using global indices
                    lo = min(global_idx, self._last_panel_idx)
                    hi = max(global_idx, self._last_panel_idx)
                    for j in range(lo, hi + 1):
                        if j < len(self.data.point_ids):
                            self._selected.add(self.data.point_ids[j])
                    self._last_panel_idx = global_idx
                    self._redraw_panel()
                    self.fig.canvas.draw_idle()
                else:
                    # Plain click — set active point
                    self.active_point_id = pid
                    self._last_panel_idx = global_idx
                    self._redraw_all()
                return

    def _on_scroll(self, event) -> None:
        # ── panel scroll ──────────────────────────────────────────────────────
        if event.inaxes is self.ax_panel:
            n = len(self.data.point_ids)
            if n <= 16:
                return
            if event.button == "up":
                self._panel_offset = max(0, self._panel_offset - 1)
            else:
                self._panel_offset = min(n - 16, self._panel_offset + 1)
            self._redraw_panel()
            self.fig.canvas.draw_idle()
            return

        # ── main image zoom ───────────────────────────────────────────────────
        if event.inaxes is not self.ax_main:
            return
        if event.xdata is None or event.ydata is None:
            return

        # scroll up = zoom in (factor < 1 shrinks the visible range)
        factor = ZOOM_FACTOR if event.button == "up" else (1.0 / ZOOM_FACTOR)

        xlim = list(self.ax_main.get_xlim())
        ylim = list(self.ax_main.get_ylim())
        xc, yc = event.xdata, event.ydata

        # Scale each bound relative to the cursor position
        new_xlim = [xc + (xlim[0] - xc) * factor, xc + (xlim[1] - xc) * factor]
        new_ylim = [yc + (ylim[0] - yc) * factor, yc + (ylim[1] - yc) * factor]

        # Clamp to image bounds.
        # imshow uses an inverted y-axis: ylim[0] > ylim[1].
        img = self._load_image(self.image_paths[self.current_idx])
        h, w = img.shape[:2]

        # x — normal direction (xlim[0] < xlim[1])
        if new_xlim[1] - new_xlim[0] >= w:
            new_xlim = [-0.5, w - 0.5]
        else:
            new_xlim[0] = max(new_xlim[0], -0.5)
            new_xlim[1] = min(new_xlim[1], w - 0.5)

        # y — inverted direction (ylim[0] > ylim[1])
        if new_ylim[0] - new_ylim[1] >= h:
            new_ylim = [h - 0.5, -0.5]
        else:
            new_ylim[0] = min(new_ylim[0], h - 0.5)
            new_ylim[1] = max(new_ylim[1], -0.5)

        self._main_xlim = new_xlim
        self._main_ylim = new_ylim
        self.ax_main.set_xlim(new_xlim)
        self.ax_main.set_ylim(new_ylim)
        self.fig.canvas.draw_idle()

    def _on_reset_zoom(self, _) -> None:
        """Return the main image to its full, unzoomed view."""
        self._main_xlim = None
        self._main_ylim = None
        self._redraw_main()
        self.fig.canvas.draw_idle()

    def _navigate_to(self, idx: int) -> None:
        if not (0 <= idx < len(self.image_paths)):
            return
        self.current_idx = idx
        self._main_xlim  = None   # reset zoom when switching images
        self._main_ylim  = None
        self._sync_active()
        self._ensure_thumb_visible(idx)
        self._redraw_all()
    def _on_prev(self, _) -> None:
        self._navigate_to(self.current_idx - 1)

    def _on_next(self, _) -> None:
        self._navigate_to(self.current_idx + 1)

    def _on_key_press(self, event) -> None:
        key = event.key or ""
        if "shift" in key:
            self._shift_held = True
        if "ctrl" in key:
            self._ctrl_held = True

    def _on_key_release(self, event) -> None:
        key = event.key or ""
        if "shift" in key:
            self._shift_held = False
        if "ctrl" in key:
            self._ctrl_held = False

    def _on_del_selected(self, _) -> None:
        """Delete all checked point IDs from every image."""
        if not self._selected:
            return
        for pid in list(self._selected):
            self.data.delete_point(pid)
        self._selected.clear()
        self._last_panel_idx = None
        if self.active_point_id not in self.data.point_ids:
            self._sync_active()
        self._dirty = True
        self._redraw_all()

    def _on_del_point(self, _) -> None:
        """Delete the active point from ALL images entirely."""
        if not self.active_point_id:
            return
        self.data.delete_point(self.active_point_id)
        self._selected.discard(self.active_point_id)
        self._last_panel_idx = None
        self._sync_active()
        self._dirty = True
        self._redraw_all()

    def _on_clr_selected(self, _) -> None:
        """Remove marks for selected points in the CURRENT image only (keep IDs)."""
        if not self._selected:
            return
        for pid in self._selected:
            self.data.unmark(pid, self.current_image_name)
        self._dirty = True
        self._redraw_all()

    def _on_clr_all(self, _) -> None:
        """Clear ALL points and marks after user confirmation."""
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        answer = messagebox.askyesno(
            "Clear All Data",
            "Are you sure you want to delete ALL points and marks?\n"
            "This cannot be undone.",
            parent=root,
        )
        root.destroy()
        if not answer:
            return
        self.data.point_ids.clear()
        self.data.marks.clear()
        self._selected.clear()
        self._last_panel_idx = None
        self.active_point_id = None
        self._dirty = True
        self._redraw_all()

    def _on_add_point(self, _) -> None:
        new_id = self.data.add_point()
        self.active_point_id = new_id
        self._dirty = True
        self._redraw_all()

    def _on_del_mark(self, _) -> None:
        if self.active_point_id:
            self.data.unmark(self.active_point_id, self.current_image_name)
            self._dirty = True
            self._redraw_all()

    def _on_save(self, _) -> None:
        self.data.save(self.json_path)
        self._dirty = False
        self._redraw_title()
        self.fig.canvas.draw_idle()

    def _on_table(self, _) -> None:
        self._show_table()

    def _on_thumb_prev(self, _) -> None:
        if self.thumb_offset > 0:
            self.thumb_offset -= 1
            self._redraw_thumbnails()
            self.fig.canvas.draw_idle()

    def _on_thumb_next(self, _) -> None:
        if self.thumb_offset + THUMBS_VISIBLE < len(self.image_paths):
            self.thumb_offset += 1
            self._redraw_thumbnails()
            self.fig.canvas.draw_idle()

    def _on_close(self, _) -> None:
        if self._dirty:
            self.data.save(self.json_path)

    # ── table window ──────────────────────────────────────────────────────────

    def _show_table(self) -> None:
        n_pts  = len(self.data.point_ids)
        n_imgs = len(self.image_names)
        if n_pts == 0:
            return

        # Filter out points with no marks in any image
        active_pids = [
            pid for pid in self.data.point_ids
            if any(self.data.get_mark(pid, img) is not None
                   for img in self.image_names)
        ]
        empty_count = len(self.data.point_ids) - len(active_pids)
        if not active_pids:
            return

        col_labels = [n[:10] for n in self.image_names]
        row_labels = active_pids

        cell_text   = []
        cell_colors = []
        for pid in active_pids:
            row_vals:   list[str] = []
            row_colors: list[str] = []
            for img_name in self.image_names:
                xy = self.data.get_mark(pid, img_name)
                if xy is not None:
                    row_vals.append(f"({xy[0]}, {xy[1]})")
                    row_colors.append("#d4f0d4")   # green — marked
                else:
                    row_vals.append("—")
                    row_colors.append("#f8f8f8")   # white — not marked
            cell_text.append(row_vals)
            cell_colors.append(row_colors)

        fig_w = max(14, n_imgs * 1.3)
        fig_h = max(3,  len(active_pids) * 0.5 + 2)
        fig_t, ax_t = plt.subplots(figsize=(fig_w, fig_h))
        ax_t.axis("off")
        ax_t.set_title("Correspondence Table", fontsize=13, pad=16, fontweight="bold")

        tbl = ax_t.table(
            cellText=cell_text,
            cellColours=cell_colors,
            rowLabels=row_labels,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.auto_set_column_width(col=list(range(n_imgs)))
        fig_t.tight_layout()
        if empty_count > 0:
            ax_t.text(
                0.01, 0.01,
                f"Note: {empty_count} point(s) with no marks excluded.",
                transform=ax_t.transAxes, fontsize=8, color="#888888", va="bottom",
            )
        plt.show(block=False)

    # ── entry point ───────────────────────────────────────────────────────────

    def run(self) -> None:
        plt.show()
