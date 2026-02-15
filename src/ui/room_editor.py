"""Task 3 – Room Creation UI – Net-based editor.

2D Net: cross-shaped unfolded room; click edges to set dimensions,
        click faces to place elements; hover preview before placement.
3D Preview: isometric wireframe with corrected orientation transforms.
Save/Load: file dialogs for Room JSON round-trip.
Units: Standard (ft) / Metric (m) toggle; internal values always in metres.
"""

from __future__ import annotations

import json
import math
from typing import Any

import dearpygui.dearpygui as dpg

from models.models import (
    Room,
    RoomElement,
    WALL_NAMES,
    UNIT_SYSTEMS,
    _ELEMENT_REGISTRY,
)
from utils.logger import setup_logger

logger = setup_logger("UI")

# ── constants ──────────────────────────────────────────────────────────────
CANVAS_W, CANVAS_H = 720, 500
MARGIN = 50
EDGE_HIT_PX = 10

ELEM_COLORS = {
    "door":   (230, 160,  50, 255),
    "window": ( 80, 180, 255, 255),
    "vent":   (100, 220, 120, 255),
}
HOVER_ALPHA = 128
ELEM_ALPHA  = 70

FACE_BG         = ( 40,  42,  54, 255)
FACE_SEL_BG     = ( 55,  58,  78, 255)
EDGE_COLOR      = (180, 180, 190, 255)
EDGE_SEL_COLOR  = (255, 220,  80, 255)
GRID_COLOR      = ( 58,  60,  72, 120)
LABEL_COLOR     = (220, 220, 220, 200)
DIM_LABEL_COLOR = (140, 190, 255, 220)
UP_ARROW_COLOR  = (255, 120, 120, 180)

# (horiz_room_attr, vert_room_attr) for each face on the unfolded net
_FACE_DIMS: dict[str, tuple[str, str]] = {
    "north":   ("width",  "height"),
    "south":   ("width",  "height"),
    "west":    ("height", "length"),
    "east":    ("height", "length"),
    "floor":   ("width",  "length"),
    "ceiling": ("width",  "length"),
}

# Faces where wall-up direction = net horizontal (u-axis).
# Element sizes (wall_w, wall_h) must be swapped to (wall_h, wall_w) so the
# tall dimension aligns with the wall's upright direction in 3D.
_UPRIGHT_SWAP: set[str] = {"west", "east"}

# Element sizes in metres: (width_along_wall, height_along_wall)
DEFAULT_ELEM_SIZE = {
    "door":   (0.9, 2.1),
    "window": (1.2, 1.0),
    "vent":   (0.3, 0.3),
}


# ── RoomEditor ─────────────────────────────────────────────────────────────

class RoomEditor:
    """Net-based room editor with 2D net view and 3D iso preview."""

    def __init__(self, units: str = "standard") -> None:
        self.room = Room()
        self._sel_face: str = "floor"
        self._elem_type: str = "door"
        self._mode: str = "2d"
        self._show_ceiling: bool = False
        self._units: str = units if units in UNIT_SYSTEMS else "standard"

        # DPG tags
        self._canvas: int | str = 0
        self._layer: int | str = 0
        self._hover_layer: int | str = 0
        self._prev_canvas: int | str = 0
        self._prev_layer: int | str = 0
        self._grp_2d: int | str = 0
        self._grp_3d: int | str = 0
        self._txt_dims: int | str = 0
        self._txt_face: int | str = 0
        self._txt_elems: int | str = 0
        self._txt_info: int | str = 0

        # Cached pixel rectangles for hit-testing
        self._rects: dict[str, tuple[float, float, float, float]] = {}
        self._sc: float = 1.0

        logger.info("RoomEditor created (units=%s)", self._units)

    # ── unit helpers ───────────────────────────────────────────────────────

    @property
    def _uf(self) -> float:
        """Display conversion factor (metres → display)."""
        return UNIT_SYSTEMS[self._units]["factor"]

    @property
    def _ul(self) -> str:
        """Unit label string."""
        return UNIT_SYSTEMS[self._units]["label"]

    def _disp(self, metres: float) -> float:
        """Convert internal metres to display value."""
        return metres * self._uf

    def _to_m(self, display_val: float) -> float:
        """Convert display value back to internal metres."""
        return display_val / self._uf

    # ── build ──────────────────────────────────────────────────────────────

    def build(self, parent: int | str) -> None:
        with dpg.group(horizontal=True, parent=parent):
            self._build_panel()
            self._build_canvases()
        with dpg.handler_registry():
            dpg.add_mouse_click_handler(
                button=dpg.mvMouseButton_Left, callback=self._on_click,
            )
            dpg.add_mouse_move_handler(callback=self._on_mouse_move)
        self._draw_net()
        logger.info("RoomEditor UI built")

    def _build_panel(self) -> None:
        with dpg.child_window(width=230, autosize_y=True):
            dpg.add_text("Room Dimensions")
            self._txt_dims = dpg.add_text(self._dims_str())
            dpg.add_text("Click an edge to edit", color=(140, 140, 140, 255))

            dpg.add_separator()
            dpg.add_text("Units")
            dpg.add_radio_button(
                items=["Standard (ft)", "Metric (m)"],
                default_value=(
                    "Standard (ft)" if self._units == "standard"
                    else "Metric (m)"
                ),
                callback=self._on_units_changed,
                horizontal=True,
            )

            dpg.add_separator()
            dpg.add_text("Selected Face")
            self._txt_face = dpg.add_text(self._sel_face)

            dpg.add_separator()
            dpg.add_text("Place Element (click face)")
            dpg.add_combo(
                items=list(_ELEMENT_REGISTRY.keys()),
                default_value=self._elem_type,
                callback=lambda s, a: self._set_elem_type(a),
            )

            dpg.add_separator()
            dpg.add_text("Elements on Face")
            self._txt_elems = dpg.add_text("(none)")
            dpg.add_button(label="Delete Last", callback=self._del_last_elem)
            dpg.add_button(label="Clear Face",  callback=self._clear_face)

            dpg.add_separator()
            dpg.add_checkbox(
                label="Show Ceiling", default_value=False,
                callback=lambda s, a: self._toggle_ceiling(a),
            )
            dpg.add_button(label="Toggle 2D / 3D", callback=self._toggle_view)

            dpg.add_separator()
            dpg.add_button(label="Save Room...", callback=self._dlg_save)
            dpg.add_button(label="Load Room...", callback=self._dlg_load)

            dpg.add_separator()
            self._txt_info = dpg.add_text("Ready")

    def _build_canvases(self) -> None:
        with dpg.group() as self._grp_2d:
            dpg.add_text(
                "2D Net  |  click edge = resize  |  click face = place element"
            )
            with dpg.drawlist(
                width=CANVAS_W, height=CANVAS_H
            ) as self._canvas:
                self._layer = dpg.add_draw_layer()
                self._hover_layer = dpg.add_draw_layer()

        with dpg.group(show=False) as self._grp_3d:
            dpg.add_text("3D Isometric Preview")
            with dpg.drawlist(
                width=CANVAS_W, height=CANVAS_H
            ) as self._prev_canvas:
                self._prev_layer = dpg.add_draw_layer()

    # ── helpers ────────────────────────────────────────────────────────────

    def _dims_str(self) -> str:
        r = self.room
        u = self._ul
        return (
            f"W = {self._disp(r.width):.2f} {u}   "
            f"L = {self._disp(r.length):.2f} {u}   "
            f"H = {self._disp(r.height):.2f} {u}"
        )

    def _elem_size_for_face(self, face: str) -> tuple[float, float]:
        """Element size in face-local (u, v) coords, swapped for upright faces."""
        sz = DEFAULT_ELEM_SIZE.get(self._elem_type, (0.5, 0.5))
        if face in _UPRIGHT_SWAP:
            return (sz[1], sz[0])
        return sz

    def _net_faces(self) -> dict[str, tuple[float, float, float, float]]:
        """Face positions on the net in room-unit coords (x, y, w, h).

        Layout (cross):
                 +-------+
                 | North |   (front wall, W x H)
        +--------+-------+--------+
        |  West  | Floor |  East  |
        | (HxL)  | (WxL) | (HxL)  |
        +--------+-------+--------+
                 | South |   (back wall, W x H)
                 +-------+
        Optional ceiling flap extends right from East.
        """
        W, L, H = self.room.width, self.room.length, self.room.height
        faces = {
            "north": (H,     0,     W, H),
            "west":  (0,     H,     H, L),
            "floor": (H,     H,     W, L),
            "east":  (H + W, H,     H, L),
            "south": (H,     H + L, W, H),
        }
        if self._show_ceiling:
            faces["ceiling"] = (H + W + H, H, W, L)
        return faces

    def _compute_transform(self):
        """Scale and offset to fit the entire net inside the canvas."""
        W, L, H = self.room.width, self.room.length, self.room.height
        net_w = 2 * H + W + (W if self._show_ceiling else 0)
        net_h = 2 * H + L
        sc = min((CANVAS_W - 2 * MARGIN) / net_w,
                 (CANVAS_H - 2 * MARGIN) / net_h)
        ox = MARGIN + ((CANVAS_W - 2 * MARGIN) - net_w * sc) / 2
        oy = MARGIN + ((CANVAS_H - 2 * MARGIN) - net_h * sc) / 2
        return sc, ox, oy

    # ── 2D net drawing ─────────────────────────────────────────────────────

    def _draw_net(self) -> None:
        dpg.delete_item(self._layer, children_only=True)
        dpg.delete_item(self._hover_layer, children_only=True)

        faces = self._net_faces()
        sc, ox, oy = self._compute_transform()
        self._sc = sc
        self._rects.clear()

        for name, (fx, fy, fw, fh) in faces.items():
            px = ox + fx * sc
            py = oy + fy * sc
            pw = fw * sc
            ph = fh * sc
            self._rects[name] = (px, py, pw, ph)

            # face background
            bg = FACE_SEL_BG if name == self._sel_face else FACE_BG
            dpg.draw_rectangle(
                (px, py), (px + pw, py + ph),
                fill=bg, color=(0, 0, 0, 0),
                parent=self._layer,
            )

            # 0.5 m grid
            gx = 0.5
            while gx < fw:
                lx = px + gx * sc
                dpg.draw_line(
                    (lx, py), (lx, py + ph),
                    color=GRID_COLOR, parent=self._layer,
                )
                gx += 0.5
            gy = 0.5
            while gy < fh:
                ly = py + gy * sc
                dpg.draw_line(
                    (px, ly), (px + pw, ly),
                    color=GRID_COLOR, parent=self._layer,
                )
                gy += 0.5

            # outline
            ec = EDGE_SEL_COLOR if name == self._sel_face else EDGE_COLOR
            dpg.draw_rectangle(
                (px, py), (px + pw, py + ph),
                color=ec, thickness=2, parent=self._layer,
            )

            # face label (top-left inside face)
            h_dim, v_dim = _FACE_DIMS[name]
            u = self._ul
            dpg.draw_text(
                (px + 4, py + 4),
                f"{name.capitalize()} "
                f"({self._disp(getattr(self.room, h_dim)):.1f}"
                f" x {self._disp(getattr(self.room, v_dim)):.1f} {u})",
                color=LABEL_COLOR, size=12, parent=self._layer,
            )

            # "up" arrow on upright-swapped faces (wall-up = net-right)
            if name in _UPRIGHT_SWAP:
                ax = px + pw - 30
                ay = py + 6
                dpg.draw_line(
                    (ax, ay + 5), (ax + 18, ay + 5),
                    color=UP_ARROW_COLOR, thickness=2, parent=self._layer,
                )
                dpg.draw_triangle(
                    (ax + 18, ay), (ax + 18, ay + 10), (ax + 24, ay + 5),
                    color=UP_ARROW_COLOR, fill=UP_ARROW_COLOR,
                    parent=self._layer,
                )
                dpg.draw_text(
                    (ax - 12, ay - 1), "up",
                    color=UP_ARROW_COLOR, size=10, parent=self._layer,
                )

            # elements
            for elem in self.room.walls.get(name, []):
                c = ELEM_COLORS.get(elem.element_type, (200, 200, 200, 255))
                ex = px + elem.pos[0] * sc
                ey = py + elem.pos[1] * sc
                ew = elem.size[0] * sc
                eh = elem.size[1] * sc
                dpg.draw_rectangle(
                    (ex, ey), (ex + ew, ey + eh),
                    color=c, fill=(*c[:3], ELEM_ALPHA),
                    thickness=2, parent=self._layer,
                )
                dpg.draw_text(
                    (ex + 2, ey + 2),
                    f"{elem.element_type[0].upper()} {elem.open_frac:.0%}",
                    color=c, size=10, parent=self._layer,
                )

        # dimension labels on key outer edges
        self._draw_dim_labels()

        # update side-panel texts
        dpg.set_value(self._txt_dims, self._dims_str())
        self._update_elems_text()

    def _draw_dim_labels(self) -> None:
        W, L, H = self.room.width, self.room.length, self.room.height
        u = self._ul

        # Width - above north top edge
        if "north" in self._rects:
            px, py, pw, _ = self._rects["north"]
            dpg.draw_text(
                (px + pw / 2 - 28, py - 20),
                f"W: {self._disp(W):.1f} {u}",
                color=DIM_LABEL_COLOR, size=13, parent=self._layer,
            )

        # Height - left of north left edge
        if "north" in self._rects:
            px, py, _, ph = self._rects["north"]
            dpg.draw_text(
                (px - 55, py + ph / 2 - 7),
                f"H: {self._disp(H):.1f} {u}",
                color=DIM_LABEL_COLOR, size=13, parent=self._layer,
            )

        # Length - left of west left edge
        if "west" in self._rects:
            px, py, _, ph = self._rects["west"]
            dpg.draw_text(
                (px - 55, py + ph / 2 - 7),
                f"L: {self._disp(L):.1f} {u}",
                color=DIM_LABEL_COLOR, size=13, parent=self._layer,
            )

    # ── hover preview ──────────────────────────────────────────────────────

    def _on_mouse_move(self, sender: Any = None, app_data: Any = None) -> None:
        dpg.delete_item(self._hover_layer, children_only=True)

        if self._mode != "2d":
            return
        if not dpg.is_item_hovered(self._canvas):
            return
        if dpg.does_item_exist("dim_popup"):
            return

        mx, my = dpg.get_drawing_mouse_pos()

        if self._hit_edge(mx, my):
            return

        face = self._hit_face(mx, my)
        if not face:
            return

        px, py, _, _ = self._rects[face]
        h_dim, v_dim = _FACE_DIMS[face]
        fw = getattr(self.room, h_dim)
        fh = getattr(self.room, v_dim)
        sz = self._elem_size_for_face(face)

        if sz[0] > fw or sz[1] > fh:
            return

        ru = (mx - px) / self._sc
        rv = (my - py) / self._sc
        ru = max(0.0, min(ru, fw - sz[0]))
        rv = max(0.0, min(rv, fh - sz[1]))
        ru = round(ru * 4) / 4
        rv = round(rv * 4) / 4

        c = ELEM_COLORS.get(self._elem_type, (200, 200, 200, 255))
        ex = px + ru * self._sc
        ey = py + rv * self._sc
        ew = sz[0] * self._sc
        eh = sz[1] * self._sc
        dpg.draw_rectangle(
            (ex, ey), (ex + ew, ey + eh),
            color=(*c[:3], 128), fill=(*c[:3], HOVER_ALPHA),
            thickness=1, parent=self._hover_layer,
        )

    # ── hit testing ────────────────────────────────────────────────────────

    def _hit_edge(self, mx: float, my: float):
        """Return (face, side, dim_attr) if click is near an edge."""
        for name, (px, py, pw, ph) in self._rects.items():
            h_dim, v_dim = _FACE_DIMS[name]
            # horizontal edges
            if px - EDGE_HIT_PX <= mx <= px + pw + EDGE_HIT_PX:
                if abs(my - py) <= EDGE_HIT_PX:
                    return name, "top", h_dim
                if abs(my - (py + ph)) <= EDGE_HIT_PX:
                    return name, "bottom", h_dim
            # vertical edges
            if py - EDGE_HIT_PX <= my <= py + ph + EDGE_HIT_PX:
                if abs(mx - px) <= EDGE_HIT_PX:
                    return name, "left", v_dim
                if abs(mx - (px + pw)) <= EDGE_HIT_PX:
                    return name, "right", v_dim
        return None

    def _hit_face(self, mx: float, my: float) -> str | None:
        for name, (px, py, pw, ph) in self._rects.items():
            if px < mx < px + pw and py < my < py + ph:
                return name
        return None

    # ── click handler ──────────────────────────────────────────────────────

    def _on_click(self, sender: Any = None, app_data: Any = None) -> None:
        if self._mode != "2d":
            return
        if not dpg.is_item_hovered(self._canvas):
            return

        mx, my = dpg.get_drawing_mouse_pos()

        # edge has priority over face
        edge = self._hit_edge(mx, my)
        if edge:
            _, _, dim_attr = edge
            self._popup_dim(dim_attr, mx, my)
            return

        face = self._hit_face(mx, my)
        if face:
            self._sel_face = face
            dpg.set_value(self._txt_face, face)
            self._place_elem(face, mx, my)
            self._draw_net()

    # ── edge dimension popup ───────────────────────────────────────────────

    def _popup_dim(self, dim_attr: str, px: float, py: float) -> None:
        tag = "dim_popup"
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)

        cur_m = getattr(self.room, dim_attr)
        cur_disp = self._disp(cur_m)
        inp_tag = f"{tag}_inp"
        ul = self._ul

        def apply(sender=None, app_data=None):
            disp_val = dpg.get_value(inp_tag)
            val_m = self._to_m(disp_val)
            val_m = round(max(0.5, min(20.0, val_m)), 3)
            setattr(self.room, dim_attr, val_m)
            logger.info(
                "Edge updated: %s = %.3f m (%.2f %s)",
                dim_attr, val_m, self._disp(val_m), ul,
            )
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
            self._clamp_elements()
            self._refresh()

        step = 0.5 if self._units == "standard" else 0.1

        with dpg.window(
            tag=tag, label=f"Set {dim_attr} ({ul})",
            pos=(int(px) + 12, int(py) + 12),
            width=220, height=80,
            no_resize=True, no_collapse=True,
        ):
            dpg.add_input_float(
                tag=inp_tag, label=ul,
                default_value=cur_disp,
                min_value=self._disp(0.5), max_value=self._disp(20.0),
                step=step, on_enter=True,
                callback=apply,
            )
            dpg.add_button(label="Apply", callback=apply)

        logger.debug("Edge popup: %s = %.2f %s", dim_attr, cur_disp, ul)

    # ── element placement ──────────────────────────────────────────────────

    def _place_elem(self, face: str, mx: float, my: float) -> None:
        px, py, _, _ = self._rects[face]
        h_dim, v_dim = _FACE_DIMS[face]
        fw = getattr(self.room, h_dim)
        fh = getattr(self.room, v_dim)
        sc = self._sc

        # pixel -> room-unit coords on face
        ru = (mx - px) / sc
        rv = (my - py) / sc

        sz = self._elem_size_for_face(face)
        orient = "swapped" if face in _UPRIGHT_SWAP else "upright"

        # reject if element can't fit on face
        if sz[0] > fw or sz[1] > fh:
            msg = f"{self._elem_type} too large for {face}"
            dpg.set_value(self._txt_info, msg)
            logger.info(msg)
            return

        # clamp + snap to 0.25 m
        ru = max(0.0, min(ru, fw - sz[0]))
        rv = max(0.0, min(rv, fh - sz[1]))
        ru = round(ru * 4) / 4
        rv = round(rv * 4) / 4

        cls = _ELEMENT_REGISTRY[self._elem_type]
        elem = cls(pos=(ru, rv), size=sz, open_frac=0.5)
        self.room.add_element(face, elem)

        logger.info(
            "Element %s oriented on %s at (%.2f, %.2f) orient=%s",
            self._elem_type, face, ru, rv, orient,
        )
        ul = self._ul
        dpg.set_value(
            self._txt_info,
            f"Placed {self._elem_type} on {face} "
            f"({self._disp(ru):.2f}, {self._disp(rv):.2f} {ul})",
        )

    # ── clamp after resize ─────────────────────────────────────────────────

    def _clamp_elements(self) -> None:
        for face in list(self.room.walls):
            if face not in _FACE_DIMS:
                continue
            h_dim, v_dim = _FACE_DIMS[face]
            fw = getattr(self.room, h_dim)
            fh = getattr(self.room, v_dim)
            keep: list[RoomElement] = []
            for e in self.room.walls[face]:
                if e.size[0] > fw or e.size[1] > fh:
                    logger.info(
                        "Removed %s from %s (no longer fits)",
                        e.element_type, face,
                    )
                    continue
                e.pos = (
                    max(0.0, min(e.pos[0], fw - e.size[0])),
                    max(0.0, min(e.pos[1], fh - e.size[1])),
                )
                keep.append(e)
            self.room.walls[face] = keep

    # ── panel helpers ──────────────────────────────────────────────────────

    def _update_elems_text(self) -> None:
        elems = self.room.walls.get(self._sel_face, [])
        ul = self._ul
        if not elems:
            dpg.set_value(self._txt_elems, "(none)")
        else:
            lines = [
                f"{i+1}. {e.element_type} "
                f"({self._disp(e.pos[0]):.1f},"
                f"{self._disp(e.pos[1]):.1f} {ul}) "
                f"{e.open_frac:.0%}"
                for i, e in enumerate(elems)
            ]
            dpg.set_value(self._txt_elems, "\n".join(lines))

    # ── simple callbacks ───────────────────────────────────────────────────

    def _set_elem_type(self, t: str) -> None:
        self._elem_type = t
        logger.debug("Element type: %s", t)

    def _del_last_elem(self) -> None:
        elems = self.room.walls.get(self._sel_face, [])
        if elems:
            removed = elems.pop()
            logger.info(
                "Removed %s from %s", removed.element_type, self._sel_face,
            )
        self._refresh()

    def _clear_face(self) -> None:
        self.room.walls[self._sel_face] = []
        logger.info("Cleared %s", self._sel_face)
        self._refresh()

    def _toggle_ceiling(self, show: bool) -> None:
        self._show_ceiling = show
        logger.info("Ceiling %s", "shown" if show else "hidden")
        self._refresh()

    def _toggle_view(self) -> None:
        if self._mode == "2d":
            self._mode = "3d"
            dpg.configure_item(self._grp_2d, show=False)
            dpg.configure_item(self._grp_3d, show=True)
            self._draw_3d()
            logger.info("Switched to 3D preview")
        else:
            self._mode = "2d"
            dpg.configure_item(self._grp_2d, show=True)
            dpg.configure_item(self._grp_3d, show=False)
            self._draw_net()
            logger.info("Switched to 2D editor")

    def _on_units_changed(self, sender: Any, app_data: str) -> None:
        self._units = "standard" if "Standard" in app_data else "metric"
        logger.info("Units changed: %s", self._units)
        self._refresh()

    def _refresh(self) -> None:
        if self._mode == "2d":
            self._draw_net()
        else:
            self._draw_3d()

    # ── 3D isometric preview ───────────────────────────────────────────────

    @staticmethod
    def _iso(
        x: float, y: float, z: float,
        sc: float, ox: float, oy: float,
    ) -> tuple[float, float]:
        a = math.pi / 6
        return (
            ox + (x - y) * math.cos(a) * sc,
            oy - z * sc + (x + y) * math.sin(a) * sc,
        )

    def _draw_3d(self) -> None:
        dpg.delete_item(self._prev_layer, children_only=True)

        W, L, H = self.room.width, self.room.length, self.room.height
        md = max(W, L, H)
        sc = (min(CANVAS_W, CANVAS_H) - 2 * MARGIN) / (md * 2.2)
        ox, oy = CANVAS_W / 2, CANVAS_H * 0.65
        iso = lambda x, y, z: self._iso(x, y, z, sc, ox, oy)

        # wireframe box
        c = [
            (0, 0, 0), (W, 0, 0), (W, L, 0), (0, L, 0),
            (0, 0, H), (W, 0, H), (W, L, H), (0, L, H),
        ]
        pts = [iso(*v) for v in c]
        for a, b in [
            (0,1),(1,2),(2,3),(3,0),
            (4,5),(5,6),(6,7),(7,4),
            (0,4),(1,5),(2,6),(3,7),
        ]:
            dpg.draw_line(
                pts[a], pts[b],
                color=EDGE_COLOR, thickness=1, parent=self._prev_layer,
            )

        # Per-face rotation matrices: (origin, u_axis, v_axis)
        # Maps face-local (u, v) → 3D via: P = origin + u*u_axis + v*v_axis
        xforms = {
            "north":   ((0, 0, H), ( 1, 0,  0), (0, 0, -1)),
            "south":   ((0, L, 0), ( 1, 0,  0), (0, 0,  1)),
            "west":    ((0, 0, H), ( 0, 0, -1), (0, 1,  0)),
            "east":    ((W, 0, 0), ( 0, 0,  1), (0, 1,  0)),
            "floor":   ((0, 0, 0), ( 1, 0,  0), (0, 1,  0)),
            "ceiling": ((0, 0, H), ( 1, 0,  0), (0, 1,  0)),
        }

        def _xf(orig, ua, va, u, v):
            return iso(
                orig[0] + u * ua[0] + v * va[0],
                orig[1] + u * ua[1] + v * va[1],
                orig[2] + u * ua[2] + v * va[2],
            )

        for face, (orig, ua, va) in xforms.items():
            for elem in self.room.walls.get(face, []):
                col = ELEM_COLORS.get(elem.element_type, (200, 200, 200, 255))
                u, v = elem.pos
                su, sv = elem.size
                corners = [(u, v), (u+su, v), (u+su, v+sv), (u, v+sv)]
                quad = [_xf(orig, ua, va, cu, cv) for cu, cv in corners]

                # Shoelace area check for degenerate projections
                area = 0.0
                for i in range(4):
                    j = (i + 1) % 4
                    area += quad[i][0] * quad[j][1] - quad[j][0] * quad[i][1]
                area = abs(area) / 2
                if area < 1.0:
                    logger.error(
                        "Degenerate projection: %s on %s (area=%.1fpx)",
                        elem.element_type, face, area,
                    )
                    continue

                logger.debug(
                    "3D: %s oriented on %s via rotation matrix",
                    elem.element_type, face,
                )
                dpg.draw_quad(
                    *quad,
                    color=col, fill=(*col[:3], 60),
                    thickness=2, parent=self._prev_layer,
                )

        # axis labels
        dpg.draw_text(
            iso(W / 2, -0.6, 0), "N",
            color=LABEL_COLOR, size=14, parent=self._prev_layer,
        )
        dpg.draw_text(
            iso(W + 0.4, L / 2, 0), "E",
            color=LABEL_COLOR, size=14, parent=self._prev_layer,
        )

    # ── save / load ────────────────────────────────────────────────────────

    def _dlg_save(self) -> None:
        with dpg.file_dialog(
            label="Save Room", callback=self._on_save,
            width=500, height=400, default_filename="room",
        ):
            dpg.add_file_extension(".json", color=(0, 255, 0, 255))

    def _on_save(self, sender: Any, app_data: dict) -> None:
        try:
            path = app_data["file_path_name"]
            if not path.endswith(".json"):
                path += ".json"
            with open(path, "w") as f:
                json.dump(self.room.to_dict(), f, indent=2)
            logger.info("Room saved: %s", path)
            dpg.set_value(self._txt_info, f"Saved: {path}")
        except Exception:
            logger.exception("Room save failed")
            dpg.set_value(self._txt_info, "Save failed - see log")

    def _dlg_load(self) -> None:
        with dpg.file_dialog(
            label="Load Room", callback=self._on_load,
            width=500, height=400,
        ):
            dpg.add_file_extension(".json", color=(0, 255, 0, 255))

    def _on_load(self, sender: Any, app_data: dict) -> None:
        try:
            path = app_data["file_path_name"]
            with open(path) as f:
                data = json.load(f)
            self.room = Room.from_dict(data)
            total = sum(len(v) for v in self.room.walls.values())
            logger.info("Room loaded: %s (%d elements)", path, total)
            dpg.set_value(self._txt_info, f"Loaded: {path}")
            self._refresh()
        except Exception:
            logger.exception("Room load failed")
            dpg.set_value(self._txt_info, "Load failed - see log")
