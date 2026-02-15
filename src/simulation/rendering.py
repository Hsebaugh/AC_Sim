"""Task 6 -- Airflow Visualization -- velocity arrows + room wireframe.

Isometric 3D view rendered on a DPG drawlist.  Velocity arrows are
subsampled from the solver grid, coloured by temperature (blue-cold,
yellow-ambient, red-hot) and scaled by speed.

Camera: right-mouse-drag to rotate, scroll-wheel to zoom.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import dearpygui.dearpygui as dpg

from models.models import Settings, WALL_NAMES
from utils.logger import setup_logger

logger = setup_logger("Render")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RENDER_W, RENDER_H = 700, 380
WIRE_COLOR = (180, 180, 190, 200)
WIRE_THICK = 1.5

# Colormap anchors: blue (cold) → yellow (ambient) → red (hot)
_COLD = (30, 100, 255)
_WARM = (255, 230, 50)
_HOT = (255, 40, 30)

ELEM_COLORS = {
    "door": (230, 160, 50, 100),
    "window": (80, 180, 255, 100),
    "vent": (100, 220, 120, 100),
}

# Face transforms: origin, u_axis, v_axis  (face-local → 3D)
_FACE_XFORMS = {
    "north":   ((0, 0, 1), (1, 0, 0), (0, 0, -1)),   # origin uses H
    "south":   ((0, 1, 0), (1, 0, 0), (0, 0, 1)),     # origin uses L
    "west":    ((0, 0, 1), (0, 0, -1), (0, 1, 0)),     # origin uses H
    "east":    ((1, 0, 0), (0, 0, 1), (0, 1, 0)),      # origin uses W
    "floor":   ((0, 0, 0), (1, 0, 0), (0, 1, 0)),
    "ceiling": ((0, 0, 1), (1, 0, 0), (0, 1, 0)),      # origin uses H
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lerp_rgb(
    c0: tuple[int, int, int],
    c1: tuple[int, int, int],
    t: float,
) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return (
        int(c0[0] + (c1[0] - c0[0]) * t),
        int(c0[1] + (c1[1] - c0[1]) * t),
        int(c0[2] + (c1[2] - c0[2]) * t),
    )


def temp_color(
    temp: float, t_lo: float, t_hi: float,
) -> tuple[int, int, int]:
    """Map temperature to RGB via blue → yellow → red colormap."""
    t_mid = (t_lo + t_hi) * 0.5
    if temp <= t_lo:
        return _COLD
    if temp >= t_hi:
        return _HOT
    if temp < t_mid:
        return _lerp_rgb(
            _COLD, _WARM, (temp - t_lo) / max(t_mid - t_lo, 1e-6),
        )
    return _lerp_rgb(
        _WARM, _HOT, (temp - t_mid) / max(t_hi - t_mid, 1e-6),
    )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseRenderer(ABC):
    """Override to implement a concrete renderer."""

    @abstractmethod
    def build(self, parent: int | str) -> None: ...

    @abstractmethod
    def render(self, solver: Any) -> None: ...


# ---------------------------------------------------------------------------
# Isometric airflow renderer
# ---------------------------------------------------------------------------

class AirflowRenderer(BaseRenderer):
    """Isometric airflow visualisation with velocity arrows."""

    def __init__(self, settings: Settings, config: dict) -> None:
        self._settings = settings

        grid = config.get("grid", {})
        nx = int(grid.get("x", 32))
        ny = int(grid.get("y", 32))

        # Sub-sampling step (LOD) and hard cap
        self._skip = max(1, min(nx, ny) // 8)
        self._max_arrows = 512
        self._render_every = 3   # render every N calls
        self._frame = 0

        # Camera state
        self._yaw = math.pi / 6
        self._pitch = math.pi / 5
        self._zoom = 1.0
        self._dragging = False
        self._last_mouse: tuple[float, float] | None = None

        # DPG tags (set during build)
        self._canvas: int | str = 0
        self._room_layer: int | str = 0
        self._arrow_layer: int | str = 0
        self._hud_layer: int | str = 0
        self._room_dirty = True

        self._arrow_scale = 15.0
        self._arrow_count = 0

        logger.info("Renderer: skip=%d max_arrows=%d", self._skip, self._max_arrows)

    # -- build ---------------------------------------------------------------

    def build(self, parent: int | str) -> None:
        with dpg.child_window(parent=parent, autosize_x=True, height=RENDER_H + 50):
            with dpg.group(horizontal=True):
                dpg.add_text("Airflow Visualization")
                dpg.add_slider_float(
                    tag="render_scale",
                    label="Arrow Scale",
                    default_value=self._arrow_scale,
                    min_value=1.0,
                    max_value=60.0,
                    width=140,
                    callback=self._on_arrow_scale,
                )
                dpg.add_text(
                    "RMB drag: rotate  |  Scroll: zoom",
                    color=(140, 140, 140),
                )

            with dpg.drawlist(
                width=RENDER_W, height=RENDER_H,
            ) as self._canvas:
                self._room_layer = dpg.add_draw_layer()
                self._arrow_layer = dpg.add_draw_layer()
                self._hud_layer = dpg.add_draw_layer()

        with dpg.handler_registry(tag="render_handlers"):
            dpg.add_mouse_down_handler(
                button=dpg.mvMouseButton_Right, callback=self._on_rdown,
            )
            dpg.add_mouse_release_handler(
                button=dpg.mvMouseButton_Right, callback=self._on_rup,
            )
            dpg.add_mouse_move_handler(callback=self._on_mmove)
            dpg.add_mouse_wheel_handler(callback=self._on_scroll)

        self._room_dirty = True
        logger.info("Renderer UI built (%dx%d)", RENDER_W, RENDER_H)

    # -- camera projection ---------------------------------------------------

    def _cam_scale(self) -> float:
        room = self._settings.room
        if room is None:
            return 1.0
        md = max(room.width, room.length, room.height, 1.0)
        return (min(RENDER_W, RENDER_H) - 80) / (md * 2.0) * self._zoom

    def _project(self, x: float, y: float, z: float) -> tuple[float, float]:
        """Project a single 3D world point to 2D screen coords."""
        room = self._settings.room
        if room is None:
            return (0.0, 0.0)

        dx = x - room.width / 2
        dy = y - room.length / 2
        dz = z - room.height / 2

        cy, sy = math.cos(self._yaw), math.sin(self._yaw)
        rx = dx * cy - dy * sy
        ry = dx * sy + dy * cy

        cp, sp = math.cos(self._pitch), math.sin(self._pitch)
        rz2 = ry * sp + dz * cp

        sc = self._cam_scale()
        return (RENDER_W / 2 + rx * sc, RENDER_H / 2 - rz2 * sc)

    def _project_arrays(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised projection for numpy arrays."""
        room = self._settings.room
        dx = x - room.width / 2
        dy = y - room.length / 2
        dz = z - room.height / 2

        cy, sy = math.cos(self._yaw), math.sin(self._yaw)
        rx = dx * cy - dy * sy
        ry = dx * sy + dy * cy

        cp, sp = math.cos(self._pitch), math.sin(self._pitch)
        rz2 = ry * sp + dz * cp

        sc = self._cam_scale()
        return (RENDER_W / 2 + rx * sc, RENDER_H / 2 - rz2 * sc)

    def _project_vec_arrays(
        self, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project velocity vectors (rotation only, no translation)."""
        cy, sy = math.cos(self._yaw), math.sin(self._yaw)
        rx = vx * cy - vy * sy
        ry = vx * sy + vy * cy

        cp, sp = math.cos(self._pitch), math.sin(self._pitch)
        rz2 = ry * sp + vz * cp

        sc = self._cam_scale()
        return (rx * sc, -rz2 * sc)

    # -- room wireframe ------------------------------------------------------

    def _draw_room(self) -> None:
        dpg.delete_item(self._room_layer, children_only=True)
        room = self._settings.room
        if room is None:
            return

        W, L, H = room.width, room.length, room.height

        # Box corners
        corners = [
            (0, 0, 0), (W, 0, 0), (W, L, 0), (0, L, 0),
            (0, 0, H), (W, 0, H), (W, L, H), (0, L, H),
        ]
        pts = [self._project(*c) for c in corners]

        for a, b in [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]:
            dpg.draw_line(
                pts[a], pts[b],
                color=WIRE_COLOR, thickness=WIRE_THICK,
                parent=self._room_layer,
            )

        # Wall labels
        for lbl, pos in [
            ("N", (W / 2, 0, H / 2)),
            ("S", (W / 2, L, H / 2)),
            ("W", (0, L / 2, H / 2)),
            ("E", (W, L / 2, H / 2)),
        ]:
            sp = self._project(*pos)
            dpg.draw_text(
                sp, lbl, color=(200, 200, 200, 150), size=14,
                parent=self._room_layer,
            )

        # Elements on walls
        self._draw_wall_elements(W, L, H)
        self._room_dirty = False

    def _draw_wall_elements(
        self, W: float, L: float, H: float,
    ) -> None:
        room = self._settings.room
        if room is None:
            return

        for face, (o_coeff, ua, va) in _FACE_XFORMS.items():
            # Compute origin from room dims
            orig = (o_coeff[0] * W, o_coeff[1] * L, o_coeff[2] * H)

            for elem in room.walls.get(face, []):
                col = ELEM_COLORS.get(
                    elem.element_type, (200, 200, 200, 100),
                )
                u, v = elem.pos
                su, sv = elem.size

                quad = []
                for cu, cv in [(u, v), (u + su, v), (u + su, v + sv), (u, v + sv)]:
                    p3 = (
                        orig[0] + cu * ua[0] + cv * va[0],
                        orig[1] + cu * ua[1] + cv * va[1],
                        orig[2] + cu * ua[2] + cv * va[2],
                    )
                    quad.append(self._project(*p3))

                dpg.draw_quad(
                    *quad,
                    color=col, fill=col, thickness=1,
                    parent=self._room_layer,
                )

    # -- arrow rendering -----------------------------------------------------

    def render(self, solver: Any) -> None:
        """Sample solver fields and draw velocity arrows."""
        self._frame += 1
        if self._frame % self._render_every != 0:
            return

        if self._room_dirty:
            self._draw_room()

        dpg.delete_item(self._arrow_layer, children_only=True)
        dpg.delete_item(self._hud_layer, children_only=True)

        room = self._settings.room
        if room is None or solver is None:
            self._draw_hud(0)
            return

        vel = solver.velocity       # (3, Nz, Ny, Nx) torch tensor
        temp = solver.temperature   # (Nz, Ny, Nx) torch tensor
        nz, ny, nx = temp.shape
        W, L, H = room.width, room.length, room.height
        dx_cell, dy_cell, dz_cell = W / nx, L / ny, H / nz
        skip = self._skip

        # Subsample → CPU numpy
        vel_sub = vel[:, ::skip, ::skip, ::skip].cpu().numpy()
        temp_sub = temp[::skip, ::skip, ::skip].cpu().numpy()
        snz, sny, snx = temp_sub.shape

        # Speed magnitude (flat)
        speed = np.sqrt(
            vel_sub[0] ** 2 + vel_sub[1] ** 2 + vel_sub[2] ** 2,
        ).ravel()

        # Filter out near-zero velocities
        mask = speed > 1e-4
        indices = np.where(mask)[0]

        if len(indices) == 0:
            self._draw_hud(0)
            return

        # Cap to max_arrows (keep fastest)
        if len(indices) > self._max_arrows:
            top = np.argpartition(
                speed[indices], -self._max_arrows,
            )[-self._max_arrows:]
            indices = indices[top]

        # Unravel to 3D sub-grid indices
        iz, iy, ix = np.unravel_index(indices, (snz, sny, snx))

        # World positions (cell centres)
        wx = (ix * skip + skip * 0.5) * dx_cell
        wy = (iy * skip + skip * 0.5) * dy_cell
        wz = (iz * skip + skip * 0.5) * dz_cell

        # Velocities at sample points
        vx = vel_sub[0].ravel()[indices]
        vy = vel_sub[1].ravel()[indices]
        vz = vel_sub[2].ravel()[indices]
        spd = speed[indices]
        temps = temp_sub.ravel()[indices]

        # Project positions (vectorised)
        sx, sy = self._project_arrays(wx, wy, wz)

        # Project velocity directions (vectorised)
        dvx, dvy = self._project_vec_arrays(vx, vy, vz)
        dlen = np.sqrt(dvx ** 2 + dvy ** 2)
        dlen = np.maximum(dlen, 1e-6)

        # Arrow length from speed, clamped
        arrow_len = np.clip(spd * self._arrow_scale, 2.0, 40.0)

        # Tip positions
        tip_x = sx + dvx / dlen * arrow_len
        tip_y = sy + dvy / dlen * arrow_len

        # Temperature range for colormap
        t_indoor = self._settings.temp_indoor
        t_outdoor = self._settings.temp_outdoor
        t_lo = min(t_indoor, t_outdoor) - 2
        t_hi = max(t_indoor, t_outdoor) + 2

        # Draw arrows
        n = len(indices)
        for i in range(n):
            r, g, b = temp_color(float(temps[i]), t_lo, t_hi)
            a = min(255, int(120 + float(spd[i]) * 300))
            dpg.draw_arrow(
                (float(tip_x[i]), float(tip_y[i])),
                (float(sx[i]), float(sy[i])),
                color=(r, g, b, a),
                thickness=1,
                size=4,
                parent=self._arrow_layer,
            )

        self._arrow_count = n
        self._draw_hud(n)
        self._draw_colorbar(t_lo, t_hi)

        # Periodic log
        if self._frame % (self._render_every * 60) == 0:
            logger.info(
                "Render: arrows=%d, skip=%d, scale=%.0f",
                n, self._skip, self._arrow_scale,
            )

    # -- HUD / colorbar ------------------------------------------------------

    def _draw_hud(self, arrow_count: int) -> None:
        dpg.draw_text(
            (8, 8),
            f"Arrows: {arrow_count}  Skip: {self._skip}",
            color=(200, 200, 200, 200), size=12,
            parent=self._hud_layer,
        )

    def _draw_colorbar(self, t_lo: float, t_hi: float) -> None:
        x0, y0, bh, bw = RENDER_W - 30, 30, 100, 12
        steps = 16
        step_h = bh / steps + 1

        for i in range(steps):
            t = t_lo + (t_hi - t_lo) * i / steps
            r, g, b = temp_color(t, t_lo, t_hi)
            fy = y0 + bh - bh * i / steps
            dpg.draw_rectangle(
                (x0, fy - step_h), (x0 + bw, fy),
                fill=(r, g, b, 200), color=(0, 0, 0, 0),
                parent=self._hud_layer,
            )

        dpg.draw_text(
            (x0 - 28, y0 - 14), f"{t_hi:.0f}C",
            color=(200, 200, 200, 180), size=10,
            parent=self._hud_layer,
        )
        dpg.draw_text(
            (x0 - 28, y0 + bh + 2), f"{t_lo:.0f}C",
            color=(200, 200, 200, 180), size=10,
            parent=self._hud_layer,
        )

    # -- public API ----------------------------------------------------------

    def invalidate_room(self) -> None:
        """Mark room wireframe for redraw (call after room/camera change)."""
        self._room_dirty = True

    # -- camera callbacks ----------------------------------------------------

    def _on_rdown(self, sender: Any = None, app_data: Any = None) -> None:
        if dpg.is_item_hovered(self._canvas):
            self._dragging = True
            self._last_mouse = None

    def _on_rup(self, sender: Any = None, app_data: Any = None) -> None:
        self._dragging = False
        self._last_mouse = None

    def _on_mmove(self, sender: Any = None, app_data: Any = None) -> None:
        if not self._dragging:
            return
        mx, my = dpg.get_mouse_pos()
        if self._last_mouse is not None:
            dx = mx - self._last_mouse[0]
            dy = my - self._last_mouse[1]
            self._yaw += dx * 0.005
            self._pitch = max(
                0.05, min(math.pi / 2 - 0.05, self._pitch - dy * 0.005),
            )
            self._room_dirty = True
        self._last_mouse = (mx, my)

    def _on_scroll(self, sender: Any = None, app_data: Any = None) -> None:
        if not dpg.is_item_hovered(self._canvas):
            return
        self._zoom *= 1.1 if app_data > 0 else 0.9
        self._zoom = max(0.3, min(5.0, self._zoom))
        self._room_dirty = True

    def _on_arrow_scale(
        self, sender: Any = None, app_data: Any = None,
    ) -> None:
        self._arrow_scale = app_data
