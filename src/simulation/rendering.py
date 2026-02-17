"""Task 6 -- Airflow Visualization -- velocity arrows + surface temps.

Isometric 3D view rendered on a DPG drawlist. Velocity arrows are
densely subsampled from the solver grid, coloured by temperature
(blue-cold, yellow-ambient, red-hot) and scaled by speed. Wall,
floor, and ceiling surfaces are painted with interpolated temperature
gradients.

Camera: LMB drag = pan, RMB drag = orbit, Scroll = zoom, R = reset.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import dearpygui.dearpygui as dpg

from models.models import Settings
from utils.logger import setup_logger

logger = setup_logger("Render")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RENDER_W, RENDER_H = 700, 430
WIRE_COLOR = (180, 180, 190, 220)
WIRE_THICK = 1.5

_COLD = (30, 100, 255)
_WARM = (255, 230, 50)
_HOT = (255, 40, 30)

_VEL_THRESHOLD = 1e-6
_MAX_ARROWS = 5000
_SURF_PATCHES = 8

ELEM_COLORS = {
    "door": (230, 160, 50, 100),
    "window": (80, 180, 255, 100),
    "vent": (100, 220, 120, 100),
}

_DEFAULT_YAW = math.pi / 6
_DEFAULT_PITCH = math.radians(35)
_DEFAULT_ZOOM = 1.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lerp_rgb(c0, c1, t):
    t = max(0.0, min(1.0, t))
    return (
        int(c0[0] + (c1[0] - c0[0]) * t),
        int(c0[1] + (c1[1] - c0[1]) * t),
        int(c0[2] + (c1[2] - c0[2]) * t),
    )

def temp_color(temp: float, t_lo: float, t_hi: float) -> tuple[int, int, int]:
    t_mid = (t_lo + t_hi) * 0.5
    if temp <= t_lo: return _COLD
    if temp >= t_hi: return _HOT
    if temp < t_mid:
        return _lerp_rgb(_COLD, _WARM, (temp - t_lo) / max(t_mid - t_lo, 1e-6))
    return _lerp_rgb(_WARM, _HOT, (temp - t_mid) / max(t_hi - t_mid, 1e-6))

# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class AirflowRenderer:
    def __init__(self, settings: Settings, config: dict) -> None:
        self._settings = settings

        grid = config.get("grid", {})
        self._skip = max(1, int(grid.get("x", 32)) // 16)
        self._max_arrows = _MAX_ARROWS
        self._render_every = 2
        self._frame = 0

        # Camera
        self._yaw = _DEFAULT_YAW
        self._pitch = _DEFAULT_PITCH
        self._zoom = _DEFAULT_ZOOM
        self._cam_x = 0.0
        self._cam_y = 0.0

        # DPG items
        self._vis_window = None
        self._canvas = None
        self._surface_layer = None
        self._room_layer = None
        self._arrow_layer = None
        self._hud_layer = None
        self._room_dirty = True

        self._arrow_scale = 15.0
        self._arrow_count = 0
        self._surf_t_min = 0.0
        self._surf_t_max = 0.0

        logger.info("Renderer initialized (skip=%d, max_arrows=%d)", self._skip, self._max_arrows)

    def build(self, parent: int | str) -> None:
        """Build visualization inside Simulation tab."""
        with dpg.child_window(
            parent=parent,
            autosize_x=True,
            height=-1,
            no_scrollbar=True,
            border=True,
        ) as self._vis_window:

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
                    "LMB: pan | RMB: orbit | Scroll: zoom | R: reset",
                    color=(140, 140, 140),
                )

            # ←←← Canvas must be created BEFORE we bind handlers
            with dpg.drawlist(width=RENDER_W, height=RENDER_H) as self._canvas:
                self._surface_layer = dpg.add_draw_layer()
                self._room_layer = dpg.add_draw_layer()
                self._arrow_layer = dpg.add_draw_layer()
                self._hud_layer = dpg.add_draw_layer()

        # Bind handler registry to this canvas (so handlers only fire here)
        dpg.bind_item_handler_registry(self._canvas, "global_render_handlers")

        self._room_dirty = True
        logger.info("Renderer UI built (%dx%d) with global handlers", RENDER_W, RENDER_H)

    # ====================== INPUT HANDLER BINDING ======================
    def bind_input_handlers(self):
        """Centralized, debuggable, modular input wiring.
        Call this once after build(). Easy to extend and log."""
        try:
            dpg.set_item_callback("mouse_left_drag",  self._on_drag)
            dpg.set_item_callback("mouse_right_drag", self._on_drag)   # same method, checks button
            dpg.set_item_callback("mouse_wheel",      self._on_scroll)
            dpg.set_item_callback("key_r",            self._on_key_r)

            logger.info("Render: Input handlers bound successfully "
                       "(left_drag, right_drag, wheel, key_r)")
        except Exception as e:
            logger.error(f"Render: Failed to bind handlers - {e}", exc_info=True)
            raise

    # ====================== CAMERA CALLBACKS ======================

    def _on_drag(self, sender, app_data):
        """LMB = Pan, RMB = Orbit"""
        if not dpg.is_item_hovered(self._canvas):
            return

        button = app_data[0]
        dx = app_data[1]
        dy = app_data[2]

        if button == dpg.mvMouseButton_Right:   # RMB = Orbit
            self._yaw += dx * 0.015
            self._pitch = max(0.1, min(1.45, self._pitch - dy * 0.015))
        elif button == dpg.mvMouseButton_Left:  # LMB = Pan
            self._cam_x += dx * 0.65
            self._cam_y -= dy * 0.65

        self._room_dirty = True

    def _on_scroll(self, sender, app_data):
        """Scroll = Zoom"""
        if not dpg.is_item_hovered(self._canvas):
            return
        self._zoom = max(0.15, self._zoom + app_data * 0.09)
        self._room_dirty = True

    def _on_key_r(self, sender, app_data):
        self.reset_camera()

    def _on_arrow_scale(self, sender, app_data):
        self._arrow_scale = app_data

    # ====================== CAMERA & DRAWING ======================

    def _cam_scale(self) -> float:
        room = self._settings.room
        if not room:
            return 1.0
        md = max(room.width, room.length, room.height, 1.0)
        return (min(RENDER_W, RENDER_H) - 80) / (md * 2.0) * self._zoom

    def _project(self, x: float, y: float, z: float) -> tuple[float, float]:
        room = self._settings.room
        if not room:
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
        return (
            RENDER_W / 2 + rx * sc + self._cam_x,
            RENDER_H / 2 - rz2 * sc + self._cam_y,
        )

    def _project_arrays(self, x, y, z):
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
        return (
            RENDER_W / 2 + rx * sc + self._cam_x,
            RENDER_H / 2 - rz2 * sc + self._cam_y,
        )

    def _depth_arrays(self, x, y, z):
        room = self._settings.room
        dx = x - room.width / 2
        dy = y - room.length / 2
        dz = z - room.height / 2

        cy, sy = math.cos(self._yaw), math.sin(self._yaw)
        ry = dx * sy + dy * cy

        cp, sp = math.cos(self._pitch), math.sin(self._pitch)
        return ry * cp - dz * sp
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
            orig = (o_coeff[0] * W, o_coeff[1] * L, o_coeff[2] * H)

            for elem in room.walls.get(face, []):
                col = ELEM_COLORS.get(
                    elem.element_type, (200, 200, 200, 100),
                )
                u, v = elem.pos
                su, sv = elem.size

                quad = []
                for cu, cv in [
                    (u, v), (u + su, v), (u + su, v + sv), (u, v + sv),
                ]:
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

    # -- surface temperature coloring ----------------------------------------

    def _draw_surface_temps(
        self, temp_np: np.ndarray, t_lo: float, t_hi: float,
    ) -> None:
        """Draw temperature-colored patches on visible wall/floor/ceiling."""
        dpg.delete_item(self._surface_layer, children_only=True)

        room = self._settings.room
        if room is None:
            return

        nz, ny, nx = temp_np.shape
        W, L, H = room.width, room.length, room.height

        # View direction for back-face culling
        vd = self._view_direction()

        surf_temps: list[float] = []

        # Each face: (name, 2D temp slice, corner_fn(u_frac, v_frac) -> 3D)
        faces = [
            ("floor",   temp_np[0, :, :],
             lambda u, v: (u * W, v * L, 0)),
            ("ceiling", temp_np[-1, :, :],
             lambda u, v: (u * W, v * L, H)),
            ("south",   temp_np[:, 0, :],
             lambda u, v: (u * W, 0, v * H)),
            ("north",   temp_np[:, -1, :],
             lambda u, v: (u * W, L, v * H)),
            ("west",    temp_np[:, :, 0],
             lambda u, v: (0, u * L, v * H)),
            ("east",    temp_np[:, :, -1],
             lambda u, v: (W, u * L, v * H)),
        ]

        for face_name, face_temp, corner_fn in faces:
            # Back-face culling: skip faces whose inner normal
            # points away from camera (dot >= 0)
            inorm = _INNER_NORMALS[face_name]
            dot = inorm[0] * vd[0] + inorm[1] * vd[1] + inorm[2] * vd[2]
            if dot >= 0:
                continue

            nv, nu = face_temp.shape
            p_u = min(_SURF_PATCHES, max(1, nu // 2))
            p_v = min(_SURF_PATCHES, max(1, nv // 2))

            for pi in range(p_u):
                for pj in range(p_v):
                    u0 = pi / p_u
                    u1 = (pi + 1) / p_u
                    v0 = pj / p_v
                    v1 = (pj + 1) / p_v

                    # Cell range for averaging
                    cu0 = int(pi * nu / p_u)
                    cu1 = max(cu0 + 1, int((pi + 1) * nu / p_u))
                    cv0 = int(pj * nv / p_v)
                    cv1 = max(cv0 + 1, int((pj + 1) * nv / p_v))

                    avg_t = float(face_temp[cv0:cv1, cu0:cu1].mean())
                    surf_temps.append(avg_t)
                    r, g, b = temp_color(avg_t, t_lo, t_hi)

                    q = [
                        self._project(*corner_fn(u0, v0)),
                        self._project(*corner_fn(u1, v0)),
                        self._project(*corner_fn(u1, v1)),
                        self._project(*corner_fn(u0, v1)),
                    ]
                    dpg.draw_quad(
                        *q,
                        color=(0, 0, 0, 0),
                        fill=(r, g, b, 120),
                        parent=self._surface_layer,
                    )

        if surf_temps:
            self._surf_t_min = min(surf_temps)
            self._surf_t_max = max(surf_temps)

    # -- sampling helpers ----------------------------------------------------

    def _sample_indices(
        self, nz: int, ny: int, nx: int,
    ) -> tuple[list[int], list[int], list[int]]:
        """Dense sampling: full z resolution, skip-based xy + boundaries.

        With skip=2 on 32x32x16 this yields 16 x 17 x 17 = 4624 samples,
        right in the 2000-5000 arrow target range.
        """
        skip = self._skip
        z = list(range(nz))                              # every z level
        y = sorted(set(range(0, ny, skip)) | {ny - 1})   # skip in y + boundary
        x = sorted(set(range(0, nx, skip)) | {nx - 1})   # skip in x + boundary
        return z, y, x

    # -- arrow rendering -----------------------------------------------------

    def render(self, solver: Any) -> None:
        """Sample solver fields and draw velocity arrows + surface temps."""
        self._frame += 1

        # Throttle rendering but always respond to camera changes
        if self._frame % self._render_every != 0 and not self._room_dirty:
            return

        if self._room_dirty:
            self._draw_room()

        dpg.delete_item(self._arrow_layer, children_only=True)
        dpg.delete_item(self._hud_layer, children_only=True)

        room = self._settings.room
        if room is None or solver is None:
            self._draw_hud(0, 0.0)
            return

        # -- get data from solver (one GPU->CPU transfer for temp) -----------
        vel, temp = solver.expose_vel_temp()
        temp_np = temp.cpu().numpy()
        nz, ny, nx = temp_np.shape
        W, L, H = room.width, room.length, room.height
        dx_cell, dy_cell, dz_cell = W / nx, L / ny, H / nz

        # Temperature range for colormap
        t_indoor = self._settings.temp_indoor
        t_outdoor = self._settings.temp_outdoor
        t_lo = min(t_indoor, t_outdoor) - 2
        t_hi = max(t_indoor, t_outdoor) + 2

        # -- surface temperature coloring ------------------------------------
        self._draw_surface_temps(temp_np, t_lo, t_hi)

        # -- dense subsampling for arrows ------------------------------------
        z_idx, y_idx, x_idx = self._sample_indices(nz, ny, nx)

        # Subsample vel on GPU, then transfer (smaller than full field)
        vel_sub = vel[:, z_idx][:, :, y_idx][:, :, :, x_idx].cpu().numpy()
        # Subsample temp from already-transferred numpy array
        temp_sub = temp_np[np.ix_(z_idx, y_idx, x_idx)]

        snz, sny, snx = temp_sub.shape
        z_arr = np.array(z_idx)
        y_arr = np.array(y_idx)
        x_arr = np.array(x_idx)

        # Speed magnitude (flat)
        speed = np.sqrt(
            vel_sub[0] ** 2 + vel_sub[1] ** 2 + vel_sub[2] ** 2,
        ).ravel()

        max_vel = float(speed.max()) if speed.size > 0 else 0.0

        # Filter by velocity threshold
        mask = speed > _VEL_THRESHOLD
        indices = np.where(mask)[0]

        if len(indices) == 0:
            logger.warning(
                "Render: No arrows -- vel max=%.6f. "
                "Open elements or enable AC/fan?",
                max_vel,
            )
            self._draw_hud(0, max_vel)
            return

        # Cap to max_arrows (keep fastest)
        if len(indices) > self._max_arrows:
            top = np.argpartition(
                speed[indices], -self._max_arrows,
            )[-self._max_arrows:]
            indices = indices[top]

        # Unravel to 3D sub-grid indices, then map to actual grid indices
        iz_sub, iy_sub, ix_sub = np.unravel_index(
            indices, (snz, sny, snx),
        )
        wx = (x_arr[ix_sub] + 0.5) * dx_cell
        wy = (y_arr[iy_sub] + 0.5) * dy_cell
        wz = (z_arr[iz_sub] + 0.5) * dz_cell

        # Velocities at sample points
        vx = vel_sub[0].ravel()[indices]
        vy = vel_sub[1].ravel()[indices]
        vz = vel_sub[2].ravel()[indices]
        spd = speed[indices]
        temps = temp_sub.ravel()[indices]

        # -- depth for sorting & alpha --------------------------------------
        depth = self._depth_arrays(wx, wy, wz)
        depth_min = float(depth.min())
        depth_max = float(depth.max())
        depth_range = max(depth_max - depth_min, 1e-6)
        depth_norm = (depth - depth_min) / depth_range  # 0=near, 1=far

        # Painter's algorithm: draw far objects first
        sort_order = np.argsort(-depth)

        # -- project positions (vectorised) ----------------------------------
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

        # -- draw arrows (back-to-front) ------------------------------------
        n = len(indices)
        for idx in sort_order:
            i = int(idx)
            r, g, b = temp_color(float(temps[i]), t_lo, t_hi)
            # Depth-based alpha: near = bright, far = dim
            d = float(depth_norm[i])
            alpha = int(80 + 175 * (1.0 - d))
            thick = 1.0 + 1.5 * (1.0 - d)
            dpg.draw_arrow(
                (float(tip_x[i]), float(tip_y[i])),
                (float(sx[i]), float(sy[i])),
                color=(r, g, b, alpha),
                thickness=thick,
                size=4,
                parent=self._arrow_layer,
            )

        self._arrow_count = n
        self._draw_hud(n, max_vel)
        self._draw_colorbar(t_lo, t_hi)

        if self._frame % 120 == 0:
            logger.info(
                "Render: Arrows=%d, MaxVel=%.1fm/s, "
                "SurfaceTempRange=%.1f-%.1f\u00b0C",
                n, max_vel, self._surf_t_min, self._surf_t_max,
            )

    # -- HUD / colorbar ------------------------------------------------------

    def _draw_hud(self, arrow_count: int, max_vel: float) -> None:
        txt = (
            f"Arrows: {arrow_count}  |  MaxVel: {max_vel:.2f}m/s  |  "
            f"SurfTemp: {self._surf_t_min:.1f}-{self._surf_t_max:.1f}\u00b0C"
        )
        dpg.draw_text(
            (8, 8), txt,
            color=(200, 200, 200, 200), size=11,
            parent=self._hud_layer,
        )
        dpg.draw_text(
            (8, RENDER_H - 18),
            "LMB: pan | RMB: orbit | Scroll: zoom | R: reset view",
            color=(120, 120, 130, 160), size=10,
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

    def reset_camera(self):
        self._yaw = _DEFAULT_YAW
        self._pitch = _DEFAULT_PITCH
        self._zoom = _DEFAULT_ZOOM
        self._cam_x = 0.0
        self._cam_y = 0.0
        self._room_dirty = True
        logger.debug("Render: Camera reset")

    def tick(self) -> None:
        """Interpolate camera values toward targets (call every frame).

        Provides light damping / momentum feel on both trackpad and mouse.
        """
        moved = False
        for attr in ("_yaw", "_pitch", "_zoom", "_pan_x", "_pan_y"):
            cur = getattr(self, attr)
            tgt = getattr(self, attr + "_target")
            diff = tgt - cur
            if abs(diff) > 1e-4:
                setattr(self, attr, cur + diff * _SMOOTH_FACTOR)
                moved = True
            elif diff != 0.0:
                setattr(self, attr, tgt)
                moved = True
        if moved:
            self._room_dirty = True

    # -- camera callbacks ----------------------------------------------------
    def _on_lclick(self, sender, app_data):
        if dpg.is_item_hovered(self._canvas):
            self._panning = True
            self._last_mouse = dpg.get_mouse_pos()

    def _on_lrelease(self, sender, app_data):
        self._panning = False
        self._last_mouse = None

    def _on_rclick(self, sender, app_data):
        if dpg.is_item_hovered(self._canvas):
            self._dragging = True
            self._last_mouse = dpg.get_mouse_pos()

    def _on_rrelease(self, sender, app_data):
        self._dragging = False
        self._last_mouse = None

    def _on_rdown(self, sender: Any = None, app_data: Any = None) -> None:
        if self._is_over_canvas():
            self._rmb_dragging = True
            self._rmb_last = None

    def _on_rup(self, sender: Any = None, app_data: Any = None) -> None:
        if self._rmb_dragging:
            self._rmb_dragging = False
            self._rmb_last = None
            logger.debug("Render: Camera orbited")

    def _on_ldown(self, sender: Any = None, app_data: Any = None) -> None:
        if self._is_over_canvas():
            self._lmb_dragging = True
            self._lmb_last = None

    def _on_lup(self, sender: Any = None, app_data: Any = None) -> None:
        if self._lmb_dragging:
            self._lmb_dragging = False
            self._lmb_last = None
            logger.debug("Render: Camera panned")

    def _on_mmove(self, sender, app_data):
        if not (self._dragging or self._panning):
            return
        mx, my = dpg.get_mouse_pos()
        if self._last_mouse is None:
            self._last_mouse = (mx, my)
            return

        dx = mx - self._last_mouse[0]
        dy = my - self._last_mouse[1]

        if self._dragging:          # RMB = Orbit
            self._yaw += dx * 0.008
            self._pitch = max(0.1, min(math.pi/2 - 0.1, self._pitch - dy * 0.008))
            self._room_dirty = True
        elif self._panning:         # LMB = Pan
            self._cam_x = getattr(self, '_cam_x', 0) + dx * 0.8
            self._cam_y = getattr(self, '_cam_y', 0) - dy * 0.8
            self._room_dirty = True

        self._last_mouse = (mx, my)

    def _on_drag(self, sender, app_data):
        """Called when mouse is dragged (LMB or RMB)"""
        if not dpg.is_item_hovered(self._canvas):
            return

        button = app_data[0]      # which button is being dragged
        dx = app_data[1]          # delta x
        dy = app_data[2]          # delta y

        if button == dpg.mvMouseButton_Right:          # RMB = Orbit
            self._yaw += dx * 0.012
            self._pitch = max(0.1, min(1.4, self._pitch - dy * 0.012))
        elif button == dpg.mvMouseButton_Left:         # LMB = Pan
            self._cam_x = getattr(self, '_cam_x', 0.0) + dx * 0.7
            self._cam_y = getattr(self, '_cam_y', 0.0) - dy * 0.7

        self._room_dirty = True

    def _on_scroll(self, sender, app_data):
        """Mouse wheel zoom"""
        if not dpg.is_item_hovered(self._canvas):
            return
        self._zoom = max(0.2, self._zoom + app_data * 0.08)
        self._room_dirty = True

    def _on_key_r(self, sender: Any = None, app_data: Any = None) -> None:
        self.reset_camera()

    def _on_arrow_scale(
        self, sender: Any = None, app_data: Any = None,
    ) -> None:
        self._arrow_scale = app_data
