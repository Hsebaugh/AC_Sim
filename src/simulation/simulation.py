"""Task 5 -- Fluid / thermal simulation on a 3-D voxel grid.

Simplified Navier-Stokes (advection, diffusion, pressure projection)
plus thermal advection-diffusion with buoyancy.  Runs on PyTorch
(MPS/Metal on Apple Silicon, CPU fallback).

Public API
----------
Solver(settings, config)   – build grid, set boundaries
solver.step(dt)            – advance one time-step
solver.velocity             – (3, Nz, Ny, Nx) tensor  [m/s]
solver.temperature          – (Nz, Ny, Nx) tensor      [°C]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn.functional as F

from models.models import Room, Settings, WALL_NAMES
from utils.logger import setup_logger

logger = setup_logger("Sim")

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        logger.info("GPU: Metal / MPS")
        return torch.device("mps")
    if torch.cuda.is_available():
        logger.info("GPU: CUDA")
        return torch.device("cuda")
    logger.info("GPU: none – using CPU")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Abstract solver base (extensible for humidity, particles, etc.)
# ---------------------------------------------------------------------------

class BaseSolver(ABC):
    """Override `step` to implement a concrete solver."""

    @abstractmethod
    def step(self, dt: float) -> None: ...

    @property
    @abstractmethod
    def velocity(self) -> torch.Tensor: ...

    @property
    @abstractmethod
    def temperature(self) -> torch.Tensor: ...


# ---------------------------------------------------------------------------
# Navier-Stokes + thermal solver
# ---------------------------------------------------------------------------

class Solver(BaseSolver):
    """Simplified incompressible NS + thermal solver on a uniform grid."""

    # Physical constants
    VISCOSITY = 1.5e-5       # kinematic viscosity of air  [m²/s]
    THERMAL_DIFF = 2.2e-5    # thermal diffusivity of air  [m²/s]
    BETA = 3.4e-3            # thermal expansion coeff     [1/K]
    GRAVITY = 9.81           # m/s²

    def __init__(self, settings: Settings, config: dict) -> None:
        self._settings = settings
        room = settings.room
        if room is None:
            raise ValueError("Settings.room must be set before creating Solver")

        # Grid resolution
        grid_cfg = config.get("grid", {})
        self._nx: int = int(grid_cfg.get("x", 32))
        self._ny: int = int(grid_cfg.get("y", 32))
        self._nz: int = int(grid_cfg.get("z", 16))

        # Physical cell sizes  [metres]
        self._dx = room.width / self._nx
        self._dy = room.length / self._ny
        self._dz = room.height / self._nz

        # Device
        self._device = _select_device()

        # State tensors  (z, y, x ordering)
        shape = (self._nz, self._ny, self._nx)
        self._vel = torch.zeros(
            (3, *shape), dtype=torch.float32, device=self._device,
        )  # [vx, vy, vz]
        self._temp = torch.full(
            shape, settings.temp_indoor, dtype=torch.float32,
            device=self._device,
        )
        self._pressure = torch.zeros(
            shape, dtype=torch.float32, device=self._device,
        )

        # Pre-compute boundary masks
        self._wall_mask = self._build_wall_mask()
        self._inflow_vel, self._inflow_temp = self._build_boundary_sources()

        # Pre-compute derived boundary masks (avoid per-step recomputation)
        self._inflow_active = self._inflow_vel.abs().sum(dim=0) > 0
        self._inflow_temp_valid = ~torch.isnan(self._inflow_temp)

        # Pre-compute grid coordinates for advection
        gz = torch.arange(self._nz, device=self._device, dtype=torch.float32)
        gy = torch.arange(self._ny, device=self._device, dtype=torch.float32)
        gx = torch.arange(self._nx, device=self._device, dtype=torch.float32)
        self._grid_z, self._grid_y, self._grid_x = torch.meshgrid(
            gz, gy, gx, indexing="ij",
        )

        logger.info(
            "Grid init: res=%dx%dx%d, cell=%.3fm, device=%s",
            self._nx, self._ny, self._nz, self._dx, self._device,
        )

    # -- public properties ---------------------------------------------------

    @property
    def velocity(self) -> torch.Tensor:
        return self._vel

    @property
    def temperature(self) -> torch.Tensor:
        return self._temp

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self._nz, self._ny, self._nx)

    @property
    def cell_size(self) -> tuple[float, float, float]:
        """Cell dimensions (dx, dy, dz) in metres."""
        return (self._dx, self._dy, self._dz)

    # -- boundary construction -----------------------------------------------

    def _build_wall_mask(self) -> torch.Tensor:
        """Boolean mask: True where cells are solid wall (no-slip)."""
        mask = torch.zeros(
            (self._nz, self._ny, self._nx),
            dtype=torch.bool, device=self._device,
        )
        # Six faces of the box
        mask[0, :, :] = True    # floor
        mask[-1, :, :] = True   # ceiling
        mask[:, 0, :] = True    # south
        mask[:, -1, :] = True   # north
        mask[:, :, 0] = True    # west
        mask[:, :, -1] = True   # east
        return mask

    def _wall_face_slice(
        self, wall: str,
    ) -> tuple[slice | int, slice, slice]:
        """Return (z, y, x) index tuple for a wall face."""
        nz, ny, nx = self._nz, self._ny, self._nx
        if wall == "floor":
            return (0, slice(None), slice(None))
        if wall == "ceiling":
            return (nz - 1, slice(None), slice(None))
        if wall == "south":
            return (slice(None), 0, slice(None))
        if wall == "north":
            return (slice(None), ny - 1, slice(None))
        if wall == "west":
            return (slice(None), slice(None), 0)
        if wall == "east":
            return (slice(None), slice(None), nx - 1)
        raise ValueError(f"Unknown wall: {wall}")

    def _element_cells(
        self, wall: str, pos: tuple[float, float], size: tuple[float, float],
    ) -> tuple[slice | int, slice, slice]:
        """Map an element's (pos, size) in metres to grid cell ranges on a wall face."""
        room = self._settings.room
        assert room is not None

        # pos/size are (u, v) in wall-local coords, origin bottom-left
        u0, v0 = pos
        su, sv = size

        if wall in ("north", "south"):
            # u → x,  v → z
            ix0 = int(u0 / room.width * self._nx)
            ix1 = int((u0 + su) / room.width * self._nx)
            iz0 = int(v0 / room.height * self._nz)
            iz1 = int((v0 + sv) / room.height * self._nz)
            ix0, ix1 = max(0, ix0), min(self._nx, ix1)
            iz0, iz1 = max(0, iz0), min(self._nz, iz1)
            y_idx = self._ny - 1 if wall == "north" else 0
            return (slice(iz0, iz1), y_idx, slice(ix0, ix1))

        if wall in ("east", "west"):
            # u → y,  v → z
            iy0 = int(u0 / room.length * self._ny)
            iy1 = int((u0 + su) / room.length * self._ny)
            iz0 = int(v0 / room.height * self._nz)
            iz1 = int((v0 + sv) / room.height * self._nz)
            iy0, iy1 = max(0, iy0), min(self._ny, iy1)
            iz0, iz1 = max(0, iz0), min(self._nz, iz1)
            x_idx = self._nx - 1 if wall == "east" else 0
            return (slice(iz0, iz1), slice(iy0, iy1), x_idx)

        if wall in ("floor", "ceiling"):
            # u → x,  v → y
            ix0 = int(u0 / room.width * self._nx)
            ix1 = int((u0 + su) / room.width * self._nx)
            iy0 = int(v0 / room.length * self._ny)
            iy1 = int((v0 + sv) / room.length * self._ny)
            ix0, ix1 = max(0, ix0), min(self._nx, ix1)
            iy0, iy1 = max(0, iy0), min(self._ny, iy1)
            z_idx = 0 if wall == "floor" else self._nz - 1
            return (z_idx, slice(iy0, iy1), slice(ix0, ix1))

        raise ValueError(f"Unknown wall: {wall}")

    def _inward_normal(self, wall: str) -> tuple[int, int, int]:
        """Unit inward-pointing normal (vx, vy, vz) for a wall."""
        normals = {
            "south": (0, 1, 0),
            "north": (0, -1, 0),
            "west":  (1, 0, 0),
            "east":  (-1, 0, 0),
            "floor": (0, 0, 1),
            "ceiling": (0, 0, -1),
        }
        return normals[wall]

    def _build_boundary_sources(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build velocity injection and temperature source tensors from element states."""
        shape = (self._nz, self._ny, self._nx)
        inflow_vel = torch.zeros(
            (3, *shape), dtype=torch.float32, device=self._device,
        )
        inflow_temp = torch.full(
            shape, float("nan"), dtype=torch.float32, device=self._device,
        )

        room = self._settings.room
        if room is None:
            return inflow_vel, inflow_temp

        t_out = self._settings.temp_outdoor

        for wall_name in WALL_NAMES:
            for elem in room.walls.get(wall_name, []):
                if elem.open_frac <= 0:
                    continue

                cells = self._element_cells(wall_name, elem.pos, elem.size)
                nx, ny, nz = self._inward_normal(wall_name)

                # Open wall cells (remove from solid mask)
                self._wall_mask[cells] = False

                # Velocity proportional to open fraction and temp difference
                speed = elem.open_frac * max(
                    0.5, abs(t_out - self._settings.temp_indoor) * 0.1,
                )
                inflow_vel[0][cells] = nx * speed
                inflow_vel[1][cells] = ny * speed
                inflow_vel[2][cells] = nz * speed

                # Temperature at opening = outdoor
                inflow_temp[cells] = t_out

                logger.debug(
                    "Boundary: %s %s open=%.0f%% speed=%.2fm/s",
                    elem.element_type, wall_name,
                    elem.open_frac * 100, speed,
                )

        return inflow_vel, inflow_temp

    def rebuild_boundaries(self) -> None:
        """Re-scan element states and update boundary masks/sources."""
        self._wall_mask = self._build_wall_mask()
        self._inflow_vel, self._inflow_temp = self._build_boundary_sources()
        self._inflow_active = self._inflow_vel.abs().sum(dim=0) > 0
        self._inflow_temp_valid = ~torch.isnan(self._inflow_temp)
        logger.debug("Boundaries rebuilt")

    # -- AC / fan injection --------------------------------------------------

    def _apply_ac_fan(self) -> None:
        """Inject AC cooling and fan velocity into the grid."""
        s = self._settings
        nz, ny, nx = self._nz, self._ny, self._nx

        if s.ac_on:
            # AC vent: top-centre of ceiling, blowing downward cool air
            cx, cy = nx // 2, ny // 2
            span = max(1, nx // 8)
            z_top = nz - 1
            sl = (z_top, slice(cy - span, cy + span), slice(cx - span, cx + span))
            speed = s.ac_speed * 0.5  # m/s
            self._vel[2][sl] = -speed           # downward
            self._temp[sl] = s.ac_temp          # cooled air

        if s.fan_on:
            # Fan: floor centre, blowing upward (circulation)
            cx, cy = nx // 2, ny // 2
            span = max(1, nx // 8)
            sl = (0, slice(cy - span, cy + span), slice(cx - span, cx + span))
            speed = s.fan_speed * 0.3
            self._vel[2][sl] = speed            # upward

    # -- physics kernels -----------------------------------------------------

    def _advect(self, field: torch.Tensor, vel: torch.Tensor, dt: float) -> torch.Tensor:
        """Semi-Lagrangian advection (first-order, grid-aligned)."""
        nx, ny, nz = self._nx, self._ny, self._nz

        # Backtrack in grid units (using cached grid coords)
        src_x = (self._grid_x - vel[0] * dt / self._dx).clamp(0, nx - 1)
        src_y = (self._grid_y - vel[1] * dt / self._dy).clamp(0, ny - 1)
        src_z = (self._grid_z - vel[2] * dt / self._dz).clamp(0, nz - 1)

        # Trilinear interpolation via grid_sample
        # grid_sample expects (N, C, D, H, W) input and (N, D, H, W, 3) grid in [-1,1]
        norm_x = 2.0 * src_x / max(nx - 1, 1) - 1.0
        norm_y = 2.0 * src_y / max(ny - 1, 1) - 1.0
        norm_z = 2.0 * src_z / max(nz - 1, 1) - 1.0
        grid = torch.stack([norm_x, norm_y, norm_z], dim=-1).unsqueeze(0)
        inp = field.unsqueeze(0).unsqueeze(0)
        out = F.grid_sample(
            inp, grid, mode="bilinear", padding_mode="border", align_corners=True,
        )
        return out.squeeze(0).squeeze(0)

    def _diffuse(self, field: torch.Tensor, coeff: float, dt: float) -> torch.Tensor:
        """Explicit diffusion (Laplacian, forward Euler)."""
        lap = torch.zeros_like(field)
        # Central differences
        lap += (torch.roll(field, 1, 2) + torch.roll(field, -1, 2) - 2 * field) / (self._dx ** 2)
        lap += (torch.roll(field, 1, 1) + torch.roll(field, -1, 1) - 2 * field) / (self._dy ** 2)
        lap += (torch.roll(field, 1, 0) + torch.roll(field, -1, 0) - 2 * field) / (self._dz ** 2)
        return field + coeff * dt * lap

    def _pressure_project(self, vel: torch.Tensor, iterations: int = 12) -> torch.Tensor:
        """Jacobi pressure projection to enforce incompressibility."""
        dx2 = self._dx ** 2
        dy2 = self._dy ** 2
        dz2 = self._dz ** 2

        # Divergence
        div = (
            (torch.roll(vel[0], -1, 2) - torch.roll(vel[0], 1, 2)) / (2 * self._dx)
            + (torch.roll(vel[1], -1, 1) - torch.roll(vel[1], 1, 1)) / (2 * self._dy)
            + (torch.roll(vel[2], -1, 0) - torch.roll(vel[2], 1, 0)) / (2 * self._dz)
        )

        p = self._pressure
        denom = 2.0 * (1.0 / dx2 + 1.0 / dy2 + 1.0 / dz2)

        for _ in range(iterations):
            p = (
                (torch.roll(p, 1, 2) + torch.roll(p, -1, 2)) / dx2
                + (torch.roll(p, 1, 1) + torch.roll(p, -1, 1)) / dy2
                + (torch.roll(p, 1, 0) + torch.roll(p, -1, 0)) / dz2
                - div
            ) / denom
            # Zero-gradient at walls
            p[self._wall_mask] = 0.0

        self._pressure = p

        # Subtract pressure gradient from velocity
        grad_px = (torch.roll(p, -1, 2) - torch.roll(p, 1, 2)) / (2 * self._dx)
        grad_py = (torch.roll(p, -1, 1) - torch.roll(p, 1, 1)) / (2 * self._dy)
        grad_pz = (torch.roll(p, -1, 0) - torch.roll(p, 1, 0)) / (2 * self._dz)

        vel[0] = vel[0] - grad_px
        vel[1] = vel[1] - grad_py
        vel[2] = vel[2] - grad_pz
        return vel

    def _apply_buoyancy(self, vel: torch.Tensor, dt: float) -> torch.Tensor:
        """Add buoyancy force from temperature differences."""
        t_ref = self._settings.temp_indoor
        force_z = self.BETA * self.GRAVITY * (self._temp - t_ref)
        vel[2] = vel[2] + force_z * dt
        return vel

    def _enforce_boundaries(self) -> None:
        """Apply no-slip on walls and inject boundary sources."""
        # No-slip: zero velocity at solid cells
        self._vel[:, self._wall_mask] = 0.0

        # Inject inflow velocity/temperature (using pre-computed masks)
        self._vel[:, self._inflow_active] = self._inflow_vel[:, self._inflow_active]
        self._temp[self._inflow_temp_valid] = self._inflow_temp[self._inflow_temp_valid]

    # -- main step -----------------------------------------------------------

    def step(self, dt: float) -> None:
        """Advance simulation by *dt* seconds."""
        # 1. External forces (AC, fan, buoyancy)
        self._apply_ac_fan()
        self._vel = self._apply_buoyancy(self._vel, dt)

        # 2. Advect velocity (each component)
        for c in range(3):
            self._vel[c] = self._advect(self._vel[c], self._vel, dt)

        # 3. Diffuse velocity
        for c in range(3):
            self._vel[c] = self._diffuse(self._vel[c], self.VISCOSITY, dt)

        # 4. Pressure projection
        self._vel = self._pressure_project(self._vel)

        # 5. Advect temperature
        self._temp = self._advect(self._temp, self._vel, dt)

        # 6. Diffuse temperature
        self._temp = self._diffuse(self._temp, self.THERMAL_DIFF, dt)

        # 7. Enforce boundary conditions
        self._enforce_boundaries()

    @property
    def expose_vel_temp(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (velocity, temperature) as CPU tensors for visualisation."""
        return self._vel.cpu(), self._temp.cpu()

    # -- utilities -----------------------------------------------------------

    def stats(self) -> dict[str, float]:
        """Return summary statistics (for logging / debug overlay)."""
        v_mag = torch.sqrt(
            self._vel[0] ** 2 + self._vel[1] ** 2 + self._vel[2] ** 2,
        )
        return {
            "avg_temp": float(self._temp.mean()),
            "max_vel": float(v_mag.max()),
            "avg_vel": float(v_mag.mean()),
        }
