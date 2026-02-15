from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from utils.logger import setup_logger

logger = setup_logger("Models")

# ---------------------------------------------------------------------------
# Wall identifiers
# ---------------------------------------------------------------------------
WALL_NAMES = ("north", "south", "east", "west", "floor", "ceiling")

# ---------------------------------------------------------------------------
# Unit systems (display-only; internal values always in metres)
# ---------------------------------------------------------------------------
UNIT_SYSTEMS: dict[str, dict[str, Any]] = {
    "metric":   {"label": "m",  "factor": 1.0,     "temp": "\u00b0C"},
    "standard": {"label": "ft", "factor": 3.28084, "temp": "\u00b0F"},
}


def c_to_f(c: float) -> float:
    """Celsius to Fahrenheit."""
    return c * 9.0 / 5.0 + 32.0


def f_to_c(f: float) -> float:
    """Fahrenheit to Celsius."""
    return (f - 32.0) * 5.0 / 9.0

# ---------------------------------------------------------------------------
# Abstract element base
# ---------------------------------------------------------------------------


class RoomElement(ABC):
    """Base for anything placed on a wall (door, window, vent, ...)."""

    def __init__(
        self,
        pos: tuple[float, float],
        size: tuple[float, float],
        open_frac: float = 0.0,
    ) -> None:
        self.pos = pos
        self.size = size
        self.open_frac = max(0.0, min(1.0, open_frac))
        logger.debug(
            "%s created: pos=%s size=%s open=%.0f%%",
            self.element_type,
            self.pos,
            self.size,
            self.open_frac * 100,
        )

    @property
    @abstractmethod
    def element_type(self) -> str:
        ...

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.element_type,
            "pos": list(self.pos),
            "size": list(self.size),
            "open_frac": self.open_frac,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> RoomElement:
        cls = _ELEMENT_REGISTRY.get(d["type"])
        if cls is None:
            logger.error("Unknown element type: %s", d["type"])
            raise ValueError(f"Unknown element type: {d['type']}")
        kwargs: dict[str, Any] = {
            "pos": tuple(d["pos"]),
            "size": tuple(d["size"]),
            "open_frac": d.get("open_frac", 0.0),
        }
        if d["type"] == "window":
            kwargs["open_from"] = d.get("open_from", "bottom")
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Concrete elements
# ---------------------------------------------------------------------------


class Door(RoomElement):
    @property
    def element_type(self) -> str:
        return "door"


class Window(RoomElement):
    def __init__(
        self,
        pos: tuple[float, float],
        size: tuple[float, float],
        open_frac: float = 0.0,
        open_from: str = "bottom",
    ) -> None:
        super().__init__(pos, size, open_frac)
        self.open_from = open_from

    @property
    def element_type(self) -> str:
        return "window"

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["open_from"] = self.open_from
        return d


class Vent(RoomElement):
    @property
    def element_type(self) -> str:
        return "vent"


_ELEMENT_REGISTRY: dict[str, type[RoomElement]] = {
    "door": Door,
    "window": Window,
    "vent": Vent,
}

# ---------------------------------------------------------------------------
# Room
# ---------------------------------------------------------------------------


@dataclass
class Room:
    width: float = 3.0
    length: float = 3.0
    height: float = 3.0
    walls: dict[str, list[RoomElement]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in WALL_NAMES:
            self.walls.setdefault(name, [])
        logger.debug(
            "Room init: %.1fx%.1fx%.1fm",
            self.width,
            self.length,
            self.height,
        )

    def add_element(self, wall: str, element: RoomElement) -> None:
        if wall not in WALL_NAMES:
            logger.error("Invalid wall name: %s", wall)
            raise ValueError(f"Invalid wall: {wall}")
        self.walls[wall].append(element)
        logger.debug("Added %s to %s wall", element.element_type, wall)

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "length": self.length,
            "height": self.height,
            "walls": {
                name: [e.to_dict() for e in elems]
                for name, elems in self.walls.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Room:
        try:
            room = cls(
                width=d["width"],
                length=d["length"],
                height=d["height"],
            )
            for wall_name, elems in d.get("walls", {}).items():
                for ed in elems:
                    room.add_element(wall_name, RoomElement.from_dict(ed))
            logger.info("Room loaded: %.1fx%.1fx%.1fm", room.width, room.length, room.height)
            return room
        except (KeyError, TypeError) as exc:
            logger.error("Room load failed: %s", exc)
            raise

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class Settings:
    temp_indoor: float = 25.0
    temp_outdoor: float = 35.0
    ac_on: bool = False
    ac_temp: float = 22.0
    ac_speed: float = 1.0
    fan_on: bool = False
    fan_speed: float = 1.0
    units: str = "standard"
    room: Room | None = None

    def __post_init__(self) -> None:
        if self.units not in UNIT_SYSTEMS:
            self.units = "standard"
        logger.debug(
            "Settings init: in=%.1fC out=%.1fC AC=%s units=%s",
            self.temp_indoor,
            self.temp_outdoor,
            "ON" if self.ac_on else "OFF",
            self.units,
        )

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "temp_indoor": self.temp_indoor,
            "temp_outdoor": self.temp_outdoor,
            "ac_on": self.ac_on,
            "ac_temp": self.ac_temp,
            "ac_speed": self.ac_speed,
            "fan_on": self.fan_on,
            "fan_speed": self.fan_speed,
            "units": self.units,
        }
        if self.room is not None:
            d["room"] = self.room.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Settings:
        try:
            room = None
            if "room" in d:
                room = Room.from_dict(d["room"])
            return cls(
                temp_indoor=d.get("temp_indoor", 25.0),
                temp_outdoor=d.get("temp_outdoor", 35.0),
                ac_on=d.get("ac_on", False),
                ac_temp=d.get("ac_temp", 22.0),
                ac_speed=d.get("ac_speed", 1.0),
                fan_on=d.get("fan_on", False),
                fan_speed=d.get("fan_speed", 1.0),
                units=d.get("units", "standard"),
                room=room,
            )
        except (KeyError, TypeError) as exc:
            logger.error("Settings load failed: %s", exc)
            raise


# ---------------------------------------------------------------------------
# Top-level JSON helpers
# ---------------------------------------------------------------------------


def save_scene(settings: Settings, path: str) -> None:
    with open(path, "w") as f:
        json.dump(settings.to_dict(), f, indent=2)
    logger.info("Scene saved: %s", path)


def load_scene(path: str) -> Settings:
    try:
        with open(path, "r") as f:
            data = json.load(f)
        settings = Settings.from_dict(data)
        logger.info("Scene loaded: %s", path)
        return settings
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Scene load error: %s", exc)
        raise
