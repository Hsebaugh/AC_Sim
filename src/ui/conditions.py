"""Task 4 -- Conditions Panel -- temperature, AC, fan, element states.

Sidebar panel for environmental conditions. All temperatures stored
internally in Celsius; displayed in F or C based on current unit system.
Element open-fraction sliders update the shared Room model directly.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import dearpygui.dearpygui as dpg

from models.models import (
    Settings,
    UNIT_SYSTEMS,
    WALL_NAMES,
    c_to_f,
    f_to_c,
)
from utils.logger import setup_logger

logger = setup_logger("UI")


class ConditionsPanel:
    """Sidebar panel for environmental conditions and element states."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

        # DPG tags
        self._elem_group: int | str = 0
        self._txt_status: int | str = 0

        logger.info("ConditionsPanel created (units=%s)", self._settings.units)

    # -- unit helpers --------------------------------------------------------

    @property
    def _units(self) -> str:
        return self._settings.units

    @property
    def _temp_label(self) -> str:
        return UNIT_SYSTEMS[self._units]["temp"]

    def _temp_to_display(self, celsius: float) -> float:
        if self._units == "standard":
            return c_to_f(celsius)
        return celsius

    def _temp_from_display(self, display: float) -> float:
        if self._units == "standard":
            return f_to_c(display)
        return display

    # -- build ---------------------------------------------------------------

    def build(self, parent: int | str) -> None:
        with dpg.child_window(width=260, autosize_y=True, parent=parent):
            dpg.add_text("Conditions")

            # -- temperatures --
            dpg.add_separator()
            dpg.add_text("Temperature")
            tl = self._temp_label
            dpg.add_input_float(
                tag="cond_temp_out",
                label=f"Outside {tl}",
                default_value=self._temp_to_display(self._settings.temp_outdoor),
                step=1.0,
                callback=self._on_temp_outdoor,
            )
            dpg.add_input_float(
                tag="cond_temp_in",
                label=f"Inside {tl}",
                default_value=self._temp_to_display(self._settings.temp_indoor),
                step=1.0,
                callback=self._on_temp_indoor,
            )

            # -- AC --
            dpg.add_separator()
            dpg.add_text("Air Conditioning")
            dpg.add_checkbox(
                tag="cond_ac_on",
                label="AC On",
                default_value=self._settings.ac_on,
                callback=self._on_ac_toggle,
            )
            dpg.add_input_float(
                tag="cond_ac_temp",
                label=f"Target {tl}",
                default_value=self._temp_to_display(self._settings.ac_temp),
                step=1.0,
                callback=self._on_ac_temp,
            )
            dpg.add_slider_int(
                tag="cond_ac_speed",
                label="Speed",
                default_value=int(self._settings.ac_speed),
                min_value=1,
                max_value=10,
                callback=self._on_ac_speed,
            )

            # -- Fan --
            dpg.add_separator()
            dpg.add_text("Fan")
            dpg.add_checkbox(
                tag="cond_fan_on",
                label="Fan On",
                default_value=self._settings.fan_on,
                callback=self._on_fan_toggle,
            )
            dpg.add_slider_int(
                tag="cond_fan_speed",
                label="Speed",
                default_value=int(self._settings.fan_speed),
                min_value=1,
                max_value=10,
                callback=self._on_fan_speed,
            )

            # -- element states --
            dpg.add_separator()
            dpg.add_text("Element States")
            dpg.add_button(
                label="Refresh Elements", callback=self._rebuild_elements,
            )
            self._elem_group = dpg.add_group()

            # -- presets --
            dpg.add_separator()
            dpg.add_button(label="Save Preset...", callback=self._dlg_save)
            dpg.add_button(label="Load Preset...", callback=self._dlg_load)

            dpg.add_separator()
            self._txt_status = dpg.add_text("Ready")

        self._rebuild_elements()
        logger.info("ConditionsPanel UI built")

    # -- temperature callbacks -----------------------------------------------

    def _on_temp_outdoor(self, sender: Any, app_data: Any) -> None:
        self._settings.temp_outdoor = self._temp_from_display(app_data)
        logger.info(
            "AC updated: outside temp=%.1f%s (%.1f\u00b0C)",
            app_data, self._temp_label, self._settings.temp_outdoor,
        )

    def _on_temp_indoor(self, sender: Any, app_data: Any) -> None:
        self._settings.temp_indoor = self._temp_from_display(app_data)
        logger.info(
            "AC updated: inside temp=%.1f%s (%.1f\u00b0C)",
            app_data, self._temp_label, self._settings.temp_indoor,
        )

    # -- AC callbacks --------------------------------------------------------

    def _on_ac_toggle(self, sender: Any, app_data: Any) -> None:
        self._settings.ac_on = app_data
        logger.info("AC updated: %s", "ON" if app_data else "OFF")

    def _on_ac_temp(self, sender: Any, app_data: Any) -> None:
        self._settings.ac_temp = self._temp_from_display(app_data)
        logger.info(
            "AC updated: target=%.1f%s", app_data, self._temp_label,
        )

    def _on_ac_speed(self, sender: Any, app_data: Any) -> None:
        self._settings.ac_speed = float(app_data)
        logger.info("AC updated: speed=%d", app_data)

    # -- fan callbacks -------------------------------------------------------

    def _on_fan_toggle(self, sender: Any, app_data: Any) -> None:
        self._settings.fan_on = app_data
        logger.info("Fan updated: %s", "ON" if app_data else "OFF")

    def _on_fan_speed(self, sender: Any, app_data: Any) -> None:
        self._settings.fan_speed = float(app_data)
        logger.info("Fan updated: speed=%d", app_data)

    # -- element states ------------------------------------------------------

    def _rebuild_elements(
        self, sender: Any = None, app_data: Any = None,
    ) -> None:
        dpg.delete_item(self._elem_group, children_only=True)

        room = self._settings.room
        if room is None:
            dpg.add_text("(no room)", parent=self._elem_group)
            return

        idx = 0
        for wall_name in WALL_NAMES:
            elements = room.walls.get(wall_name, [])
            for i, elem in enumerate(elements):
                tag = f"elem_state_{idx}"
                label = f"{wall_name}: {elem.element_type} #{i + 1}"
                dpg.add_slider_float(
                    tag=tag,
                    label=label,
                    default_value=elem.open_frac * 100,
                    min_value=0.0,
                    max_value=100.0,
                    format="%.0f%%",
                    callback=self._on_elem_state,
                    user_data=(wall_name, i),
                    parent=self._elem_group,
                )
                if elem.element_type == "window":
                    combo_tag = f"elem_open_from_{idx}"
                    dpg.add_combo(
                        tag=combo_tag,
                        items=["bottom", "top", "full"],
                        default_value=getattr(elem, "open_from", "bottom"),
                        label="Open from",
                        callback=self._on_window_mode,
                        user_data=(wall_name, i),
                        parent=self._elem_group,
                    )
                idx += 1

        if idx == 0:
            dpg.add_text("(no elements)", parent=self._elem_group)

        logger.debug("Element states rebuilt: %d elements", idx)

    def _on_elem_state(
        self, sender: Any, app_data: Any, user_data: Any,
    ) -> None:
        wall_name, elem_idx = user_data
        room = self._settings.room
        if room and wall_name in room.walls:
            elems = room.walls[wall_name]
            if elem_idx < len(elems):
                elems[elem_idx].open_frac = app_data / 100.0
                logger.info(
                    "Element %s %s #%d open: %.0f%%",
                    elems[elem_idx].element_type, wall_name,
                    elem_idx + 1, app_data,
                )

    def _on_window_mode(
        self, sender: Any, app_data: Any, user_data: Any,
    ) -> None:
        wall_name, elem_idx = user_data
        room = self._settings.room
        if room and wall_name in room.walls:
            elems = room.walls[wall_name]
            if elem_idx < len(elems) and hasattr(elems[elem_idx], "open_from"):
                elems[elem_idx].open_from = app_data
                logger.info(
                    "Window %s #%d open from: %s",
                    wall_name, elem_idx + 1, app_data,
                )

    # -- presets -------------------------------------------------------------

    def _dlg_save(self, sender: Any = None, app_data: Any = None) -> None:
        with dpg.file_dialog(
            label="Save Conditions", callback=self._on_save,
            width=500, height=400, default_filename="conditions",
        ):
            dpg.add_file_extension(".json", color=(0, 255, 0, 255))

    def _on_save(self, sender: Any, app_data: dict) -> None:
        try:
            path = app_data["file_path_name"]
            if not path.endswith(".json"):
                path += ".json"
            data = self._settings.to_dict()
            data.pop("room", None)
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info("Conditions saved: %s", path)
            dpg.set_value(self._txt_status, f"Saved: {path}")
        except Exception:
            logger.exception("Conditions save failed")
            dpg.set_value(self._txt_status, "Save failed - see log")

    def _dlg_load(self, sender: Any = None, app_data: Any = None) -> None:
        with dpg.file_dialog(
            label="Load Conditions", callback=self._on_load,
            width=500, height=400,
        ):
            dpg.add_file_extension(".json", color=(0, 255, 0, 255))

    def _on_load(self, sender: Any, app_data: dict) -> None:
        try:
            path = app_data["file_path_name"]
            with open(path) as f:
                data = json.load(f)
            self._settings.temp_indoor = data.get(
                "temp_indoor", self._settings.temp_indoor,
            )
            self._settings.temp_outdoor = data.get(
                "temp_outdoor", self._settings.temp_outdoor,
            )
            self._settings.ac_on = data.get("ac_on", self._settings.ac_on)
            self._settings.ac_temp = data.get(
                "ac_temp", self._settings.ac_temp,
            )
            self._settings.ac_speed = data.get(
                "ac_speed", self._settings.ac_speed,
            )
            self._settings.fan_on = data.get("fan_on", self._settings.fan_on)
            self._settings.fan_speed = data.get(
                "fan_speed", self._settings.fan_speed,
            )
            self._refresh_ui()
            tl = self._temp_label
            logger.info(
                "Conditions loaded, temp=%.0f%s",
                self._temp_to_display(self._settings.temp_indoor), tl,
            )
            dpg.set_value(self._txt_status, f"Loaded: {path}")
        except Exception:
            logger.exception("Conditions load failed")
            dpg.set_value(self._txt_status, "Load failed - see log")

    # -- refresh -------------------------------------------------------------

    def _refresh_ui(self) -> None:
        """Sync DPG widgets with current settings values."""
        tl = self._temp_label
        dpg.set_value(
            "cond_temp_out",
            self._temp_to_display(self._settings.temp_outdoor),
        )
        dpg.configure_item("cond_temp_out", label=f"Outside {tl}")
        dpg.set_value(
            "cond_temp_in",
            self._temp_to_display(self._settings.temp_indoor),
        )
        dpg.configure_item("cond_temp_in", label=f"Inside {tl}")
        dpg.set_value("cond_ac_on", self._settings.ac_on)
        dpg.set_value(
            "cond_ac_temp",
            self._temp_to_display(self._settings.ac_temp),
        )
        dpg.configure_item("cond_ac_temp", label=f"Target {tl}")
        dpg.set_value("cond_ac_speed", int(self._settings.ac_speed))
        dpg.set_value("cond_fan_on", self._settings.fan_on)
        dpg.set_value("cond_fan_speed", int(self._settings.fan_speed))
        self._rebuild_elements()

    def on_units_changed(self, new_units: str) -> None:
        """Called when units toggle changes in room editor."""
        self._settings.units = new_units
        self._refresh_ui()
        logger.info("Conditions units synced: %s", new_units)

    def refresh(self) -> None:
        """Public: refresh all UI from current settings/room state."""
        self._refresh_ui()
