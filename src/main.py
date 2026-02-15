import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import dearpygui.dearpygui as dpg

from utils.logger import setup_logger
from utils.config import load_config
from models.models import Settings
from ui.room_editor import RoomEditor
from ui.conditions import ConditionsPanel

logger = setup_logger("Main")


def main():
    logger.info("AC Sim starting")

    config = load_config()
    logger.info(
        "Config loaded: FPS=%d, Grid=%dx%dx%d",
        config["fps"],
        config["grid"]["x"],
        config["grid"]["y"],
        config["grid"]["z"],
    )

    units = config.get("units", "standard")
    settings = Settings(
        temp_indoor=config["temperature"]["indoor"],
        temp_outdoor=config["temperature"]["outdoor"],
        units=units,
    )

    # -- DPG init ------------------------------------------------------------
    dpg.create_context()
    dpg.create_viewport(
        title="AC Sim",
        width=1300,
        height=600,
        resizable=True,
    )

    with dpg.window(tag="primary"):
        main_row = dpg.add_group(horizontal=True)

    editor = RoomEditor(units=units)
    editor.build(parent=main_row)

    settings.room = editor.room

    conditions = ConditionsPanel(settings=settings)
    conditions.build(parent=main_row)

    # -- cross-panel sync ----------------------------------------------------
    def sync_units(new_units: str) -> None:
        settings.units = new_units
        conditions.on_units_changed(new_units)

    def sync_room() -> None:
        settings.room = editor.room
        conditions.refresh()

    editor.on_units_changed = sync_units
    editor.on_room_changed = sync_room

    # -- run -----------------------------------------------------------------
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("primary", True)

    logger.info("AC Sim ready")

    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()

    dpg.destroy_context()
    logger.info("AC Sim shutdown")


if __name__ == "__main__":
    main()
