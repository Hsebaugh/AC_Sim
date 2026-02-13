import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import dearpygui.dearpygui as dpg

from utils.logger import setup_logger
from utils.config import load_config
from ui.room_editor import RoomEditor

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

    # -- DPG init ------------------------------------------------------------
    dpg.create_context()
    dpg.create_viewport(
        title="AC Sim",
        width=1020,
        height=600,
        resizable=True,
    )

    with dpg.window(tag="primary"):
        editor = RoomEditor()
        editor.build(parent="primary")

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("primary", True)

    logger.info("AC Sim ready")

    # -- render loop (non-blocking) ------------------------------------------
    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()

    dpg.destroy_context()
    logger.info("AC Sim shutdown")


if __name__ == "__main__":
    main()
