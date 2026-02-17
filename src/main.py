import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

import dearpygui.dearpygui as dpg

from utils.logger import setup_logger
from utils.config import load_config
from models.models import Settings
from ui.room_editor import RoomEditor
from ui.conditions import ConditionsPanel
from simulation.simulation import Solver
from simulation.rendering import AirflowRenderer

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

    # -- DPG init ------------------------------------------------------------
    dpg.create_context()

    # -- Create ALL objects FIRST (critical order) ---------------------------
    settings = Settings(
        temp_indoor=config["temperature"]["indoor"],
        temp_outdoor=config["temperature"]["outdoor"],
        units=units,
    )

    editor = RoomEditor(units=units)
    conditions = ConditionsPanel(settings=settings)
    renderer = AirflowRenderer(settings=settings, config=config)



    dpg.create_viewport(
        title="AC Sim",
        width=1400,
        height=1100,
        resizable=True,
    )

    # -- Create tabbed layout ------------------------------------------------
    with dpg.window(tag="primary", no_scrollbar=True, no_title_bar=False):
        with dpg.tab_bar(tag="main_tabs"):

            # ==================== TAB 1: Room Design & Conditions ====================
            with dpg.tab(label="Room Design & Conditions", tag="tab_design"):
                with dpg.group(horizontal=True, tag="design_row"):
                    pass  # will be filled after objects are created

            # ==================== TAB 2: Simulation (Default) ====================
            with dpg.tab(label="Simulation", tag="tab_sim"):
                pass  # will be filled after objects are created

    # -- Populate Tab 1 ------------------------------------------------------
    editor.build(parent="design_row")
    conditions.build(parent="design_row")

    settings.room = editor.room

    # -- Simulation state ----------------------------------------------------
    solver: Solver | None = None
    sim_running = False
    target_fps = config["fps"]
    frame_dt = 1.0 / target_fps
    log_interval = target_fps * 5
    frame_count = 0

    # -- Callbacks -----------------------------------------------------------
    def _init_solver() -> Solver:
        nonlocal solver
        solver = Solver(settings=settings, config=config)
        renderer.invalidate_room()
        logger.info("Solver created: res=%dx%dx%d", *solver.shape)
        return solver

    def _on_sim_toggle(sender, app_data):
        nonlocal sim_running, solver, frame_count
        sim_running = app_data
        if sim_running:
            if solver is None:
                _init_solver()
            else:
                solver.rebuild_boundaries()
            frame_count = 0
            dpg.set_value("sim_status", "Sim: Running")
            logger.info("Simulation started")
        else:
            dpg.set_value("sim_status", "Sim: Paused")
            logger.info("Simulation paused")

    def _on_sim_reset(sender=None, app_data=None):
        nonlocal solver, sim_running, frame_count
        sim_running = False
        dpg.set_value("sim_toggle", False)
        solver = _init_solver()
        frame_count = 0
        renderer.reset_camera()
        dpg.set_value("sim_status", "Sim: Reset")
        logger.info("Simulation reset")

    # -- Populate Tab 2 (Simulation) -----------------------------------------
    with dpg.group(horizontal=True, parent="tab_sim"):
        dpg.add_checkbox(
            tag="sim_toggle",
            label="Run Simulation",
            default_value=False,
            callback=_on_sim_toggle,
        )
        dpg.add_button(label="Reset Simulation", callback=_on_sim_reset)
        dpg.add_spacer(width=30)
        dpg.add_text("Sim: Idle", tag="sim_status")
        dpg.add_spacer(width=20)
        dpg.add_text("FPS: --", tag="sim_fps")
        dpg.add_spacer(width=20)
        dpg.add_text("Avg Temp: -- °C", tag="sim_avg_temp")
        dpg.add_spacer(width=20)
        dpg.add_text("Max Vel: -- m/s", tag="sim_max_vel")

    dpg.add_separator(parent="tab_sim")

    # Large rendering area that fills the rest of the tab
    renderer.build(parent="tab_sim")

    # -- Cross-panel synchronization -----------------------------------------
    def sync_units(new_units: str):
        settings.units = new_units
        conditions.on_units_changed(new_units)

    def sync_room():
        settings.room = editor.room
        conditions.refresh()
        renderer.invalidate_room()

    editor.on_units_changed = sync_units
    editor.on_room_changed = sync_room

    # -- Final setup ---------------------------------------------------------
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("primary", True)

    # Default to Simulation tab
    dpg.set_value("main_tabs", "tab_sim")

    logger.info("AC Sim ready - Tabbed layout active (default: Simulation)")

    # -- Main render loop ----------------------------------------------------
    last_time = time.perf_counter()

    while dpg.is_dearpygui_running():
        now = time.perf_counter()
        dt = now - last_time
        last_time = now

        if sim_running and solver is not None:
            solver.step(frame_dt)
            frame_count += 1

            renderer.render(solver)

            fps = 1.0 / max(dt, 1e-6)
            dpg.set_value("sim_fps", f"FPS: {fps:.0f}")

            if frame_count % log_interval == 0:
                st = solver.stats()
                dpg.set_value("sim_avg_temp", f"Avg Temp: {st['avg_temp']:.1f} °C")
                dpg.set_value("sim_max_vel", f"Max Vel: {st['max_vel']:.2f} m/s")
                logger.info(
                    "Sim: step=%d AvgTemp=%.1f°C MaxVel=%.2fm/s FPS=%.0f",
                    frame_count, st["avg_temp"], st["max_vel"], fps,
                )

        renderer.tick()
        dpg.render_dearpygui_frame()

    dpg.destroy_context()
    logger.info("AC Sim shutdown")


if __name__ == "__main__":
    main()