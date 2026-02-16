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
    settings = Settings(
        temp_indoor=config["temperature"]["indoor"],
        temp_outdoor=config["temperature"]["outdoor"],
        units=units,
    )

    # -- DPG init ------------------------------------------------------------
    dpg.create_context()
    dpg.create_viewport(
        title="AC Sim",
        width=1400,
        height=1100,
        resizable=True,
    )

    # -- Tabbed layout -------------------------------------------------------
    with dpg.window(tag="primary", no_scrollbar=True):
        with dpg.tab_bar(tag="main_tabs"):
            with dpg.tab(label="Room Design & Conditions", tag="tab_design"):
                design_row = dpg.add_group(horizontal=True)
            with dpg.tab(label="Simulation", tag="tab_sim"):
                pass  # content added below after callbacks are defined

    # -- Tab 1: Room Design & Conditions -------------------------------------
    editor = RoomEditor(units=units)
    editor.build(parent=design_row)

    settings.room = editor.room

    conditions = ConditionsPanel(settings=settings)
    conditions.build(parent=design_row)

    # -- simulation state ----------------------------------------------------
    solver: Solver | None = None
    sim_running = False
    target_fps = config["fps"]
    frame_dt = 1.0 / target_fps
    log_interval = target_fps * 5  # log stats every 5 seconds
    frame_count = 0

    # -- renderer ------------------------------------------------------------
    renderer = AirflowRenderer(settings=settings, config=config)

    def _init_solver() -> Solver:
        nonlocal solver
        solver = Solver(settings=settings, config=config)
        renderer.invalidate_room()
        logger.info(
            "Solver created: res=%dx%dx%d, FPS target=%d",
            *solver.shape, target_fps,
        )
        return solver

    def _on_sim_toggle(sender, app_data) -> None:
        nonlocal sim_running, solver, frame_count
        sim_running = app_data
        if sim_running:
            if solver is None:
                _init_solver()
            else:
                solver.rebuild_boundaries()
            frame_count = 0
            dpg.set_value("sim_status", "Sim: Running")
            logger.info("Sim started")
        else:
            dpg.set_value("sim_status", "Sim: Paused")
            logger.info("Sim paused")

    def _on_sim_reset(sender=None, app_data=None) -> None:
        nonlocal solver, sim_running, frame_count
        sim_running = False
        dpg.set_value("sim_toggle", False)
        solver = _init_solver()
        frame_count = 0
        renderer.reset_camera()
        dpg.set_value("sim_status", "Sim: Reset")
        logger.info("Sim reset")

    # -- Tab 2: Simulation controls + render viewport ------------------------
    with dpg.group(horizontal=True, parent="tab_sim"):
        dpg.add_checkbox(
            tag="sim_toggle",
            label="Run Simulation",
            default_value=False,
            callback=_on_sim_toggle,
        )
        dpg.add_button(label="Reset Simulation", callback=_on_sim_reset)
        dpg.add_spacer(width=20)
        dpg.add_text("Sim: Idle", tag="sim_status")
        dpg.add_spacer(width=10)
        dpg.add_text("FPS: --", tag="sim_fps")
        dpg.add_spacer(width=10)
        dpg.add_text("Avg Temp: --", tag="sim_avg_temp")
        dpg.add_spacer(width=10)
        dpg.add_text("Max Vel: --", tag="sim_max_vel")

    dpg.add_separator(parent="tab_sim")
    renderer.build(parent="tab_sim")

    # -- cross-panel sync ----------------------------------------------------
    def sync_units(new_units: str) -> None:
        settings.units = new_units
        conditions.on_units_changed(new_units)

    def sync_room() -> None:
        settings.room = editor.room
        conditions.refresh()
        renderer.invalidate_room()

    editor.on_units_changed = sync_units
    editor.on_room_changed = sync_room

    # -- run -----------------------------------------------------------------
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("primary", True)
    dpg.set_value("main_tabs", "tab_sim")

    logger.info("AC Sim ready")
    logger.info("UI: Tabbed layout initialized, default tab=Simulation")

    last_time = time.perf_counter()

    while dpg.is_dearpygui_running():
        now = time.perf_counter()
        dt = now - last_time
        last_time = now

        if sim_running and solver is not None:
            solver.step(frame_dt)
            frame_count += 1

            # Render airflow visualisation
            renderer.render(solver)

            # Update HUD
            fps = 1.0 / max(dt, 1e-6)
            dpg.set_value("sim_fps", f"FPS: {fps:.0f}")

            if frame_count % log_interval == 0:
                st = solver.stats()
                dpg.set_value("sim_avg_temp", f"Avg Temp: {st['avg_temp']:.1f}C")
                dpg.set_value("sim_max_vel", f"Max Vel: {st['max_vel']:.2f}m/s")
                logger.info(
                    "Sim: step=%d AvgTemp=%.1fC MaxVel=%.2fm/s FPS=%.0f",
                    frame_count, st["avg_temp"], st["max_vel"], fps,
                )

        renderer.tick()
        dpg.render_dearpygui_frame()

    dpg.destroy_context()
    logger.info("AC Sim shutdown")


if __name__ == "__main__":
    main()
