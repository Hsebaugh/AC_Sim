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

    # === GLOBAL INPUT HANDLER REGISTRY (Item_handler_debug pattern) ===
    # Explicit tags = easy, debuggable, scalable binding
    if not dpg.does_item_exist("global_render_handlers"):
        with dpg.handler_registry(tag="global_render_handlers"):
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                threshold=0.0,
                callback=None,
                tag="mouse_left_drag"          # ← key fix
            )
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Right,
                threshold=0.0,
                callback=None,
                tag="mouse_right_drag"         # ← key fix
            )
            dpg.add_mouse_wheel_handler(
                callback=None,
                tag="mouse_wheel"              # ← key fix
            )
            dpg.add_key_press_handler(
                key=dpg.mvKey_R,
                callback=None,
                tag="key_r"                    # ← key fix
            )
        logger.info("UI: Global render handlers registry created with tags")

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

    # -- Instantiate UI and Simulation components ----------------------------
    settings = Settings(
        temp_indoor=config["temperature"]["indoor"],
        temp_outdoor=config["temperature"]["outdoor"],
        units=units,
    )

    editor = RoomEditor(units=units)
    conditions = ConditionsPanel(settings=settings)
    renderer = AirflowRenderer(settings=settings, config=config)

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

    # Large rendering area
    renderer.build(parent="tab_sim")

    # === BIND INPUT HANDLERS (after canvas exists) ===
    renderer.bind_input_handlers()          # ← modular call

    # -- Cross-panel synchronization -----------------------------------------
    def sync_units(new_units: str):
        settings.units = new_units
        conditions.on_units_changed(new_units)

    def sync_room():
        settings.room = editor.room
        conditions.refresh()
        renderer.invalidate_room()
    logger.info('Start: Sync Units')
    editor.on_units_changed = sync_units
    logger.info('Start Sync_room')
    editor.on_room_changed = sync_room

    logger.info('Start Final Setup')
    # -- Final setup ---------------------------------------------------------
    logger.info('Start setup_derapygui()')
    dpg.setup_dearpygui()
    logger.info('Start show_viewport()')
    dpg.show_viewport()
    logger.info('Start set_primary_window')
    dpg.set_primary_window("primary", True)

    # Default to Simulation tab
    dpg.set_value("main_tabs", "tab_sim")
    # This resolves the canvas size so drawing appears in the full view area
    logger.info('Start Force Initial Layout')
    renderer.force_initial_layout()

    logger.info("AC Sim ready - Tabbed layout active (default: Simulation)")

    # -- Main render loop ----------------------------------------------------
    last_time = time.perf_counter()

    while dpg.is_dearpygui_running():
        now = time.perf_counter()
        dt = now - last_time
        last_time = now

        if frame_count < 30:                  # safe here — UI is already visible
            renderer._update_size()
            if renderer._size_valid:
                logger.info("Renderer: Canvas size resolved to %dx%d → full simulation view active",
                           renderer._w, renderer._h)

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