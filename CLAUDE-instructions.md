Project: Airflow Simulation in Python
Goal: Build modular Python app for room airflow sim. User designs 2D/3D rooms, sets conditions, runs sim at >=60 FPS on M4 MacBook Pro using GPU. Visualize with temp-colored velocity arrows. Extendable for multi-rooms, objects, humidity.
Tech Stack:

UI: Dear PyGui (resizable, GPU).
Rendering: ModernGL/PyOpenGL.
Sim: NumPy/SciPy for CFD approx; PyTorch/CuPy for GPU.
Logging: Python logging module; concise for AI feedback.
Avoid heavy deps; M4 compatible.

Principles:

Modular: Separate UI, sim, models, utils.
Scalable: Abstract classes for extensions.
Debug: Log at DEBUG/INFO/ERROR; catch root causes.
Optimize: Vectorize ops; batch GPU; profile FPS.

Tasks (Implement Sequentially)
1. Setup

Env: Python 3.x venv; install dearpygui, numpy, scipy, moderngl, torch.
Structure: /src/main.py, /ui, /simulation, /models, /utils (logger, JSON), /tests, /logs.
Boilerplate: main.py init logger, config (JSON for defaults: FPS=60, grid=32x32x16).
Log: File+console; format "LEVEL: Module: Message".

2. Models

Room: Class with width, length, height (default 3m); lists for walls (doors/windows: pos, size, open%).
Elements: Door/Window/Vent classes (pos, size, state).
Settings: Class for temps (in/out), AC (on/off/temp/speed), fan.
Serialize: To/from JSON for save/load.

3. UI - Room Creation

2D Editor: Panel for dims; interactive grid to add/edit doors/windows.
3D Preview: Extrude to mesh; viewport with edit toggle.
File: Save/load dialogs.

4. UI - Conditions

Panel: Inputs for temps, AC, window states (slider 0-100%).
Link to Room/Settings.
Presets: Save/load JSON.

5. Simulation

Grid: 3D voxels; init with room bounds.
Physics: Simplified NS (advection/diffusion/pressure); thermo heat transfer.
Boundaries: Inflow/outflow at openings; buoyancy from temps.
GPU: Torch tensors; step per frame.
Fallback: CPU.

6. Rendering/Vis

Embed GL in DPG window (resizable).
Draw: Room mesh; sample grid for arrows (len=vel, color=blue-yellow-red).
Controls: Pause, camera.
FPS: Monitor/log; optimize LOD/batching.

7. Logging/Debug

Concise: e.g., "Sim: AvgTemp=22C, FPS=65; Arrows: Inflow blue vel=2m/s".
Errors: Try/except; log trace/root.
AI Feedback: Describe UI/sim states visually in logs.

8. Test/Optimize

Units: Pytest for models/sim.
Profile: cProfile; vectorize NumPy.
Integrate: E2E tests.

9. Future

Extensions: MultiRoom graph; Obstacles; Fans (velocity fields); Humidity (add var).

Implementation Guidelines for Claude:

One file/module per prompt if possible.
Output code only; no explanations unless error.
Use <thinking> for planning; keep brief.
Clear context after task: /clear.