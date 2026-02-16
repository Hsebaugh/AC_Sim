## Project: Airflow Simulation in Python

**Goal**: Build modular Python app for room airflow sim. User designs 2D net/3D rooms, sets conditions, runs sim at >=60 FPS on M4 MacBook Pro using GPU. Visualize with temp-colored velocity arrows. Extendable for multi-rooms, objects, humidity.

**Tech Stack**:
- UI: Dear PyGui (resizable, GPU).
- Rendering: ModernGL/PyOpenGL.
- Sim: NumPy/SciPy for CFD approx; PyTorch/CuPy for GPU.
- Logging: Python logging module; concise for AI feedback.
- Avoid heavy deps; M4 compatible.

**Principles**:
- Modular: Separate UI, sim, models, utils.
- Scalable: Abstract classes for extensions.
- Debug: Log at DEBUG/INFO/ERROR; catch root causes.
- Optimize: Vectorize ops; batch GPU; profile FPS.
- UI Do's/Don'ts: Use drawlist for dynamic net/hover; matrix transforms for 3D orientation; event-driven clicks; no blocking loops.

**Camera & Input Do's**: 
Use **item-specific handlers** attached directly to the drawlist (not global handler_registry). Place the visualization in its **own child_window** with `no_scrollbar=True` and `border=True`. Support both trackpad and wired mouse: LMB=pan, RMB=orbit, Scroll=zoom, "R"=reset view.


## Tasks (Implement Sequentially; Fix Current Issues First)

### 1. Setup (Complete)
- Env, structure, boilerplate, logger.

### 2. Models (Complete)
- Room/Elements/Settings with JSON; add units enum (Standard/ft, Metric/m; default Standard) to Settings/Room for dims conversions.

### 3. UI - Room Creation (Partially Complete; Fix Issues)
- 3.1: 2D Net Editor: Canvas for cross net (floor center, walls around, optional ceiling); clickable edges for dim input (popup float, Enter update; consistent shared edges); face clicks for element placement (select type: door/window/vent; input pos/size/open%; hover 50% opacity preview rect).
- 3.2: 3D Projection: Map net to isometric mesh; fix element orientation (use transforms/rotations per face—e.g., north vertical no 90deg flip); viewport with toggle 2D/3D, show ceiling checkbox.
- 3.3: Save/Load: File dialogs; include units in JSON.

### 4. UI - Conditions
- Panel: Inputs for temps (in/out), AC (on/off/temp/speed), window states (slider 0-100%, top/bottom).
- Link to Room/Settings; convert units.
- Presets: Save/load JSON.

### 5. Simulation
- Grid: 3D voxels; init with room bounds.
- Physics: Simplified NS (advection/diffusion/pressure); thermo heat transfer.
- Boundaries: Inflow/outflow at openings; buoyancy from temps.
- GPU: Torch tensors; step per frame.
- Fallback: CPU.

### 6. Rendering/Vis (Partially Complete — Critical Fix Needed)
- 6.1: Isometric 3D viewport (current DPG drawlist + projection).
- 6.2: Dense airflow arrows: Sample heavily from 3D grid (target 2000–5000 arrows), project to screen with proper 3D rotation.
- 6.3: Temperature zones: Color room walls/floor/ceiling with temp gradient (blue-cold → yellow-ambient → red-hot) like reference image.
- 6.4: Camera: Yaw/pitch/zoom, smooth controls.
- 6.5: Performance: Adaptive skip, LOD, batch drawing; log arrow count + FPS.

### 7. Logging/Debug
- Concise: e.g., "Sim: AvgTemp=22C, FPS=65; Arrows: Inflow blue vel=2m/s".
- Errors: Try/except; log trace/root.
- AI Feedback: Describe UI/sim states visually in logs (e.g., "UI: Placed door on north, oriented vertical").

### 8. Test/Optimize
- Units: Pytest for models/sim.
- Profile: cProfile; vectorize NumPy.
- Integrate: E2E tests.

### 9. Future
- Extensions: MultiRoom graph; Obstacles; Fans (velocity fields); Humidity (add var).

**Implementation Guidelines for Claude**:
- One file/module per prompt if possible.
- Output code only; no explanations unless error.
- Use <thinking> for planning; keep brief.
- Clear context after task: /clear.