# ThrillRide 3D

**FCG Mini Project — Interactive Roller Coaster Simulation Using OpenGL**

A C++/OpenGL desktop simulation demonstrating 3D transformations, Catmull–Rom curve generation, animation, perspective projection, fixed-function lighting, and keyboard/mouse interaction. The project is configured for Visual Studio on Windows and uses FreeGLUT and GLU.

## Features

- One asymmetric, closed circuit with a tall lift, steep first drop, elevated hills, a sweeping 180-degree turn, camelback, S-turn section, and broad return; no loops or corkscrews.
- Dense centripetal Catmull–Rom sampling, parallel-transported 3D frames, curvature-based banking, and arc-length train movement with interpolated poses.
- Cross ties and supports placed by track distance, uniform cylindrical steel rails, metallic connectors, and diagonal bracing.
- Three articulated train cars with red bodywork, individual seats, headrests, lap bars, wheel assemblies, and couplers.
- Landscaped terrain, an oval walking path, trees, bushes, infield flower beds, small park buildings, an entrance sign, distant green hills, a sky gradient, and distance fog.
- Elapsed-time animation with smoothly adjusted speed and filtered ride-camera movement; Orbit, Chase, and Front Seat modes remain available.
- Orbit, chase, and front-seat cameras; orbit controls support mouse drag, arrow keys, and wheel zoom.
- Ride status, camera, speed, and track progress in the HUD; pause, reset, adjustable speed, and lighting controls.

## Controls

| Input | Action |
| --- | --- |
| `Space` | Pause / resume the ride |
| `+` / `-` | Increase / decrease ride speed |
| `C` | Cycle orbit, chase, and front-seat cameras |
| Left mouse drag | Rotate the orbit camera |
| Mouse wheel | Zoom the orbit camera |
| Arrow keys | Adjust orbit camera angle |
| `L` | Toggle scene lighting |
| `H` | Toggle the help/status display |
| `R` | Reset ride and speed |
| `Esc` | Quit |

## Build and run (Visual Studio)

1. Install the **Desktop development with C++** workload and the Windows 10/11 SDK.
2. Install the FreeGLUT development package used by your existing Visual Studio project. The project already contains FreeGLUT include and library paths for x64 Debug and Release; if the package is in a different location, update those paths in **Project Properties → C/C++ → General** and **Linker → General**. Keep the matching architecture and ensure `freeglut.dll` is beside the executable or available on `PATH`.
3. Open `Thrilerider 3D.slnx`, choose **x64 / Debug** or **x64 / Release**, then build and run. The project links the existing `freeglut.lib`, `opengl32.lib`, and `glu32.lib` dependencies.

The project uses the legacy OpenGL compatibility API provided by Windows OpenGL, so it does not require a separate OpenGL SDK. FreeGLUT is required for windowing and input.
