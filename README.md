# D3X

Data-oriented orbital mechanics simulation framework. C++ core for performance, Python scripting for mission design.

## Dependencies

<!-- DEPENDENCIES -->
* attrs
* cfgv
* contourpy
* cycler
* d3x
* distlib
* filelock
* fonttools
* gherkin-official
* glcontext
* glfw
* identify
* iniconfig
* kiwisolver
* mako
* markdown-it-py
* markupsafe
* matplotlib
* mdurl
* moderngl
* nodeenv
* numpy
* packaging
* parse
* parse-type
* pillow
* platformdirs
* pluggy
* pre-commit
* pybind11-stubgen
* pyglm
* pygments
* pyparsing
* pytest
* pytest-bdd
* pytest-rich
* pytest-sugar
* python-dateutil
* pyyaml
* rich
* ruff
* six
* termcolor
* typing-extensions
* virtualenv
<!-- /DEPENDENCIES -->

## Installation

<!-- INSTALL -->
```bash
# Install with uv (recommended)
uv sync --dev

# Legacy install (pip)
pip install -e ".[dev]"

# Run tests
uv run pytest

# Run visualization example
uv run python examples/earth_moon.py
```
<!-- /INSTALL -->

## Quick Start

```python
import d3x

# Create simulation
world = d3x.World()
world.add_body(pos=(0, 0, 0), vel=(0, 0, 0), mass=d3x.constants.M_SOL)
world.add_body(pos=(d3x.constants.AU, 0, 0), vel=(0, 29780, 0), mass=d3x.constants.M_EARTH)

# Simulate 24 hours (1-hour steps)
for _ in range(24):
    d3x.step_rk4(world, 3600)

print(f"Earth position: {world.px[1]:.3e} m")
print(f"Energy conserved: {world.total_energy():.6e} J")
```

<!-- API -->
## API Reference

### Quick Start
```python
from d3x import (
    StepResult,
    Vec3,
    Vec3Like,
    World,
    compute_gravity,
    constants,
    step_dopri54,
    step_leapfrog,
    step_rk4,
)
```

## Class `StepResult`
| Member | Type | Description |
|:-------|:-----|:------------|
| `dt_next` | Property | - |
| `dt_used` | Property | - |
| `error_estimate` | Property | - |

## Class `Vec3`
| Member | Type | Description |
|:-------|:-----|:------------|
| `magnitude()` | Method | - |
| `x` | Property | - |
| `y` | Property | - |
| `z` | Property | - |

## Class `World`
| Member | Type | Description |
|:-------|:-----|:------------|
| `add_body(pos, vel, mass)` | Method | Add a body with position [m], velocity [m/s], and mass [kg]. Returns body index. |
| `angular_momentum()` | Method | Total angular momentum vector [kg·m²/s] |
| `clear()` | Method | Remove all bodies and reset time |
| `kinetic_energy()` | Method | Total kinetic energy [J] |
| `potential_energy()` | Method | Total gravitational potential energy [J] |
| `reserve(n)` | Method | Pre-allocate memory for n bodies |
| `total_energy()` | Method | Total mechanical energy [J] |
| `count` | Property | Number of bodies in the simulation |
| `mass` | Property | Masses [kg] (numpy view) |
| `px` | Property | Position x-components [m] (numpy view) |
| `py_` | Property | Position y-components [m] (numpy view, named py_ to avoid collision) |
| `pz` | Property | Position z-components [m] (numpy view) |
| `time` | Property | Current simulation time [s] |
| `vx` | Property | Velocity x-components [m/s] (numpy view) |
| `vy` | Property | Velocity y-components [m/s] (numpy view) |
| `vz` | Property | Velocity z-components [m/s] (numpy view) |

### `fn compute_gravity(world)`
Compute gravitational accelerations for all bodies

### `fn compute_gravity(world, softening)`
Compute gravitational accelerations with softening parameter

### `fn step_dopri54(world, dt, tol)`
Advance simulation using adaptive Dormand-Prince 5(4) method

### `fn step_leapfrog(world, dt)`
Advance simulation using symplectic leapfrog (requires pre-computed accelerations)

### `fn step_rk4(world, dt)`
Advance simulation by dt seconds using 4th-order Runge-Kutta

<!-- /API -->

## Integrators

<!-- INTEGRATORS -->
| Integrator | Type | Best For |
|------------|------|----------|
| `step_rk4(world, dt)` | Fixed-step | General purpose, smooth trajectories |
| `step_dopri54(world, dt, tol)` | Adaptive | Variable dynamics, close encounters |
| `step_leapfrog(world, dt)` | Symplectic | Long-term stability, energy conservation |
<!-- /INTEGRATORS -->

<!-- CONSTANTS -->
## Constants

| Name | Value | Unit |
|------|-------|------|
| `AU` | 149597870700.0 | m |
| `DAY` | 86400.0 | s |
| `G` | 6.6743e-11 | m³/(kg·s²) |
| `MU_EARTH` | 398600542309999.94 | m³/s² |
| `MU_SUN` | 1.3274648755999998e+20 | m³/s² |
| `M_EARTH` | 5.97217e+24 | kg |
| `M_MARS` | 6.4171e+23 | kg |
| `M_MOON` | 7.342e+22 | kg |
| `M_SUN` | 1.98892e+30 | kg |
<!-- /CONSTANTS -->

## Test Coverage

Behavior-driven tests using Gherkin + pytest-bdd:

<!-- FEATURES -->
| Feature | Scenarios | Description |
|---------|-----------|-------------|
| World Container | 6 | Body management, SoA arrays, zero-copy numpy |
| Gravity | 5 | Inverse-square law, Newton's 3rd, superposition |
| Integrators | 7 | RK4, DOPRI54 adaptive, leapfrog symplectic |
| Conservation | 7 | Energy, angular momentum, integrator comparison |
<!-- /FEATURES -->

```bash
make test
# or
uv run pytest
```

## Architecture

- **SoA Layout**: Positions/velocities as separate `px[], py[], pz[]` arrays for cache efficiency
- **Zero-copy**: numpy arrays view C++ memory directly via pybind11
- **Precision**: `double` throughout, single typedef to change

## Syncing This README

Sections between `<!-- BINDING -->` markers are auto-generated:

```bash
python scripts/sync_readme.py         # Update README
python scripts/sync_readme.py --check # Verify in sync (CI)
```
