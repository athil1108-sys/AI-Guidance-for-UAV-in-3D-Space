# Implementation Plan - Phase 1 2D UAV Pursuit-Guidance Simulator (PNG Baseline)

This document outlines the design and implementation plan for Phase 1 of reproducing and extending Chithapuram, Jeppu & Aswani Kumar (IC3I 2014) *"Artificial Intelligence Guidance for Unmanned Aerial Vehicles in Three Dimensional Space"*.

In this phase, we build a clean, modular, 2D pursuit-guidance simulation framework using NumPy, Matplotlib, and Pytest. It implements classical Proportional Navigation Guidance (PNG) with zero-order hold guidance commands and exact analytical point of closest approach (CPA) interpolation.

## User Review Required

> [!IMPORTANT]
> - **Zero-Order Hold (ZOH)**: As specified, guidance commands $A_n$ are computed once per timestep $dt$ and kept constant during RK4 integration substeps.
> - **CPA Interpolation**: Miss distance is calculated by interpolating within the final timestep using $t_{cpa} = \text{clamp}(-\frac{\mathbf{r} \cdot \mathbf{v}}{\|\mathbf{v}\|^2}, 0, dt)$ to prevent discrete timestep quantization errors.
> - **Package Structure**: Clean functional and object-oriented design using Python 3.10+ dataclasses and strict type hints.

## Open Questions

> [!NOTE]
> None at this time. All physics, guidance equations, and simulation requirements match the prompt specifications.

## Proposed Changes

We will create a Python package `uav_guidance` along with `tests/` and an entry script `run_baseline.py`.

```
.
├── uav_guidance/
│   ├── __init__.py
│   ├── kinematics.py    # VehicleState, integration (RK4/Euler), equations of motion
│   ├── seeker.py        # SeekerData, analytical & finite-difference LOS rate / closing velocity
│   ├── png.py           # GuidanceConfig, PNG guidance law & acceleration saturation
│   ├── simulation.py    # SimulationEngine, trajectory logger, CPA calculation & episode runner
│   ├── scenarios.py     # Scenario dataclass, default scenario, grid search generator & OA hook
│   └── plotting.py      # Plotter for 2D trajectories, LOS rate, An, and Range vs time
├── tests/
│   ├── test_kinematics.py
│   ├── test_seeker.py
│   ├── test_png.py
│   ├── test_simulation.py
│   └── test_scenarios.py
└── run_baseline.py      # Main entry point running baseline engagement & showing plots
```

---

### `uav_guidance` Package

#### [NEW] [uav_guidance/__init__.py](file:///d:/Ai%20Project/uav_guidance/__init__.py)
Exposes package public API (`VehicleState`, `SeekerData`, `PNGGuidance`, `SimulationRunner`, `Scenario`, `plot_engagement`).

#### [NEW] [uav_guidance/kinematics.py](file:///d:/Ai%20Project/uav_guidance/kinematics.py)
- `VehicleState`: Dataclass holding `x: float`, `y: float`, `v: float` (constant speed), `psi: float` (heading angle in radians).
- Derives velocity vector $(v_x, v_y) = (V \cos\psi, V \sin\psi)$.
- Equations of motion:
  $$\frac{dx}{dt} = V \cos\psi, \quad \frac{dy}{dt} = V \sin\psi, \quad \frac{d\psi}{dt} = \frac{A_n}{V}$$
- Step function `integrate_step(state, An, dt, method="rk4") -> VehicleState`:
  - `rk4`: 4th order Runge-Kutta integration of state $[x, y, \psi]$ with constant $A_n$.
  - `euler`: Forward Euler integration step for comparison.

#### [NEW] [uav_guidance/seeker.py](file:///d:/Ai%20Project/uav_guidance/seeker.py)
- `SeekerData`: Dataclass holding `range: float`, `los_angle: float`, `los_rate: float`, `closing_velocity: float`.
- `compute_seeker_data(uav_state, target_state, prev_uav_state=None, prev_target_state=None, dt=None, use_finite_difference=False) -> SeekerData`:
  - **Analytical Method**:
    $$\mathbf{r} = (X_t - X_m, Y_t - Y_m)$$
    $$\mathbf{v} = (V_{xt} - V_{xm}, V_{yt} - V_{ym})$$
    $$R = \|\mathbf{r}\| = \sqrt{r_x^2 + r_y^2}$$
    $$\lambda = \text{atan2}(r_y, r_x)$$
    $$\dot{\lambda} = \frac{r_x v_y - r_y v_x}{R^2}$$
    $$V_c = -\frac{\mathbf{r} \cdot \mathbf{v}}{R} = -\frac{r_x v_x + r_y v_y}{R}$$
  - **Finite-Difference Method** (when `use_finite_difference=True`):
    $$\dot{\lambda}_{fd} = \frac{\text{wrap\_angle}(\lambda - \lambda_{prev})}{dt}$$
    $$V_{c,fd} = -\frac{R - R_{prev}}{dt}$$

#### [NEW] [uav_guidance/png.py](file:///d:/Ai%20Project/uav_guidance/png.py)
- `GuidanceConfig`: Dataclass with `N: float = 4.0`, `an_max: float = 100.0`.
- `GuidanceOutput`: Dataclass with `commanded_an: float`, `applied_an: float`.
- `PNGGuidanceLaw`:
  - Computes commanded acceleration:
    $$A_{n,cmd} = N \cdot V_c \cdot \dot{\lambda}$$
  - Saturates to range $[-A_{n,max}, +A_{n,max}]$.

#### [NEW] [uav_guidance/simulation.py](file:///d:/Ai%20Project/uav_guidance/simulation.py)
- `TrajectoryLog`: Structured log containing NumPy arrays for `time`, `uav_x`, `uav_y`, `target_x`, `target_y`, `range`, `los_angle`, `los_rate`, `closing_velocity`, `commanded_an`, `applied_an`.
- `SimulationResult`: Dataclass containing `trajectory: TrajectoryLog`, `miss_distance: float`, `intercept_time: float`, `termination_reason: str`, `t_cpa: float`, `peak_commanded_an: float`.
- `run_simulation(scenario: Scenario, integration_method: str = "rk4", use_finite_difference: False) -> SimulationResult`:
  - Main loop using fixed $dt$.
  - Applies ZOH for guidance commands.
  - Applies target maneuver $TN$ when $t \ge maneuver\_time$.
  - Termination condition check:
    1. $V_c < 0$ (target receding)
    2. $R > max\_range$
    3. $t \ge t\_max$
  - True Point of Closest Approach (CPA) calculation at termination:
    $$t_{cpa} = \text{clamp}\left(-\frac{\mathbf{r} \cdot \mathbf{v}}{\|\mathbf{v}\|^2}, 0, dt\right)$$
    $$\text{miss\_distance} = \|\mathbf{r} + \mathbf{v} \cdot t_{cpa}\|$$

#### [NEW] [uav_guidance/scenarios.py](file:///d:/Ai%20Project/uav_guidance/scenarios.py)
- `Scenario`: Dataclass containing:
  - `uav_x`, `uav_y`, `uav_speed`, `uav_heading`
  - `target_x`, `target_y`, `target_speed`, `target_heading`
  - `maneuver_time`, `maneuver_accel` ($TN$)
  - `dt`, `t_max`, `max_range`
  - `N`, `an_max`
- `make_default_scenario() -> Scenario`:
  - Baseline engagement: UAV at (0,0), V=300 m/s; Target at (3000, 4000), V=200 m/s heading towards origin/angle, maneuver_time=5.0s, TN=30 m/s^2.
- `generate_grid_scenarios(v_target_list, launch_angle_list, maneuver_accel_list, maneuver_time_list)`: Full-factorial generator.
- `generate_orthogonal_array_scenarios()`: Hook for future phase.

#### [NEW] [uav_guidance/plotting.py](file:///d:/Ai%20Project/uav_guidance/plotting.py)
- `plot_engagement(result: SimulationResult, title: str = "UAV Pursuit Engagement", show: bool = True, save_path: str | None = None)`:
  - Creates 4 subplots in a 2x2 grid:
    1. Trajectory (XY plane) with start & intercept markers.
    2. LOS rate vs time ($\text{deg/s}$ or $\text{rad/s}$).
    3. Commanded vs Applied Normal Acceleration vs time ($\text{m/s}^2$).
    4. Range vs time ($\text{m}$).
  - Uses clean styling, clear unit labels, and explicit legends.

---

### `tests` Package

#### [NEW] [tests/test_kinematics.py](file:///d:/Ai%20Project/tests/test_kinematics.py)
- Integrator accuracy check.
- Straight-line heading preservation with $A_n = 0$.
- Circular trajectory turning radius $R = V^2 / A_n$ check.

#### [NEW] [tests/test_seeker.py](file:///d:/Ai%20Project/tests/test_seeker.py)
- Cross-validation between analytical vs finite-difference LOS rate and closing velocity.

#### [NEW] [tests/test_png.py](file:///d:/Ai%20Project/tests/test_png.py)
- PNG acceleration calculation test ($N \cdot V_c \cdot \dot{\lambda}$).
- Saturation check at $\pm A_{n,max}$.

#### [NEW] [tests/test_simulation.py](file:///d:/Ai%20Project/tests/test_simulation.py)
- **Head-on non-maneuvering target**: PNG drives LOS rate to 0 and miss distance < 1 m.
- **Collision-course geometry**: LOS rate remains near 0 and commanded $A_n$ near 0.
- **RK4 vs Euler**: Verify close agreement at small $dt$.
- **Maneuvering vs non-maneuvering**: Verify maneuvering target yields larger miss distance.
- **CPA interpolation**: Test $t_{cpa}$ and miss distance against hand-calculated straight-line scenario.

#### [NEW] [tests/test_scenarios.py](file:///d:/Ai%20Project/tests/test_scenarios.py)
- Test default scenario generation and grid factorial generator output count.

---

### Demo Entry Point

#### [NEW] [run_baseline.py](file:///d:/Ai%20Project/run_baseline.py)
- Runs `make_default_scenario()`.
- Executes simulation using `run_simulation()`.
- Prints summary statistics:
  - Miss Distance (m)
  - Time of Intercept (s)
  - Peak Commanded Acceleration ($\text{m/s}^2$)
  - Termination Reason
- Displays the 4-panel diagnostic plot.

---

## Verification Plan

### Automated Tests
Run pytest across the entire test suite:
```powershell
pytest tests/ -v
```

### Manual Verification
Execute `run_baseline.py`:
```powershell
python run_baseline.py
```
Verify stdout outputs correct miss distance (< 1m for successful guidance) and that matplotlib figure pops up with 4 clean plots.
