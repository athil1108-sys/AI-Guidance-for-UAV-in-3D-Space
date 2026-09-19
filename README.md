# Artificial Intelligence Guidance for Unmanned Aerial Vehicles in Three Dimensional Space

Phase 1 implementation reproducing and extending Chithapuram, Jeppu & Aswani Kumar, *"Artificial Intelligence Guidance for Unmanned Aerial Vehicles in Three Dimensional Space"* (IC3I 2014).

This phase provides a 2D UAV pursuit-guidance baseline simulator implementing classical Proportional Navigation Guidance (PNG) with numerical integration (RK4/Euler), analytical seeker derivatives, scenario parameter sweeps, and Point of Closest Approach (CPA) sub-step interpolation.

## Repository Structure

```
.
├── uav_guidance/
│   ├── __init__.py
│   ├── kinematics.py    # UAV & Target state representation and RK4/Euler integrators
│   ├── seeker.py        # Relative kinematics: Range, LOS angle, analytical & FD LOS rate / Vc
│   ├── png.py           # PNG guidance law and acceleration saturation
│   ├── simulation.py    # Main episode runner, zero-order hold execution, CPA interpolation
│   ├── scenarios.py     # Scenario parameters, default scenario, full-factorial grid generator
│   └── plotting.py      # Telemetry visualizer (Trajectory, LOS rate, Acceleration, Range)
├── tests/
│   ├── test_kinematics.py
│   ├── test_seeker.py
│   ├── test_png.py
│   ├── test_simulation.py
│   └── test_scenarios.py
├── run_baseline.py      # Entry point demo executing baseline maneuvering target engagement
├── README.md
└── .gitignore
```

## Setup & Quickstart

### Requirements
- Python 3.10+
- NumPy
- Matplotlib
- Pytest (for running tests)

### Run Baseline Engagement
```bash
python run_baseline.py
```

### Run Unit Tests
```bash
python -m pytest tests/ -v
```

## Math & Physics Highlights
- **State derivatives**:
  $$\frac{dx}{dt} = V \cos\psi, \quad \frac{dy}{dt} = V \sin\psi, \quad \frac{d\psi}{dt} = \frac{A_n}{V}$$
- **Analytical Seeker**:
  $$\dot{\lambda} = \frac{r_x v_y - r_y v_x}{R^2}, \quad V_c = -\frac{\mathbf{r} \cdot \mathbf{v}}{R}$$
- **PNG Guidance Law**:
  $$A_{n,\text{cmd}} = N \cdot V_c \cdot \dot{\lambda} \quad (\text{saturated to } \pm A_{n,\text{max}})$$
- **Point of Closest Approach (CPA) Interpolation**:
  $$t_{\text{cpa}} = \text{clamp}\left(-\frac{\mathbf{r} \cdot \mathbf{v}}{\|\mathbf{v}\|^2}, 0, dt\right), \quad d_{\text{miss}} = \|\mathbf{r} + \mathbf{v} \cdot t_{\text{cpa}}\|$$
