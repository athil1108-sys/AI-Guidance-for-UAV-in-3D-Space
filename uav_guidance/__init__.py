"""
UAV Pursuit Guidance Simulator (Phase 1 Baseline).

Implements 2D proportional navigation guidance (PNG) against stationary,
constant-heading, and maneuvering targets.
"""

from uav_guidance.kinematics import VehicleState, integrate_step
from uav_guidance.seeker import SeekerData, compute_seeker_data
from uav_guidance.png import GuidanceConfig, GuidanceOutput, compute_png_guidance
from uav_guidance.scenarios import (
    Scenario,
    make_default_scenario,
    generate_grid_scenarios,
)
from uav_guidance.simulation import (
    TrajectoryLog,
    SimulationResult,
    run_simulation,
    calculate_cpa_interpolation,
)
from uav_guidance.plotting import plot_engagement

__all__ = [
    "VehicleState",
    "integrate_step",
    "SeekerData",
    "compute_seeker_data",
    "GuidanceConfig",
    "GuidanceOutput",
    "compute_png_guidance",
    "Scenario",
    "make_default_scenario",
    "generate_grid_scenarios",
    "TrajectoryLog",
    "SimulationResult",
    "run_simulation",
    "calculate_cpa_interpolation",
    "plot_engagement",
]
