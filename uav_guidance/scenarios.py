"""
Scenarios module for generating engagement initial conditions and parameter sweeps.

Factors from Chithapuram et al. (IC3I 2014):
    1. Target velocity
    2. Target launch / aspect angle
    3. Target maneuver acceleration (TN)
    4. Target maneuver time / range
"""

import itertools
import math
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Scenario:
    """Engagement scenario parameters.

    Attributes:
        uav_x: UAV initial X position (m).
        uav_y: UAV initial Y position (m).
        uav_speed: UAV speed V_m (m/s).
        uav_heading: UAV initial heading angle psi_m (rad).
        target_x: Target initial X position (m).
        target_y: Target initial Y position (m).
        target_speed: Target speed V_t (m/s).
        target_heading: Target initial heading angle psi_t (rad).
        maneuver_time: Time t after which target applies constant normal acceleration TN (s).
        maneuver_accel: Target maneuver acceleration TN (m/s^2).
        dt: Simulation timestep (s).
        t_max: Maximum simulation run time (s).
        max_range: Maximum allowable range before termination (m).
        N: PNG Navigation constant.
        an_max: UAV maximum lateral acceleration limit (m/s^2).
    """

    uav_x: float = 0.0
    uav_y: float = 0.0
    uav_speed: float = 300.0
    uav_heading: float = 0.9272952180016122  # atan2(4000, 3000) ~ 53.13 deg
    target_x: float = 3000.0
    target_y: float = 4000.0
    target_speed: float = 200.0
    target_heading: float = -2.214297435588181  # atan2(-4000, -3000) = pi + los (head-on)
    maneuver_time: float = 5.0
    maneuver_accel: float = 30.0
    dt: float = 0.01
    t_max: float = 100.0
    max_range: float = 20000.0
    N: float = 4.0
    an_max: float = 100.0


def make_default_scenario() -> Scenario:
    """Generate the default baseline engagement scenario against a maneuvering target."""
    return Scenario()


def generate_grid_scenarios(
    target_velocities: List[float],
    target_headings: List[float],
    maneuver_accels: List[float],
    maneuver_times: List[float],
    base_scenario: Optional[Scenario] = None,
) -> List[Scenario]:
    """Generate a full-factorial grid of scenarios over key experimental parameters.

    Args:
        target_velocities: List of target speeds (m/s).
        target_headings: List of target initial headings (rad).
        maneuver_accels: List of target maneuver accelerations TN (m/s^2).
        maneuver_times: List of maneuver onset times (s).
        base_scenario: Base Scenario template. Defaults to make_default_scenario().

    Returns:
        List of Scenario objects covering all Cartesian combinations.
    """
    if base_scenario is None:
        base_scenario = make_default_scenario()

    scenarios = []
    for v_t, psi_t, tn, t_m in itertools.product(
        target_velocities, target_headings, maneuver_accels, maneuver_times
    ):
        sc = Scenario(
            uav_x=base_scenario.uav_x,
            uav_y=base_scenario.uav_y,
            uav_speed=base_scenario.uav_speed,
            uav_heading=base_scenario.uav_heading,
            target_x=base_scenario.target_x,
            target_y=base_scenario.target_y,
            target_speed=v_t,
            target_heading=psi_t,
            maneuver_time=t_m,
            maneuver_accel=tn,
            dt=base_scenario.dt,
            t_max=base_scenario.t_max,
            max_range=base_scenario.max_range,
            N=base_scenario.N,
            an_max=base_scenario.an_max,
        )
        scenarios.append(sc)

    return scenarios


def generate_orthogonal_array_scenarios() -> List[Scenario]:
    """Hook for Orthogonal Array (Taguchi design) scenario selection.

    To be implemented in Phase 2 / future experimental design extension.
    Reduces full-factorial combinations to an optimal orthogonal matrix.
    """
    # TODO: Implement Taguchi Orthogonal Array (L9/L16) parameter sampling
    raise NotImplementedError(
        "Orthogonal Array scenario selection is scheduled for a future phase."
    )
