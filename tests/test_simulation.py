"""
Unit tests for simulation engine module.
"""

import math
import numpy as np
import pytest

from uav_guidance.kinematics import VehicleState
from uav_guidance.scenarios import Scenario, make_default_scenario
from uav_guidance.simulation import (
    calculate_cpa_interpolation,
    run_simulation,
)


def test_head_on_non_maneuvering_target():
    """Head-on non-maneuvering target: PNG should drive LOS rate toward zero and produce miss distance < 1m."""
    sc = Scenario(
        uav_x=0.0,
        uav_y=0.0,
        uav_speed=300.0,
        uav_heading=0.0,
        target_x=5000.0,
        target_y=0.0,
        target_speed=200.0,
        target_heading=math.pi,
        maneuver_time=100.0,
        maneuver_accel=0.0,
        dt=0.01,
        t_max=50.0,
    )

    res = run_simulation(sc, integration_method="rk4")

    # Miss distance must be < 1.0 m
    assert res.miss_distance < 1.0, f"Miss distance was {res.miss_distance} m"
    # Final LOS rate should be near zero
    assert abs(res.trajectory.los_rate[-1]) < 1e-3


def test_collision_course_geometry():
    """Non-maneuvering target on collision course geometry: LOS rate and commanded An should stay near zero throughout."""
    # UAV at (0,0), speed 300
    # Target at (4000, 3000), distance 5000, speed 200
    # LOS angle lambda = atan2(3000, 4000) = 0.6435011087932844
    # Target heading psi_t = pi (heading towards -X)
    # Collision condition: V_m * sin(psi_m - lambda) = V_t * sin(psi_t - lambda)
    # sin(psi_t - lambda) = sin(pi - 0.6435011) = 0.6
    # 300 * sin(psi_m - lambda) = 200 * 0.6 = 120 => sin(psi_m - lambda) = 0.4
    # psi_m = lambda + arcsin(0.4) = 0.6435011087932844 + 0.41151684606748806 = 1.0550179548607725
    los_angle = math.atan2(3000.0, 4000.0)
    psi_m = los_angle + math.asin(0.4)
    psi_t = math.pi

    sc = Scenario(
        uav_x=0.0,
        uav_y=0.0,
        uav_speed=300.0,
        uav_heading=psi_m,
        target_x=4000.0,
        target_y=3000.0,
        target_speed=200.0,
        target_heading=psi_t,
        maneuver_time=100.0,
        maneuver_accel=0.0,
        dt=0.01,
        t_max=50.0,
    )

    res = run_simulation(sc, integration_method="rk4")

    # Command An should stay near zero (< 1 m/s^2 peak)
    assert res.peak_commanded_an < 1.0, f"Peak commanded An was {res.peak_commanded_an}"
    # Max LOS rate magnitude should be near zero (< 1e-3 rad/s)
    max_los_rate = np.max(np.abs(res.trajectory.los_rate))
    assert max_los_rate < 1e-3, f"Max LOS rate was {max_los_rate}"
    # Miss distance < 0.1 m
    assert res.miss_distance < 0.1


def test_analytic_vs_finite_difference_simulation():
    """Verify simulation run with analytic vs finite-difference seeker yield similar results."""
    sc = make_default_scenario()

    res_analytic = run_simulation(sc, use_finite_difference=False)
    res_fd = run_simulation(sc, use_finite_difference=True)

    # Intercept time and miss distance should match within close tolerance
    assert math.isclose(res_analytic.intercept_time, res_fd.intercept_time, abs_tol=0.05)
    assert math.isclose(res_analytic.miss_distance, res_fd.miss_distance, abs_tol=0.5)


def test_rk4_vs_euler_integration():
    """RK4 and Euler give similar miss distances at small dt."""
    sc = make_default_scenario()

    res_rk4 = run_simulation(sc, integration_method="rk4")
    res_euler = run_simulation(sc, integration_method="euler")

    assert math.isclose(res_rk4.miss_distance, res_euler.miss_distance, abs_tol=0.5)
    assert math.isclose(res_rk4.intercept_time, res_euler.intercept_time, abs_tol=0.05)


def test_maneuvering_vs_non_maneuvering_target():
    """A maneuvering target produces a larger miss distance than a non-maneuvering target under identical initial conditions."""
    sc_base = make_default_scenario()
    # Scenario with maneuvering target (TN = 30 m/s^2)
    res_maneuver = run_simulation(sc_base)

    # Scenario with non-maneuvering target (TN = 0 m/s^2)
    sc_non_maneuver = Scenario(
        uav_x=sc_base.uav_x,
        uav_y=sc_base.uav_y,
        uav_speed=sc_base.uav_speed,
        uav_heading=sc_base.uav_heading,
        target_x=sc_base.target_x,
        target_y=sc_base.target_y,
        target_speed=sc_base.target_speed,
        target_heading=sc_base.target_heading,
        maneuver_time=sc_base.maneuver_time,
        maneuver_accel=0.0,
        dt=sc_base.dt,
        t_max=sc_base.t_max,
    )
    res_non_maneuver = run_simulation(sc_non_maneuver)

    assert res_maneuver.miss_distance >= res_non_maneuver.miss_distance


def test_cpa_interpolation_hand_computed():
    """Verify t_cpa interpolation against a hand-computed straight-line case.

    Case:
      UAV at (0, 0), velocity = (300, 0) m/s
      Target at (10, 5), velocity = (-200, 0) m/s
      Relative position r = (10, 5) m
      Relative velocity v = (-500, 0) m/s
      t_cpa = -(10*-500 + 5*0) / ((-500)^2 + 0^2) = 5000 / 250000 = 0.02 s
      At t_cpa = 0.02 s:
        r(t_cpa) = (10 - 500*0.02, 5 + 0) = (0, 5) m
      Expected miss distance = 5.0 m.
    """
    uav_state = VehicleState(x=0.0, y=0.0, speed=300.0, psi=0.0)
    target_state = VehicleState(x=10.0, y=5.0, speed=200.0, psi=math.pi)

    t_cpa, miss_dist = calculate_cpa_interpolation(uav_state, target_state, dt=0.05)

    assert math.isclose(t_cpa, 0.02, abs_tol=1e-7)
    assert math.isclose(miss_dist, 5.0, abs_tol=1e-7)
