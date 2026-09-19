"""
Unit tests for kinematics.py module.
"""

import math
import pytest
from uav_guidance.kinematics import (
    VehicleState,
    integrate_euler,
    integrate_rk4,
    integrate_step,
    wrap_angle,
)


def test_wrap_angle():
    assert math.isclose(wrap_angle(0.0), 0.0)
    assert math.isclose(wrap_angle(3 * math.pi), math.pi, abs_tol=1e-7)
    assert math.isclose(wrap_angle(-3 * math.pi), -math.pi, abs_tol=1e-7)


def test_vehicle_state_velocity():
    state = VehicleState(x=100.0, y=200.0, speed=300.0, psi=0.0)
    assert math.isclose(state.vx, 300.0)
    assert math.isclose(state.vy, 0.0)

    state_90 = VehicleState(x=0.0, y=0.0, speed=200.0, psi=math.pi / 2)
    assert math.isclose(state_90.vx, 0.0, abs_tol=1e-7)
    assert math.isclose(state_90.vy, 200.0)


def test_straight_line_integration():
    state = VehicleState(x=0.0, y=0.0, speed=100.0, psi=0.0)
    dt = 0.1
    # Zero acceleration -> straight line along +X
    new_euler = integrate_euler(state, an=0.0, dt=dt)
    new_rk4 = integrate_rk4(state, an=0.0, dt=dt)

    assert math.isclose(new_euler.x, 10.0)
    assert math.isclose(new_euler.y, 0.0)
    assert math.isclose(new_rk4.x, 10.0)
    assert math.isclose(new_rk4.y, 0.0)


def test_circular_turn_rk4_accuracy():
    # Constant lateral acceleration An causes uniform circular motion with radius R = V^2 / An
    v = 100.0
    an = 10.0
    radius = (v * v) / an  # 1000 m
    omega = an / v  # 0.1 rad/s

    state = VehicleState(x=0.0, y=0.0, speed=v, psi=0.0)
    dt = 0.01
    num_steps = 100
    for _ in range(num_steps):
        state = integrate_step(state, an=an, dt=dt, method="rk4")

    expected_psi = omega * (num_steps * dt)  # 0.1 * 1.0 = 0.1 rad
    assert math.isclose(state.psi, expected_psi, abs_tol=1e-5)
