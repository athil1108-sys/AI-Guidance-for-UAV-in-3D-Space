"""
Unit tests for seeker.py module.
"""

import math
import pytest
from uav_guidance.kinematics import VehicleState, integrate_rk4
from uav_guidance.seeker import SeekerData, compute_seeker_data


def test_analytical_seeker_geometry():
    uav = VehicleState(x=0.0, y=0.0, speed=300.0, psi=0.0)
    target = VehicleState(x=3000.0, y=4000.0, speed=200.0, psi=math.pi)

    seeker = compute_seeker_data(uav, target)

    # Range = sqrt(3000^2 + 4000^2) = 5000
    assert math.isclose(seeker.range, 5000.0)
    # LOS angle = atan2(4000, 3000) ~ 0.927295 rad
    assert math.isclose(seeker.los_angle, math.atan2(4000, 3000))

    # Rel pos = (3000, 4000)
    # UAV vel = (300, 0), Target vel = (-200, 0)
    # Rel vel = (-500, 0)
    # dlambda/dt = (3000 * 0 - 4000 * (-500)) / 5000^2 = 2000000 / 25000000 = 0.08 rad/s
    assert math.isclose(seeker.los_rate, 0.08)

    # Vc = -(3000 * (-500) + 4000 * 0) / 5000 = 1500000 / 5000 = 300 m/s
    assert math.isclose(seeker.closing_velocity, 300.0)


def test_analytic_vs_finite_difference_agreement():
    uav = VehicleState(x=0.0, y=0.0, speed=300.0, psi=0.5)
    target = VehicleState(x=4000.0, y=3000.0, speed=200.0, psi=2.5)

    dt = 0.001  # Small dt for accurate finite differences
    uav_next = integrate_rk4(uav, an=10.0, dt=dt)
    target_next = integrate_rk4(target, an=-15.0, dt=dt)

    seeker_0 = compute_seeker_data(uav, target)
    seeker_1_analytic = compute_seeker_data(uav_next, target_next)
    seeker_1_fd = compute_seeker_data(
        uav_next, target_next, prev_seeker_data=seeker_0, dt=dt, use_finite_difference=True
    )

    # Verify analytical and finite difference agree within tight tolerance
    assert math.isclose(seeker_1_analytic.los_rate, seeker_1_fd.los_rate, rel_tol=1e-3, abs_tol=1e-4)
    assert math.isclose(seeker_1_analytic.closing_velocity, seeker_1_fd.closing_velocity, rel_tol=1e-3, abs_tol=1e-2)
