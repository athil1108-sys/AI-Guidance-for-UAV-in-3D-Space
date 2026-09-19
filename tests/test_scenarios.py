"""
Unit tests for scenarios.py module.
"""

import pytest
from uav_guidance.scenarios import (
    Scenario,
    generate_grid_scenarios,
    generate_orthogonal_array_scenarios,
    make_default_scenario,
)


def test_make_default_scenario():
    sc = make_default_scenario()
    assert isinstance(sc, Scenario)
    assert sc.uav_speed == 300.0
    assert sc.target_speed == 200.0
    assert sc.maneuver_accel == 30.0


def test_generate_grid_scenarios():
    target_vels = [150.0, 250.0]
    target_headings = [0.0, 3.14]
    maneuver_accels = [0.0, 20.0]
    maneuver_times = [2.0, 5.0]

    scenarios = generate_grid_scenarios(
        target_velocities=target_vels,
        target_headings=target_headings,
        maneuver_accels=maneuver_accels,
        maneuver_times=maneuver_times,
    )

    # 2 x 2 x 2 x 2 = 16 scenarios
    assert len(scenarios) == 16
    assert isinstance(scenarios[0], Scenario)


def test_orthogonal_array_hook():
    with pytest.raises(NotImplementedError):
        generate_orthogonal_array_scenarios()
