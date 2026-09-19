"""
Unit tests for png.py guidance law module.
"""

import math
import pytest
from uav_guidance.png import GuidanceConfig, compute_png_guidance
from uav_guidance.seeker import SeekerData


def test_png_guidance_calculation():
    seeker = SeekerData(range=5000.0, los_angle=0.5, los_rate=0.05, closing_velocity=400.0)
    config = GuidanceConfig(N=4.0, an_max=100.0)

    output = compute_png_guidance(seeker, config)

    # An = N * Vc * dlambda/dt = 4.0 * 400.0 * 0.05 = 80.0 m/s^2
    assert math.isclose(output.commanded_an, 80.0)
    assert math.isclose(output.applied_an, 80.0)


def test_png_guidance_saturation():
    seeker = SeekerData(range=5000.0, los_angle=0.5, los_rate=0.1, closing_velocity=400.0)
    config = GuidanceConfig(N=4.0, an_max=100.0)

    output = compute_png_guidance(seeker, config)

    # An_raw = 4.0 * 400.0 * 0.1 = 160.0 m/s^2 > 100.0
    assert math.isclose(output.commanded_an, 160.0)
    assert math.isclose(output.applied_an, 100.0)
