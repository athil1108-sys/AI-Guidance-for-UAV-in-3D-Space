"""
Proportional Navigation Guidance (PNG) law module.

PNG command equation (eq. 1/2 from paper):
    An_command = N * Vc * (dlambda/dt)
"""

from dataclasses import dataclass
from uav_guidance.seeker import SeekerData


@dataclass(frozen=True)
class GuidanceConfig:
    """Configuration for PNG guidance law.

    Attributes:
        N: Navigation constant (dimensionless). Paper default = 4.0.
        an_max: Maximum normal acceleration magnitude (m/s^2). Paper default = 100.0.
    """

    N: float = 4.0
    an_max: float = 100.0


@dataclass(frozen=True)
class GuidanceOutput:
    """Output from PNG guidance law.

    Attributes:
        commanded_an: Raw commanded normal acceleration (m/s^2).
        applied_an: Saturated normal acceleration within [-an_max, +an_max] (m/s^2).
    """

    commanded_an: float
    applied_an: float


def compute_png_guidance(
    seeker_data: SeekerData,
    config: GuidanceConfig = GuidanceConfig(),
) -> GuidanceOutput:
    """Compute Proportional Navigation Guidance command.

    # eq. 1/2: An_command = N * Vc * dlambda/dt

    Args:
        seeker_data: Current relative seeker measurements (Vc, los_rate).
        config: Guidance law configuration parameters (N, an_max).

    Returns:
        GuidanceOutput containing raw commanded and saturated applied acceleration.
    """
    # eq. 1: An_command = N * Vc * dlambda/dt
    commanded_an = config.N * seeker_data.closing_velocity * seeker_data.los_rate

    # Action-space saturation to [-an_max, +an_max]
    an_max = config.an_max
    if commanded_an > an_max:
        applied_an = an_max
    elif commanded_an < -an_max:
        applied_an = -an_max
    else:
        applied_an = commanded_an

    return GuidanceOutput(commanded_an=commanded_an, applied_an=applied_an)
