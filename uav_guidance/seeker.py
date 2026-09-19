"""
Seeker module for relative kinematics calculation: Range, LOS angle, LOS rate, and Closing Velocity.
"""

import math
from dataclasses import dataclass
from typing import Optional
from uav_guidance.kinematics import VehicleState, wrap_angle


@dataclass(frozen=True)
class SeekerData:
    """Relative kinematics measured by the UAV seeker at a given timestep.

    Attributes:
        range: Distance between UAV and Target, R (m).
        los_angle: Line-of-sight angle lambda measured from +X axis (rad).
        los_rate: Time rate of change of LOS angle, dlambda/dt (rad/s).
        closing_velocity: Closing speed Vc = -dR/dt (m/s).
    """

    range: float
    los_angle: float
    los_rate: float
    closing_velocity: float


def compute_seeker_data(
    uav_state: VehicleState,
    target_state: VehicleState,
    prev_seeker_data: Optional[SeekerData] = None,
    dt: Optional[float] = None,
    use_finite_difference: bool = False,
) -> SeekerData:
    """Compute relative kinematics between UAV and Target.

    Args:
        uav_state: Current state of the UAV.
        target_state: Current state of the Target.
        prev_seeker_data: Previous timestep's SeekerData (needed if finite difference is requested).
        dt: Timestep duration in seconds (needed if finite difference is requested).
        use_finite_difference: If True, computes LOS rate and closing velocity using finite differences.
                                Defaults to False (analytical calculation).

    Returns:
        SeekerData object containing range, los_angle, los_rate, and closing_velocity.
    """
    rx = target_state.x - uav_state.x
    ry = target_state.y - uav_state.y

    r_mag = math.sqrt(rx * rx + ry * ry)
    los_angle = math.atan2(ry, rx)

    if r_mag < 1e-9:
        # Avoid division by zero when range is virtually zero
        return SeekerData(
            range=r_mag,
            los_angle=los_angle,
            los_rate=0.0,
            closing_velocity=0.0,
        )

    if use_finite_difference:
        if prev_seeker_data is None or dt is None or dt <= 0:
            # Fall back to analytical on first frame or invalid dt
            vx_rel = target_state.vx - uav_state.vx
            vy_rel = target_state.vy - uav_state.vy
            los_rate = (rx * vy_rel - ry * vx_rel) / (r_mag * r_mag)
            closing_velocity = -(rx * vx_rel + ry * vy_rel) / r_mag
        else:
            # Finite difference calculation
            d_lambda = wrap_angle(los_angle - prev_seeker_data.los_angle)
            los_rate = d_lambda / dt
            closing_velocity = -(r_mag - prev_seeker_data.range) / dt
    else:
        # Analytical computation:
        # rel_pos = (Xt - Xm, Yt - Ym)
        # rel_vel = (Vxt - Vxm, Vyt - Vym)
        # dlambda/dt = (rel_pos.x * rel_vel.y - rel_pos.y * rel_vel.x) / R^2
        # Vc = -(rel_pos . rel_vel) / R
        vx_rel = target_state.vx - uav_state.vx
        vy_rel = target_state.vy - uav_state.vy

        los_rate = (rx * vy_rel - ry * vx_rel) / (r_mag * r_mag)
        closing_velocity = -(rx * vx_rel + ry * vy_rel) / r_mag

    return SeekerData(
        range=r_mag,
        los_angle=los_angle,
        los_rate=los_rate,
        closing_velocity=closing_velocity,
    )
