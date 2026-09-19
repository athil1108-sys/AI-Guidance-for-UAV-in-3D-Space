"""
Simulation engine module for UAV pursuit-guidance.

Implements fixed timestep main loop, ZOH guidance command execution,
and true Point of Closest Approach (CPA) interpolation upon termination.
"""

from dataclasses import dataclass
from typing import Literal, Tuple
import numpy as np

from uav_guidance.kinematics import VehicleState, integrate_step
from uav_guidance.png import GuidanceConfig, compute_png_guidance
from uav_guidance.scenarios import Scenario
from uav_guidance.seeker import SeekerData, compute_seeker_data


@dataclass(frozen=True)
class TrajectoryLog:
    """Complete trajectory log recorded during simulation episode.

    All fields are 1D NumPy arrays of matching length.
    """

    time: np.ndarray
    uav_x: np.ndarray
    uav_y: np.ndarray
    uav_speed: np.ndarray
    uav_heading: np.ndarray
    target_x: np.ndarray
    target_y: np.ndarray
    target_speed: np.ndarray
    target_heading: np.ndarray
    range: np.ndarray
    los_angle: np.ndarray
    los_rate: np.ndarray
    closing_velocity: np.ndarray
    commanded_an: np.ndarray
    applied_an: np.ndarray


@dataclass(frozen=True)
class SimulationResult:
    """Outcome and trajectory payload of a simulation run.

    Attributes:
        trajectory: TrajectoryLog containing complete episode state history.
        miss_distance: True interpolated miss distance at point of closest approach (m).
        intercept_time: Total engagement duration / time of closest approach (s).
        termination_reason: Reason for simulation termination.
        t_cpa: Sub-step interpolation time of closest approach within final step [0, dt] (s).
        peak_commanded_an: Maximum absolute commanded lateral acceleration (m/s^2).
    """

    trajectory: TrajectoryLog
    miss_distance: float
    intercept_time: float
    termination_reason: str
    t_cpa: float
    peak_commanded_an: float


def calculate_cpa_interpolation(
    uav_state: VehicleState,
    target_state: VehicleState,
    dt: float,
) -> Tuple[float, float]:
    """Compute true point of closest approach (CPA) by interpolating within final timestep.

    CRITICAL NOTE / REPRODUCTION LESSON:
    In engagement simulations with small dt (e.g. 0.01 s) and high closing speeds (hundreds of m/s),
    simply using the range at the last discrete timestep overestimates the miss distance by meters.
    To get the true miss distance, we model relative position as r(tau) = r + v * tau for tau in [0, dt].
    Setting d/d(tau) |r + v * tau|^2 = 0 yields:
        t_cpa = -(r . v) / (v . v)
    clamped to [0, dt], and true miss distance = |r + v * t_cpa|.

    Args:
        uav_state: UAV state at start of final step.
        target_state: Target state at start of final step.
        dt: Timestep duration (s).

    Returns:
        Tuple of (t_cpa, miss_distance).
    """
    rx = target_state.x - uav_state.x
    ry = target_state.y - uav_state.y
    r = np.array([rx, ry], dtype=np.float64)

    vx_rel = target_state.vx - uav_state.vx
    vy_rel = target_state.vy - uav_state.vy
    v = np.array([vx_rel, vy_rel], dtype=np.float64)

    v_sq = float(np.dot(v, v))
    if v_sq > 1e-9:
        t_cpa_raw = -float(np.dot(r, v)) / v_sq
        t_cpa = float(np.clip(t_cpa_raw, 0.0, dt))
    else:
        t_cpa = 0.0

    miss_vec = r + v * t_cpa
    miss_distance = float(np.linalg.norm(miss_vec))
    return (t_cpa, miss_distance)


def run_simulation(
    scenario: Scenario,
    integration_method: Literal["rk4", "euler"] = "rk4",
    use_finite_difference: bool = False,
) -> SimulationResult:
    """Run an engagement simulation episode to completion.

    Args:
        scenario: Engagement scenario parameters.
        integration_method: "rk4" or "euler" for vehicle state propagation.
        use_finite_difference: If True, uses finite difference for seeker derivatives.

    Returns:
        SimulationResult containing recorded trajectory, miss distance, and statistics.
    """
    uav_state = VehicleState(
        x=scenario.uav_x,
        y=scenario.uav_y,
        speed=scenario.uav_speed,
        psi=scenario.uav_heading,
    )
    target_state = VehicleState(
        x=scenario.target_x,
        y=scenario.target_y,
        speed=scenario.target_speed,
        psi=scenario.target_heading,
    )

    guidance_config = GuidanceConfig(N=scenario.N, an_max=scenario.an_max)

    time_hist = []
    uav_x_hist, uav_y_hist = [], []
    uav_speed_hist, uav_psi_hist = [], []
    target_x_hist, target_y_hist = [], []
    target_speed_hist, target_psi_hist = [], []
    range_hist, los_angle_hist = [], []
    los_rate_hist, Vc_hist = [], []
    cmd_an_hist, app_an_hist = [], []

    current_time = 0.0
    dt = scenario.dt
    prev_seeker: SeekerData | None = None
    prev_uav_state: VehicleState = uav_state
    prev_target_state: VehicleState = target_state
    step_count = 0

    termination_reason = "max_time_exceeded"
    t_cpa = 0.0
    miss_distance = 0.0

    while current_time <= scenario.t_max:
        # Seeker measurements at beginning of timestep
        seeker = compute_seeker_data(
            uav_state=uav_state,
            target_state=target_state,
            prev_seeker_data=prev_seeker,
            dt=dt,
            use_finite_difference=use_finite_difference,
        )

        # Check termination criteria (allow at least 1 step to initialize)
        if step_count > 0 and seeker.closing_velocity < 0.0:
            termination_reason = "closing_velocity_negative"
            # Compute CPA interpolation using state at start of final interval (prev step)
            t_cpa, miss_distance = calculate_cpa_interpolation(
                prev_uav_state, prev_target_state, dt
            )
            break

        if seeker.range > scenario.max_range:
            termination_reason = "max_range_exceeded"
            t_cpa, miss_distance = calculate_cpa_interpolation(
                prev_uav_state, prev_target_state, dt
            )
            break

        # Compute PNG guidance command (Zero-Order Hold for duration dt)
        guidance = compute_png_guidance(seeker, guidance_config)

        # Record trajectory state
        time_hist.append(current_time)
        uav_x_hist.append(uav_state.x)
        uav_y_hist.append(uav_state.y)
        uav_speed_hist.append(uav_state.speed)
        uav_psi_hist.append(uav_state.psi)
        target_x_hist.append(target_state.x)
        target_y_hist.append(target_state.y)
        target_speed_hist.append(target_state.speed)
        target_psi_hist.append(target_state.psi)
        range_hist.append(seeker.range)
        los_angle_hist.append(seeker.los_angle)
        los_rate_hist.append(seeker.los_rate)
        Vc_hist.append(seeker.closing_velocity)
        cmd_an_hist.append(guidance.commanded_an)
        app_an_hist.append(guidance.applied_an)

        # Determine target maneuver acceleration
        target_an = (
            scenario.maneuver_accel
            if current_time >= scenario.maneuver_time
            else 0.0
        )

        # Save states prior to numerical integration step
        prev_uav_state = uav_state
        prev_target_state = target_state

        # Integrate vehicle states over dt with ZOH guidance commands
        uav_state = integrate_step(
            uav_state, guidance.applied_an, dt, method=integration_method
        )
        target_state = integrate_step(
            target_state, target_an, dt, method=integration_method
        )

        prev_seeker = seeker
        current_time += dt
        step_count += 1

    else:
        # Loop exited due to t_max reached
        termination_reason = "max_time_exceeded"
        t_cpa, miss_distance = calculate_cpa_interpolation(
            prev_uav_state, prev_target_state, dt
        )

    last_logged_time = time_hist[-1] if time_hist else current_time
    intercept_time = last_logged_time + t_cpa

    trajectory = TrajectoryLog(
        time=np.array(time_hist, dtype=np.float64),
        uav_x=np.array(uav_x_hist, dtype=np.float64),
        uav_y=np.array(uav_y_hist, dtype=np.float64),
        uav_speed=np.array(uav_speed_hist, dtype=np.float64),
        uav_heading=np.array(uav_psi_hist, dtype=np.float64),
        target_x=np.array(target_x_hist, dtype=np.float64),
        target_y=np.array(target_y_hist, dtype=np.float64),
        target_speed=np.array(target_speed_hist, dtype=np.float64),
        target_heading=np.array(target_psi_hist, dtype=np.float64),
        range=np.array(range_hist, dtype=np.float64),
        los_angle=np.array(los_angle_hist, dtype=np.float64),
        los_rate=np.array(los_rate_hist, dtype=np.float64),
        closing_velocity=np.array(Vc_hist, dtype=np.float64),
        commanded_an=np.array(cmd_an_hist, dtype=np.float64),
        applied_an=np.array(app_an_hist, dtype=np.float64),
    )

    peak_cmd_an = (
        float(np.max(np.abs(trajectory.commanded_an)))
        if len(trajectory.commanded_an) > 0
        else 0.0
    )

    return SimulationResult(
        trajectory=trajectory,
        miss_distance=miss_distance,
        intercept_time=intercept_time,
        termination_reason=termination_reason,
        t_cpa=t_cpa,
        peak_commanded_an=peak_cmd_an,
    )
