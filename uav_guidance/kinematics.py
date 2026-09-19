"""
Kinematics module for UAV and Target state representation and numerical integration.

Equations of motion for a point mass moving at constant speed V with normal acceleration An:
    dx/dt = V * cos(psi)
    dy/dt = V * sin(psi)
    dpsi/dt = An / V
"""

import math
from dataclasses import dataclass
from typing import Literal, Tuple


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class VehicleState:
    """State of a vehicle (UAV or Target) at a specific time.

    Attributes:
        x: Position along X axis (m).
        y: Position along Y axis (m).
        speed: Constant speed V (m/s).
        psi: Heading angle from +X axis (rad).
    """

    x: float
    y: float
    speed: float
    psi: float

    @property
    def vx(self) -> float:
        """X component of velocity vector (m/s)."""
        return self.speed * math.cos(self.psi)

    @property
    def vy(self) -> float:
        """Y component of velocity vector (m/s)."""
        return self.speed * math.sin(self.psi)

    @property
    def velocity(self) -> Tuple[float, float]:
        """Velocity vector (vx, vy) in m/s."""
        return (self.vx, self.vy)

    @property
    def position(self) -> Tuple[float, float]:
        """Position vector (x, y) in m."""
        return (self.x, self.y)


def state_derivatives(state: VehicleState, an: float) -> Tuple[float, float, float]:
    """Compute state derivatives (dx/dt, dy/dt, dpsi/dt).

    Args:
        state: Current vehicle state.
        an: Applied normal acceleration (m/s^2), perpendicular to velocity.

    Returns:
        Tuple of (dx_dt, dy_dt, dpsi_dt).
    """
    dx_dt = state.speed * math.cos(state.psi)
    dy_dt = state.speed * math.sin(state.psi)
    dpsi_dt = an / state.speed if state.speed > 0 else 0.0
    return (dx_dt, dy_dt, dpsi_dt)


def integrate_euler(state: VehicleState, an: float, dt: float) -> VehicleState:
    """Integrate vehicle state using Forward Euler method.

    Args:
        state: Initial vehicle state.
        an: Normal acceleration held constant over dt (m/s^2).
        dt: Integration time step (s).

    Returns:
        Updated VehicleState after time dt.
    """
    dx_dt, dy_dt, dpsi_dt = state_derivatives(state, an)
    new_x = state.x + dx_dt * dt
    new_y = state.y + dy_dt * dt
    new_psi = wrap_angle(state.psi + dpsi_dt * dt)
    return VehicleState(x=new_x, y=new_y, speed=state.speed, psi=new_psi)


def integrate_rk4(state: VehicleState, an: float, dt: float) -> VehicleState:
    """Integrate vehicle state using 4th-order Runge-Kutta (RK4) method.

    Guidance / acceleration command is held constant over the timestep dt
    (Zero-Order Hold).

    Args:
        state: Initial vehicle state.
        an: Normal acceleration held constant over dt (m/s^2).
        dt: Integration time step (s).

    Returns:
        Updated VehicleState after time dt.
    """

    def f(x: float, y: float, psi: float) -> Tuple[float, float, float]:
        dx = state.speed * math.cos(psi)
        dy = state.speed * math.sin(psi)
        dpsi = an / state.speed if state.speed > 0 else 0.0
        return (dx, dy, dpsi)

    # Substep 1
    k1_x, k1_y, k1_psi = f(state.x, state.y, state.psi)

    # Substep 2
    k2_x, k2_y, k2_psi = f(
        state.x + 0.5 * dt * k1_x,
        state.y + 0.5 * dt * k1_y,
        state.psi + 0.5 * dt * k1_psi,
    )

    # Substep 3
    k3_x, k3_y, k3_psi = f(
        state.x + 0.5 * dt * k2_x,
        state.y + 0.5 * dt * k2_y,
        state.psi + 0.5 * dt * k2_psi,
    )

    # Substep 4
    k4_x, k4_y, k4_psi = f(
        state.x + dt * k3_x,
        state.y + dt * k3_y,
        state.psi + dt * k3_psi,
    )

    new_x = state.x + (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
    new_y = state.y + (dt / 6.0) * (k1_y + 2.0 * k2_y + 2.0 * k3_y + k4_y)
    new_psi = wrap_angle(
        state.psi + (dt / 6.0) * (k1_psi + 2.0 * k2_psi + 2.0 * k3_psi + k4_psi)
    )

    return VehicleState(x=new_x, y=new_y, speed=state.speed, psi=new_psi)


def integrate_step(
    state: VehicleState,
    an: float,
    dt: float,
    method: Literal["rk4", "euler"] = "rk4",
) -> VehicleState:
    """Integrate state by dt using selected numerical scheme.

    Args:
        state: Initial vehicle state.
        an: Normal acceleration held constant over dt (m/s^2).
        dt: Integration time step (s).
        method: "rk4" or "euler".

    Returns:
        Updated VehicleState after time dt.
    """
    if method.lower() == "rk4":
        return integrate_rk4(state, an, dt)
    elif method.lower() == "euler":
        return integrate_euler(state, an, dt)
    else:
        raise ValueError(f"Unknown integration method: {method}")
