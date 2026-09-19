"""
Plotting module for visualization of UAV pursuit guidance simulation runs.

Generates a 4-panel diagnostic figure:
  1. UAV vs Target trajectory in the inertial XY plane with intercept markers.
  2. LOS rate vs time.
  3. Commanded vs Applied Lateral Acceleration vs time.
  4. Range vs time.
"""

from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

from uav_guidance.simulation import SimulationResult


def plot_engagement(
    result: SimulationResult,
    title: str = "2D UAV Pursuit-Guidance Engagement (PNG Baseline)",
    show: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot simulation engagement trajectory and guidance telemetry.

    Args:
        result: SimulationResult returned by run_simulation.
        title: Overall title for figure.
        show: If True, calls plt.show().
        save_path: If provided, saves figure to this file path.
        figsize: Figure dimensions (width, height) in inches.

    Returns:
        Tuple of (Figure, Axes array).
    """
    traj = result.trajectory
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Panel 1: XY Trajectory
    ax1 = axes[0, 0]
    ax1.plot(traj.uav_x, traj.uav_y, "b-", label="UAV Trajectory", linewidth=2)
    ax1.plot(traj.target_x, traj.target_y, "r--", label="Target Trajectory", linewidth=2)
    # Start markers
    ax1.plot(traj.uav_x[0], traj.uav_y[0], "bo", markersize=8, label="UAV Start")
    ax1.plot(traj.target_x[0], traj.target_y[0], "ro", markersize=8, label="Target Start")
    # Intercept markers
    if len(traj.uav_x) > 0:
        ax1.plot(traj.uav_x[-1], traj.uav_y[-1], "bx", markersize=10, markeredgewidth=3, label="UAV Intercept")
        ax1.plot(traj.target_x[-1], traj.target_y[-1], "rx", markersize=10, markeredgewidth=3, label="Target Intercept")

    ax1.set_xlabel("X Position (m)")
    ax1.set_ylabel("Y Position (m)")
    ax1.set_title("Trajectory in XY Plane")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="best", fontsize=9)
    ax1.axis("equal")

    # Panel 2: LOS Rate vs Time
    ax2 = axes[0, 1]
    los_rate_deg = np.rad2deg(traj.los_rate)
    ax2.plot(traj.time, los_rate_deg, "g-", label="LOS Rate (dλ/dt)", linewidth=1.8)
    ax2.axhline(0, color="black", linestyle=":", alpha=0.7)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("LOS Rate (deg/s)")
    ax2.set_title("Line-of-Sight (LOS) Rate vs Time")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="best", fontsize=9)

    # Panel 3: Commanded vs Applied Acceleration vs Time
    ax3 = axes[1, 0]
    ax3.plot(traj.time, traj.commanded_an, "c-", label="Commanded An", linewidth=1.8)
    ax3.plot(traj.time, traj.applied_an, "m--", label="Applied An (Saturated)", linewidth=1.8)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Normal Acceleration (m/s²)")
    ax3.set_title("Normal Acceleration vs Time")
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend(loc="best", fontsize=9)

    # Panel 4: Range vs Time
    ax4 = axes[1, 1]
    ax4.plot(traj.time, traj.range, "k-", label="Range (R)", linewidth=1.8)
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Range (m)")
    ax4.set_title("Relative Range vs Time")
    ax4.grid(True, linestyle="--", alpha=0.6)
    ax4.legend(loc="best", fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, axes
