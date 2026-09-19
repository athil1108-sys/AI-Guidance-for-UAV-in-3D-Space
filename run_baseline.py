"""
Demo entry point: run_baseline.py

Runs one default baseline engagement against a maneuvering target implementing
Proportional Navigation Guidance (PNG), prints miss distance, intercept time,
and peak commanded lateral acceleration, and displays the 4-panel telemetry plots.
"""

from uav_guidance import (
    make_default_scenario,
    plot_engagement,
    run_simulation,
)


def main() -> None:
    print("=" * 60)
    print("2D UAV Pursuit-Guidance Simulator (PNG Baseline - Phase 1)")
    print("Paper: Chithapuram, Jeppu & Aswani Kumar (IC3I 2014)")
    print("=" * 60)

    # 1. Generate default scenario against maneuvering target
    scenario = make_default_scenario()
    print("\n[Scenario Initial Conditions]")
    print(f"  UAV Position:        ({scenario.uav_x:.1f}, {scenario.uav_y:.1f}) m")
    print(f"  UAV Speed:           {scenario.uav_speed:.1f} m/s")
    print(f"  UAV Heading:         {scenario.uav_heading:.4f} rad")
    print(f"  Target Position:     ({scenario.target_x:.1f}, {scenario.target_y:.1f}) m")
    print(f"  Target Speed:        {scenario.target_speed:.1f} m/s")
    print(f"  Target Heading:      {scenario.target_heading:.4f} rad")
    print(f"  Maneuver Time:       t >= {scenario.maneuver_time:.1f} s")
    print(f"  Maneuver Accel TN:   {scenario.maneuver_accel:.1f} m/s^2")
    print(f"  PNG Constant N:      {scenario.N:.1f}")
    print(f"  Max Acceleration:    {scenario.an_max:.1f} m/s^2")
    print(f"  Timestep dt:         {scenario.dt:.3f} s")

    # 2. Run simulation
    print("\nRunning simulation with RK4 integration and Zero-Order Hold guidance...")
    result = run_simulation(scenario, integration_method="rk4")

    # 3. Print performance metrics
    print("\n" + "-" * 40)
    print("SIMULATION RESULTS")
    print("-" * 40)
    print(f"  Termination Reason:       {result.termination_reason}")
    print(f"  Miss Distance:            {result.miss_distance:.4f} m")
    print(f"  Time of Intercept:        {result.intercept_time:.4f} s")
    print(f"  Peak Commanded An:        {result.peak_commanded_an:.2f} m/s^2")
    print(f"  Substep CPA Offset (t_cpa):{result.t_cpa:.5f} s")
    print("-" * 40)

    # 4. Display 4-panel diagnostic plot
    print("\nGenerating engagement plots...")
    plot_engagement(
        result,
        title="PNG Baseline Engagement vs Maneuvering Target (IC3I 2014 Reproduction)",
        show=True,
        save_path="baseline_engagement.png",
    )
    print("Saved plot to baseline_engagement.png")


if __name__ == "__main__":
    main()
