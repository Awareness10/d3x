#!/usr/bin/env python3
"""Earth-Moon system visualization example."""

import argparse

import d3x
from d3x.viz import Viewer

# Speed presets: (dt, steps_per_frame, description)
SPEED_PRESETS = {
    "slow": (60.0, 2, "Educational pace - 1 orbit in ~2 min"),
    "normal": (120.0, 6, "Default - 1 orbit in ~30 sec"),
    "fast": (300.0, 10, "Fast forward - 1 orbit in ~8 sec"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Earth-Moon orbital simulation")
    parser.add_argument(
        "--speed",
        "-s",
        choices=["slow", "normal", "fast"],
        default="normal",
        help="Simulation speed preset (default: normal)",
    )
    parser.add_argument(
        "--dt", type=float, help="Custom physics timestep in seconds (overrides preset)"
    )
    parser.add_argument("--steps", type=int, help="Custom steps per frame (overrides preset)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Get speed settings
    dt, steps_per_frame, desc = SPEED_PRESETS[args.speed]
    if args.dt is not None:
        dt = args.dt
    if args.steps is not None:
        steps_per_frame = args.steps

    # Create simulation
    world = d3x.World()

    # Earth at origin
    world.add_body(
        pos=(0.0, 0.0, 0.0),
        vel=(0.0, 0.0, 0.0),
        mass=5.972e24,
    )

    # Moon in circular orbit
    moon_distance = 384400e3  # meters
    moon_velocity = 1022.0  # m/s (circular orbit velocity)
    world.add_body(
        pos=(moon_distance, 0.0, 0.0),
        vel=(0.0, moon_velocity, 0.0),
        mass=7.342e22,
    )

    print("D3X Earth-Moon Simulation")
    print(f"Speed: {args.speed} - {desc}")
    print(f"  dt={dt}s, {steps_per_frame} steps/frame")
    print()
    print("Controls:")
    print("  Left drag      - Orbit camera")
    print("  Right/Mid drag - Pan")
    print("  Scroll         - Zoom")
    print("  Click body     - Select & track")
    print("  1-9            - Select body")
    print("  0/Backspace    - Deselect")
    print("  R              - Reset view")
    print("  G              - Toggle grid")
    print("  A              - Toggle axes")
    print("  Space          - Pause")
    print("  ESC            - Exit")
    print()

    with Viewer(world, title="Earth-Moon System", show_fps=True) as viewer:
        while viewer.running:
            if not viewer.paused:
                for _ in range(steps_per_frame):
                    d3x.step_rk4(world, dt)

            viewer.update()


if __name__ == "__main__":
    main()
