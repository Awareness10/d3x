#!/usr/bin/env python3
"""Inner solar system visualization example."""

import argparse

import d3x
from d3x.viz import Viewer

SPEED_PRESETS = {
    "slow": (43200.0, 1, "1 Earth year in ~12 min"),
    "normal": (86400.0, 2, "1 Earth year in ~3 min"),
    "fast": (172800.0, 4, "1 Earth year in ~45 sec"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Inner solar system simulation")
    parser.add_argument(
        "--speed",
        "-s",
        choices=["slow", "normal", "fast"],
        default="normal",
        help="Simulation speed preset",
    )
    parser.add_argument("--dt", type=float, help="Custom timestep (seconds)")
    parser.add_argument("--steps", type=int, help="Custom steps per frame")
    return parser.parse_args()


def main():
    args = parse_args()
    dt, steps_per_frame, desc = SPEED_PRESETS[args.speed]
    if args.dt:
        dt = args.dt
    if args.steps:
        steps_per_frame = args.steps

    world = d3x.World()

    # Sun at origin
    world.add_body(pos=(0, 0, 0), vel=(0, 0, 0), mass=d3x.constants.M_SUN)

    # Planetary data: (name, semi-major axis [AU], orbital velocity [m/s], mass [kg])
    planets = [
        ("Mercury", 0.387, 47870, 3.285e23),
        ("Venus", 0.723, 35020, 4.867e24),
        ("Earth", 1.000, 29780, 5.972e24),
        ("Mars", 1.524, 24070, 6.417e23),
    ]

    for _name, au, vel, mass in planets:
        r = au * d3x.constants.AU
        world.add_body(pos=(r, 0, 0), vel=(0, vel, 0), mass=mass)

    print("D3X Inner Solar System")
    print(f"Bodies: Sun + {len(planets)} planets")
    print(f"Speed: {args.speed} - {desc}")
    print()
    print("Controls: Left-drag=orbit, Scroll=zoom, R=reset, Space=pause, ESC=exit")
    print()

    with Viewer(world, title="Inner Solar System", trail_length=800, show_fps=True) as viewer:
        while viewer.running:
            if not viewer.paused:
                for _ in range(steps_per_frame):
                    d3x.step_rk4(world, dt)

            viewer.update()


if __name__ == "__main__":
    main()
