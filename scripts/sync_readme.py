#!/usr/bin/env python3
"""Sync README.md sections with actual code.

Sections between <!-- BINDING_NAME --> and <!-- /BINDING_NAME --> are
automatically regenerated from the codebase.

Usage:
    python scripts/sync_readme.py         # Update README.md
    python scripts/sync_readme.py --check # Verify README is in sync
"""

import re
import subprocess
import sys
from pathlib import Path

from api_gen import generate_api, generate_constants

ROOT = Path(__file__).parents[1]
README_PATH = ROOT / "README.md"
FEATURES_DIR = ROOT / "tests" / "features"


def generate_features() -> str:
    """Generate feature summary from Gherkin files."""
    lines = ["| Feature | Scenarios | Description |", "|---------|-----------|-------------|"]

    feature_info = [
        ("world.feature", "World Container", "Body management, SoA arrays, zero-copy numpy"),
        ("gravity.feature", "Gravity", "Inverse-square law, Newton's 3rd, superposition"),
        ("integrators.feature", "Integrators", "RK4, DOPRI54 adaptive, leapfrog symplectic"),
        ("conservation.feature", "Conservation", "Energy, angular momentum, integrator comparison"),
    ]

    for filename, name, desc in feature_info:
        path = FEATURES_DIR / filename
        if path.exists():
            content = path.read_text()
            scenario_count = len(re.findall(r"^\s*Scenario:", content, re.MULTILINE))
            lines.append(f"| {name} | {scenario_count} | {desc} |")

    return "\n".join(lines)


def generate_install() -> str:
    """Generate installation instructions."""
    return """```bash
# Install with uv (recommended)
uv sync --dev

# Alternative legacy install (pip)
pip install -e ".[dev]"

# Run visualization example
uv run python examples/earth_moon.py
```"""


def generate_integrators() -> str:
    """Generate integrators comparison table."""
    return """| Integrator | Type | Best For |
|------------|------|----------|
| `step_rk4(world, dt)` | Fixed-step | General purpose, smooth trajectories |
| `step_dopri54(world, dt, tol)` | Adaptive | Variable dynamics, close encounters |
| `step_leapfrog(world, dt)` | Symplectic | Long-term stability, energy conservation |"""


def generate_dependencies() -> str:
    """Generate top-level dependencies from pip."""

    cmd = ["uv", "pip", "list", "--format=freeze"]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )

    deps = result.stdout.strip().split("\n")

    def remove_version(x: str) -> str:
        return x.split("==")[0]

    lines = []
    for line in deps:
        package_name = remove_version(line)
        lines.append(f"* {package_name}")

    return "\n".join(lines)


# Registry of binding generators
GENERATORS = {
    "CONSTANTS": generate_constants,
    "API": generate_api,
    "FEATURES": generate_features,
    "INSTALL": generate_install,
    "INTEGRATORS": generate_integrators,
    "DEPENDENCIES": generate_dependencies,
}


def sync_readme(check_only: bool = False) -> bool:
    """Sync README.md with generated content.

    Args:
        check_only: If True, don't write changes, just check if in sync.

    Returns:
        True if README is in sync (or was updated), False if out of sync.
    """
    content = README_PATH.read_text()
    original = content

    # Pattern: <!-- NAME -->\n...content...\n<!-- /NAME -->
    pattern = r"(<!-- ([A-Z_]+) -->\n)(.*?)(<!-- /\2 -->)"

    def replacer(match):
        prefix = match.group(1)
        name = match.group(2)
        suffix = match.group(4)

        if name in GENERATORS:
            new_content = GENERATORS[name]()
            return f"{prefix}{new_content}\n{suffix}"
        return match.group(0)

    content = re.sub(pattern, replacer, content, flags=re.DOTALL)

    if content == original:
        print("README.md is in sync")
        return True

    if check_only:
        print("README.md is out of sync. Run: python scripts/sync_readme.py")
        return False

    README_PATH.write_text(content)
    print("README.md updated")
    return True


if __name__ == "__main__":
    check_only = "--check" in sys.argv
    success = sync_readme(check_only)
    sys.exit(0 if success else 1)
