"""
Physical constants (SI units)
"""

from __future__ import annotations

__all__ = [
    "AU",
    "DAY",
    "G",
    "MU_EARTH",
    "MU_SUN",
    "M_EARTH",
    "M_MARS",
    "M_MOON",
    "M_SUN",
]

AU: float = 149_597_870_700.0
"""@unit m"""

DAY: float = 86400.0
"""@unit s"""

G: float = 6.6743e-11
"""@unit m³/(kg·s²)"""

MU_EARTH: float = 398_600_542_309_999.94
"""@unit m³/s²"""

MU_SUN: float = 1.3274648755999998e20
"""@unit m³/s²"""

M_EARTH: float = 5.97217e24
"""@unit kg"""

M_MARS: float = 6.4171e23
"""@unit kg"""

M_MOON: float = 7.342e22
"""@unit kg"""

M_SUN: float = 1.98892e30
"""@unit kg"""
