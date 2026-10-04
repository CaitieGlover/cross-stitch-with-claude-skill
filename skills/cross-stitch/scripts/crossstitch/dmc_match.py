"""Nearest-thread lookup.

Colour distance defaults to CIEDE2000 (``metric="lab"``), which tracks how
different two colours look far better than straight RGB distance. Plain
Euclidean RGB distance is available with ``metric="rgb"``.

As a rough guide to CIEDE2000 values: under 2 is hard to tell apart, 2-5 is
a noticeable but close match, 5-10 is clearly different, and over 10 reads
as a different colour.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Optional, Sequence

import numpy as np

from .grid import PaletteEntry, hex_to_rgb, normalise_thread_number

_DATA = os.path.join(os.path.dirname(__file__), "data", "dmc.csv")


@dataclass(frozen=True)
class Thread:
    number: str  # bare DMC number, e.g. "310", "B5200", "Ecru"
    name: str
    hex: str

    @property
    def rgb(self) -> tuple[int, int, int]:
        return hex_to_rgb(self.hex)

    @property
    def label(self) -> str:
        return f"DMC {self.number}"

    def palette_entry(self, strands: int = 2) -> PaletteEntry:
        return PaletteEntry(number=self.label, name=self.name, color=self.hex, strands=strands)


@lru_cache(maxsize=1)
def dmc_threads() -> tuple[Thread, ...]:
    with open(_DATA, newline="") as f:
        return tuple(Thread(r["number"], r["name"], r["hex"]) for r in csv.DictReader(f))


def lookup(number: str) -> Thread:
    """Find a DMC thread by number. Accepts '310', 'DMC 310', 'ecru', 'white', 'b5200'."""
    key = normalise_thread_number(str(number))
    if key.startswith("DMC "):
        key = key[4:]
    if key == "BLANC":
        key = "WHITE"
    for t in dmc_threads():
        if t.number.upper() == key:
            return t
    raise KeyError(f"unknown DMC thread {number!r}")


# ---------------------------------------------------------------- colour maths


def rgb_to_lab(rgb) -> np.ndarray:
    """sRGB (0-255, shape (..., 3)) to CIE L*a*b* under D65."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = c @ m.T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def delta_e2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 difference; inputs broadcast against each other, shape (..., 3)."""
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cbar7 = ((C1 + C2) / 2) ** 7
    G = 0.5 * (1 - np.sqrt(Cbar7 / (Cbar7 + 25.0**7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)
    dhp = np.where(C1p * C2p == 0, 0, dhp)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2))

    Lbarp = (L1 + L2) / 2
    Cbarp = (C1p + C2p) / 2
    hsum = h1p + h2p
    hbarp = np.where(np.abs(h1p - h2p) > 180, (hsum + 360) / 2, hsum / 2)
    hbarp = np.where(hbarp >= 360, hbarp - 360, hbarp)
    hbarp = np.where(C1p * C2p == 0, hsum, hbarp)

    T = (
        1
        - 0.17 * np.cos(np.radians(hbarp - 30))
        + 0.24 * np.cos(np.radians(2 * hbarp))
        + 0.32 * np.cos(np.radians(3 * hbarp + 6))
        - 0.20 * np.cos(np.radians(4 * hbarp - 63))
    )
    dtheta = 30 * np.exp(-(((hbarp - 275) / 25) ** 2))
    Cbarp7 = Cbarp**7
    Rc = 2 * np.sqrt(Cbarp7 / (Cbarp7 + 25.0**7))
    Sl = 1 + 0.015 * (Lbarp - 50) ** 2 / np.sqrt(20 + (Lbarp - 50) ** 2)
    Sc = 1 + 0.045 * Cbarp
    Sh = 1 + 0.015 * Cbarp * T
    Rt = -np.sin(np.radians(2 * dtheta)) * Rc
    return np.sqrt(
        (dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh)
    )


def distance_matrix(colours_rgb, candidates_rgb, metric: str = "lab") -> np.ndarray:
    """Pairwise distances, shape (len(colours), len(candidates))."""
    a = np.asarray(colours_rgb, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(candidates_rgb, dtype=np.float64).reshape(-1, 3)
    if metric == "rgb":
        return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    if metric == "lab":
        return delta_e2000(rgb_to_lab(a)[:, None, :], rgb_to_lab(b)[None, :, :])
    raise ValueError(f"unknown metric {metric!r}; use 'lab' or 'rgb'")


def nearest(
    colours_rgb,
    candidates: Optional[Sequence[Thread]] = None,
    metric: str = "lab",
) -> tuple[list[Thread], np.ndarray]:
    """For each colour, the nearest thread among ``candidates`` (default: all DMC).

    Returns (threads, distances).
    """
    cands = list(candidates) if candidates is not None else list(dmc_threads())
    if not cands:
        raise ValueError("no candidate threads to match against")
    d = distance_matrix(colours_rgb, [t.rgb for t in cands], metric)
    best = d.argmin(axis=1)
    return [cands[i] for i in best], d[np.arange(len(best)), best]


def nearest_one(rgb, candidates: Optional[Iterable[Thread]] = None, metric: str = "lab") -> tuple[Thread, float]:
    threads, dist = nearest([rgb], list(candidates) if candidates is not None else None, metric)
    return threads[0], float(dist[0])
