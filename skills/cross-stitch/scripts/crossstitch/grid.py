"""Shared internal representation of a cross-stitch pattern.

A pattern is a palette-indexed bitmap: ``cells[y, x]`` holds the palette index
of the full stitch at that position, and 0 means "no stitch" (bare cloth),
matching the OXS convention that palette index 0 is the cloth colour.

Backstitches are kept as line segments alongside the grid. Anything the skill
does not model directly (part stitches, knots, beads, vendor-specific
elements) is carried through untouched in ``extras`` so a file can be read,
edited and written back without losing data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import xml.etree.ElementTree as ET

import numpy as np

CLOTH = 0


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.strip().lstrip("#")
    if len(value) != 6:
        raise ValueError(f"expected a 6-digit hex colour, got {value!r}")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def rgb_to_hex(rgb) -> str:
    r, g, b = (int(round(c)) for c in rgb)
    return f"{r:02X}{g:02X}{b:02X}"


@dataclass
class PaletteEntry:
    """One thread colour. ``number`` is e.g. "DMC 310"; "cloth" for index 0."""

    number: str
    name: str
    color: str  # 6-digit hex, no '#'
    strands: int = 2
    attrs: dict = field(default_factory=dict)  # other attributes, preserved on write
    children: list = field(default_factory=list)  # e.g. <blend> elements, preserved on write

    @property
    def rgb(self) -> tuple[int, int, int]:
        return hex_to_rgb(self.color)


@dataclass
class Backstitch:
    """A line from (x1, y1) to (x2, y2) in grid-corner coordinates.

    Coordinates are on the grid lines, so a backstitch along the top edge of
    cell (0, 0) runs from (0, 0) to (1, 0). Values may be fractional.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    palindex: int
    objecttype: str = "backstitch"
    attrs: dict = field(default_factory=dict)


@dataclass
class Pattern:
    cells: np.ndarray  # shape (height, width), integer palette indices
    palette: list[PaletteEntry]
    backstitches: list[Backstitch] = field(default_factory=list)
    properties: dict = field(default_factory=dict)
    # Elements the skill does not model (partstitches, ornaments, ...),
    # kept as parsed XML in their original order.
    extras: list[ET.Element] = field(default_factory=list)

    @classmethod
    def blank(cls, width: int, height: int, cloth_hex: str = "FFFFFF", title: str = "") -> "Pattern":
        cells = np.zeros((height, width), dtype=np.int32)
        palette = [PaletteEntry(number="cloth", name="cloth", color=cloth_hex)]
        props = {"charttitle": title} if title else {}
        return cls(cells=cells, palette=palette, properties=props)

    @property
    def width(self) -> int:
        return int(self.cells.shape[1])

    @property
    def height(self) -> int:
        return int(self.cells.shape[0])

    @property
    def title(self) -> str:
        return self.properties.get("charttitle", "")

    def stitch_counts(self) -> dict[int, int]:
        """Full-stitch count per palette index, excluding cloth."""
        idx, counts = np.unique(self.cells, return_counts=True)
        return {int(i): int(c) for i, c in zip(idx, counts) if i != CLOTH}

    def backstitch_counts(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for b in self.backstitches:
            out[b.palindex] = out.get(b.palindex, 0) + 1
        return out

    def find_color(self, number: str) -> Optional[int]:
        """Palette index of a thread by number (e.g. "DMC 310"), ignoring spacing/case."""
        key = normalise_thread_number(number)
        for i, entry in enumerate(self.palette):
            if i != CLOTH and normalise_thread_number(entry.number) == key:
                return i
        return None

    def add_color(self, entry: PaletteEntry) -> int:
        """Return the index of ``entry``, appending it only if not already present."""
        existing = self.find_color(entry.number)
        if existing is not None:
            return existing
        self.palette.append(entry)
        return len(self.palette) - 1

    def rgb_image(self) -> np.ndarray:
        """(height, width, 3) uint8 array of stitch colours; cloth shows as cloth colour."""
        lut = np.array([e.rgb for e in self.palette], dtype=np.uint8)
        return lut[self.cells]


def normalise_thread_number(number: str) -> str:
    """'DMC    310' and 'dmc 310' both become 'DMC 310'."""
    return " ".join(number.split()).upper()
