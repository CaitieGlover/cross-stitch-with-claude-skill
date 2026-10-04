"""Render a pattern to a PNG so it can be checked by eye.

The preview shows the chart (one square per stitch, grid lines every stitch
and heavier lines every 10), backstitches, and a thread key with stitch
counts. If the source image is given it is shown alongside for comparison.
"""

from __future__ import annotations

from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from .grid import CLOTH, Pattern

GRID_LIGHT = (200, 200, 200)
GRID_HEAVY = (90, 90, 90)
TEXT = (30, 30, 30)
PAGE = (255, 255, 255)


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def render_chart(pattern: Pattern, cell: int = 10) -> Image.Image:
    w, h = pattern.width, pattern.height
    img = Image.fromarray(pattern.rgb_image(), "RGB").resize((w * cell, h * cell), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (w * cell + 1, h * cell + 1), PAGE)
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    if cell >= 4:
        for x in range(w + 1):
            if x % 10:
                draw.line([(x * cell, 0), (x * cell, h * cell)], fill=GRID_LIGHT)
        for y in range(h + 1):
            if y % 10:
                draw.line([(0, y * cell), (w * cell, y * cell)], fill=GRID_LIGHT)
    for x in range(0, w + 1, 10):
        draw.line([(x * cell, 0), (x * cell, h * cell)], fill=GRID_HEAVY)
    for y in range(0, h + 1, 10):
        draw.line([(0, y * cell), (w * cell, y * cell)], fill=GRID_HEAVY)
    draw.line([(w * cell, 0), (w * cell, h * cell)], fill=GRID_HEAVY)
    draw.line([(0, h * cell), (w * cell, h * cell)], fill=GRID_HEAVY)
    line_w = max(2, cell // 4)
    for b in pattern.backstitches:
        colour = pattern.palette[b.palindex].rgb
        draw.line([(b.x1 * cell, b.y1 * cell), (b.x2 * cell, b.y2 * cell)], fill=colour, width=line_w)
    return canvas


def render_key(pattern: Pattern, row_height: int = 22) -> Image.Image:
    counts = pattern.stitch_counts()
    bs = pattern.backstitch_counts()
    rows = [i for i in range(len(pattern.palette)) if i != CLOTH]
    font = _font(14)
    width = 360
    img = Image.new("RGB", (width, row_height * (len(rows) + 2)), PAGE)
    draw = ImageDraw.Draw(img)
    draw.text((8, 4), f"{pattern.width} x {pattern.height} stitches, {len(rows)} colours", fill=TEXT, font=font)
    for n, i in enumerate(rows, start=1):
        e = pattern.palette[i]
        y = n * row_height + 4
        draw.rectangle([8, y, 8 + row_height - 6, y + row_height - 6], fill=e.rgb, outline=GRID_HEAVY)
        label = f"{' '.join(e.number.split())}  {e.name}  ({counts.get(i, 0)}"
        label += f", {bs[i]} backstitch)" if bs.get(i) else ")"
        draw.text((8 + row_height + 4, y), label, fill=TEXT, font=font)
    return img


def render_preview(pattern: Pattern, source: Optional[Image.Image] = None, cell: Optional[int] = None) -> Image.Image:
    """Chart with thread key underneath; the source image (if given) on the left."""
    if cell is None:
        cell = max(4, min(14, 1000 // max(pattern.width, pattern.height)))
    chart = render_chart(pattern, cell)
    key = render_key(pattern)
    panels = []
    if source is not None:
        src = source.convert("RGBA")
        scale = chart.height / src.height
        src = src.resize((max(1, round(src.width * scale)), chart.height), Image.Resampling.LANCZOS)
        flat = Image.new("RGB", src.size, PAGE)
        flat.paste(src, mask=src.split()[3])
        panels.append(flat)
    panels.append(chart)
    gap = 16
    top_w = sum(p.width for p in panels) + gap * (len(panels) + 1)
    total_w = max(top_w, key.width + 2 * gap)
    total_h = gap + chart.height + gap + key.height + gap
    out = Image.new("RGB", (total_w, total_h), PAGE)
    x = gap
    for p in panels:
        out.paste(p, (x, gap))
        x += p.width + gap
    out.paste(key, (gap, gap + chart.height + gap))
    return out
