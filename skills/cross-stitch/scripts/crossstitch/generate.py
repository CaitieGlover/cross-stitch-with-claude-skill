"""Image -> cross-stitch pattern.

Pipeline: load -> crop -> downsample to the stitch grid -> mark background
as unstitched -> k-means colour reduction -> match to DMC -> clean up stray
stitches -> Pattern.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
from PIL import Image, ImageOps

from . import dmc_match
from .dmc_match import Thread
from .grid import CLOTH, PaletteEntry, Pattern, hex_to_rgb

# Below these widths, warn that small features will not survive.
DETAIL_WARN_WIDTH = 60
SEVERE_DETAIL_WARN_WIDTH = 35
# CIEDE2000 distance under which a colour counts as "the fabric".
BACKGROUND_TOLERANCE = 10.0
# How far (Lab units) an edge cell may sit from the background-to-subject
# blend line and still be treated as a blend.
HALO_RESIDUAL = 6.0
# An isolated stitch is merged into its neighbours only if they are this close in colour.
DESPECKLE_MAX_DELTA = 12.0


@dataclass
class GenerateOptions:
    width: int = 80
    max_colors: int = 15
    # "auto": leave the background unstitched if it is close to the fabric colour.
    # "remove": always leave the background (traced from the image edges) unstitched.
    # "keep": stitch everything.
    background: str = "auto"
    fabric: str = "FFFFFF"
    crop: Optional[tuple[int, int, int, int]] = None  # left, top, right, bottom in source pixels
    despeckle: bool = True
    # Tidy the one-stitch blended ring between subject and background.
    clean_edges: bool = True
    metric: str = "lab"
    stitches_per_inch: int = 14
    title: str = ""
    seed: int = 0


@dataclass
class GenerateResult:
    pattern: Pattern
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def load_image(source: Union[str, Image.Image]) -> Image.Image:
    img = Image.open(source) if isinstance(source, str) else source
    img = ImageOps.exif_transpose(img)
    return img.convert("RGBA")


def downsample(img: Image.Image, width: int) -> np.ndarray:
    """Resize to ``width`` stitches (square stitches, aspect kept). Returns RGBA uint8 array."""
    height = max(1, round(width * img.height / img.width))
    # Area averaging: each stitch is the mean colour of the pixels it covers.
    # Unlike sharpening filters, it never overshoots at edges, so outline
    # cells are clean blends that clean_halo() can recognise.
    small = img.resize((width, height), Image.Resampling.BOX)
    return np.asarray(small, dtype=np.uint8)


# ---------------------------------------------------------------- background


def _border_pixels(mask_shape) -> list[tuple[int, int]]:
    h, w = mask_shape
    pts = [(0, x) for x in range(w)] + [(h - 1, x) for x in range(w)]
    pts += [(y, 0) for y in range(h)] + [(y, w - 1) for y in range(h)]
    return pts


def _flood_from_border(similar: np.ndarray) -> np.ndarray:
    """Cells connected to the image edge through ``similar`` cells (4-connected)."""
    h, w = similar.shape
    out = np.zeros_like(similar)
    queue = deque()
    for y, x in _border_pixels(similar.shape):
        if similar[y, x] and not out[y, x]:
            out[y, x] = True
            queue.append((y, x))
    while queue:
        y, x = queue.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and similar[ny, nx] and not out[ny, nx]:
                out[ny, nx] = True
                queue.append((ny, nx))
    return out


def background_mask(lab: np.ndarray, opaque: np.ndarray, mode: str, fabric_rgb) -> tuple[np.ndarray, Optional[str]]:
    """Return (mask of cells to leave unstitched, note for the user or None)."""
    empty = ~opaque
    if mode == "keep":
        return empty, None
    h, w = opaque.shape
    border = [(y, x) for y, x in _border_pixels((h, w)) if opaque[y, x]]
    if not border:
        return empty, None

    border_lab = np.array([lab[y, x] for y, x in border])
    # Dominant border colour: the border pixel with the most near neighbours on the border.
    d = dmc_match.delta_e2000(border_lab[:, None, :], border_lab[None, :, :])
    support = (d < BACKGROUND_TOLERANCE).sum(axis=1)
    bg_lab = border_lab[support.argmax()]
    share = support.max() / len(border)
    if share < 0.5:
        return empty, None

    fabric_lab = dmc_match.rgb_to_lab(np.asarray(fabric_rgb))
    if mode == "auto":
        if float(dmc_match.delta_e2000(bg_lab, fabric_lab)) > BACKGROUND_TOLERANCE:
            return empty, (
                "The image background is not close to the fabric colour, so it has been "
                "stitched. Use the 'remove' background option to leave it unstitched instead."
            )
    dist = dmc_match.delta_e2000(lab, bg_lab)
    bg = _flood_from_border(opaque & (dist < BACKGROUND_TOLERANCE))
    return empty | bg, None


def clean_halo(lab: np.ndarray, unstitched: np.ndarray, background_lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fix the one-stitch ring where downsampling blended the subject into the background.

    A cell touching the background whose colour lies on the line between the
    background and a neighbouring subject colour is a blend. If it is mostly
    background it is left unstitched; otherwise it takes the subject colour.
    Returns updated (lab, unstitched) arrays.
    """
    h, w = unstitched.shape
    lab = lab.copy()
    unstitched = unstitched.copy()
    bg = unstitched
    changes = []
    for y in range(h):
        for x in range(w):
            if bg[y, x]:
                continue
            neigh = [(y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy or dx) and 0 <= y + dy < h and 0 <= x + dx < w]
            if not any(bg[p] for p in neigh):
                continue
            c = lab[y, x]
            best = None
            # Compare against subject colours up to two cells away.
            reach = [(y + dy, x + dx) for dy in range(-2, 3) for dx in range(-2, 3) if (dy or dx) and 0 <= y + dy < h and 0 <= x + dx < w]
            for p in reach:
                if bg[p]:
                    continue
                s = lab[p]
                seg = s - background_lab
                denom = float(seg @ seg)
                if denom < 1e-6:
                    continue
                t = float((c - background_lab) @ seg) / denom
                if not 0.0 < t < 1.0:
                    continue
                residual = float(np.linalg.norm(c - (background_lab + t * seg)))
                if residual < HALO_RESIDUAL and (best is None or residual < best[0]):
                    best = (residual, t, s)
            if best is not None:
                changes.append((y, x, best[1], best[2]))
    for y, x, t, s in changes:
        if t < 0.5:
            unstitched[y, x] = True
        else:
            lab[y, x] = s
    return lab, unstitched


# ---------------------------------------------------------------- quantisation


def assign_nearest_centre(points: np.ndarray, centres: np.ndarray) -> np.ndarray:
    return ((points[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)


def kmeans(points: np.ndarray, k: int, seed: int = 0, iterations: int = 25) -> np.ndarray:
    """Plain k-means with k-means++ seeding. Returns cluster centres, shape (<=k, d)."""
    rng = np.random.default_rng(seed)
    unique = np.unique(points, axis=0)
    if len(unique) <= k:
        return unique.astype(np.float64)
    pts = points.astype(np.float64)
    centres = [pts[rng.integers(len(pts))]]
    d2 = ((pts - centres[0]) ** 2).sum(axis=1)
    for _ in range(1, k):
        if d2.sum() == 0:
            break
        centres.append(pts[rng.choice(len(pts), p=d2 / d2.sum())])
        d2 = np.minimum(d2, ((pts - centres[-1]) ** 2).sum(axis=1))
    centres = np.array(centres)
    for _ in range(iterations):
        labels = assign_nearest_centre(pts, centres)
        new = np.array([pts[labels == i].mean(axis=0) if np.any(labels == i) else centres[i] for i in range(len(centres))])
        if np.allclose(new, centres, atol=1e-3):
            break
        centres = new
    return centres


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    delta = 6 / 29
    f = np.stack([fx, fy, fz], axis=-1)
    xyz = np.where(f > delta, f**3, 3 * delta**2 * (f - 4 / 29)) * np.array([0.95047, 1.0, 1.08883])
    m_inv = np.array(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ]
    )
    c = np.clip(xyz @ m_inv.T, 0, 1)
    c = np.where(c > 0.0031308, 1.055 * c ** (1 / 2.4) - 0.055, 12.92 * c)
    return np.clip(c * 255, 0, 255)


def merge_clusters(centres: np.ndarray, counts: np.ndarray, k: int) -> np.ndarray:
    """Merge clusters pairwise (Ward's criterion) until ``k`` remain.

    Ward's cost grows with both colour distance and cluster size, so similar
    shades of a large area merge long before a small, distinctive colour
    (an eye, a flower centre) is absorbed into its surroundings.
    """
    centres = [c.astype(np.float64) for c in centres]
    counts = [float(n) for n in counts]
    while len(centres) > k:
        best, pair = None, None
        for i in range(len(centres)):
            for j in range(i + 1, len(centres)):
                cost = counts[i] * counts[j] / (counts[i] + counts[j]) * float(((centres[i] - centres[j]) ** 2).sum())
                if best is None or cost < best:
                    best, pair = cost, (i, j)
        i, j = pair
        n = counts[i] + counts[j]
        centres[i] = (centres[i] * counts[i] + centres[j] * counts[j]) / n
        counts[i] = n
        del centres[j], counts[j]
    return np.array(centres)


def choose_threads(lab_pixels: np.ndarray, max_colors: int, metric: str, seed: int) -> list[Thread]:
    # Over-cluster first, then merge down, so small distinctive colours survive.
    k0 = min(max(3 * max_colors, 24), 64)
    centres = kmeans(lab_pixels, k0, seed=seed)
    labels = assign_nearest_centre(lab_pixels, centres)
    counts = np.bincount(labels, minlength=len(centres))
    present = counts > 0
    centres = merge_clusters(centres[present], counts[present], max_colors)
    threads, _ = dmc_match.nearest(_lab_to_rgb(centres), metric=metric)
    unique: list[Thread] = []
    for t in threads:
        if t not in unique:
            unique.append(t)
    return unique


def assign(lab_pixels: np.ndarray, threads: list[Thread]) -> np.ndarray:
    """Index into ``threads`` of the nearest thread (CIEDE2000) for each pixel."""
    thread_lab = dmc_match.rgb_to_lab(np.array([t.rgb for t in threads], dtype=np.float64))
    best = np.zeros(len(lab_pixels), dtype=np.int32)
    chunk = 20000
    for start in range(0, len(lab_pixels), chunk):
        part = lab_pixels[start : start + chunk]
        d = dmc_match.delta_e2000(part[:, None, :], thread_lab[None, :, :])
        best[start : start + chunk] = d.argmin(axis=1)
    return best


# ---------------------------------------------------------------- clean-up


def despeckle(cells: np.ndarray, palette_lab: np.ndarray, max_delta: float = DESPECKLE_MAX_DELTA) -> tuple[np.ndarray, int]:
    """Replace isolated stitches (no same-colour 8-neighbour) by the most common
    neighbouring colour, but only where that colour is within ``max_delta``.

    Deliberate single-stitch details such as an eye highlight are very
    different from their surroundings and are kept.
    """
    h, w = cells.shape
    out = cells.copy()
    changed = 0
    for y in range(h):
        for x in range(w):
            c = cells[y, x]
            if c == CLOTH:
                continue
            neigh = [
                cells[ny, nx]
                for ny in range(max(0, y - 1), min(h, y + 2))
                for nx in range(max(0, x - 1), min(w, x + 2))
                if (ny, nx) != (y, x)
            ]
            if c in neigh:
                continue
            stitched = [n for n in neigh if n != CLOTH]
            if not stitched:
                continue
            values, counts = np.unique(stitched, return_counts=True)
            candidate = int(values[counts.argmax()])
            if float(dmc_match.delta_e2000(palette_lab[c], palette_lab[candidate])) <= max_delta:
                out[y, x] = candidate
                changed += 1
    return out, changed


# ---------------------------------------------------------------- main entry


def generate_from_image(source: Union[str, Image.Image], options: Optional[GenerateOptions] = None) -> GenerateResult:
    opts = options or GenerateOptions()
    if opts.width < 5:
        raise ValueError("width must be at least 5 stitches")
    if not 1 <= opts.max_colors <= 60:
        raise ValueError("max_colors must be between 1 and 60")
    if opts.background not in ("auto", "remove", "keep"):
        raise ValueError("background must be 'auto', 'remove' or 'keep'")

    warnings: list[str] = []
    img = load_image(source)
    src_size = img.size
    if opts.crop:
        img = img.crop(opts.crop)
    if img.width < opts.width:
        warnings.append(
            f"The source image is only {img.width}px wide but the pattern is {opts.width} stitches wide, "
            "so it has been enlarged and may look soft."
        )

    rgba = downsample(img, opts.width)
    h, w = rgba.shape[:2]
    rgb = rgba[..., :3].astype(np.float64)
    opaque = rgba[..., 3] >= 128
    lab = dmc_match.rgb_to_lab(rgb)

    fabric_rgb = hex_to_rgb(opts.fabric)
    unstitched, note = background_mask(lab, opaque, opts.background, fabric_rgb)
    if note:
        warnings.append(note)
    if opts.clean_edges and unstitched.any():
        # Edges blend towards the colour that was removed; transparent areas
        # carry no colour, so use the fabric for them.
        removed = unstitched & opaque
        if removed.any():
            edge_lab = lab[removed].mean(axis=0)
        else:
            edge_lab = dmc_match.rgb_to_lab(np.asarray(fabric_rgb, dtype=np.float64))
        lab, unstitched = clean_halo(lab, unstitched, edge_lab)
    stitched = ~unstitched

    pattern = Pattern.blank(w, h, cloth_hex=opts.fabric.upper(), title=opts.title)
    pattern.properties["stitchesperinch"] = str(opts.stitches_per_inch)
    if not stitched.any():
        warnings.append("Every cell was treated as background, so the pattern is empty.")
        return GenerateResult(pattern, warnings, _stats(pattern, src_size, opts, 0))

    pix = lab[stitched]
    threads = choose_threads(pix, opts.max_colors, opts.metric, opts.seed)
    which = assign(pix, threads)

    # Order palette by usage (most-used first) and build the grid.
    counts = np.bincount(which, minlength=len(threads))
    order = [i for i in np.argsort(-counts, kind="stable") if counts[i] > 0]
    pal_index = {old: new + 1 for new, old in enumerate(order)}
    for i in order:
        pattern.palette.append(threads[i].palette_entry())
    cells = np.zeros((h, w), dtype=np.int32)
    cells[stitched] = np.array([pal_index[i] for i in which], dtype=np.int32)

    removed = 0
    if opts.despeckle:
        pal_lab = dmc_match.rgb_to_lab(np.array([e.rgb for e in pattern.palette], dtype=np.float64))
        cells, removed = despeckle(cells, pal_lab)
    pattern.cells = cells
    _drop_unused(pattern)

    if w < SEVERE_DETAIL_WARN_WIDTH:
        warnings.append(
            f"At {w} stitches wide, faces, eyes and other small features will be lost or reduced to "
            "single stitches. Consider a width of 60 or more if the subject has fine detail."
        )
    elif w < DETAIL_WARN_WIDTH:
        warnings.append(
            f"At {w} stitches wide, fine detail (eyes, facial features, small text) may not read clearly. "
            "Check the preview; a wider pattern keeps more detail."
        )
    return GenerateResult(pattern, warnings, _stats(pattern, src_size, opts, removed))


def _drop_unused(pattern: Pattern) -> None:
    used = set(np.unique(pattern.cells).tolist()) | {b.palindex for b in pattern.backstitches} | {CLOTH}
    keep = [i for i in range(len(pattern.palette)) if i in used]
    if len(keep) == len(pattern.palette):
        return
    remap = np.zeros(len(pattern.palette), dtype=np.int32)
    for new, old in enumerate(keep):
        remap[old] = new
    pattern.cells = remap[pattern.cells]
    pattern.palette = [pattern.palette[i] for i in keep]


def _stats(pattern: Pattern, src_size, opts: GenerateOptions, despeckled: int) -> dict:
    spi = opts.stitches_per_inch
    counts = pattern.stitch_counts()
    return {
        "width": pattern.width,
        "height": pattern.height,
        "source_size": list(src_size),
        "colours": len(pattern.palette) - 1,
        "total_stitches": int(sum(counts.values())),
        "finished_size_in": [round(pattern.width / spi, 1), round(pattern.height / spi, 1)],
        "finished_size_cm": [round(pattern.width / spi * 2.54, 1), round(pattern.height / spi * 2.54, 1)],
        "stitches_per_inch": spi,
        "isolated_stitches_merged": despeckled,
        "threads": [
            {"index": i, "number": e.number, "name": e.name, "hex": e.color, "stitches": counts.get(i, 0)}
            for i, e in enumerate(pattern.palette)
            if i != CLOTH
        ],
    }
