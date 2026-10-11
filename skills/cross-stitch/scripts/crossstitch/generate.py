"""Image -> cross-stitch pattern.

Pipeline: load -> crop -> mark background as unstitched -> k-means colour
reduction -> match to DMC -> one colour per stitch -> clean up stray
stitches -> Pattern.

Two ways of picking each stitch's colour, chosen automatically:

- "illustration" (flat-colour artwork, clip art, logos): every source pixel
  is matched to a thread first, then each stitch takes the thread covering
  most of its square. No in-between colours are invented at edges, so
  outlines stay crisp and colour slots are not wasted on blends.
- "photo": each stitch is the average colour of its square, then matched.
  Averaging suits gradients and texture.
"""

from __future__ import annotations

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
# How far (RGB units) an edge cell may sit from the background-to-subject
# blend line and still be treated as a blend. Area averaging mixes colours
# in RGB, so blends lie on straight lines in RGB, not in Lab.
HALO_RESIDUAL = 14.0
# Illustration mode samples each stitch on a grid of up to this many pixels square.
SAMPLES_PER_STITCH = 8
STYLES = ("auto", "illustration", "photo")
# An isolated stitch is merged into its neighbours only if they are this close in colour.
DESPECKLE_MAX_DELTA = 12.0
# A thread used for fewer stitches than this share of the pattern (or
# MIN_THREAD_STITCHES, whichever is larger) is folded into the closest
# remaining thread, if that thread is within RARE_MERGE_MAX_DELTA.
MIN_THREAD_SHARE = 0.005
MIN_THREAD_STITCHES = 10
RARE_MERGE_MAX_DELTA = 15.0


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
    # "auto" picks "illustration" for flat-colour artwork and "photo" otherwise.
    style: str = "auto"


@dataclass
class GenerateResult:
    pattern: Pattern
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def load_image(source: Union[str, Image.Image]) -> Image.Image:
    img = Image.open(source) if isinstance(source, str) else source
    img = ImageOps.exif_transpose(img)
    return img.convert("RGBA")


def grid_height(img: Image.Image, width: int) -> int:
    return max(1, round(width * img.height / img.width))


def detect_style(img: Image.Image) -> str:
    """'illustration' for flat-colour artwork, 'photo' otherwise.

    Flat artwork has no colour change between most neighbouring pixels, so
    the median neighbour difference is near zero; photos have texture and
    noise everywhere.
    """
    small = img.convert("RGB")
    small.thumbnail((400, 400))
    lab = dmc_match.rgb_to_lab(np.asarray(small, dtype=np.float64))
    dx = np.linalg.norm(np.diff(lab, axis=1), axis=2)[:-1, :]
    dy = np.linalg.norm(np.diff(lab, axis=0), axis=2)[:, :-1]
    step = np.maximum(dx, dy)
    return "illustration" if float(np.median(step)) < 0.8 else "photo"


def downsample(img: Image.Image, width: int, samples: int = 1) -> np.ndarray:
    """Resize to ``width`` stitches (square stitches, aspect kept), with
    ``samples`` x ``samples`` pixels per stitch. Returns RGBA uint8 array."""
    height = grid_height(img, width)
    width, height = width * samples, height * samples
    # Area averaging: each stitch is the mean colour of the pixels it covers.
    # Unlike sharpening filters, it never overshoots at edges, so outline
    # cells are clean blends that clean_halo() can recognise.
    small = img.resize((width, height), Image.Resampling.BOX)
    return np.asarray(small, dtype=np.uint8)


# ---------------------------------------------------------------- background


def _border_mask(shape) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = True
    return m


def _dilate(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _flood_from_border(similar: np.ndarray) -> np.ndarray:
    """Cells connected to the image edge through ``similar`` cells (4-connected)."""
    out = similar & _border_mask(similar.shape)
    while True:
        grown = _dilate(out) & similar
        if np.array_equal(grown, out):
            return out
        out = grown


def background_mask(
    lab: np.ndarray, opaque: np.ndarray, mode: str, fabric_rgb
) -> tuple[np.ndarray, Optional[str], Optional[np.ndarray]]:
    """Return (mask of cells to leave unstitched, note for the user or None,
    Lab colour of the removed background or None)."""
    empty = ~opaque
    if mode == "keep":
        return empty, None, None
    border_lab = lab[_border_mask(opaque.shape) & opaque]
    if not len(border_lab):
        return empty, None, None
    if len(border_lab) > 600:
        border_lab = border_lab[np.linspace(0, len(border_lab) - 1, 600).astype(int)]

    # Dominant border colour: the border pixel with the most near neighbours on the border.
    d = dmc_match.delta_e2000(border_lab[:, None, :], border_lab[None, :, :])
    support = (d < BACKGROUND_TOLERANCE).sum(axis=1)
    bg_lab = border_lab[support.argmax()]
    share = support.max() / len(border_lab)
    if share < 0.5:
        return empty, None, None

    fabric_lab = dmc_match.rgb_to_lab(np.asarray(fabric_rgb))
    if mode == "auto":
        if float(dmc_match.delta_e2000(bg_lab, fabric_lab)) > BACKGROUND_TOLERANCE:
            return empty, (
                "The image background is not close to the fabric colour, so it has been "
                "stitched. Use the 'remove' background option to leave it unstitched instead."
            ), None
    dist = dmc_match.delta_e2000(lab, bg_lab)
    bg = _flood_from_border(opaque & (dist < BACKGROUND_TOLERANCE))
    return empty | bg, None, bg_lab


def clean_halo(rgb: np.ndarray, unstitched: np.ndarray, background_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fix the one-stitch ring where downsampling blended the subject into the background.

    A cell touching the background whose colour lies on the line between the
    background and a neighbouring subject colour is a blend. If it is mostly
    background it is left unstitched; otherwise it takes the subject colour.
    Works in RGB because area averaging blends colours in RGB.
    Returns updated (rgb, unstitched) arrays.
    """
    h, w = unstitched.shape
    rgb = rgb.copy()
    background_rgb = np.asarray(background_rgb, dtype=np.float64)
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
            c = rgb[y, x]
            best = None
            # Compare against subject colours up to two cells away.
            reach = [(y + dy, x + dx) for dy in range(-2, 3) for dx in range(-2, 3) if (dy or dx) and 0 <= y + dy < h and 0 <= x + dx < w]
            for p in reach:
                if bg[p]:
                    continue
                s = rgb[p]
                seg = s - background_rgb
                denom = float(seg @ seg)
                if denom < 1e-6:
                    continue
                t = float((c - background_rgb) @ seg) / denom
                if not 0.0 < t < 1.0:
                    continue
                residual = float(np.linalg.norm(c - (background_rgb + t * seg)))
                if residual < HALO_RESIDUAL and (best is None or residual < best[0]):
                    best = (residual, t, s)
            if best is not None:
                changes.append((y, x, best[1], best[2]))
    for y, x, t, s in changes:
        if t < 0.5:
            unstitched[y, x] = True
        else:
            rgb[y, x] = s
    return rgb, unstitched


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


def fold_rare_threads(cells: np.ndarray, palette_lab: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Merge threads used for only a handful of stitches into a similar thread.

    A colour that appears a few times and looks like another colour already
    in the pattern costs the stitcher a skein for almost nothing. Rare
    colours that are clearly different (an eye, a highlight) are kept.
    Returns (cells, palette indices that were folded away).
    """
    cells = cells.copy()
    folded: list[int] = []
    while True:
        counts = np.bincount(cells.ravel(), minlength=len(palette_lab))
        counts[CLOTH] = 0
        total = int(counts.sum())
        if not total:
            return cells, folded
        limit = max(MIN_THREAD_STITCHES, MIN_THREAD_SHARE * total)
        used = [i for i in range(1, len(palette_lab)) if counts[i] > 0]
        best = None
        for i in sorted(used, key=lambda i: counts[i]):
            if counts[i] >= limit or len(used) < 2:
                break
            others = [j for j in used if j != i]
            d = dmc_match.delta_e2000(palette_lab[i][None, :], palette_lab[others])
            j = int(np.argmin(d))
            if float(d[j]) <= RARE_MERGE_MAX_DELTA:
                best = (i, others[j])
                break
        if best is None:
            return cells, folded
        cells[cells == best[0]] = best[1]
        folded.append(best[0])


# ---------------------------------------------------------------- main entry


def generate_from_image(source: Union[str, Image.Image], options: Optional[GenerateOptions] = None) -> GenerateResult:
    opts = options or GenerateOptions()
    if opts.width < 5:
        raise ValueError("width must be at least 5 stitches")
    if not 1 <= opts.max_colors <= 60:
        raise ValueError("max_colors must be between 1 and 60")
    if opts.background not in ("auto", "remove", "keep"):
        raise ValueError("background must be 'auto', 'remove' or 'keep'")
    if opts.style not in STYLES:
        raise ValueError(f"style must be one of {', '.join(STYLES)}")

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
    style = detect_style(img) if opts.style == "auto" else opts.style
    w, h = opts.width, grid_height(img, opts.width)
    fabric_rgb = hex_to_rgb(opts.fabric)

    pattern = Pattern.blank(w, h, cloth_hex=opts.fabric.upper(), title=opts.title)
    pattern.properties["stitchesperinch"] = str(opts.stitches_per_inch)

    if style == "illustration":
        cells, threads, note = _cells_illustration(img, w, h, opts, fabric_rgb)
    else:
        cells, threads, note = _cells_photo(img, w, h, opts, fabric_rgb)
    if note:
        warnings.append(note)
    if cells is None:
        warnings.append("Every cell was treated as background, so the pattern is empty.")
        return GenerateResult(pattern, warnings, _stats(pattern, src_size, opts, 0, style))

    # Order palette by usage (most-used first).
    counts = np.bincount(cells[cells >= 0], minlength=len(threads))
    order = [i for i in np.argsort(-counts, kind="stable") if counts[i] > 0]
    lut = np.zeros(len(threads) + 1, dtype=np.int32)  # index -1 (unstitched) -> cloth
    for new, old in enumerate(order):
        lut[old] = new + 1
        pattern.palette.append(threads[old].palette_entry())
    pattern.cells = lut[cells]

    removed = 0
    pal_lab = dmc_match.rgb_to_lab(np.array([e.rgb for e in pattern.palette], dtype=np.float64))
    if opts.despeckle:
        pattern.cells, removed = despeckle(pattern.cells, pal_lab)
    pattern.cells, folded = fold_rare_threads(pattern.cells, pal_lab)
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
    return GenerateResult(pattern, warnings, _stats(pattern, src_size, opts, removed, style))


def _cells_photo(img, w, h, opts, fabric_rgb):
    """Average colour per stitch. Returns (cells with -1 for unstitched, threads, note)."""
    rgba = downsample(img, w)
    rgb = rgba[..., :3].astype(np.float64)
    opaque = rgba[..., 3] >= 128
    lab = dmc_match.rgb_to_lab(rgb)
    unstitched, note, bg_lab = background_mask(lab, opaque, opts.background, fabric_rgb)
    if opts.clean_edges and unstitched.any():
        # Edges blend towards the colour that was removed; transparent areas
        # carry no colour, so use the fabric for them.
        removed = unstitched & opaque
        edge_rgb = rgb[removed].mean(axis=0) if removed.any() else np.asarray(fabric_rgb, dtype=np.float64)
        rgb, unstitched = clean_halo(rgb, unstitched, edge_rgb)
        lab = dmc_match.rgb_to_lab(rgb)
    stitched = ~unstitched
    if not stitched.any():
        return None, [], note
    pix = lab[stitched]
    threads = choose_threads(pix, opts.max_colors, opts.metric, opts.seed)
    cells = np.full((h, w), -1, dtype=np.int32)
    cells[stitched] = assign(pix, threads)
    return cells, threads, note


def _cells_illustration(img, w, h, opts, fabric_rgb):
    """Majority thread per stitch. Returns (cells with -1 for unstitched, threads, note)."""
    n = max(1, min(SAMPLES_PER_STITCH, img.width // w, img.height // h))
    rgba = downsample(img, w, samples=n)
    opaque = rgba[..., 3] >= 128
    lab = dmc_match.rgb_to_lab(rgba[..., :3].astype(np.float64))
    unstitched, note, _ = background_mask(lab, opaque, opts.background, fabric_rgb)
    stitched = ~unstitched
    if not stitched.any():
        return None, [], note

    pix = lab[stitched]
    if len(pix) > 60000:
        pix_sample = pix[np.random.default_rng(opts.seed).choice(len(pix), 60000, replace=False)]
    else:
        pix_sample = pix
    threads = choose_threads(pix_sample, opts.max_colors, opts.metric, opts.seed)
    labels = np.full(stitched.shape, -1, dtype=np.int32)
    labels[stitched] = assign(pix, threads)

    # Vote within each stitch's n x n block; slot 0 counts unstitched pixels.
    k = len(threads) + 1
    ys, xs = np.indices(labels.shape)
    cell_id = (ys // n) * w + (xs // n)
    votes = np.bincount((cell_id * k + labels + 1).ravel(), minlength=h * w * k).reshape(h, w, k)
    winner = votes.argmax(axis=2) - 1
    if not (winner >= 0).any():
        return None, [], note
    return winner.astype(np.int32), threads, note


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


def _stats(pattern: Pattern, src_size, opts: GenerateOptions, despeckled: int, style: str) -> dict:
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
        "style": style,
        "threads": [
            {"index": i, "number": e.number, "name": e.name, "hex": e.color, "stitches": counts.get(i, 0)}
            for i, e in enumerate(pattern.palette)
            if i != CLOTH
        ],
    }
