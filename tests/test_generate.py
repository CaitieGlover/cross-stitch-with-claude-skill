import numpy as np
from PIL import Image, ImageDraw

from crossstitch import read_oxs, validate_file, write_oxs
from crossstitch.generate import GenerateOptions, despeckle, generate_from_image, merge_clusters
from crossstitch.dmc_match import rgb_to_lab
from crossstitch.preview import render_preview


def flag_image():
    """Red circle with a small blue dot, on a white background."""
    img = Image.new("RGB", (400, 300), "white")
    draw = ImageDraw.Draw(img)
    draw.ellipse([100, 50, 300, 250], fill=(200, 30, 40))
    draw.ellipse([190, 140, 210, 160], fill=(20, 40, 160))
    return img


def test_generate_basic_shape(tmp_path):
    result = generate_from_image(flag_image(), GenerateOptions(width=40, max_colors=4))
    p = result.pattern
    assert (p.width, p.height) == (40, 30)
    main = p.palette[1]  # most-used thread
    assert main.rgb[0] > 150 and main.rgb[1] < 80 and main.rgb[2] < 80
    # White background left unstitched; corners are cloth.
    assert p.cells[0, 0] == 0 and p.cells[-1, -1] == 0
    # Centre of the circle holds the small blue dot.
    blue = p.palette[p.cells[15, 20]]
    assert blue.rgb[2] > blue.rgb[0]
    out = tmp_path / "flag.oxs"
    write_oxs(p, out)
    assert validate_file(out) == []
    assert np.array_equal(read_oxs(out).cells, p.cells)


def test_no_pale_halo_around_subject():
    p = generate_from_image(flag_image(), GenerateOptions(width=40, max_colors=6)).pattern
    for e in p.palette[1:]:
        r, g, b = e.rgb
        assert not (r > 220 and g > 180 and b > 180), f"pale blend colour {e.number} left in pattern"


def test_background_modes():
    keep = generate_from_image(flag_image(), GenerateOptions(width=20, background="keep")).pattern
    assert (keep.cells > 0).all()
    dark = Image.new("RGB", (200, 200), (10, 20, 80))
    ImageDraw.Draw(dark).ellipse([50, 50, 150, 150], fill=(250, 220, 0))
    auto = generate_from_image(dark, GenerateOptions(width=20))
    assert (auto.pattern.cells > 0).all()
    assert any("not close to the fabric" in w for w in auto.warnings)
    removed = generate_from_image(dark, GenerateOptions(width=20, background="remove")).pattern
    assert removed.cells[0, 0] == 0 and removed.cells[10, 10] > 0


def test_transparent_pixels_are_unstitched():
    img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    ImageDraw.Draw(img).rectangle([25, 25, 75, 75], fill=(0, 128, 0, 255))
    p = generate_from_image(img, GenerateOptions(width=20)).pattern
    assert p.cells[0, 0] == 0 and p.cells[10, 10] > 0
    assert len(p.palette) == 2


def test_palette_respects_max_colors():
    rng = np.random.default_rng(1)
    noise = Image.fromarray(rng.integers(0, 255, (60, 60, 3), dtype=np.uint8))
    p = generate_from_image(noise, GenerateOptions(width=30, max_colors=8, background="keep")).pattern
    assert 1 <= len(p.palette) - 1 <= 8
    used = set(np.unique(p.cells).tolist()) - {0}
    assert used == set(range(1, len(p.palette)))


def test_small_distinct_colour_survives_merge():
    # Many near-identical browns, one small green cluster.
    browns = np.array([[40, 10, 20], [42, 11, 21], [44, 12, 22], [46, 10, 24]], dtype=float)
    green = np.array([[50, -30, 30]], dtype=float)
    centres = np.vstack([browns, green])
    counts = np.array([1000, 1000, 1000, 1000, 20])
    merged = merge_clusters(centres, counts, 2)
    assert any(c[1] < -20 for c in merged)


def test_despeckle_keeps_contrasting_details():
    pal = rgb_to_lab(np.array([[255, 255, 255], [0, 0, 0], [250, 250, 250], [10, 10, 10]], dtype=float))
    cells = np.full((5, 5), 1)
    cells[2, 2] = 2  # white highlight in black: kept
    cells[0, 4] = 3  # near-black speck in black: merged
    out, changed = despeckle(cells, pal)
    assert out[2, 2] == 2
    assert out[0, 4] == 1
    assert changed == 1


def test_detail_warning_for_narrow_patterns():
    r = generate_from_image(flag_image(), GenerateOptions(width=30))
    assert any("detail" in w for w in r.warnings)
    r = generate_from_image(flag_image(), GenerateOptions(width=80))
    assert not any("detail" in w for w in r.warnings)


def test_preview_renders():
    img = flag_image()
    p = generate_from_image(img, GenerateOptions(width=40)).pattern
    prev = render_preview(p, img)
    assert prev.width > 400 and prev.height > 300
