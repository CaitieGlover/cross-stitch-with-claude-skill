import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from crossstitch import Backstitch, Pattern, read_oxs, validate, validate_file, write_oxs
from crossstitch.dmc_match import lookup
from crossstitch.oxs_io import OXSError, to_bytes

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "third_party_style.oxs")


def small_pattern():
    p = Pattern.blank(5, 4, title="Test")
    red = p.add_color(lookup("321").palette_entry())
    black = p.add_color(lookup("310").palette_entry())
    p.cells[1, 1] = red
    p.cells[2, 3] = black
    p.backstitches.append(Backstitch(0, 0, 2, 0, black))
    p.backstitches.append(Backstitch(0.5, 1.5, 1.5, 2.5, red))
    return p


def test_read_third_party_style_file():
    p = read_oxs(FIXTURE)
    assert (p.width, p.height) == (8, 6)
    assert p.title == "Fixture"
    assert [e.number for e in p.palette] == ["cloth", "DMC    310", "DMC    321", "DMC    939"]
    assert p.palette[0].color == "F5F0E1"
    assert p.stitch_counts() == {2: 4, 3: 1, 1: 1}
    assert p.cells[1, 1] == 2 and p.cells[5, 7] == 1
    # Decimal commas are accepted.
    assert p.backstitches[2].x1 == 0.5 and p.backstitches[2].y2 == 4.5
    assert p.find_color("dmc 321") == 2


def test_round_trip_preserves_everything(tmp_path):
    p = read_oxs(FIXTURE)
    out = tmp_path / "rt.oxs"
    write_oxs(p, out)
    q = read_oxs(out)
    assert np.array_equal(p.cells, q.cells)
    assert [(e.number, e.color) for e in p.palette] == [(e.number, e.color) for e in q.palette]
    assert [(b.x1, b.y1, b.x2, b.y2, b.palindex) for b in p.backstitches] == [
        (b.x1, b.y1, b.x2, b.y2, b.palindex) for b in q.backstitches
    ]
    root = ET.parse(out).getroot()
    # Unmodelled elements survive: part stitches, knots, blends, vendor extensions.
    assert root.find("partstitches/partstitch").get("direction") == "2"
    assert root.find("ornaments_inc_knots_and_beads/object").get("objecttype") == "knot"
    assert len(root.findall("palette/palette_item/blend")) == 2
    assert root.find("vendor_extension").get("foo") == "bar"
    assert root.find("properties").get("author") == "Test"
    assert root.find("properties").get("stitchesperinch") == "16"
    assert validate_file(out) == []


def test_written_file_structure(tmp_path):
    out = tmp_path / "small.oxs"
    write_oxs(small_pattern(), out)
    root = ET.parse(out).getroot()
    assert root.tag == "chart"
    order = [c.tag for c in root]
    assert order[:8] == [
        "format", "properties", "palette", "fullstitches", "partstitches",
        "backstitches", "ornaments_inc_knots_and_beads", "commentboxes",
    ]
    props = root.find("properties")
    assert props.get("chartwidth") == "5" and props.get("chartheight") == "4"
    assert props.get("palettecount") == "2"
    items = root.findall("palette/palette_item")
    assert items[0].get("index") == "0" and items[0].get("number") == "cloth"
    assert items[1].get("number") == "DMC 321" and items[1].get("color") == "C72B3B"
    stitches = root.findall("fullstitches/stitch")
    assert [(s.get("x"), s.get("y"), s.get("palindex")) for s in stitches] == [("1", "1", "1"), ("3", "2", "2")]
    bs = root.findall("backstitches/backstitch")
    assert bs[1].get("x1") == "0.5" and bs[1].get("y2") == "2.5"
    # All element and attribute names are lowercase, per the format notes.
    for el in root.iter():
        assert el.tag == el.tag.lower()
        assert all(k == k.lower() for k in el.attrib)
    assert validate_file(out) == []


def test_empty_pattern_still_has_mandatory_elements(tmp_path):
    out = tmp_path / "empty.oxs"
    write_oxs(Pattern.blank(3, 3), out)
    root = ET.parse(out).getroot()
    for tag in ("properties", "palette", "fullstitches", "backstitches"):
        assert root.find(tag) is not None
    assert root.find("properties").get("palettecount") == "0"


def test_add_color_reuses_existing_entry():
    p = small_pattern()
    n = len(p.palette)
    assert p.add_color(lookup("DMC 321").palette_entry()) == 1
    assert len(p.palette) == n


def test_validate_catches_bad_references():
    p = small_pattern()
    p.cells[0, 0] = 9
    assert validate(p)
    with pytest.raises(OXSError):
        write_oxs(p, os.devnull)
    p.cells[0, 0] = 0
    p.backstitches.append(Backstitch(0, 0, 99, 0, 1))
    assert validate(p)


def test_non_contiguous_palette_indices_are_renumbered():
    xml = b"""<chart><properties chartwidth="2" chartheight="1"/><palette>
      <palette_item index="0" number="cloth" name="cloth" color="FFFFFF"/>
      <palette_item index="5" number="DMC 310" name="Black" color="000000"/></palette>
      <fullstitches><stitch x="1" y="0" palindex="5"/></fullstitches>
      <backstitches><backstitch x1="0" y1="0" x2="1" y2="0" palindex="5"/></backstitches></chart>"""
    p = read_oxs(xml)
    assert len(p.palette) == 2
    assert p.cells[0, 1] == 1
    assert p.backstitches[0].palindex == 1
    assert b'palettecount="1"' in to_bytes(p)
