"""Read and write OXS (Open Cross Stitch) files.

Structure written, following the format published by Ursa Software and the
files their software produces::

    <chart>
      <format .../>                      standard comment block
      <properties chartwidth=... chartheight=... palettecount=... .../>
      <palette>
        <palette_item index="0" number="cloth" .../>
        <palette_item index="1" number="DMC 310" name="Black" color="000000" .../>
      </palette>
      <fullstitches><stitch x="0" y="0" palindex="1"/>...</fullstitches>
      <partstitches/>
      <backstitches><backstitch x1= y1= x2= y2= palindex= objecttype="backstitch" sequence="0"/>...</backstitches>
      <ornaments_inc_knots_and_beads/>
      <commentboxes/>
    </chart>

Reading is lenient, since files come from many programs: element and
attribute names are matched case-insensitively, decimal commas are accepted,
and elements this module does not model are preserved and written back.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Union

import numpy as np

from .grid import CLOTH, Backstitch, PaletteEntry, Pattern

SOFTWARE_NAME = "cross-stitch-with-claude-skill"
SOFTWARE_VERSION = "0.1.0"

FORMAT_COMMENTS = [
    "Designed to allow interchange of basic pattern data between any cross stitch style software",
    "the 'properties' section establishes size, copyright, authorship and software used",
    "The features of each software package varies, but using XML each can pick out the things it can deal with, while ignoring others",
    "The basic items are :",
    "'palette'..a set of colors used in the design: palettecount excludes cloth color, which is item 0",
    "'fullstitches'.. simple crosses",
    "'backstitches'.. lines/objects with a start and end point",
    "(There is a wide variety of ways of treating part stitches, knots, beads and so on.)",
    "Colors are expressed in hex RGB format.",
    "Decimal numbers use US/UK format where '.' is the indicator - eg 0.5 is 'half'",
    "For readability, please use words not enumerations",
    "The properties, fullstitches, and backstitches elements should be considered mandatory, even if empty",
    "element and attribute names are always lowercase",
]

# Elements written in this order; unknown extras follow them.
_CANONICAL_EXTRAS = ["partstitches", "ornaments_inc_knots_and_beads", "commentboxes"]
_PALINDEX_ATTRS = ("palindex", "palindex1", "palindex2")


class OXSError(ValueError):
    pass


# ---------------------------------------------------------------- reading


def _lower_attrs(el: ET.Element) -> dict:
    return {k.lower(): v for k, v in el.attrib.items()}


def _num(value: str) -> float:
    return float(value.strip().replace(",", "."))


def _int(value: str) -> int:
    return int(round(_num(value)))


def _lowercase_tree(el: ET.Element) -> ET.Element:
    el.tag = el.tag.lower()
    el.attrib = _lower_attrs(el)
    for child in el:
        _lowercase_tree(child)
    return el


def read_oxs(source: Union[str, os.PathLike, bytes]) -> Pattern:
    """Parse an OXS file (path, or raw bytes) into a Pattern."""
    if isinstance(source, (bytes, bytearray)):
        root = ET.fromstring(source)
    else:
        root = ET.parse(source).getroot()
    root = _lowercase_tree(root)
    if root.tag != "chart":
        raise OXSError(f"root element is <{root.tag}>, expected <chart>")

    properties: dict = {}
    palette_items: list[tuple[int, PaletteEntry]] = []
    raw_stitches: list[tuple[int, int, int]] = []
    backstitches: list[Backstitch] = []
    extras: list[ET.Element] = []

    for child in root:
        tag = child.tag
        if tag == "format":
            continue  # regenerated on write
        if tag == "properties":
            properties = dict(child.attrib)
        elif tag == "palette":
            for item in child:
                if item.tag != "palette_item":
                    continue
                a = dict(item.attrib)
                index = _int(a.pop("index", str(len(palette_items))))
                entry = PaletteEntry(
                    number=a.pop("number", "").strip(),
                    name=a.pop("name", "").strip(),
                    color=a.pop("color", "FFFFFF").strip().lstrip("#").upper(),
                    strands=_int(a.pop("strands", "2") or "2"),
                    attrs=a,
                    children=list(item),
                )
                palette_items.append((index, entry))
        elif tag == "fullstitches":
            for st in child:
                if st.tag != "stitch":
                    continue
                a = st.attrib
                try:
                    raw_stitches.append((_int(a["x"]), _int(a["y"]), _int(a["palindex"])))
                except (KeyError, ValueError):
                    continue
        elif tag == "backstitches":
            for bs in child:
                if bs.tag != "backstitch":
                    continue
                a = dict(bs.attrib)
                try:
                    backstitches.append(
                        Backstitch(
                            x1=_num(a.pop("x1")),
                            y1=_num(a.pop("y1")),
                            x2=_num(a.pop("x2")),
                            y2=_num(a.pop("y2")),
                            palindex=_int(a.pop("palindex")),
                            objecttype=a.pop("objecttype", "backstitch"),
                            attrs=a,
                        )
                    )
                except (KeyError, ValueError):
                    continue
        else:
            extras.append(child)

    if not palette_items:
        palette_items = [(0, PaletteEntry(number="cloth", name="cloth", color="FFFFFF"))]

    # Re-number the palette densely (0..n) in index order, remapping references.
    palette_items.sort(key=lambda p: p[0])
    remap = {old: new for new, (old, _) in enumerate(palette_items)}
    palette = [entry for _, entry in palette_items]
    if any(old != new for old, new in remap.items()):
        for b in backstitches:
            b.palindex = remap.get(b.palindex, b.palindex)
        for el in extras:
            for node in el.iter():
                for key in _PALINDEX_ATTRS:
                    if key in node.attrib:
                        try:
                            node.attrib[key] = str(remap.get(_int(node.attrib[key]), _int(node.attrib[key])))
                        except ValueError:
                            pass

    width = _int(properties["chartwidth"]) if "chartwidth" in properties else 0
    height = _int(properties["chartheight"]) if "chartheight" in properties else 0
    if raw_stitches:
        width = max(width, max(s[0] for s in raw_stitches) + 1)
        height = max(height, max(s[1] for s in raw_stitches) + 1)

    cells = np.zeros((height, width), dtype=np.int32)
    for x, y, p in raw_stitches:
        p = remap.get(p, p)
        if 0 <= x < width and 0 <= y < height and 0 <= p < len(palette):
            cells[y, x] = p

    return Pattern(cells=cells, palette=palette, backstitches=backstitches, properties=properties, extras=extras)


# ---------------------------------------------------------------- writing


def _fmt(value: float) -> str:
    """Format a coordinate with '.' decimals and no trailing zeros."""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def to_element(pattern: Pattern) -> ET.Element:
    chart = ET.Element("chart")

    fmt = ET.SubElement(chart, "format")
    for i, text in enumerate(FORMAT_COMMENTS, start=1):
        fmt.set(f"comments{i:02d}", text)

    props = {
        "oxsversion": "1.0",
        "software": SOFTWARE_NAME,
        "software_version": SOFTWARE_VERSION,
        "chartheight": str(pattern.height),
        "chartwidth": str(pattern.width),
        "charttitle": "",
        "author": "",
        "copyright": "",
        "instructions": "",
        "stitchesperinch": "14",
        "stitchesperinch_y": "",
        "palettecount": str(len(pattern.palette) - 1),
    }
    for key, value in pattern.properties.items():
        if key not in ("chartheight", "chartwidth", "palettecount", "oxsversion", "software", "software_version"):
            props[key] = value
    if not props["stitchesperinch_y"]:
        props["stitchesperinch_y"] = props["stitchesperinch"]
    ET.SubElement(chart, "properties", props)

    pal = ET.SubElement(chart, "palette")
    for index, entry in enumerate(pattern.palette):
        attrs = {
            "index": str(index),
            "number": entry.number if index != CLOTH else (entry.number or "cloth"),
            "name": entry.name,
            "color": entry.color.upper(),
            "printcolor": entry.color.upper(),
            "blendcolor": "nil",
            "comments": "",
            "strands": str(entry.strands),
            "dashpattern": "",
            "bsstrands": str(entry.strands),
            "bscolor": entry.color.upper(),
        }
        attrs.update(entry.attrs)
        item = ET.SubElement(pal, "palette_item", attrs)
        item.extend(entry.children)

    full = ET.SubElement(chart, "fullstitches")
    ys, xs = np.nonzero(pattern.cells)
    for y, x in zip(ys.tolist(), xs.tolist()):  # row-major order
        ET.SubElement(full, "stitch", {"x": str(x), "y": str(y), "palindex": str(int(pattern.cells[y, x]))})

    extras_by_tag = {el.tag: el for el in pattern.extras}

    def emit_extra(tag: str) -> None:
        chart.append(extras_by_tag.get(tag, ET.Element(tag)))

    emit_extra("partstitches")

    bss = ET.SubElement(chart, "backstitches")
    for b in pattern.backstitches:
        attrs = {
            "x1": _fmt(b.x1),
            "x2": _fmt(b.x2),
            "y1": _fmt(b.y1),
            "y2": _fmt(b.y2),
            "palindex": str(b.palindex),
            "objecttype": b.objecttype,
            "sequence": "0",
        }
        attrs.update(b.attrs)
        ET.SubElement(bss, "backstitch", attrs)

    emit_extra("ornaments_inc_knots_and_beads")
    emit_extra("commentboxes")
    for el in pattern.extras:
        if el.tag not in _CANONICAL_EXTRAS:
            chart.append(el)
    return chart


def to_bytes(pattern: Pattern) -> bytes:
    chart = to_element(pattern)
    ET.indent(chart, space="  ")
    body = ET.tostring(chart, encoding="unicode")
    return ('<?xml version="1.0" encoding="UTF-8"?>\n' + body + "\n").encode("utf-8")


def write_oxs(pattern: Pattern, path: Union[str, os.PathLike]) -> None:
    problems = validate(pattern)
    if problems:
        raise OXSError("refusing to write invalid pattern:\n  " + "\n  ".join(problems))
    with open(path, "wb") as f:
        f.write(to_bytes(pattern))


# ---------------------------------------------------------------- validation


def validate(pattern: Pattern) -> list[str]:
    """Return a list of problems that would make the OXS output invalid. Empty means OK."""
    problems = []
    if pattern.width < 1 or pattern.height < 1:
        problems.append(f"chart size {pattern.width}x{pattern.height} is empty")
    n = len(pattern.palette)
    if n < 1:
        problems.append("palette has no cloth entry (index 0)")
    if pattern.cells.size and (pattern.cells.min() < 0 or pattern.cells.max() >= n):
        problems.append(f"full stitches reference palette indices outside 0..{n - 1}")
    for i, entry in enumerate(pattern.palette):
        c = entry.color
        if len(c) != 6 or any(ch not in "0123456789ABCDEFabcdef" for ch in c):
            problems.append(f"palette item {i} has invalid colour {c!r}")
    for b in pattern.backstitches:
        if not (0 < b.palindex < n):
            problems.append(f"backstitch {b} references palette index {b.palindex}")
            break
        if not all(0 <= v <= lim for v, lim in ((b.x1, pattern.width), (b.x2, pattern.width), (b.y1, pattern.height), (b.y2, pattern.height))):
            problems.append(f"backstitch {b} lies outside the chart")
            break
    return problems


def validate_file(path: Union[str, os.PathLike]) -> list[str]:
    """Structural checks on a written file, including the mandatory elements."""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        return [f"not well-formed XML: {e}"]
    problems = []
    if root.tag != "chart":
        problems.append(f"root element is <{root.tag}>, expected <chart>")
    for tag in ("properties", "palette", "fullstitches", "backstitches"):
        if root.find(tag) is None:
            problems.append(f"missing mandatory <{tag}> element")
    props = root.find("properties")
    palette = root.find("palette")
    if props is not None and palette is not None:
        items = palette.findall("palette_item")
        if props.get("palettecount") != str(len(items) - 1):
            problems.append(f"palettecount={props.get('palettecount')} but palette has {len(items) - 1} thread colours")
    if not problems:
        problems.extend(validate(read_oxs(path)))
    return problems
