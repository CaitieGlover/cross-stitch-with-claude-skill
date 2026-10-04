# cross-stitch-with-claude-skill

A Claude Code skill for creating cross-stitch patterns in the OXS (Open Cross
Stitch) format. OXS is an open XML format read by MacStitch, WinStitch,
KXStitch, Pattern Maker, Embroiderly and other cross-stitch software.

## Status

| Capability | State |
| --- | --- |
| Reference image to OXS pattern, DMC colours, PNG preview | Working |
| Read and write OXS files from other software | Working |
| Combining several reference images | Planned |
| Editing an existing pattern (motifs, resize, region recolour) | Planned |
| Recolouring to the threads you own (stash) | Planned |

## Install

As a plugin, from inside Claude Code:

```
/plugin marketplace add caitieglover/cross-stitch-with-claude-skill
/plugin install cross-stitch@cross-stitch-with-claude
```

Or copy `skills/cross-stitch/` into your project's `.claude/skills/`
directory (or `~/.claude/skills/` for all projects).

Then install the Python dependencies:

```
pip install -r skills/cross-stitch/scripts/requirements.txt
```

## Use

Give Claude an image and ask for a cross-stitch pattern, for example
"make an 80-stitch-wide cross-stitch pattern of this photo with at most 15
colours". Claude runs the generator, checks the preview, and returns the
`.oxs` file, a preview PNG and the thread list.

The generator can also be run directly:

```
python skills/cross-stitch/scripts/generate_pattern.py photo.jpg \
    --out photo.oxs --preview photo.png --width 80 --colors 15
```

## Layout

```
.claude-plugin/              plugin and marketplace manifests
skills/cross-stitch/
  SKILL.md                   instructions Claude follows
  scripts/
    generate_pattern.py      command-line entry point
    crossstitch/
      grid.py                shared pattern representation
      oxs_io.py              OXS reading, writing and validation
      dmc_match.py           DMC table and colour matching (CIEDE2000)
      generate.py            image to pattern pipeline
      preview.py             PNG preview rendering
      data/dmc.csv           DMC thread colours (see data/README.md)
tests/                       pytest suite and fixtures
```

## Development

```
pip install numpy Pillow pytest
pytest
```
