---
name: cross-stitch
description: Create cross-stitch patterns in the OXS (Open Cross Stitch) format from a reference image, with DMC thread colours and a PNG preview. Use when the user wants to turn a picture into a cross-stitch chart, make a cross-stitch or embroidery pattern, or produce an .oxs file for MacStitch, WinStitch, KXStitch, Pattern Maker, Embroiderly or similar software.
---

# Cross-stitch patterns (OXS)

This skill turns a reference image into a cross-stitch pattern saved as an
`.oxs` file, the open XML format read by most cross-stitch software. Colours
are matched to DMC stranded cotton.

All the work is done by tested Python code in `scripts/` next to this file.
Do not hand-write OXS XML or work out colour matches in conversation; run the
scripts.

## Setup

The scripts need Python 3.9+ with numpy and Pillow:

```bash
pip install -r <skill-dir>/scripts/requirements.txt
```

`<skill-dir>` is the directory containing this SKILL.md.

## What is available now

- **Generate from an image**: supported.
- Generate from a text prompt alone: ask the user for a reference image
  (a photo, drawing or clip art of what they want). Do not try to draw or
  invent a base image yourself.
- Combining several images, editing an existing .oxs file, recolouring, and
  remapping to the user's own thread stash: **not built yet**. Say so plainly
  if asked, rather than improvising.

## Generating a pattern from an image

1. Agree the basics with the user if they have not said:
   - **Width in stitches.** 50-150 is typical. On 14-count fabric, 70
     stitches is 5 in / 12.7 cm. Faces, eyes and small text need roughly 60+
     stitches across the whole design; small motifs can go lower.
   - **Maximum colours.** 10-20 is typical; default 15. The result often
     uses fewer: near-duplicate colours, and similar colours used for only a
     few stitches, are merged so the stitcher is not buying a skein for a
     handful of stitches.
   - **Fabric colour** if not white, and fabric count if not 14.
   - If only part of the image is wanted, a crop box in source pixels.

2. Run:

   ```bash
   python <skill-dir>/scripts/generate_pattern.py IMAGE \
       --out NAME.oxs --preview NAME.png \
       --width 80 --colors 15 --title "Title"
   ```

   Other options:
   - `--style auto|illustration|photo`. `illustration` gives each stitch the
     single thread that covers most of its square, which keeps flat-colour
     artwork crisp with no in-between blend colours. `photo` averages each
     square, which suits gradients and texture. `auto` (default) picks from
     the image; the summary's `style` field says which was used. Override it
     if the result looks wrong for the image.
   - `--fabric HEX` (default `FFFFFF`)
   - `--count N` (stitches per inch, default 14)
   - `--crop left,top,right,bottom`
   - `--background auto|remove|keep`. `auto` leaves the background
     unstitched only when it is close to the fabric colour. `remove` always
     leaves the edge-connected background unstitched. `keep` stitches
     everything.
   - `--keep-edges` (photo style) turns off clean-up of the blended one-stitch outline
     between subject and background. Use it if a pale edge of the subject
     has been trimmed.
   - `--no-despeckle` keeps every isolated single stitch.

   The script prints a JSON summary: size, finished size in inches and cm,
   each thread with its stitch count, `warnings`, and `validation_problems`.
   `validation_problems` must be empty; if it is not, report it and do not
   hand the file over as valid.

3. **Look at the preview PNG** before replying. It shows the source image
   beside the chart and the thread key. Check that:
   - the subject reads clearly at this size,
   - key colours survived (for example, eye colour, a flower centre, a logo
     colour). Colour reduction can absorb a small but important colour into
     its surroundings. If that happens, re-run with more colours (try +5) and
     check again.
   - the background was handled the way the user expects.

4. Reply with:
   - the `.oxs` and preview paths,
   - size in stitches and the finished size,
   - number of colours and the thread list (DMC number, name, stitch count),
   - every warning from the summary, in plain words. If fine detail is at
     risk, say which features may suffer and suggest a width that would
     help. Do not drop or soften warnings.

## Using the Python API directly

```python
import sys; sys.path.insert(0, "<skill-dir>/scripts")
from crossstitch import read_oxs, write_oxs, validate_file
from crossstitch.generate import GenerateOptions, generate_from_image
from crossstitch.preview import render_preview
from crossstitch import dmc_match

result = generate_from_image("photo.jpg", GenerateOptions(width=80, max_colors=15))
write_oxs(result.pattern, "out.oxs")
render_preview(result.pattern).save("out.png")
dmc_match.lookup("310")            # Thread(number='310', name='Black', hex='000000')
dmc_match.nearest_one((200, 30, 40))
```

`Pattern.cells` is a `(height, width)` numpy array of palette indices; 0 is
unstitched cloth. `Pattern.palette[0]` is the cloth colour. Backstitches are
line segments on grid-corner coordinates.

## Notes on the format

- Mandatory elements `properties`, `fullstitches` and `backstitches` are
  always written, even when empty.
- Palette index 0 is the cloth; `palettecount` excludes it.
- Reading accepts files from other programs, including decimal commas and
  padded thread numbers such as `DMC    310`. Elements the skill does not
  model (part stitches, French knots, beads, vendor extensions) are kept
  and written back unchanged.
