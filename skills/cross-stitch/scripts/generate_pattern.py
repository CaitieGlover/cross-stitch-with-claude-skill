#!/usr/bin/env python3
"""Turn a reference image into an OXS cross-stitch pattern plus a PNG preview.

Example:
    python generate_pattern.py photo.jpg --width 80 --colors 15 --out fox.oxs --preview fox.png

Prints a JSON summary (size, threads, stitch counts, warnings) to stdout.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crossstitch.generate import GenerateOptions, generate_from_image, load_image  # noqa: E402
from crossstitch.oxs_io import validate_file, write_oxs  # noqa: E402
from crossstitch.preview import render_preview  # noqa: E402


def parse_crop(value):
    parts = [int(p) for p in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("crop must be left,top,right,bottom")
    return tuple(parts)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("image", help="reference image (PNG, JPEG, ...)")
    p.add_argument("--out", required=True, help="output .oxs path")
    p.add_argument("--preview", help="output .png preview path")
    p.add_argument("--width", type=int, default=80, help="pattern width in stitches (default 80)")
    p.add_argument("--colors", type=int, default=15, help="maximum number of thread colours (default 15)")
    p.add_argument("--background", choices=["auto", "remove", "keep"], default="auto",
                   help="auto: leave background unstitched if it matches the fabric; remove: always; keep: never")
    p.add_argument("--fabric", default="FFFFFF", help="fabric colour as hex (default FFFFFF)")
    p.add_argument("--crop", type=parse_crop, help="crop the source first: left,top,right,bottom in pixels")
    p.add_argument("--count", type=int, default=14, help="fabric count in stitches per inch (default 14)")
    p.add_argument("--title", default="", help="chart title")
    p.add_argument("--metric", choices=["lab", "rgb"], default="lab", help="colour distance (default lab = CIEDE2000)")
    p.add_argument("--no-despeckle", action="store_true", help="keep isolated single stitches as they are")
    p.add_argument("--keep-edges", action="store_true", help="do not tidy blended outline stitches next to the background")
    args = p.parse_args(argv)

    opts = GenerateOptions(
        width=args.width,
        max_colors=args.colors,
        background=args.background,
        fabric=args.fabric.lstrip("#"),
        crop=args.crop,
        despeckle=not args.no_despeckle,
        clean_edges=not args.keep_edges,
        metric=args.metric,
        stitches_per_inch=args.count,
        title=args.title,
    )
    result = generate_from_image(args.image, opts)
    write_oxs(result.pattern, args.out)
    problems = validate_file(args.out)
    if args.preview:
        source = load_image(args.image)
        if args.crop:
            source = source.crop(args.crop)
        render_preview(result.pattern, source).save(args.preview)

    summary = dict(result.stats)
    summary["oxs"] = args.out
    summary["preview"] = args.preview
    summary["warnings"] = result.warnings
    summary["validation_problems"] = problems
    print(json.dumps(summary, indent=2))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
