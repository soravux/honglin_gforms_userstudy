#!/usr/bin/env python3
"""Create a comparison image: grid_1 | tpose | grid_2 (horizontally stacked)."""

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from make_grid import (
    compute_crop_params,
    crop_single,
    equalize_lightness,
    get_eligible_images,
    get_tpose_image,
    make_grid_from_params,
    parse_grid_size,
)

LABEL_TOP_PADDING = 8
LABEL_A = "Method A"
LABEL_REFERENCE = "Reference"
LABEL_B = "Method B"


def _parse_pct(s: str, name: str, min_val: float = 0, max_val: float = 100) -> float:
    v = float(s)
    if not (min_val <= v <= max_val):
        raise argparse.ArgumentTypeError(f"{name} must be between {min_val} and {max_val}")
    return v


def main():
    parser = argparse.ArgumentParser(
        description="Stack two folder grids with a T-pose reference in between.",
    )
    parser.add_argument("folder1", type=Path, help="First folder (also supplies the T-pose image)")
    parser.add_argument("folder2", type=Path, help="Second folder")
    parser.add_argument(
        "--grid-size",
        type=parse_grid_size,
        default="5x3",
        metavar="N|RxC",
        help="Grid size: integer for NxN (e.g. 5), or RxC (e.g. 3x5) (default: 5)",
    )
    parser.add_argument(
        "--padding", type=int, default=0,
        help="Padding in pixels around the union bounding box (default: 20)",
    )
    parser.add_argument(
        "--no-equalize", action="store_true",
        help="Disable L* histogram equalization",
    )
    parser.add_argument(
        "--tpose-size",
        type=lambda s: _parse_pct(s, "tpose-size"),
        default=40.0,
        metavar="%",
        help="Scale of center tpose image as %% (100=full height, 50=half) (default: 100)",
    )
    parser.add_argument(
        "--tpose-vertical",
        type=lambda s: _parse_pct(s, "vertical"),
        default=25.0,
        metavar="%",
        help="Vertical alignment of tpose: 0=top, 50=center, 100=bottom (default: 50)",
    )
    parser.add_argument(
        "--label-size",
        type=int,
        default=108,
        metavar="PT",
        help="Label font size in points (default: 24)",
    )
    args = parser.parse_args()

    folder1: Path = args.folder1.resolve()
    folder2: Path = args.folder2.resolve()
    for f in (folder1, folder2):
        if not f.is_dir():
            print(f"Error: {f} is not a directory", file=sys.stderr)
            sys.exit(1)

    grid_rows, grid_cols = args.grid_size
    padding = args.padding
    equalize = not args.no_equalize

    # --- Folder 1: grid + crop params (reused for the tpose image) ---
    imgs1 = get_eligible_images(folder1)
    if not imgs1:
        print(f"No eligible images in {folder1}", file=sys.stderr)
        sys.exit(1)
    crop_box1, cell_size1 = compute_crop_params(imgs1, padding)
    print(f"Folder 1: {len(imgs1)} images, cell {cell_size1[0]}x{cell_size1[1]}")
    grid1 = make_grid_from_params(
        imgs1, crop_box1, cell_size1, grid_rows, grid_cols, equalize
    )

    # --- T-pose from folder 1 ---
    tpose_path = get_tpose_image(folder1)
    if tpose_path is None:
        print(f"No *_tpose* image found in {folder1}", file=sys.stderr)
        sys.exit(1)
    print(f"T-pose: {tpose_path.name}")
    tpose_cropped = crop_single(tpose_path, crop_box1)
    grid_h = grid1.size[1]
    tw, th = tpose_cropped.size
    scale_full = grid_h / th
    scale = (args.tpose_size / 100.0) * scale_full
    new_w = round(tw * scale)
    new_h = round(th * scale)
    tpose_resized = tpose_cropped.resize((new_w, new_h), Image.LANCZOS)
    if equalize:
        tpose_resized = equalize_lightness(tpose_resized)
    # Pad to column (new_w, grid_h) with vertical alignment
    tpose_y = int((args.tpose_vertical / 100.0) * (grid_h - new_h))
    tpose_column = Image.new("RGB", (new_w, grid_h), (255, 255, 255))
    tpose_column.paste(tpose_resized, (0, tpose_y))

    # --- Folder 2: grid ---
    imgs2 = get_eligible_images(folder2)
    if not imgs2:
        print(f"No eligible images in {folder2}", file=sys.stderr)
        sys.exit(1)
    crop_box2, cell_size2 = compute_crop_params(imgs2, padding)
    print(f"Folder 2: {len(imgs2)} images, cell {cell_size2[0]}x{cell_size2[1]}")
    grid2 = make_grid_from_params(
        imgs2, crop_box2, cell_size2, grid_rows, grid_cols, equalize
    )

    # --- Horizontal stack: grid1 | tpose_column | grid2 ---
    w1, w_tpose, w2 = grid1.size[0], tpose_column.size[0], grid2.size[0]
    total_w = w1 + w_tpose + w2
    content_h = grid_h

    # Load label font and compute label strip height from font size
    font = None
    for path in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        try:
            font = ImageFont.truetype(path, args.label_size)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()
    # Label strip height: fit tallest label + padding
    _dummy = Image.new("RGB", (1, 1))
    _draw = ImageDraw.Draw(_dummy)
    max_text_h = 0
    for t in (LABEL_A, LABEL_REFERENCE, LABEL_B):
        b = _draw.textbbox((0, 0), t, font=font)
        max_text_h = max(max_text_h, b[3] - b[1])
    label_h = max_text_h + LABEL_TOP_PADDING

    result_h = label_h + content_h
    result = Image.new("RGB", (total_w, result_h), (255, 255, 255))
    result.paste(grid1, (0, label_h))
    result.paste(tpose_column, (w1, label_h))
    result.paste(grid2, (w1 + w_tpose, label_h))

    # Labels above each section (centered horizontally in each section)
    draw = ImageDraw.Draw(result)
    for x_center, text in [
        (w1 // 2, LABEL_A),
        (w1 + w_tpose // 2, LABEL_REFERENCE),
        (w1 + w_tpose + w2 // 2, LABEL_B),
    ]:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw_, th_ = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = x_center - tw_ // 2
        ty = (label_h - th_) // 2
        draw.text((tx, ty), text, fill=(0, 0, 0), font=font)

    # Subtle vertical dividers between sections
    bar_w = 2
    bar_color = (220, 220, 220)
    for x_left in (w1, w1 + w_tpose):
        draw.rectangle(
            (x_left, label_h, x_left + bar_w, result_h),
            fill=bar_color,
        )

    stem = f"{folder1.name}_vs_{folder2.name}"
    out_dir = folder1.parent
    for fmt, ext in [("JPEG", ".jpg")]:
        out = out_dir / f"{stem}{ext}"
        save_kw = {"quality": 86} if fmt == "JPEG" else {}
        result.save(out, format=fmt, **save_kw)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()

# Which set of poses appears more anatomically plausible and natural for the given animal mesh?
# Which set of poses is more diverse, with a larger range of different poses?
