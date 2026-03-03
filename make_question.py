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


def create_comparison_image(
    folder1: Path,
    folder2: Path,
    grid_size=(5, 3),
    padding: int = 0,
    equalize: bool = True,
    tpose_size: float = 40.0,
    tpose_vertical: float = 25.0,
    label_size: int = 108,
    label_a: str = LABEL_A,
    label_reference: str = LABEL_REFERENCE,
    label_b: str = LABEL_B,
    skip_text: bool = False,
    output_path: Path | None = None,
) -> Path:
    folder1 = Path(folder1).resolve()
    folder2 = Path(folder2).resolve()
    for folder in (folder1, folder2):
        if not folder.is_dir():
            raise ValueError(f"{folder} is not a directory")

    grid_rows, grid_cols = grid_size

    imgs1 = get_eligible_images(folder1)
    if not imgs1:
        raise ValueError(f"No eligible images in {folder1}")
    crop_box1, cell_size1 = compute_crop_params(imgs1, padding)
    grid1 = make_grid_from_params(
        imgs1, crop_box1, cell_size1, grid_rows, grid_cols, equalize
    )

    tpose_path = get_tpose_image(folder1)
    if tpose_path is None:
        raise ValueError(f"No *_tpose* image found in {folder1}")
    tpose_cropped = crop_single(tpose_path, crop_box1)
    grid_h = grid1.size[1]
    tw, th = tpose_cropped.size
    scale_full = grid_h / th
    scale = (tpose_size / 100.0) * scale_full
    new_w = round(tw * scale)
    new_h = round(th * scale)
    tpose_resized = tpose_cropped.resize((new_w, new_h), Image.LANCZOS)
    if equalize:
        tpose_resized = equalize_lightness(tpose_resized)
    tpose_y = int((tpose_vertical / 100.0) * (grid_h - new_h))
    tpose_column = Image.new("RGB", (new_w, grid_h), (255, 255, 255))
    tpose_column.paste(tpose_resized, (0, tpose_y))

    imgs2 = get_eligible_images(folder2)
    if not imgs2:
        raise ValueError(f"No eligible images in {folder2}")
    crop_box2, cell_size2 = compute_crop_params(imgs2, padding)
    grid2 = make_grid_from_params(
        imgs2, crop_box2, cell_size2, grid_rows, grid_cols, equalize
    )

    w1, w_tpose, w2 = grid1.size[0], tpose_column.size[0], grid2.size[0]
    total_w = w1 + w_tpose + w2
    content_h = grid_h

    font = None
    for path in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        try:
            font = ImageFont.truetype(path, label_size)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()

    label_h = 0
    if not skip_text:
        _dummy = Image.new("RGB", (1, 1))
        _draw = ImageDraw.Draw(_dummy)
        max_text_h = 0
        for text in (label_a, label_reference, label_b):
            b = _draw.textbbox((0, 0), text, font=font)
            max_text_h = max(max_text_h, b[3] - b[1])
        label_h = max_text_h + LABEL_TOP_PADDING

    result_h = label_h + content_h
    result = Image.new("RGB", (total_w, result_h), (255, 255, 255))
    result.paste(grid1, (0, label_h))
    result.paste(tpose_column, (w1, label_h))
    result.paste(grid2, (w1 + w_tpose, label_h))

    draw = ImageDraw.Draw(result)
    if not skip_text:
        for x_center, text in [
            (w1 // 2, label_a),
            (w1 + w_tpose // 2, label_reference),
            (w1 + w_tpose + w2 // 2, label_b),
        ]:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            tx = x_center - text_w // 2
            ty = (label_h - text_h) // 2
            draw.text((tx, ty), text, fill=(0, 0, 0), font=font)

    bar_w = 2
    bar_color = (220, 220, 220)
    for x_left in (w1, w1 + w_tpose):
        draw.rectangle(
            (x_left, label_h, x_left + bar_w, result_h),
            fill=bar_color,
        )

    if output_path is None:
        stem = f"{folder1.name}_vs_{folder2.name}"
        output_path = folder1.parent / f"{stem}.jpg"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path, format="JPEG", quality=86)
    return output_path


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
    parser.add_argument(
        "--skip-text",
        action="store_true",
        help="Skip rendering top text labels in the output image",
    )
    args = parser.parse_args()

    try:
        out = create_comparison_image(
            folder1=args.folder1,
            folder2=args.folder2,
            grid_size=args.grid_size,
            padding=args.padding,
            equalize=not args.no_equalize,
            tpose_size=args.tpose_size,
            tpose_vertical=args.tpose_vertical,
            label_size=args.label_size,
            skip_text=args.skip_text,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Saved {out}")


if __name__ == "__main__":
    main()

# Which set of poses appears more anatomically plausible and natural for the given animal mesh?
# Which set of poses is more diverse, with a larger range of different poses?
