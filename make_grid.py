#!/usr/bin/env python3
"""Create a grid of cropped object images from a folder of renders."""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Pixels to discard from every edge of the source images.
# Removes rendered text labels and any border artifacts.
BORDER_DISCARD = 30

# Pixels below this value (per channel) are considered non-background.
WHITE_THRESHOLD = 240


def load_image(path: Path) -> Image.Image:
    """Load a PNG, composite RGBA onto white, return RGB."""
    img = Image.open(path)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert("RGB")


def discard_border(img: Image.Image) -> Image.Image:
    """Remove BORDER_DISCARD pixels from every edge."""
    w, h = img.size
    return img.crop((
        BORDER_DISCARD, BORDER_DISCARD,
        w - BORDER_DISCARD, h - BORDER_DISCARD,
    ))


def find_object_bbox(img_array: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return (x_min, y_min, x_max, y_max) of non-white pixels, or None."""
    mask = np.any(img_array[:, :, :3] < WHITE_THRESHOLD, axis=2)
    if not mask.any():
        return None
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    x_min, x_max = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
    return x_min, y_min, x_max, y_max


CropBox = tuple[int, int, int, int]
CellSize = tuple[int, int]


def compute_crop_params(
    image_paths: list[Path], padding: int
) -> tuple[CropBox, CellSize]:
    """Scan all images and return the union crop box and resulting cell size.

    Returns
    -------
    crop_box : (x_min, y_min, x_max, y_max) in the border-discarded coordinate space.
    cell_size : (width, height) of each cropped cell.
    """
    bboxes: list[tuple[int, int, int, int]] = []
    inner_size: tuple[int, int] | None = None

    for path in image_paths:
        inner = discard_border(load_image(path))
        if inner_size is None:
            inner_size = inner.size
        bbox = find_object_bbox(np.array(inner))
        if bbox is not None:
            bboxes.append(bbox)

    if not bboxes or inner_size is None:
        raise ValueError("No objects detected in any image.")

    ux_min = max(0, min(b[0] for b in bboxes) - padding)
    uy_min = max(0, min(b[1] for b in bboxes) - padding)
    ux_max = min(inner_size[0], max(b[2] for b in bboxes) + 1 + padding)
    uy_max = min(inner_size[1], max(b[3] for b in bboxes) + 1 + padding)

    crop_box: CropBox = (ux_min, uy_min, ux_max, uy_max)
    cell_size: CellSize = (ux_max - ux_min, uy_max - uy_min)
    return crop_box, cell_size


def crop_single(path: Path, crop_box: CropBox) -> Image.Image:
    """Load one image, discard its border, and apply the given crop box."""
    return discard_border(load_image(path)).crop(crop_box)


def build_grid(
    image_paths: list[Path],
    crop_box: CropBox,
    cell_size: CellSize,
    grid_rows: int,
    grid_cols: int,
) -> Image.Image:
    """Assemble a grid of shape grid_rows x grid_cols from the given images."""
    cell_w, cell_h = cell_size
    grid_img = Image.new(
        "RGB",
        (grid_cols * cell_w, grid_rows * cell_h),
        (255, 255, 255),
    )
    for idx, path in enumerate(image_paths):
        cell = crop_single(path, crop_box)
        row, col = divmod(idx, grid_cols)
        grid_img.paste(cell, (col * cell_w, row * cell_h))
    return grid_img


def equalize_lightness(image: Image.Image) -> Image.Image:
    """Histogram-equalize the L* channel in CIE-Lab, ignoring white background."""
    rgb_arr = np.array(image)
    fg_mask = np.any(rgb_arr[:, :, :3] < WHITE_THRESHOLD, axis=2)
    if not fg_mask.any():
        return image

    lab_arr = np.array(image.convert("LAB"))
    L = lab_arr[:, :, 0].copy()
    fg_L = L[fg_mask]

    hist, _ = np.histogram(fg_L, bins=256, range=(0, 256))
    cdf = hist.cumsum()
    cdf_min = cdf[cdf > 0].min()
    n_fg = int(fg_mask.sum())
    if n_fg <= cdf_min:
        return image

    lut = np.round((cdf - cdf_min) / (n_fg - cdf_min) * 255).clip(0, 255).astype(np.uint8)
    L[fg_mask] = lut[fg_L]
    lab_arr[:, :, 0] = L

    return Image.fromarray(lab_arr, mode="LAB").convert("RGB")


# ---------------------------------------------------------------------------
# Public helpers for other scripts
# ---------------------------------------------------------------------------

def parse_grid_size(s: str) -> tuple[int, int]:
    """Parse --grid-size: integer '5' -> (5, 5); '3x5' or '5x4' -> (rows, cols)."""
    s = s.strip().lower()
    if "x" in s:
        parts = s.split("x")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(f"Invalid grid size: {s!r} (use e.g. 5 or 3x5)")
        try:
            r, c = int(parts[0].strip()), int(parts[1].strip())
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid grid size: {s!r}")
    else:
        try:
            n = int(s)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid grid size: {s!r}")
        r = c = n
    if r < 1 or c < 1:
        raise argparse.ArgumentTypeError(f"Grid dimensions must be positive: {s!r}")
    return (r, c)


def get_eligible_images(folder: Path) -> list[Path]:
    """Return sorted list of PNG paths whose filename contains ``_rep_``."""
    return sorted(p for p in folder.glob("*.png") if "_rep_" in p.name)


def get_tpose_image(folder: Path) -> Path | None:
    """Return the first ``*_tpose*`` PNG found in *folder*, or None."""
    matches = sorted(folder.glob("*_tpose*.png"))
    return matches[0] if matches else None


def make_grid(
    folder: Path,
    grid_rows: int = 5,
    grid_cols: int | None = None,
    padding: int = 20,
    equalize: bool = True,
) -> Image.Image:
    """High-level: build a grid image from a folder. Returns a PIL Image."""
    if grid_cols is None:
        grid_cols = grid_rows
    all_images = get_eligible_images(folder)
    if not all_images:
        raise ValueError(f"No eligible PNG images in {folder}")
    crop_box, cell_size = compute_crop_params(all_images, padding)
    return make_grid_from_params(
        all_images, crop_box, cell_size, grid_rows, grid_cols, equalize
    )


def make_grid_from_params(
    all_images: list[Path],
    crop_box: CropBox,
    cell_size: CellSize,
    grid_rows: int = 5,
    grid_cols: int | None = None,
    equalize: bool = True,
) -> Image.Image:
    """Build a grid from pre-computed crop parameters (avoids redundant scans)."""
    if grid_cols is None:
        grid_cols = grid_rows
    num_cells = grid_rows * grid_cols
    if len(all_images) > num_cells:
        selected = random.sample(all_images, num_cells)
    else:
        selected = all_images
    grid_img = build_grid(selected, crop_box, cell_size, grid_rows, grid_cols)
    if equalize:
        grid_img = equalize_lightness(grid_img)
    return grid_img


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Create an NxN grid of cropped object images."
    )
    parser.add_argument("folder", type=Path, help="Folder containing PNG images")
    parser.add_argument(
        "--grid-size",
        type=parse_grid_size,
        default="5",
        metavar="N|RxC",
        help="Grid size: single integer for NxN (e.g. 5), or RxC (e.g. 3x5) (default: 5)",
    )
    parser.add_argument(
        "--padding", type=int, default=20,
        help="Padding in pixels around the union bounding box (default: 20)",
    )
    parser.add_argument(
        "--no-equalize", action="store_true",
        help="Disable L* histogram equalization",
    )
    args = parser.parse_args()

    folder: Path = args.folder.resolve()
    if not folder.is_dir():
        print(f"Error: {folder} is not a directory", file=sys.stderr)
        sys.exit(1)

    equalize = not args.no_equalize
    all_images = get_eligible_images(folder)
    if not all_images:
        print(
            "No eligible PNG images found (must contain '_rep_' in filename).",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Found {len(all_images)} eligible images.")

    crop_box, cell_size = compute_crop_params(all_images, args.padding)
    print(
        f"Union crop region: {crop_box[0:2]}–{crop_box[2:4]}, "
        f"cell size {cell_size[0]}x{cell_size[1]}"
    )

    grid_rows, grid_cols = args.grid_size
    grid_img = make_grid_from_params(
        all_images, crop_box, cell_size, grid_rows, grid_cols, equalize
    )

    for fmt, ext in [("AVIF", ".avif"), ("PNG", ".png"), ("JPEG", ".jpg"    )]:
        out = folder / f"{folder.name}_grid{ext}"
        save_kw = {"quality": 86} if fmt == "JPEG" else {}
        save_kw = {"quality": 70} if fmt == "AVIF" else {}
        grid_img.save(out, format=fmt, **save_kw)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
