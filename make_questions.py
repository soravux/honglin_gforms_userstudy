#!/usr/bin/env python3

import argparse
import json
import random
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from make_question import create_comparison_image

OURS_METHOD = "Vips"


def _sanitize_filename_part(value: str) -> str:
	value = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
	value = re.sub(r"-+", "-", value).strip("-")
	return value or "item"


def scan_results(root_folder: Path):
	individual_to_methods: dict[tuple[str, str, str], dict[str, Path]] = {}
	all_methods = set()

	for individual_group_dir in sorted(root_folder.iterdir()):
		if not individual_group_dir.is_dir():
			continue
		for individual_subgroup_dir in sorted(individual_group_dir.iterdir()):
			if not individual_subgroup_dir.is_dir():
				continue
			for method_dir in sorted(individual_subgroup_dir.iterdir()):
				if not method_dir.is_dir():
					continue
				method_name = method_dir.name
				all_methods.add(method_name)

				for individual_dir in sorted(method_dir.iterdir()):
					if not individual_dir.is_dir():
						continue
					individual_name = individual_dir.name
					key = (individual_group_dir.name, individual_subgroup_dir.name, individual_name)
					individual_to_methods.setdefault(key, {})[method_name] = individual_dir

	baselines = sorted(method for method in all_methods if method != OURS_METHOD)

	individuals = []
	for (individual_group, individual_subgroup, individual_name), methods in sorted(individual_to_methods.items()):
		individuals.append(
			{
				"individual_group": individual_group,
				"individual_subgroup": individual_subgroup,
				"individual_name": individual_name,
				"available_methods": sorted(methods.keys()),
			}
		)

	return baselines, individuals, individual_to_methods


def _parse_methods_arg(raw_methods: str | None) -> set[str] | None:
	if not raw_methods:
		return None
	methods = {m.strip() for m in raw_methods.split(",") if m.strip()}
	return methods or None


def _parse_sample_names_arg(raw_sample_names: str | None) -> set[str] | None:
	if not raw_sample_names:
		return None
	sample_names = {name.strip() for name in raw_sample_names.split(",") if name.strip()}
	return sample_names or None


def _filter_methods_data(
	individual_to_methods: dict[tuple[str, str, str], dict[str, Path]],
	all_methods: set[str],
	included_methods: set[str] | None,
):
	if included_methods is None:
		return individual_to_methods, all_methods

	missing_methods = sorted(included_methods - all_methods)
	if missing_methods:
		raise SystemExit(
			"Error: --methods includes method(s) not found under root folder: "
			+ ", ".join(missing_methods)
		)

	if OURS_METHOD not in included_methods:
		raise SystemExit(f"Error: --methods must include '{OURS_METHOD}'")

	filtered_individual_to_methods: dict[tuple[str, str, str], dict[str, Path]] = {}
	for key, methods in individual_to_methods.items():
		filtered = {name: path for name, path in methods.items() if name in included_methods}
		if filtered:
			filtered_individual_to_methods[key] = filtered

	return filtered_individual_to_methods, included_methods


def _filter_samples_data(
	individual_to_methods: dict[tuple[str, str, str], dict[str, Path]],
	included_sample_names: set[str] | None,
):
	if included_sample_names is None:
		return individual_to_methods

	available_sample_names = {individual_name for (_, _, individual_name) in individual_to_methods}
	missing_sample_names = sorted(included_sample_names - available_sample_names)
	if missing_sample_names:
		raise SystemExit(
			"Error: --sample-names includes sample(s) not found under root folder: "
			+ ", ".join(missing_sample_names)
		)

	filtered_individual_to_methods: dict[tuple[str, str, str], dict[str, Path]] = {}
	for key, methods in individual_to_methods.items():
		individual_name = key[2]
		if individual_name in included_sample_names:
			filtered_individual_to_methods[key] = methods

	return filtered_individual_to_methods


def build_comparisons(
	individual_to_methods: dict[tuple[str, str, str], dict[str, Path]],
	baselines: list[str],
	output_folder: Path,
	grid_size: tuple[int, int],
	padding: int,
	equalize: bool,
	tpose_size: float,
	tpose_vertical: float,
	label_size: int,
):
	comparison_specs = []

	for (individual_group, individual_subgroup, individual_name), methods in sorted(individual_to_methods.items()):
		if OURS_METHOD not in methods:
			continue

		for baseline in baselines:
			if baseline not in methods:
				continue

			comparison_specs.append(
				{
					"individual_group": individual_group,
					"individual_subgroup": individual_subgroup,
					"individual_name": individual_name,
					"method_ours": OURS_METHOD,
					"method_baseline": baseline,
					"ours_folder": methods[OURS_METHOD],
					"baseline_folder": methods[baseline],
				}
			)

	random.shuffle(comparison_specs)

	comparisons = []
	for comparison_idx, spec in enumerate(tqdm(comparison_specs), start=1):
		individual_group = spec["individual_group"]
		individual_subgroup = spec["individual_subgroup"]
		individual_name = spec["individual_name"]
		baseline = spec["method_baseline"]

		if random.choice((True, False)):
			method_left = OURS_METHOD
			method_right = baseline
			folder_left = spec["ours_folder"]
			folder_right = spec["baseline_folder"]
		else:
			method_left = baseline
			method_right = OURS_METHOD
			folder_left = spec["baseline_folder"]
			folder_right = spec["ours_folder"]

		safe_group = _sanitize_filename_part(individual_group)
		safe_subgroup = _sanitize_filename_part(individual_subgroup)
		safe_individual = _sanitize_filename_part(individual_name)
		safe_method_left = _sanitize_filename_part(method_left)
		safe_method_right = _sanitize_filename_part(method_right)
		image_name = (
			f"question_{comparison_idx:05d}_"
			f"{safe_group}__{safe_subgroup}__{safe_individual}__"
			f"left-{safe_method_left}__right-{safe_method_right}.jpg"
		)
		output_path = output_folder / image_name

		create_comparison_image(
			folder1=folder_left,
			folder2=folder_right,
			grid_size=grid_size,
			padding=padding,
			equalize=equalize,
			tpose_size=tpose_size,
			tpose_vertical=tpose_vertical,
			label_size=label_size,
			label_a="Method A",
			label_b="Method B",
			output_path=output_path,
		)

		comparisons.append(
			{
				"comparison_id": comparison_idx,
				"individual_group": individual_group,
				"individual_subgroup": individual_subgroup,
				"individual_name": individual_name,
				"method_ours": OURS_METHOD,
				"method_baseline": baseline,
				"ours_folder": str(spec["ours_folder"]),
				"baseline_folder": str(spec["baseline_folder"]),
				"method_left": method_left,
				"method_right": method_right,
				"left_folder": str(folder_left),
				"right_folder": str(folder_right),
				"ours_is_left": method_left == OURS_METHOD,
				"image_filename": image_name,
				"image_path": str(output_path),
			}
		)

	return comparisons


def main():
	parser = argparse.ArgumentParser(
		description="Generate all Vips-vs-baseline user-study comparison images and a JSON manifest.",
	)
	parser.add_argument("root_folder", type=Path, help="Root folder with structure group/subgroup/method/individual")
	parser.add_argument("output_folder", type=Path, help="Where to write generated comparison images")
	parser.add_argument(
		"--manifest",
		type=Path,
		default=None,
		help="Output JSON manifest path (default: <output_folder>/comparisons.json)",
	)
	parser.add_argument("--grid-size", default="5x3", help="Grid size, e.g. 5 or 5x3")
	parser.add_argument("--padding", type=int, default=0, help="Crop padding in pixels")
	parser.add_argument("--no-equalize", action="store_true", help="Disable lightness equalization")
	parser.add_argument("--tpose-size", type=float, default=40.0, help="Tpose size percentage")
	parser.add_argument("--tpose-vertical", type=float, default=25.0, help="Tpose vertical alignment percentage")
	parser.add_argument("--label-size", type=int, default=108, help="Label font size")
	parser.add_argument(
		"--tutorial_image",
		type=Path,
		default=Path("../tutorial.jpg"),
		help="Path to tutorial image to copy into output folder as tutorial.jpg",
	)
	parser.add_argument(
		"--methods",
		type=str,
		default=None,
		help="Comma-separated method names to include (must include Vips), e.g. 'Vips,MethodA,MethodB'",
	)
	parser.add_argument(
		"--sample_names",
		type=str,
		default=None,
		help="Comma-separated sample names to include, e.g. 'sample_001,sample_014'",
	)
	args = parser.parse_args()

	root_folder = args.root_folder.resolve()
	output_folder = args.output_folder.resolve()
	if not root_folder.is_dir():
		raise SystemExit(f"Error: {root_folder} is not a directory")
	output_folder.mkdir(parents=True, exist_ok=True)

	tutorial_image_path = args.tutorial_image.resolve()
	if not tutorial_image_path.is_file():
		raise SystemExit(f"Error: tutorial image not found: {tutorial_image_path}")
	shutil.copy2(tutorial_image_path, output_folder / "tutorial.jpg")

	from make_grid import parse_grid_size

	grid_size = parse_grid_size(args.grid_size)
	baselines, individuals, individual_to_methods = scan_results(root_folder)
	included_methods = _parse_methods_arg(args.methods)
	included_sample_names = _parse_sample_names_arg(args.sample_names)
	if included_methods is not None:
		individual_to_methods, selected_methods = _filter_methods_data(
			individual_to_methods=individual_to_methods,
			all_methods=set(baselines + [OURS_METHOD]),
			included_methods=included_methods,
		)
		baselines = sorted(method for method in selected_methods if method != OURS_METHOD)

	individual_to_methods = _filter_samples_data(
		individual_to_methods=individual_to_methods,
		included_sample_names=included_sample_names,
	)

	individuals = []
	for (individual_group, individual_subgroup, individual_name), methods in sorted(individual_to_methods.items()):
		individuals.append(
			{
				"individual_group": individual_group,
				"individual_subgroup": individual_subgroup,
				"individual_name": individual_name,
				"available_methods": sorted(methods.keys()),
			}
		)

	comparisons = build_comparisons(
		individual_to_methods=individual_to_methods,
		baselines=baselines,
		output_folder=output_folder,
		grid_size=grid_size,
		padding=args.padding,
		equalize=not args.no_equalize,
		tpose_size=args.tpose_size,
		tpose_vertical=args.tpose_vertical,
		label_size=args.label_size,
	)

	manifest_path = args.manifest.resolve() if args.manifest else (output_folder / "comparisons.json")
	manifest = {
		"created_at_utc": datetime.now(timezone.utc).isoformat(),
		"root_folder": str(root_folder),
		"output_folder": str(output_folder),
		"ours_method": OURS_METHOD,
		"included_methods": sorted(included_methods) if included_methods is not None else None,
		"included_sample_names": sorted(included_sample_names) if included_sample_names is not None else None,
		"baselines": baselines,
		"individuals": individuals,
		"comparisons": comparisons,
	}
	manifest_path.parent.mkdir(parents=True, exist_ok=True)
	manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

	print(f"Found {len(individuals)} individuals")
	print(f"Found {len(baselines)} baselines: {', '.join(baselines) if baselines else '(none)'}")
	print(f"Generated {len(comparisons)} comparisons")
	print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
	main()
