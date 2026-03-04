#!/usr/bin/env python3

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import choix

DEFAULT_MANIFEST_PATH = Path("../data/user_study_manifest.json")
DEFAULT_RESULT_PATH = Path("../data/results/responses_1.csv")

CHOICE_A = "method a"
CHOICE_B = "method b"


def _parse_individual_weights(raw_weights: list[str] | None) -> dict[str, int]:
    if not raw_weights:
        return {}

    weights: dict[str, int] = {}
    for item in raw_weights:
        if "=" not in item:
            raise SystemExit(
                "Error: each --individual-weights entry must be in the format individual_name=weight"
            )

        individual_name, weight_raw = item.split("=", 1)
        individual_name = individual_name.strip()
        weight_raw = weight_raw.strip()
        if not individual_name:
            raise SystemExit("Error: individual name in --individual-weights cannot be empty")

        try:
            weight = int(weight_raw)
        except ValueError as exc:
            raise SystemExit(
                f"Error: invalid weight '{weight_raw}' for individual '{individual_name}'; expected integer"
            ) from exc

        if weight < 0:
            raise SystemExit(
                f"Error: weight for individual '{individual_name}' must be >= 0"
            )

        weights[individual_name] = weight

    return weights

def _load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.is_file():
        raise SystemExit(f"Error: manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _load_rows(csv_path: Path) -> tuple[list[str], list[list[str]]]:
    if not csv_path.is_file():
        raise SystemExit(f"Error: results CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    if not rows:
        raise SystemExit(f"Error: empty CSV file: {csv_path}")

    header = rows[0]
    data_rows = rows[1:]
    return header, data_rows


def _ordered_comparisons(manifest: dict) -> list[dict]:
    comparisons = manifest.get("comparisons", [])
    if not comparisons:
        raise SystemExit("Error: no comparisons found in manifest")

    return sorted(comparisons, key=lambda item: item["comparison_id"])


def _select_comparison_indices(
    comparisons: list[dict],
    included_individuals: set[str] | None,
) -> list[int]:
    if included_individuals is None:
        return list(range(len(comparisons)))

    selected_indices: list[int] = []
    for idx, comparison in enumerate(comparisons):
        if comparison.get("individual_name") in included_individuals:
            selected_indices.append(idx)

    return selected_indices


def _detect_answer_columns(header: list[str], rows: list[list[str]], n_columns: int) -> list[int]:
    question_headers = {
        "which set of poses appears more anatomically plausible and natural for the given human or animal mesh?",
        "which set of poses is more diverse, with a larger range of different poses? ignore completely unrecognizable poses.",
    }

    answer_columns: list[int] = []
    for col_idx in range(n_columns):
        header_value = ""
        if col_idx < len(header):
            header_value = header[col_idx].strip().lower()

        values = set()
        for row in rows:
            if col_idx >= len(row):
                continue
            value = row[col_idx].strip().lower()
            if value:
                values.add(value)

        is_answer_by_header = header_value in question_headers
        is_answer_by_values = bool(values) and values.issubset({CHOICE_A, CHOICE_B})
        if is_answer_by_header or is_answer_by_values:
            answer_columns.append(col_idx)

    if not answer_columns:
        raise SystemExit("Error: could not find any answer columns with values 'Method A'/'Method B'")

    return answer_columns


def build_pairwise_data(
    manifest: dict,
    header: list[str],
    csv_rows: list[list[str]],
    n_columns: int,
    included_individuals: set[str] | None = None,
    individual_weights: dict[str, int] | None = None,
):
    individual_weights = individual_weights or {}
    comparisons = _ordered_comparisons(manifest)
    sides = [(item["method_left"], item["method_right"]) for item in comparisons]
    comparison_individuals = [item.get("individual_name", "") for item in comparisons]
    answer_columns = _detect_answer_columns(header=header, rows=csv_rows, n_columns=n_columns)
    base_answer_columns = list(answer_columns)

    n_comparisons = len(sides)
    if len(answer_columns) % n_comparisons != 0:
        raise SystemExit(
            "Error: number of answer columns is not divisible by number of manifest comparisons. "
            f"answer_columns={len(answer_columns)}, comparisons={n_comparisons}"
        )

    questions_per_comparison = len(answer_columns) // n_comparisons
    if questions_per_comparison <= 0:
        raise SystemExit("Error: invalid questions-per-comparison inferred from CSV")

    selected_comparison_indices = _select_comparison_indices(comparisons, included_individuals)

    filtered_sides = [sides[idx] for idx in selected_comparison_indices]
    filtered_individuals = [comparison_individuals[idx] for idx in selected_comparison_indices]
    filtered_answer_columns: list[int] = []
    for comparison_idx in selected_comparison_indices:
        start = comparison_idx * questions_per_comparison
        end = start + questions_per_comparison
        filtered_answer_columns.extend(answer_columns[start:end])

    sides = filtered_sides
    comparison_individuals = filtered_individuals
    answer_columns = filtered_answer_columns

    question_labels: list[str] = []
    for question_idx in range(questions_per_comparison):
        col_idx = base_answer_columns[question_idx]
        label = header[col_idx].strip() if col_idx < len(header) else ""
        question_labels.append(label or f"Question {question_idx + 1}")

    data_named: list[tuple[str, str]] = []
    data_by_question_named: list[list[tuple[str, str]]] = [[] for _ in range(questions_per_comparison)]
    for row in csv_rows:
        for answer_idx, col_idx in enumerate(answer_columns):
            if col_idx >= len(row):
                continue
            answer = row[col_idx].strip().lower()
            if not answer:
                continue

            comparison_idx = answer_idx // questions_per_comparison
            method_left, method_right = sides[comparison_idx]

            if answer == CHOICE_A:
                winner, loser = method_left, method_right
            elif answer == CHOICE_B:
                winner, loser = method_right, method_left
            else:
                continue

            individual_name = comparison_individuals[comparison_idx]
            weight = individual_weights.get(individual_name, 1)
            question_idx = answer_idx % questions_per_comparison
            for _ in range(weight):
                data_named.append((winner, loser))
                data_by_question_named[question_idx].append((winner, loser))

    return data_named, data_by_question_named, question_labels, questions_per_comparison


def _compute_pair_win_loss(
    data: list[tuple[int, int]],
    id_to_method: dict[int, str],
) -> list[tuple[str, str, int, int]]:
    directional_counts: dict[tuple[int, int], int] = defaultdict(int)
    for winner_id, loser_id in data:
        directional_counts[(winner_id, loser_id)] += 1

    method_ids = sorted(id_to_method.keys())
    pair_rows: list[tuple[str, str, int, int]] = []
    for idx_a, method_a_id in enumerate(method_ids):
        for method_b_id in method_ids[idx_a + 1 :]:
            wins_a = directional_counts[(method_a_id, method_b_id)]
            wins_b = directional_counts[(method_b_id, method_a_id)]
            if wins_a == 0 and wins_b == 0:
                continue
            pair_rows.append(
                (
                    id_to_method[method_a_id],
                    id_to_method[method_b_id],
                    wins_a,
                    wins_b,
                )
            )

    return pair_rows


def _print_strengths_with_names(strengths, id_to_method: dict[int, str]) -> None:
    # print("Estimated parameters (strengths):", strengths)
    print("Strengths by method (Luce Spectral Ranking):")
    for method_id in sorted(id_to_method.keys()):
        method_name = id_to_method[method_id]
        print(f"  {method_name}: {strengths[method_id]:.6f}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract win/lose method pairs from Google Form CSV + manifest")
    parser.add_argument(
        "--manifest",
        type=Path,
        nargs="+",
        default=[DEFAULT_MANIFEST_PATH],
        help="One or more paths to comparisons manifest JSON files",
    )
    parser.add_argument(
        "--results",
        type=Path,
        nargs="+",
        default=[DEFAULT_RESULT_PATH],
        help="One or more paths to Google Form CSV exports",
    )
    parser.add_argument(
        "--individuals",
        nargs="+",
        default=None,
        help="Optional list of individual names to include (space-separated)",
    )
    parser.add_argument(
        "--individual-weights",
        nargs="+",
        default=None,
        help="Optional per-individual integer weights as individual_name=weight (space-separated)",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    manifest_paths = [path.resolve() for path in args.manifest]
    result_paths = [path.resolve() for path in args.results]
    if len(manifest_paths) != len(result_paths):
        raise SystemExit(
            "Error: --manifest and --results must have the same number of files "
            f"(got {len(manifest_paths)} manifests and {len(result_paths)} result files)"
        )

    manifests = [_load_manifest(path) for path in manifest_paths]

    included_individuals = set(args.individuals) if args.individuals else None
    individual_weights = _parse_individual_weights(args.individual_weights)

    available_individuals: set[str] = set()
    for manifest in manifests:
        for comparison in _ordered_comparisons(manifest):
            name = comparison.get("individual_name", "")
            if name:
                available_individuals.add(name)

    if included_individuals is not None:
        missing = sorted(included_individuals - available_individuals)
        if missing:
            raise SystemExit(
                "Error: --individuals includes name(s) not found in manifest comparisons: "
                + ", ".join(missing)
            )

    unknown_weight_individuals = sorted(set(individual_weights.keys()) - available_individuals)
    if unknown_weight_individuals:
        raise SystemExit(
            "Error: --individual-weights includes individual(s) not found in manifest comparisons: "
            + ", ".join(unknown_weight_individuals)
        )

    data_named_all: list[tuple[str, str]] = []
    data_by_question_named_all: list[list[tuple[str, str]]] | None = None
    question_labels: list[str] | None = None
    questions_per_comparison: int | None = None

    for manifest, result_path in zip(manifests, result_paths):
        header, csv_rows = _load_rows(result_path)
        data_named, data_by_question_named, labels, qpc = build_pairwise_data(
            manifest=manifest,
            header=header,
            csv_rows=csv_rows,
            n_columns=len(header),
            included_individuals=included_individuals,
            individual_weights=individual_weights,
        )

        if questions_per_comparison is None:
            questions_per_comparison = qpc
            question_labels = labels
            data_by_question_named_all = [[] for _ in range(qpc)]
        elif qpc != questions_per_comparison:
            raise SystemExit(
                "Error: inconsistent questions-per-comparison across files: "
                f"expected {questions_per_comparison}, got {qpc} for {result_path}"
            )

        data_named_all.extend(data_named)
        for question_idx in range(qpc):
            data_by_question_named_all[question_idx].extend(data_by_question_named[question_idx])

    if questions_per_comparison is None or data_by_question_named_all is None or question_labels is None:
        raise SystemExit("Error: failed to load any manifest/results data")

    all_methods = sorted({method for winner, loser in data_named_all for method in (winner, loser)})
    method_to_id = {method_name: idx for idx, method_name in enumerate(all_methods)}
    id_to_method = {idx: method_name for method_name, idx in method_to_id.items()}

    data = [(method_to_id[winner], method_to_id[loser]) for winner, loser in data_named_all]
    data_by_question = [
        [(method_to_id[winner], method_to_id[loser]) for winner, loser in question_data]
        for question_data in data_by_question_named_all
    ]

    if not data:
        raise SystemExit("Error: no valid win/lose pairs extracted from results")

    print("Method IDs:", method_to_id)
    if included_individuals is not None:
        print(f"Included individuals: {sorted(included_individuals)}")
    if individual_weights:
        print(f"Individual weights: {dict(sorted(individual_weights.items()))}")
    print(f"Input files: {len(manifest_paths)} manifest(s), {len(result_paths)} results file(s)")
    print(f"Questions per comparison (inferred): {questions_per_comparison}")
    print(f"Extracted pairs: {len(data)}")

    print("\n=== Overall ===")
    overall_params = choix.ilsr_pairwise(len(method_to_id), data)
    _print_strengths_with_names(overall_params, id_to_method)
    print("Wins by method pair:")
    for method_a, method_b, wins_a, wins_b in _compute_pair_win_loss(data, id_to_method):
        print(f"  {method_a} vs {method_b}: {wins_a} - {wins_b}")

    for question_idx, question_data in enumerate(data_by_question):
        question_label = question_labels[question_idx] if question_idx < len(question_labels) else f"Question {question_idx + 1}"
        print(f"\n=== Question {question_idx + 1} ===")
        print(f"Prompt: {question_label}")
        print(f"Pairs: {len(question_data)}")

        if not question_data:
            print("Estimated parameters (strengths): (no data)")
            print("Wins by method pair: (no data)")
            continue

        try:
            strengths = choix.ilsr_pairwise(len(method_to_id), question_data)
        except ValueError as e:
            print(f"Error estimating parameters for question {question_idx + 1}: {e}")
            strengths = [0.0] * len(method_to_id)

        _print_strengths_with_names(strengths, id_to_method)
        print("Wins by method pair:")
        for method_a, method_b, wins_a, wins_b in _compute_pair_win_loss(question_data, id_to_method):
            print(f"  {method_a} vs {method_b}: {wins_a} - {wins_b}")


if __name__ == "__main__":
    main()
