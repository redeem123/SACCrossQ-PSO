from __future__ import annotations

import argparse
import csv
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import friedmanchisquare, rankdata

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }
)


@dataclass
class ScenarioRankingResult:
    scenario_id: int
    source_path: Path
    algorithms: list[str]
    values: np.ndarray
    ranks_per_run: np.ndarray
    avg_ranks: np.ndarray
    std_ranks: np.ndarray
    chi_square: float
    p_value: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run-level Friedman ranking analysis from scenario CSV exports."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory containing TTest_AllRuns_Scenario*.csv outputs.",
    )
    parser.add_argument(
        "--scenarios",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Scenario IDs to process (default: 1 2 3).",
    )
    return parser.parse_args()


def scenario_csv_candidates(results_dir: Path, scenario_id: int) -> list[Path]:
    candidates: list[Path] = []

    deterministic = results_dir / f"TTest_AllRuns_Scenario{scenario_id}.csv"
    if deterministic.exists():
        candidates.append(deterministic)

    candidates.extend(results_dir.glob(f"TTest_AllRuns_Scenario{scenario_id}_*.csv"))
    unique_candidates = sorted(
        {candidate.resolve() for candidate in candidates},
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return unique_candidates


def _pick_first_key(fieldnames: Iterable[str], candidates: Iterable[str]) -> str | None:
    normalized = {name.strip().lower(): name for name in fieldnames}
    for candidate in candidates:
        key = normalized.get(candidate.lower())
        if key:
            return key
    return None


def load_run_level_matrix(csv_path: Path) -> tuple[list[str], np.ndarray]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")

        run_key = _pick_first_key(reader.fieldnames, ["Run", "run"])
        alg_key = _pick_first_key(reader.fieldnames, ["Algorithm", "algorithm"])
        fitness_key = _pick_first_key(
            reader.fieldnames,
            [
                "GlobalPathFitness",
                "globalpathfitness",
                "global_fitness_total",
                "MetricValue",
            ],
        )

        if not run_key or not alg_key or not fitness_key:
            raise ValueError(
                f"CSV missing required columns (Run/Algorithm/GlobalPathFitness): {csv_path}"
            )

        run_to_values: dict[str, dict[str, float]] = {}
        algorithm_order: list[str] = []

        for row in reader:
            run_value = row.get(run_key, "")
            alg = row.get(alg_key, "")
            metric_value = row.get(fitness_key, "")

            if not run_value or not alg:
                continue

            try:
                score = float(metric_value)
            except (TypeError, ValueError):
                continue

            if not math.isfinite(score):
                continue

            if alg not in algorithm_order:
                algorithm_order.append(alg)

            run_to_values.setdefault(run_value, {})[alg] = score

    if not run_to_values:
        raise ValueError(f"No usable run-level rows in {csv_path}")

    common_algorithms = set.intersection(*(set(v.keys()) for v in run_to_values.values()))
    algorithms = [alg for alg in algorithm_order if alg in common_algorithms]

    if len(algorithms) < 2:
        raise ValueError(
            f"Need at least two algorithms with complete run pairing in {csv_path}"
        )

    def run_sort_key(value: str):
        try:
            return (0, int(value))
        except ValueError:
            return (1, value)

    complete_runs: list[str] = []
    for run_id in sorted(run_to_values.keys(), key=run_sort_key):
        run_entry = run_to_values[run_id]
        if all(alg in run_entry for alg in algorithms):
            complete_runs.append(run_id)

    if len(complete_runs) < 2:
        raise ValueError(
            f"Need at least two complete paired runs for Friedman test in {csv_path}"
        )

    values = np.array(
        [[run_to_values[run_id][alg] for alg in algorithms] for run_id in complete_runs],
        dtype=float,
    )

    return algorithms, values


def compute_ranks(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ranks_per_run = np.array([rankdata(row, method="average") for row in values], dtype=float)
    avg_ranks = ranks_per_run.mean(axis=0)
    std_ranks = ranks_per_run.std(axis=0)
    return ranks_per_run, avg_ranks, std_ranks


def run_friedman(values: np.ndarray) -> tuple[float, float]:
    if values.shape[0] < 2 or values.shape[1] < 2:
        return float("nan"), float("nan")
    stat, pval = friedmanchisquare(*[values[:, i] for i in range(values.shape[1])])
    return float(stat), float(pval)


def plot_ranks(
    algorithms: list[str],
    avg_ranks: np.ndarray,
    std_ranks: np.ndarray,
    title: str | None,
    output_path: Path,
) -> None:
    order = np.argsort(avg_ranks)
    algs_sorted = [algorithms[i] for i in order]
    avg_sorted = avg_ranks[order]
    std_sorted = std_ranks[order]

    fig_height = max(3.5, 0.45 * len(algs_sorted))
    fig, ax = plt.subplots(figsize=(7.5, fig_height))
    ax.barh(
        algs_sorted,
        avg_sorted,
        color="#4C78A8",
        edgecolor="#2B2B2B",
        linewidth=0.6,
        alpha=0.85,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Average Rank (lower is better)")
    if title:
        ax.set_title(title)

    ax.grid(axis="x", linestyle="--", alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)

    for y_idx, (val, std_val) in enumerate(zip(avg_sorted, std_sorted, strict=False)):
        ax.text(val + 0.03, y_idx, f"{val:.2f} ± {std_val:.2f}", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_scenario_summary(result: ScenarioRankingResult, output_path: Path) -> None:
    lines = [
        f"Scenario {result.scenario_id} Friedman test (run-level paired fitness)",
        f"Source: {result.source_path}",
        f"Runs used: {result.values.shape[0]}",
        f"Algorithms: {result.values.shape[1]}",
        f"Chi-square statistic: {result.chi_square:.6f}",
        f"p-value: {result.p_value:.6g}",
        "",
        "Average ranks (lower is better):",
    ]

    for idx in np.argsort(result.avg_ranks):
        lines.append(
            f"{result.algorithms[idx]}: {result.avg_ranks[idx]:.4f} ± {result.std_ranks[idx]:.4f}"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_combined_descriptive_summary(
    output_path: Path,
    algorithms: list[str],
    avg_ranks: np.ndarray,
    std_ranks: np.ndarray,
    scenario_ids: list[int],
    pooled_runs: int,
) -> None:
    lines = [
        "Combined average ranks (descriptive only; pooled run-level ranks across scenarios)",
        f"Scenarios pooled: {', '.join(str(s) for s in scenario_ids)}",
        f"Total pooled run blocks: {pooled_runs}",
        "Inferential claims should use per-scenario Friedman tests.",
        "",
        "Average ranks (lower is better):",
    ]

    for idx in np.argsort(avg_ranks):
        lines.append(f"{algorithms[idx]}: {avg_ranks[idx]:.4f} ± {std_ranks[idx]:.4f}")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def copy_combined_figure_to_paper(base_dir: Path, combined_figure: Path) -> None:
    paper_figures_dir = base_dir / "paper" / "figures" / "statistics"
    paper_figures_dir.mkdir(parents=True, exist_ok=True)
    destination = paper_figures_dir / combined_figure.name
    shutil.copy2(combined_figure, destination)


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parents[1]
    results_dir = args.results_dir or (base_dir / "outputs" / "results")
    results_dir = results_dir.resolve()

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    scenario_results: list[ScenarioRankingResult] = []

    for scenario_id in args.scenarios:
        candidates = scenario_csv_candidates(results_dir, scenario_id)
        if not candidates:
            print(f"[skip] Scenario {scenario_id}: no TTest_AllRuns CSV found")
            continue

        selected_path: Path | None = None
        selected_algorithms: list[str] | None = None
        selected_values: np.ndarray | None = None
        best_runs = -1
        candidate_errors: list[str] = []

        for candidate in candidates:
            try:
                algorithms, values = load_run_level_matrix(candidate)
            except ValueError as exc:
                candidate_errors.append(f"{candidate.name}: {exc}")
                continue

            num_runs = values.shape[0]
            if num_runs > best_runs:
                best_runs = num_runs
                selected_path = candidate
                selected_algorithms = algorithms
                selected_values = values

        if selected_path is None or selected_algorithms is None or selected_values is None:
            print(f"[skip] Scenario {scenario_id}: no usable run-level CSV candidate")
            for message in candidate_errors:
                print(f"  - {message}")
            continue

        csv_path = selected_path
        algorithms = selected_algorithms
        values = selected_values

        ranks_per_run, avg_ranks, std_ranks = compute_ranks(values)
        chi_square, p_value = run_friedman(values)

        scenario_result = ScenarioRankingResult(
            scenario_id=scenario_id,
            source_path=csv_path,
            algorithms=algorithms,
            values=values,
            ranks_per_run=ranks_per_run,
            avg_ranks=avg_ranks,
            std_ranks=std_ranks,
            chi_square=chi_square,
            p_value=p_value,
        )
        scenario_results.append(scenario_result)

        scenario_plot = results_dir / f"Friedman_Ranking_Scenario{scenario_id}.pdf"
        scenario_summary = results_dir / f"Friedman_Test_Scenario{scenario_id}.txt"

        plot_ranks(
            algorithms,
            avg_ranks,
            std_ranks,
            f"Scenario {scenario_id}",
            scenario_plot,
        )
        write_scenario_summary(scenario_result, scenario_summary)

        print(
            f"Scenario {scenario_id}: chi2={chi_square:.4f}, p={p_value:.6g}, "
            f"runs={values.shape[0]}, algs={values.shape[1]}"
        )

    if not scenario_results:
        raise RuntimeError("No scenarios had usable run-level data for Friedman analysis.")

    common_algorithms = set(scenario_results[0].algorithms)
    for scenario_result in scenario_results[1:]:
        common_algorithms &= set(scenario_result.algorithms)

    if len(common_algorithms) < 2:
        raise RuntimeError(
            "Unable to build combined descriptive ranking: fewer than two shared algorithms across scenarios."
        )

    ordered_common = [
        alg for alg in scenario_results[0].algorithms if alg in common_algorithms
    ]

    pooled_rank_rows = []
    pooled_scenarios = []
    for scenario_result in scenario_results:
        index_map = [scenario_result.algorithms.index(alg) for alg in ordered_common]
        pooled_rank_rows.append(scenario_result.ranks_per_run[:, index_map])
        pooled_scenarios.append(scenario_result.scenario_id)

    pooled_ranks = np.vstack(pooled_rank_rows)
    combined_avg = pooled_ranks.mean(axis=0)
    combined_std = pooled_ranks.std(axis=0)

    combined_plot = results_dir / "Friedman_Ranking_Combined.pdf"
    combined_summary = results_dir / "Friedman_Test_Combined.txt"

    plot_ranks(
        ordered_common,
        combined_avg,
        combined_std,
        "Combined (Descriptive Pooled Ranks)",
        combined_plot,
    )
    write_combined_descriptive_summary(
        combined_summary,
        ordered_common,
        combined_avg,
        combined_std,
        pooled_scenarios,
        pooled_ranks.shape[0],
    )

    default_results_dir = (base_dir / "outputs" / "results").resolve()
    if results_dir == default_results_dir:
        copy_combined_figure_to_paper(base_dir, combined_plot)
    else:
        print(
            "Skipped paper figure sync because a non-default results directory was used:"
            f" {results_dir}"
        )

    print(
        "Combined descriptive ranking generated from pooled run-level ranks "
        f"across scenarios {pooled_scenarios} (runs={pooled_ranks.shape[0]})."
    )


if __name__ == "__main__":
    main()
