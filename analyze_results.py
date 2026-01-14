"""Analysis and visualization of reinforcement learning experiment results.

This script aggregates results from multiple training runs, computes statistics,
generates learning curves, and creates comparison plots and policy visualizations.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def read_csv_rows(file_path: str | Path) -> list[dict]:
    """Read all rows from a CSV file as dictionaries.

    Args:
        file_path: Path to the CSV file.

    Returns:
        List of dictionaries, one per row, with keys from the CSV header.
    """
    with open(file_path, "r", newline="") as file_handle:
        csv_reader = csv.DictReader(file_handle)
        return list(csv_reader)


def to_float(value: str, default: float = np.nan) -> float:
    """Safely convert a string to float, returning a default on failure.

    Args:
        value: String value to convert.
        default: Value to return if conversion fails.

    Returns:
        Float value or default if conversion fails.
    """
    try:
        return float(value)
    except Exception:
        return default


def find_runs(results_directory: str | Path) -> list[dict]:
    """Find and parse all training runs in a results directory.

    Searches recursively for config.json files, loads their configurations and
    corresponding metrics CSV files, and returns run information.

    Args:
        results_directory: Root directory containing training run subdirectories.

    Returns:
        List of dictionaries, each containing run metadata and metrics.
    """
    runs = []
    for config_path in Path(results_directory).rglob("config.json"):
        run_directory = config_path.parent
        metrics_path = run_directory / "metrics.csv"
        if not metrics_path.exists():
            continue

        config = json.loads(config_path.read_text())
        metrics_rows = read_csv_rows(metrics_path)
        if not metrics_rows:
            continue

        last_row = metrics_rows[-1]
        algorithm = config.get("algo")
        reward_mode = config.get("reward")
        seed = int(config.get("seed", -1))

        runs.append(
            {
                "run_dir": str(run_directory),
                "algo": algorithm,
                "reward": reward_mode,
                "seed": seed,
                "cfg": config,
                "rows": metrics_rows,
                "last": last_row,
            }
        )
    return runs


def extract_final_metrics(run: dict) -> dict:
    """Extract final evaluation metrics from a training run.

    Args:
        run: Dictionary containing run information including the last metrics row.

    Returns:
        Dictionary of final metrics (step, eval return, win/draw/loss rates, etc.).
    """
    last_row = run["last"]
    return {
        "final_step": int(float(last_row.get("step", 0))),
        "final_eval_return": to_float(last_row.get("eval_mean_return")),
        "win_rate": to_float(last_row.get("win_rate")),
        "draw_rate": to_float(last_row.get("draw_rate")),
        "loss_rate": to_float(last_row.get("loss_rate")),
        "mean_len": to_float(last_row.get("mean_len")),
    }


def write_csv(
    file_path: str | Path, fieldnames: list[str], rows: list[dict]
) -> None:
    """Write rows to a CSV file.

    Args:
        file_path: Path where CSV will be written.
        fieldnames: List of column names.
        rows: List of dictionaries to write as rows.
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", newline="") as file_handle:
        csv_writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        csv_writer.writeheader()
        for row in rows:
            csv_writer.writerow(row)


def aggregate_final_metrics(runs: list[dict]) -> list[dict]:
    """Aggregate final metrics across seeds for each (algorithm, reward) pair.

    Args:
        runs: List of run dictionaries.

    Returns:
        List of aggregated results with mean and std across seeds.
    """
    metrics_by_config = defaultdict(list)
    for run in runs:
        key = (run["algo"], run["reward"])
        metrics_by_config[key].append(extract_final_metrics(run))

    aggregated_results = []
    for (algorithm, reward_mode), metrics_list in sorted(metrics_by_config.items()):
        returns_array = np.array([metric["final_eval_return"] for metric in metrics_list], dtype=float)
        win_rates_array = np.array([metric["win_rate"] for metric in metrics_list], dtype=float)
        draw_rates_array = np.array([metric["draw_rate"] for metric in metrics_list], dtype=float)
        loss_rates_array = np.array([metric["loss_rate"] for metric in metrics_list], dtype=float)
        lengths_array = np.array([metric["mean_len"] for metric in metrics_list], dtype=float)

        aggregated_results.append(
            {
                "algo": algorithm,
                "reward": reward_mode,
                "n_seeds": len(metrics_list),
                "mean_eval_return": float(np.nanmean(returns_array)),
                "std_eval_return": float(np.nanstd(returns_array)),
                "mean_win_rate": float(np.nanmean(win_rates_array)),
                "std_win_rate": float(np.nanstd(win_rates_array)),
                "mean_draw_rate": float(np.nanmean(draw_rates_array)),
                "std_draw_rate": float(np.nanstd(draw_rates_array)),
                "mean_loss_rate": float(np.nanmean(loss_rates_array)),
                "std_loss_rate": float(np.nanstd(loss_rates_array)),
                "mean_ep_len": float(np.nanmean(lengths_array)),
                "std_ep_len": float(np.nanstd(lengths_array)),
            }
        )
    return aggregated_results


def best_reward_per_algo(aggregated_rows: list[dict]) -> dict:
    """Find the best reward mode for each algorithm.

    Args:
        aggregated_rows: List of aggregated result dictionaries.

    Returns:
        Dictionary mapping algorithm name to its best configuration.
    """
    best_configs: dict[str, dict[str, Any]] = {}
    for row in aggregated_rows:
        algorithm = row["algo"]
        if algorithm not in best_configs or row["mean_eval_return"] > best_configs[algorithm]["mean_eval_return"]:
            best_configs[algorithm] = row
    return best_configs


def extract_learning_curves(runs: list[dict]) -> dict:
    """Extract learning curves (eval return vs step) from all runs.

    Args:
        runs: List of run dictionaries.

    Returns:
        Nested dictionary: {(algo, reward): {seed: [(step, value), ...]}}
    """
    learning_curves: dict[tuple[str, str], dict[int, list[tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        key = (run["algo"], run["reward"])
        seed = run["seed"]
        points = []
        for row in run["rows"]:
            step = int(float(row.get("step", 0)))
            value = to_float(row.get("eval_mean_return"))
            if math.isnan(value):
                continue
            points.append((step, value))
        points.sort(key=lambda point: point[0])
        learning_curves[key][seed] = points
    return learning_curves


def aggregate_learning_curve(curve_data_for_config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate learning curves across seeds.

    Args:
        curve_data_for_config: Dictionary mapping seed to list of (step, value) points.

    Returns:
        Tuple of (steps, means, stds) as numpy arrays.
    """
    step_to_values = defaultdict(list)
    for _, points in curve_data_for_config.items():
        for step, value in points:
            step_to_values[step].append(value)

    steps = np.array(sorted(step_to_values.keys()), dtype=int)
    means = np.array([np.mean(step_to_values[step]) for step in steps], dtype=float)
    std_devs = np.array([np.std(step_to_values[step]) for step in steps], dtype=float)

    return steps, means, std_devs


def plot_learning_curves(learning_curves: dict, output_directory: str | Path) -> None:
    """Plot and save learning curves for each algorithm.

    Args:
        learning_curves: Dictionary from extract_learning_curves.
        output_directory: Directory where plots will be saved.
    """
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    algorithms = sorted({key[0] for key in learning_curves.keys()})
    reward_modes = sorted({key[1] for key in learning_curves.keys()})

    for algorithm in algorithms:
        figure = plt.figure(figsize=(8, 5), constrained_layout=True)
        axis = figure.add_subplot(111)

        for reward_mode in reward_modes:
            key = (algorithm, reward_mode)
            if key not in learning_curves:
                continue

            steps, means, std_devs = aggregate_learning_curve(learning_curves[key])
            axis.plot(steps, means, label=reward_mode)
            axis.fill_between(steps, means - std_devs, means + std_devs, alpha=0.15)

        axis.set_title(f"Learning Curves: {algorithm}")
        axis.set_xlabel("training step")
        axis.set_ylabel("eval mean return (true reward)")
        axis.legend()
        figure.savefig(output_path / f"curve_{algorithm}.png", dpi=160)
        plt.close(figure)


def plot_best_comparison(learning_curves: dict, best_configs: dict, output_directory: str | Path) -> None:
    """Plot comparison of best configurations for each algorithm.

    Args:
        learning_curves: Dictionary from extract_learning_curves.
        best_configs: Dictionary mapping algorithm to best configuration.
        output_directory: Directory where plot will be saved.
    """
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(8, 5), constrained_layout=True)
    axis = figure.add_subplot(111)

    for algorithm, config_row in best_configs.items():
        reward_mode = config_row["reward"]
        key = (algorithm, reward_mode)
        if key not in learning_curves:
            continue

        steps, means, std_devs = aggregate_learning_curve(learning_curves[key])
        axis.plot(steps, means, label=f"{algorithm} ({reward_mode})")
        axis.fill_between(steps, means - std_devs, means + std_devs, alpha=0.15)

    axis.set_title("Best Reward Variant per Algorithm")
    axis.set_xlabel("training step")
    axis.set_ylabel("eval mean return (true reward)")
    axis.legend()
    figure.savefig(output_path / "comparison_best.png", dpi=160)
    plt.close(figure)


def pick_run(runs: list[dict], algorithm: str, reward_mode: str, preferred_seed: int = 0) -> dict | None:
    """Select a specific run from the results.

    Args:
        runs: List of run dictionaries.
        algorithm: Algorithm name to filter by.
        reward_mode: Reward mode to filter by.
        preferred_seed: Preferred seed to select if multiple runs match.

    Returns:
        Matching run dictionary, or None if no match found.
    """
    candidates = [run for run in runs if run["algo"] == algorithm and run["reward"] == reward_mode]
    if not candidates:
        return None

    for run in candidates:
        if run["seed"] == preferred_seed:
            return run

    return sorted(candidates, key=lambda run: run["seed"])[0]


def find_policy_image(run: dict | None) -> str | None:
    """Find the policy visualization image for a run.

    Args:
        run: Run dictionary containing run_dir.

    Returns:
        Path to policy image file, or None if not found.
    """
    if run is None:
        return None

    figures_directory = Path(run["run_dir"]) / "figures"
    if not figures_directory.exists():
        return None

    for image_name in ["policy_doubleq.png", "policy_a2c.png"]:
        image_path = figures_directory / image_name
        if image_path.exists():
            return str(image_path)

    # Fallback: any policy*.png file
    policy_images = sorted(figures_directory.glob("policy*.png"))
    return str(policy_images[0]) if policy_images else None


def create_policy_montage(runs: list[dict], best_configs: dict, output_directory: str | Path) -> None:
    """Create a montage of policy visualizations.

    Args:
        runs: List of run dictionaries.
        best_configs: Dictionary mapping algorithm to best configuration.
        output_directory: Directory where montage will be saved.
    """
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    algorithm_list = ["doubleq", "a2c"]
    image_paths = []
    subplot_titles = []

    for algorithm in algorithm_list:
        # Get r0 (baseline) run
        baseline_run = pick_run(runs, algorithm, "r0", preferred_seed=0)
        best_reward_mode = best_configs.get(algorithm, {}).get("reward", "r0")
        best_run = pick_run(runs, algorithm, best_reward_mode, preferred_seed=0)

        baseline_image = find_policy_image(baseline_run)
        best_image = find_policy_image(best_run)

        image_paths.append(baseline_image)
        subplot_titles.append(f"{algorithm} r0")
        image_paths.append(best_image)
        subplot_titles.append(f"{algorithm} {best_reward_mode}")

    figure = plt.figure(figsize=(12, 8), constrained_layout=True)
    for subplot_index in range(4):
        axis = figure.add_subplot(2, 2, subplot_index + 1)
        image_path = image_paths[subplot_index]
        if image_path is None:
            axis.text(0.5, 0.5, "missing", ha="center", va="center")
            axis.set_axis_off()
            axis.set_title(subplot_titles[subplot_index])
            continue

        image_data = mpimg.imread(image_path)
        axis.imshow(image_data)
        axis.set_axis_off()
        axis.set_title(subplot_titles[subplot_index])

    figure.savefig(output_path / "policy_montage.png", dpi=160)
    plt.close(figure)


def write_report(aggregated_rows: list[dict], best_configs: dict, output_directory: str | Path) -> None:
    """Generate a markdown report summarizing results.

    Args:
        aggregated_rows: List of aggregated result dictionaries.
        best_configs: Dictionary mapping algorithm to best configuration.
        output_directory: Directory where report will be saved.
    """
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    def format_mean_std(mean: float, std_dev: float) -> str:
        return f"{mean:.4f} ± {std_dev:.4f}"

    lines = []
    lines.append("# Blackjack Results Summary\n")
    lines.append("## Best reward per algorithm\n")

    for algorithm, config_row in best_configs.items():
        lines.append(
            f"- **{algorithm}** best reward: **{config_row['reward']}** "
            f"(mean return {config_row['mean_eval_return']:.4f})\n"
        )

    lines.append("\n## Final metrics (mean ± std over seeds)\n")
    lines.append(
        "| algo | reward | n | eval_return | win_rate | draw_rate | loss_rate | ep_len |\n"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")

    for row in aggregated_rows:
        lines.append(
            f"| {row['algo']} | {row['reward']} | {row['n_seeds']} | "
            f"{format_mean_std(row['mean_eval_return'], row['std_eval_return'])} | "
            f"{format_mean_std(row['mean_win_rate'], row['std_win_rate'])} | "
            f"{format_mean_std(row['mean_draw_rate'], row['std_draw_rate'])} | "
            f"{format_mean_std(row['mean_loss_rate'], row['std_loss_rate'])} | "
            f"{format_mean_std(row['mean_ep_len'], row['std_ep_len'])} |\n"
        )

    (output_path / "report.md").write_text("".join(lines))


def main() -> None:
    """Main analysis script entry point.

    Parses command-line arguments, loads all training runs, computes aggregate
    statistics, and generates all visualizations and reports.
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--results_dir", type=str, default="results")
    argument_parser.add_argument("--outdir", type=str, default="analysis")
    args = argument_parser.parse_args()

    runs = find_runs(args.results_dir)
    if not runs:
        raise SystemExit(f"No runs found under {args.results_dir}")

    output_directory = Path(args.outdir)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Create per-run summary
    run_summary_rows = []
    for run in runs:
        metrics = extract_final_metrics(run)
        run_summary_rows.append(
            {
                "algo": run["algo"],
                "reward": run["reward"],
                "seed": run["seed"],
                "final_step": metrics["final_step"],
                "final_eval_return": metrics["final_eval_return"],
                "win_rate": metrics["win_rate"],
                "draw_rate": metrics["draw_rate"],
                "loss_rate": metrics["loss_rate"],
                "mean_len": metrics["mean_len"],
                "run_dir": run["run_dir"],
            }
        )

    write_csv(
        output_directory / "summary_by_run.csv",
        [
            "algo",
            "reward",
            "seed",
            "final_step",
            "final_eval_return",
            "win_rate",
            "draw_rate",
            "loss_rate",
            "mean_len",
            "run_dir",
        ],
        sorted(run_summary_rows, key=lambda row: (row["algo"], row["reward"], row["seed"])),
    )

    # Create aggregated summary
    aggregated_rows = aggregate_final_metrics(runs)
    write_csv(
        output_directory / "summary_by_algo_reward.csv",
        list(aggregated_rows[0].keys()),
        aggregated_rows,
    )

    # Find best configurations
    best_configs = best_reward_per_algo(aggregated_rows)

    # Generate all visualizations
    learning_curves = extract_learning_curves(runs)
    plot_learning_curves(learning_curves, output_directory / "figures")
    plot_best_comparison(learning_curves, best_configs, output_directory / "figures")
    create_policy_montage(runs, best_configs, output_directory / "figures")
    write_report(aggregated_rows, best_configs, output_directory)

    print("Wrote:")
    print(" -", output_directory / "summary_by_run.csv")
    print(" -", output_directory / "summary_by_algo_reward.csv")
    print(" -", output_directory / "report.md")
    print(" -", output_directory / "figures" / "curve_doubleq.png")
    print(" -", output_directory / "figures" / "curve_a2c.png")
    print(" -", output_directory / "figures" / "comparison_best.png")
    print(" -", output_directory / "figures" / "policy_montage.png")


if __name__ == "__main__":
    main()
