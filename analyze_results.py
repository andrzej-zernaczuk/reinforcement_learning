import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def read_csv_rows(path):
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        return list(r)


def to_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def find_runs(results_dir):
    runs = []
    for cfg_path in Path(results_dir).rglob("config.json"):
        run_dir = cfg_path.parent
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        rows = read_csv_rows(metrics_path)
        if not rows:
            continue
        last = rows[-1]
        algo = cfg.get("algo")
        reward = cfg.get("reward")
        seed = int(cfg.get("seed", -1))
        runs.append(
            {
                "run_dir": str(run_dir),
                "algo": algo,
                "reward": reward,
                "seed": seed,
                "cfg": cfg,
                "rows": rows,
                "last": last,
            }
        )
    return runs


def extract_final_metrics(run):
    last = run["last"]
    return {
        "final_step": int(float(last.get("step", 0))),
        "final_eval_return": to_float(last.get("eval_mean_return")),
        "win_rate": to_float(last.get("win_rate")),
        "draw_rate": to_float(last.get("draw_rate")),
        "loss_rate": to_float(last.get("loss_rate")),
        "mean_len": to_float(last.get("mean_len")),
    }


def write_csv(path, fieldnames, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def agg_final(runs):
    by = defaultdict(list)
    for run in runs:
        key = (run["algo"], run["reward"])
        by[key].append(extract_final_metrics(run))

    out = []
    for (algo, reward), ms in sorted(by.items()):
        arr_ret = np.array([m["final_eval_return"] for m in ms], dtype=float)
        arr_win = np.array([m["win_rate"] for m in ms], dtype=float)
        arr_draw = np.array([m["draw_rate"] for m in ms], dtype=float)
        arr_loss = np.array([m["loss_rate"] for m in ms], dtype=float)
        arr_len = np.array([m["mean_len"] for m in ms], dtype=float)
        out.append(
            {
                "algo": algo,
                "reward": reward,
                "n_seeds": len(ms),
                "mean_eval_return": float(np.nanmean(arr_ret)),
                "std_eval_return": float(np.nanstd(arr_ret)),
                "mean_win_rate": float(np.nanmean(arr_win)),
                "std_win_rate": float(np.nanstd(arr_win)),
                "mean_draw_rate": float(np.nanmean(arr_draw)),
                "std_draw_rate": float(np.nanstd(arr_draw)),
                "mean_loss_rate": float(np.nanmean(arr_loss)),
                "std_loss_rate": float(np.nanstd(arr_loss)),
                "mean_ep_len": float(np.nanmean(arr_len)),
                "std_ep_len": float(np.nanstd(arr_len)),
            }
        )
    return out


def best_reward_per_algo(agg_rows):
    best = {}
    for row in agg_rows:
        algo = row["algo"]
        if algo not in best or row["mean_eval_return"] > best[algo]["mean_eval_return"]:
            best[algo] = row
    return best


def series_by_algo_reward(runs):
    series = defaultdict(lambda: defaultdict(list))
    for run in runs:
        key = (run["algo"], run["reward"])
        seed = run["seed"]
        pts = []
        for row in run["rows"]:
            step = int(float(row.get("step", 0)))
            val = to_float(row.get("eval_mean_return"))
            if math.isnan(val):
                continue
            pts.append((step, val))
        pts.sort(key=lambda x: x[0])
        series[key][seed] = pts
    return series


def aggregate_series(series_for_key):
    step_to_vals = defaultdict(list)
    for _, pts in series_for_key.items():
        for step, val in pts:
            step_to_vals[step].append(val)
    steps = np.array(sorted(step_to_vals.keys()), dtype=int)
    means = np.array([np.mean(step_to_vals[s]) for s in steps], dtype=float)
    stds = np.array([np.std(step_to_vals[s]) for s in steps], dtype=float)
    return steps, means, stds


def plot_curves(series, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    algos = sorted({k[0] for k in series.keys()})
    rewards = sorted({k[1] for k in series.keys()})

    for algo in algos:
        fig = plt.figure(figsize=(8, 5), constrained_layout=True)
        ax = fig.add_subplot(111)
        for reward in rewards:
            key = (algo, reward)
            if key not in series:
                continue
            steps, means, stds = aggregate_series(series[key])
            ax.plot(steps, means, label=reward)
            ax.fill_between(steps, means - stds, means + stds, alpha=0.15)
        ax.set_title(f"Learning Curves: {algo}")
        ax.set_xlabel("training step")
        ax.set_ylabel("eval mean return (true reward)")
        ax.legend()
        fig.savefig(outdir / f"curve_{algo}.png", dpi=160)
        plt.close(fig)


def plot_best_comparison(series, best, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8, 5), constrained_layout=True)
    ax = fig.add_subplot(111)

    for algo, row in best.items():
        reward = row["reward"]
        key = (algo, reward)
        if key not in series:
            continue
        steps, means, stds = aggregate_series(series[key])
        ax.plot(steps, means, label=f"{algo} ({reward})")
        ax.fill_between(steps, means - stds, means + stds, alpha=0.15)

    ax.set_title("Best Reward Variant per Algorithm")
    ax.set_xlabel("training step")
    ax.set_ylabel("eval mean return (true reward)")
    ax.legend()
    fig.savefig(outdir / "comparison_best.png", dpi=160)
    plt.close(fig)


def pick_run(runs, algo, reward, preferred_seed=0):
    candidates = [r for r in runs if r["algo"] == algo and r["reward"] == reward]
    if not candidates:
        return None
    for r in candidates:
        if r["seed"] == preferred_seed:
            return r
    return sorted(candidates, key=lambda r: r["seed"])[0]


def find_policy_image(run):
    if run is None:
        return None
    figdir = Path(run["run_dir"]) / "figures"
    if not figdir.exists():
        return None
    for name in ["policy_doubleq.png", "policy_a2c.png"]:
        p = figdir / name
        if p.exists():
            return str(p)
    pngs = sorted(figdir.glob("policy*.png"))
    return str(pngs[0]) if pngs else None


def montage_policies(runs, best, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    algo_list = ["doubleq", "a2c"]
    imgs = []
    titles = []

    for algo in algo_list:
        r0 = pick_run(runs, algo, "r0", preferred_seed=0)
        best_r = best.get(algo, {}).get("reward", "r0")
        rb = pick_run(runs, algo, best_r, preferred_seed=0)

        p0 = find_policy_image(r0)
        pb = find_policy_image(rb)

        imgs.append(p0)
        titles.append(f"{algo} r0")
        imgs.append(pb)
        titles.append(f"{algo} {best_r}")

    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    for i in range(4):
        ax = fig.add_subplot(2, 2, i + 1)
        if imgs[i] is None:
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
            ax.set_axis_off()
            ax.set_title(titles[i])
            continue
        im = mpimg.imread(imgs[i])
        ax.imshow(im)
        ax.set_axis_off()
        ax.set_title(titles[i])
    fig.savefig(outdir / "policy_montage.png", dpi=160)
    plt.close(fig)


def write_report(agg_rows, best, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def fmt(m, s):
        return f"{m:.4f} ± {s:.4f}"

    lines = []
    lines.append("# Blackjack Results Summary\n")
    lines.append("## Best reward per algorithm\n")
    for algo, row in best.items():
        lines.append(
            f"- **{algo}** best reward: **{row['reward']}** (mean return {row['mean_eval_return']:.4f})\n"
        )

    lines.append("\n## Final metrics (mean ± std over seeds)\n")
    lines.append(
        "| algo | reward | n | eval_return | win_rate | draw_rate | loss_rate | ep_len |\n"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for row in agg_rows:
        lines.append(
            f"| {row['algo']} | {row['reward']} | {row['n_seeds']} | "
            f"{fmt(row['mean_eval_return'], row['std_eval_return'])} | "
            f"{fmt(row['mean_win_rate'], row['std_win_rate'])} | "
            f"{fmt(row['mean_draw_rate'], row['std_draw_rate'])} | "
            f"{fmt(row['mean_loss_rate'], row['std_loss_rate'])} | "
            f"{fmt(row['mean_ep_len'], row['std_ep_len'])} |\n"
        )

    (outdir / "report.md").write_text("".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default="results")
    ap.add_argument("--outdir", type=str, default="analysis")
    args = ap.parse_args()

    runs = find_runs(args.results_dir)
    if not runs:
        raise SystemExit(f"No runs found under {args.results_dir}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    run_rows = []
    for r in runs:
        m = extract_final_metrics(r)
        run_rows.append(
            {
                "algo": r["algo"],
                "reward": r["reward"],
                "seed": r["seed"],
                "final_step": m["final_step"],
                "final_eval_return": m["final_eval_return"],
                "win_rate": m["win_rate"],
                "draw_rate": m["draw_rate"],
                "loss_rate": m["loss_rate"],
                "mean_len": m["mean_len"],
                "run_dir": r["run_dir"],
            }
        )

    write_csv(
        outdir / "summary_by_run.csv",
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
        sorted(run_rows, key=lambda x: (x["algo"], x["reward"], x["seed"])),
    )

    agg_rows = agg_final(runs)
    write_csv(
        outdir / "summary_by_algo_reward.csv",
        list(agg_rows[0].keys()),
        agg_rows,
    )

    best = best_reward_per_algo(agg_rows)

    series = series_by_algo_reward(runs)
    plot_curves(series, outdir / "figures")
    plot_best_comparison(series, best, outdir / "figures")
    montage_policies(runs, best, outdir / "figures")
    write_report(agg_rows, best, outdir)

    print("Wrote:")
    print(" -", outdir / "summary_by_run.csv")
    print(" -", outdir / "summary_by_algo_reward.csv")
    print(" -", outdir / "report.md")
    print(" -", outdir / "figures" / "curve_doubleq.png")
    print(" -", outdir / "figures" / "curve_a2c.png")
    print(" -", outdir / "figures" / "comparison_best.png")
    print(" -", outdir / "figures" / "policy_montage.png")


if __name__ == "__main__":
    main()
