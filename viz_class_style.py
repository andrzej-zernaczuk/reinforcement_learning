import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.a2c_gae import A2CConfig, A2CGAEAgent
from src.doubleq import DoubleQAgent, DoubleQConfig
from src.features import OBS_DIM, obs_to_onehot


def make_surface_data(df: pd.DataFrame, value_col: str):
    xs = np.array(sorted(df["player_sum"].unique()))
    ys = np.array(sorted(df["dealer_card"].unique()))
    Z = np.full((xs.size, ys.size), np.nan, dtype=float)
    lut = {
        (int(r.player_sum), int(r.dealer_card)): float(r[value_col])
        for _, r in df.iterrows()
    }
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            Z[i, j] = lut.get((int(x), int(y)), np.nan)
    X, Y = np.meshgrid(ys, xs)
    return X, Y, Z


def plot_surface(
    df: pd.DataFrame,
    usable_ace: bool,
    value_col: str,
    title: str,
    outpath: str,
    invert_x=True,
):
    d = df[df["usable_ace"] == usable_ace].copy()
    d = d[(d["player_sum"] >= 12) & (d["player_sum"] <= 21)]
    X, Y, Z = make_surface_data(d, value_col=value_col)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True)

    ax.set_xlabel("dealer_showing")
    ax.set_ylabel("player_sum")
    ax.set_zlabel(value_col)
    ax.set_title(f"{title} | usable_ace={usable_ace}")
    if invert_x:
        ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_epsilon(cfg: DoubleQConfig, outpath: str, episodes=200000):
    eps = []
    for ep in range(episodes + 1):
        if cfg.eps_decay_episodes <= 0:
            e = cfg.eps_end
        else:
            frac = min(1.0, ep / float(cfg.eps_decay_episodes))
            e = cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)
        eps.append(e)

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    ax.plot(np.arange(episodes + 1), eps)
    ax.set_xlabel("episode")
    ax.set_ylabel("epsilon")
    ax.set_title("Epsilon schedule (DoubleQ)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def build_doubleq_df(ckpt_path: str):
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)

    cfg = DoubleQConfig(**ckpt["cfg"])
    agent = DoubleQAgent(n_actions=ckpt["n_actions"], cfg=cfg, seed=0)
    agent.QA.update(ckpt["QA"])
    agent.QB.update(ckpt["QB"])

    rows = []
    for usable in [False, True]:
        for ps in range(4, 22):
            for dc in range(1, 11):
                s = (ps, dc, usable)
                qa = agent.QA[s]
                qb = agent.QB[s]
                q = (qa + qb) / 2.0
                rows.append(
                    {
                        "player_sum": ps,
                        "dealer_card": dc,
                        "usable_ace": usable,
                        "Q_stick": float(q[0]),
                        "Q_hit": float(q[1]),
                        "V": float(np.max(q)),
                    }
                )
    return pd.DataFrame(rows), cfg


def build_a2c_df(ckpt_path: str, cfg: A2CConfig):
    agent = A2CGAEAgent(obs_dim=OBS_DIM, n_actions=2, cfg=cfg, seed=0)
    agent.load(ckpt_path)

    net = getattr(agent, "net", None)
    if net is None:
        raise AttributeError("Could not find agent.net on A2CGAEAgent")

    net.eval()
    device = getattr(net, "device", None)
    if device is None:
        device = (
            torch.device(cfg.device) if hasattr(cfg, "device") else torch.device("cpu")
        )

    rows = []
    for usable in [False, True]:
        for ps in range(4, 22):
            for dc in range(1, 11):
                obs = (ps, dc, int(usable))
                x = obs_to_onehot(obs)

                x_t = torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(
                    0
                )

                out = net(x_t)
                if isinstance(out, tuple) and len(out) == 2:
                    logits, val = out
                elif isinstance(out, dict):
                    logits = out.get(
                        "logits", out.get("pi_logits", out.get("policy_logits"))
                    )
                    val = out.get("value", out.get("v"))
                    if logits is None or val is None:
                        raise ValueError(
                            f"Unexpected net dict keys: {list(out.keys())}"
                        )
                else:
                    raise ValueError("Unexpected net output type/format")

                probs = torch.softmax(logits, dim=-1).detach().cpu().numpy().reshape(-1)

                rows.append(
                    {
                        "player_sum": ps,
                        "dealer_card": dc,
                        "usable_ace": usable,
                        "V": float(val.detach().cpu().item()),
                        "pi_hit": float(probs[1]),
                        "pi_stick": float(probs[0]),
                    }
                )

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["doubleq", "a2c"], required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="viz_class")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.algo == "doubleq":
        df, cfg = build_doubleq_df(args.checkpoint)
        plot_surface(
            df, False, "V", "DoubleQ: V(s)", str(outdir / "doubleq_V_false.png")
        )
        plot_surface(df, True, "V", "DoubleQ: V(s)", str(outdir / "doubleq_V_true.png"))
        plot_surface(
            df,
            False,
            "Q_stick",
            "DoubleQ: Q(s,stick)",
            str(outdir / "doubleq_Qstick_false.png"),
        )
        plot_surface(
            df,
            True,
            "Q_stick",
            "DoubleQ: Q(s,stick)",
            str(outdir / "doubleq_Qstick_true.png"),
        )
        plot_surface(
            df,
            False,
            "Q_hit",
            "DoubleQ: Q(s,hit)",
            str(outdir / "doubleq_Qhit_false.png"),
        )
        plot_surface(
            df,
            True,
            "Q_hit",
            "DoubleQ: Q(s,hit)",
            str(outdir / "doubleq_Qhit_true.png"),
        )
        plot_epsilon(cfg, str(outdir / "doubleq_epsilon.png"), episodes=200000)

        df.to_csv(outdir / "doubleq_values.csv", index=False)

    else:
        cfg = A2CConfig(
            lr=0.0010948770705738267,
            gamma=0.95,
            gae_lambda=0.97,
            entropy_coef=0.0,
            hidden_sizes=(64, 64),
            device=args.device,
        )
        df = build_a2c_df(args.checkpoint, cfg)
        plot_surface(df, False, "V", "A2C-GAE: V(s)", str(outdir / "a2c_V_false.png"))
        plot_surface(df, True, "V", "A2C-GAE: V(s)", str(outdir / "a2c_V_true.png"))
        plot_surface(
            df,
            False,
            "pi_hit",
            "A2C-GAE: π(hit|s)",
            str(outdir / "a2c_pi_hit_false.png"),
        )
        plot_surface(
            df, True, "pi_hit", "A2C-GAE: π(hit|s)", str(outdir / "a2c_pi_hit_true.png")
        )

        df.to_csv(outdir / "a2c_values.csv", index=False)

    print(f"Saved figures to: {outdir}")


if __name__ == "__main__":
    main()
