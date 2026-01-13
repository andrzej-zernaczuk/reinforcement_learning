# Advanced Reinforcement Learning on Gymnasium Blackjack (Double Q-learning vs A2C-GAE)

This project compares two RL paradigms on `Blackjack-v1`:

- **Value-based TD (tabular): Double Q-learning**
- **Actor-Critic (neural): A2C with GAE(λ)**

It also supports **reward shaping** variants (R0/R1/R2/R3) and **limited systematic tuning**.

## Install

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

python -m pip install -U pip
pip install -r requirements.txt
```
or
```bash
uv sync
```

> NOTE: Blackjack doesn't need pygame. Avoid `gymnasium[classic-control]` for this project.

## Quick start (single run)

Train **DoubleQ** on baseline reward (R0) and evaluate:

```bash
python -m src.run --algo doubleq --reward r0 --train_episodes 200000 --eval_episodes 20000 --seed 0
```

Train **A2C-GAE** on baseline reward (R0) and evaluate:

```bash
python -m src.run --algo a2c --reward r0 --train_steps 300000 --rollout_steps 256 --eval_episodes 20000 --seed 0
```

Reward modes:

- `r0`: baseline env reward
- `r1`: step penalty (use `--step_penalty`)
- `r2`: extra bust penalty (use `--bust_penalty`)
- `r3`: potential-based shaping (use `--gamma` and potential based on player_sum)

## Limited systematic tuning

### DoubleQ grid search (small)

```bash
python -m src.tune --algo doubleq --reward r0 --episodes 150000 --eval_episodes 20000
```

### A2C random search (small)

```bash
python -m src.tune --algo a2c --reward r0 --steps 200000 --eval_episodes 20000 --trials 12
```

All results are written to `results/` as CSV + JSON configs.

## Outputs to share with me

- `results/*.csv`
- `figures/*.png` (optional)
- the printed "BEST" config line from tuning
