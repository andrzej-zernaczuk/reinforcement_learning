# Advanced Reinforcement Learning on Gymnasium Blackjack

A comprehensive reinforcement learning project comparing two fundamental RL paradigms on the classic Blackjack environment:

- **Value-based Tabular TD**: Double Q-learning
- **Policy-based Neural Actor-Critic**: A2C with Generalized Advantage Estimation (GAE)

This project explores reward shaping strategies and provides systematic hyperparameter tuning capabilities.

## Prerequisites

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management. Install uv first:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Alternative: via pip
pip install uv
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd reinforcement_learning

# Install dependencies (creates virtual environment automatically)
uv sync

# Verify installation
uv run python -c "import gymnasium; print('Ready!')"
```

The project requires Python 3.13+ and will automatically install all dependencies including:
- `gymnasium` - RL environments
- `torch` - Neural network implementation
- `numpy` - Numerical computations
- `matplotlib` - Visualization
- `pandas` - Data analysis
- `tqdm` - Progress bars

## Project Structure

```
reinforcement_learning/
├── src/
│   ├── doubleq.py        # Double Q-learning agent (tabular)
│   ├── a2c_gae.py        # A2C with GAE agent (neural)
│   ├── envs.py           # Environment creation & reward shaping
│   ├── features.py       # Observation encoding (one-hot)
│   ├── training.py       # Shared rollout collection utilities
│   ├── metrics.py        # CSV logging for training metrics
│   ├── eval.py           # Agent evaluation utilities
│   ├── utils.py          # Common utilities (seeds, I/O, etc.)
│   ├── viz.py            # Policy visualization
│   ├── run.py            # Main training script
│   └── tune.py           # Hyperparameter tuning
├── analyze_results.py    # Results aggregation & visualization
├── pyproject.toml        # Project dependencies (uv)
└── README.md             # This file
```

## Quick Start

### Training Single Agents

**Train Double Q-learning** (200K episodes, baseline reward):

```bash
uv run python -m src.run \
  --algo doubleq \
  --reward r0 \
  --train_episodes 200000 \
  --eval_episodes 20000 \
  --seed 0
```

**Train A2C-GAE** (300K steps, baseline reward):

```bash
uv run python -m src.run \
  --algo a2c \
  --reward r0 \
  --train_steps 300000 \
  --rollout_steps 256 \
  --eval_episodes 20000 \
  --seed 0
```

Results are saved to `results/<algo>_<reward>_seed<N>_<timestamp>/`:
- `config.json` - Full configuration
- `metrics.csv` - Training and evaluation metrics
- `figures/policy_*.png` - Policy visualization heatmaps

### Reward Shaping Modes

The project supports four reward modes for exploring different learning objectives:

| Mode | Description | Parameters |
|------|-------------|------------|
| `r0` | **Baseline** - Original environment rewards | None |
| `r1` | **Step penalty** - Encourages shorter episodes | `--step_penalty 0.01` |
| `r2` | **Bust penalty** - Extra penalty for losing/busting | `--bust_penalty 0.5` |
| `r3` | **Potential-based shaping** - Uses hand value potential | `--gamma 0.99` |

Example with reward shaping:

```bash
uv run python -m src.run \
  --algo doubleq \
  --reward r1 \
  --step_penalty 0.02 \
  --train_episodes 200000
```

**Note**: All evaluations use the true (unshaped) reward for unbiased performance measurement.

## Hyperparameter Tuning

### Double Q-learning Grid Search

Explores a predefined grid of hyperparameters across 3 random seeds:

```bash
uv run python -m src.tune \
  --algo doubleq \
  --reward r0 \
  --episodes 150000 \
  --eval_episodes 20000
```

Tuned hyperparameters:
- `alpha`: Learning rate [0.05, 0.1, 0.2]
- `gamma`: Discount factor [0.95, 0.99, 1.0]
- `eps_end`: Final epsilon [0.01, 0.05]
- `eps_decay_episodes`: Decay schedule [50k, 200k]

### A2C Random Search

Samples random configurations with log-uniform learning rate sampling:

```bash
uv run python -m src.tune \
  --algo a2c \
  --reward r0 \
  --steps 200000 \
  --rollout_steps 256 \
  --trials 12 \
  --eval_episodes 20000
```

Tuned hyperparameters:
- `lr`: Learning rate [1e-4, 3e-3] (log-uniform)
- `gamma`: Discount factor [0.95, 0.99]
- `gae_lambda`: GAE lambda [0.90, 0.95, 0.97]
- `entropy_coef`: Entropy bonus [0.0, 0.01, 0.02]
- `hidden_sizes`: Network architecture [(64,64), (128,128)]

Results are saved to `results/tune_<algo>_<reward>_<timestamp>/`:
- `tuning_results.csv` - All configurations tried
- `best_config.json` - Best configuration found

## Analyzing Results

After running multiple experiments, aggregate and visualize results:

```bash
uv run python analyze_results.py \
  --results_dir results \
  --outdir analysis
```

Generates:
- `summary_by_run.csv` - Per-run final metrics
- `summary_by_algo_reward.csv` - Aggregated statistics (mean ± std)
- `report.md` - Human-readable summary
- `figures/curve_*.png` - Learning curves per algorithm
- `figures/comparison_best.png` - Best configurations comparison
- `figures/policy_montage.png` - Policy visualizations

## Understanding the Algorithms

### Double Q-learning

A **tabular value-based** method that maintains two Q-tables to reduce overestimation bias. On each update, one table selects the best action while the other evaluates it, decoupling selection from evaluation.

**Key hyperparameters:**
- `alpha` (0.1): Learning rate - how quickly Q-values update
- `gamma` (0.99): Discount factor - importance of future rewards
- `epsilon` (1.0→0.05): Exploration rate with linear decay

**Strengths:**
- Simple and interpretable
- Guaranteed convergence in tabular settings
- No function approximation errors

**Limitations:**
- Cannot generalize to unseen states
- Memory grows with state space size

### A2C with GAE

A **policy gradient** method that learns both a policy (actor) and value function (critic) using a neural network. Uses Generalized Advantage Estimation to balance bias and variance in advantage computation.

**Key hyperparameters:**
- `lr` (3e-4): Learning rate for Adam optimizer
- `gamma` (0.99): Discount factor for returns
- `gae_lambda` (0.95): GAE λ - bias-variance tradeoff (higher = less bias, more variance)
- `entropy_coef` (0.01): Entropy bonus for exploration

**Strengths:**
- Can generalize via function approximation
- Learns stochastic policies
- On-policy stability

**Limitations:**
- Sample inefficient compared to off-policy methods
- Sensitive to hyperparameters
- Requires more tuning than tabular methods

### Reward Shaping Theory

**Potential-based shaping** (mode `r3`) is theoretically sound - it transforms rewards using a potential function Φ(s) without changing the optimal policy:

```
r'(s, a, s') = r(s, a, s') + γ·Φ(s') - Φ(s)
```

For Blackjack, we use `Φ(s) = player_sum / 21`, providing denser feedback about hand quality while preserving optimality guarantees (Ng et al., 1999).

## Common Options

### Environment Variations

```bash
--natural       # Natural blackjack: blackjack pays 1.5x (default: 1.0x)
--sab          # Show actual dealer card instead of value
```

### Double Q-learning

```bash
--alpha 0.1                    # Learning rate
--gamma 0.99                   # Discount factor
--eps_start 1.0               # Initial epsilon
--eps_end 0.05                # Final epsilon
--eps_decay_episodes 100000   # Decay duration
--train_episodes 200000       # Training episodes
--eval_every_episodes 25000   # Evaluation frequency
```

### A2C-GAE

```bash
--lr 3e-4                  # Learning rate
--gamma 0.99               # Discount factor
--gae_lambda 0.95          # GAE lambda parameter
--entropy_coef 0.01        # Entropy bonus coefficient
--value_coef 0.5           # Value loss coefficient
--max_grad_norm 0.5        # Gradient clipping
--hidden1 128              # First hidden layer size
--hidden2 128              # Second hidden layer size
--train_steps 300000       # Total training steps
--rollout_steps 256        # Steps per rollout (batch size)
--eval_every_updates 10    # Evaluation frequency
--device cpu               # Device (cpu or cuda)
```