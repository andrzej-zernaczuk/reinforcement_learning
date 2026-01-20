#!/usr/bin/env bash
set -euo pipefail

EVAL_EPISODES=20000

# Consistent 500k training budget
DQ_EPISODES=500000
A2C_STEPS=500000
A2C_ROLLOUT=256

# ---- Tuned DoubleQ params ----
DQ_ALPHA=0.05
DQ_GAMMA=0.95
DQ_EPS_START=1.0
DQ_EPS_END=0.05
DQ_EPS_DECAY=50000

# ---- Tuned A2C params ----
A2C_LR=0.0010948770705738267
A2C_GAMMA=0.95
A2C_GAE_LAMBDA=0.97
A2C_ENTROPY=0.0
A2C_H1=64
A2C_H2=64

for seed in 0 1 2; do
  for reward in r0 r1 r2; do

    EXTRA_ARGS=()
    if [ "$reward" = "r1" ]; then
      EXTRA_ARGS+=(--step_penalty 0.01)
    elif [ "$reward" = "r2" ]; then
      EXTRA_ARGS+=(--bust_penalty 0.5)
    fi

    echo "=== DoubleQ | $reward | seed=$seed ==="
    if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
      uv run python -m src.run \
        --algo doubleq \
        --reward "$reward" \
        --train_episodes "$DQ_EPISODES" \
        --eval_episodes "$EVAL_EPISODES" \
        --seed "$seed" \
        --alpha "$DQ_ALPHA" \
        --gamma "$DQ_GAMMA" \
        --eps_start "$DQ_EPS_START" \
        --eps_end "$DQ_EPS_END" \
        --eps_decay_episodes "$DQ_EPS_DECAY" \
        "${EXTRA_ARGS[@]}"
    else
      uv run python -m src.run \
        --algo doubleq \
        --reward "$reward" \
        --train_episodes "$DQ_EPISODES" \
        --eval_episodes "$EVAL_EPISODES" \
        --seed "$seed" \
        --alpha "$DQ_ALPHA" \
        --gamma "$DQ_GAMMA" \
        --eps_start "$DQ_EPS_START" \
        --eps_end "$DQ_EPS_END" \
        --eps_decay_episodes "$DQ_EPS_DECAY"
    fi

    echo "=== A2C | $reward | seed=$seed ==="
    if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
      uv run python -m src.run \
        --algo a2c \
        --reward "$reward" \
        --train_steps "$A2C_STEPS" \
        --rollout_steps "$A2C_ROLLOUT" \
        --eval_episodes "$EVAL_EPISODES" \
        --seed "$seed" \
        --lr "$A2C_LR" \
        --gamma "$A2C_GAMMA" \
        --gae_lambda "$A2C_GAE_LAMBDA" \
        --entropy_coef "$A2C_ENTROPY" \
        --hidden1 "$A2C_H1" \
        --hidden2 "$A2C_H2" \
        "${EXTRA_ARGS[@]}"
    else
      uv run python -m src.run \
        --algo a2c \
        --reward "$reward" \
        --train_steps "$A2C_STEPS" \
        --rollout_steps "$A2C_ROLLOUT" \
        --eval_episodes "$EVAL_EPISODES" \
        --seed "$seed" \
        --lr "$A2C_LR" \
        --gamma "$A2C_GAMMA" \
        --gae_lambda "$A2C_GAE_LAMBDA" \
        --entropy_coef "$A2C_ENTROPY" \
        --hidden1 "$A2C_H1" \
        --hidden2 "$A2C_H2"
    fi

  done
done
