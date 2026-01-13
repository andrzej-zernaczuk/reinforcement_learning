import argparse
import os
import pickle
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from src.a2c_gae import A2CConfig, A2CGAEAgent
from src.doubleq import DoubleQAgent, DoubleQConfig
from src.envs import RewardConfig, RewardShapingWrapper
from src.features import OBS_DIM, obs_to_onehot


def clear_screen():
    print("\033[2J\033[H", end="")


def outcome_from_return(ep_return: float) -> str:
    if ep_return > 0:
        return "WIN"
    if ep_return < 0:
        return "LOSS"
    return "DRAW"


def make_watch_env(seed, natural, sab, reward_cfg, render_mode):
    env = gym.make("Blackjack-v1", natural=natural, sab=sab, render_mode=render_mode)
    env.reset(seed=seed)
    if reward_cfg is not None and reward_cfg.mode != "r0":
        env = RewardShapingWrapper(env, reward_cfg)
    return env


def save_doubleq(path, agent, cfg):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(
            {
                "QA": dict(agent.QA),
                "QB": dict(agent.QB),
                "cfg": cfg.__dict__,
                "n_actions": agent.n_actions,
            },
            f,
        )


def load_doubleq(path, seed=0):
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    cfg = DoubleQConfig(**ckpt["cfg"])
    agent = DoubleQAgent(n_actions=ckpt["n_actions"], cfg=cfg, seed=seed)
    agent.QA.update(ckpt["QA"])
    agent.QB.update(ckpt["QB"])
    agent.epsilon = 0.0
    return agent, cfg


def train_doubleq(env, cfg, episodes, seed):
    agent = DoubleQAgent(n_actions=env.action_space.n, cfg=cfg, seed=seed)
    for _ in range(episodes):
        obs, _ = env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            a = agent.act(obs, train=True)
            nxt, r, terminated, truncated, info = env.step(a)
            agent.update(obs, a, r, terminated, nxt)
            obs = nxt
        agent.end_episode()
    agent.epsilon = 0.0
    return agent


def train_a2c(env, cfg, steps, rollout_steps, seed):
    agent = A2CGAEAgent(
        obs_dim=OBS_DIM, n_actions=env.action_space.n, cfg=cfg, seed=seed
    )

    def collect(rollout_steps_):
        obs, _ = env.reset()
        bo, ba, br, bd, bv, blp = [], [], [], [], [], []
        for _ in range(rollout_steps_):
            x = obs_to_onehot(obs)
            a, logp, v, ent = agent.act(x, train=True)
            nxt, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            bo.append(x)
            ba.append(a)
            br.append(float(r))
            bd.append(float(done))
            bv.append(float(v.item()))
            blp.append(float(logp.item()))
            obs = nxt
            if done:
                obs, _ = env.reset()
        x_last = obs_to_onehot(obs)
        _, _, last_v, _ = agent.act(x_last, train=False)
        return {
            "obs": np.asarray(bo, dtype=np.float32),
            "actions": np.asarray(ba, dtype=np.int64),
            "rewards": np.asarray(br, dtype=np.float32),
            "dones": np.asarray(bd, dtype=np.float32),
            "values": np.asarray(bv, dtype=np.float32),
            "logprobs": np.asarray(blp, dtype=np.float32),
            "last_value": float(last_v.item()),
        }

    updates = steps // rollout_steps
    for _ in range(updates):
        batch = collect(rollout_steps)
        agent.update(batch)
    return agent


def set_window_caption(text: str):
    try:
        import pygame

        pygame.display.set_caption(text)
    except Exception:
        pass


def play_human(env, act_fn, episodes, delay):
    bankroll = 0.0
    wins = draws = losses = 0

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        terminated = truncated = False
        ep_return = 0.0
        t = 0

        while not (terminated or truncated):
            env.render()

            caption = f"Ep {ep}/{episodes} | step {t} | ep={ep_return:+.2f} | bank={bankroll:+.2f} | W/D/L={wins}/{draws}/{losses}"
            set_window_caption(caption)

            a = act_fn(obs)
            obs, r, terminated, truncated, info = env.step(a)
            ep_return += float(info.get("true_reward", r))
            t += 1

            if delay > 0:
                time.sleep(delay)

        bankroll += ep_return
        out = "WIN" if ep_return > 0 else ("LOSS" if ep_return < 0 else "DRAW")
        if out == "WIN":
            wins += 1
        elif out == "LOSS":
            losses += 1
        else:
            draws += 1

        env.render()
        caption = f"Ep {ep} RESULT: {out} | ep={ep_return:+.2f} | bank={bankroll:+.2f} | W/D/L={wins}/{draws}/{losses}"
        set_window_caption(caption)

        print(caption, flush=True)
        time.sleep(max(delay, 0.8))


def play_rgb(env, act_fn, episodes, delay):
    bankroll = 0.0
    wins = draws = losses = 0

    plt.ion()
    fig, ax = plt.subplots()
    im = None

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        terminated = truncated = False
        ep_return = 0.0
        t = 0

        frame = env.render()
        if frame is None:
            raise RuntimeError("rgb_array render returned None.")

        if im is None:
            im = ax.imshow(frame)
            ax.set_axis_off()
        else:
            im.set_data(frame)

        ax.set_title(
            f"Episode {ep}/{episodes} | step {t} | ep_return={ep_return:.2f} | bankroll={bankroll:.2f} | W/D/L={wins}/{draws}/{losses}"
        )
        fig.canvas.draw()
        plt.pause(max(0.001, delay))

        while not (terminated or truncated):
            a = act_fn(obs)
            obs, r, terminated, truncated, info = env.step(a)
            r_true = float(info.get("true_reward", r))
            ep_return += r_true
            t += 1

            frame = env.render()
            if frame is not None:
                im.set_data(frame)
                ax.set_title(
                    f"Episode {ep}/{episodes} | step {t} | ep_return={ep_return:.2f} | bankroll={bankroll:.2f} | W/D/L={wins}/{draws}/{losses}"
                )
                fig.canvas.draw()
                plt.pause(max(0.001, delay))

        bankroll += ep_return
        out = outcome_from_return(ep_return)
        if out == "WIN":
            wins += 1
        elif out == "LOSS":
            losses += 1
        else:
            draws += 1

        ax.set_title(
            f"Episode {ep} result: {out} | ep_return={ep_return:.2f} | bankroll={bankroll:.2f} | W/D/L={wins}/{draws}/{losses}"
        )
        fig.canvas.draw()
        plt.pause(max(0.001, delay))
        time.sleep(max(delay, 0.8))

    plt.ioff()
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["doubleq", "a2c"], required=True)
    ap.add_argument("--reward", choices=["r0", "r1", "r2", "r3"], default="r0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--natural", action="store_true")
    ap.add_argument("--sab", action="store_true")
    ap.add_argument("--step_penalty", type=float, default=0.01)
    ap.add_argument("--bust_penalty", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=0.95)

    ap.add_argument("--checkpoint", type=str, default="")
    ap.add_argument("--train_episodes", type=int, default=200000)
    ap.add_argument("--train_steps", type=int, default=300000)
    ap.add_argument("--rollout_steps", type=int, default=256)

    ap.add_argument("--play_episodes", type=int, default=5)
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--render", choices=["human", "rgb", "none"], default="human")

    args = ap.parse_args()

    reward_cfg = RewardConfig(
        mode=args.reward,
        step_penalty=args.step_penalty,
        bust_penalty=args.bust_penalty,
        gamma=args.gamma,
    )

    if args.render == "human":
        render_mode = "human"
    elif args.render == "rgb":
        render_mode = "rgb_array"
    else:
        render_mode = None

    env = make_watch_env(args.seed, args.natural, args.sab, reward_cfg, render_mode)

    if args.algo == "doubleq":
        cfg = DoubleQConfig(
            alpha=0.05,
            gamma=0.95,
            eps_start=1.0,
            eps_end=0.05,
            eps_decay_episodes=50000,
        )
        if args.checkpoint and os.path.exists(args.checkpoint):
            agent, cfg = load_doubleq(args.checkpoint, seed=args.seed)
        else:
            agent = train_doubleq(env, cfg, args.train_episodes, args.seed)
            if args.checkpoint:
                save_doubleq(args.checkpoint, agent, cfg)

        act_fn = lambda o: agent.greedy_action(o)

    else:
        cfg = A2CConfig(
            lr=0.0010948770705738267,
            gamma=0.95,
            gae_lambda=0.97,
            entropy_coef=0.0,
            hidden_sizes=(64, 64),
            device=args.device,
        )
        if args.checkpoint and os.path.exists(args.checkpoint):
            agent = A2CGAEAgent(
                obs_dim=OBS_DIM, n_actions=env.action_space.n, cfg=cfg, seed=args.seed
            )
            agent.load(args.checkpoint)
        else:
            agent = train_a2c(env, cfg, args.train_steps, args.rollout_steps, args.seed)
            if args.checkpoint:
                os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
                agent.save(args.checkpoint)

        act_fn = lambda o: agent.act(obs_to_onehot(o), train=False)[0]

    if args.render == "human":
        play_human(env, act_fn, args.play_episodes, args.delay)
    elif args.render == "rgb":
        play_rgb(env, act_fn, args.play_episodes, args.delay)

    env.close()


if __name__ == "__main__":
    main()
