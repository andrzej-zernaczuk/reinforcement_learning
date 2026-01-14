"""Interactive agent visualization tool for watching trained RL agents play Blackjack.

This module provides utilities for training, saving, loading, and visualizing
reinforcement learning agents playing Blackjack in real-time. Supports both
Double Q-learning and A2C-GAE agents with various render modes.
"""

import argparse
import os
import pickle
import time
from typing import Any, Callable

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.spaces import Discrete

from src.a2c_gae import A2CConfig, A2CGAEAgent
from src.doubleq import DoubleQAgent, DoubleQConfig
from src.envs import RewardConfig, RewardShapingWrapper
from src.features import OBS_DIM, obs_to_onehot


def clear_screen() -> None:
    """Clear the terminal screen using ANSI escape codes."""
    print("\033[2J\033[H", end="")


def outcome_from_return(episode_return: float) -> str:
    """Convert episode return to outcome string.

    Args:
        episode_return: Total return received in the episode.

    Returns:
        "WIN" for positive return, "LOSS" for negative, "DRAW" for zero.
    """
    if episode_return > 0:
        return "WIN"
    if episode_return < 0:
        return "LOSS"
    return "DRAW"


def make_watch_env(
    seed: int,
    natural: bool,
    sab: bool,
    reward_config: RewardConfig | None,
    render_mode: str | None,
) -> gym.Env:
    """Create a Blackjack environment for watching agents play.

    Args:
        seed: Random seed for environment reproducibility.
        natural: Whether natural blackjacks pay 1.5x.
        sab: Whether to show actual dealer card.
        reward_config: Optional reward shaping configuration. If None or mode="r0",
            no reward shaping is applied.
        render_mode: Rendering mode ("human", "rgb_array", or None).

    Returns:
        Configured Gymnasium Blackjack environment.
    """
    environment = gym.make("Blackjack-v1", natural=natural, sab=sab, render_mode=render_mode)
    environment.reset(seed=seed)
    if reward_config is not None and reward_config.mode != "r0":
        environment = RewardShapingWrapper(environment, reward_config)
    return environment


def save_doubleq(file_path: str, agent: DoubleQAgent, config: DoubleQConfig) -> None:
    """Save a Double Q-learning agent to disk.

    Args:
        file_path: Path where the agent will be saved.
        agent: Double Q-learning agent to save.
        config: Agent configuration.
    """
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    with open(file_path, "wb") as file_handle:
        pickle.dump(
            {
                "q_table_a": dict(agent.q_table_a),
                "q_table_b": dict(agent.q_table_b),
                "config": config.__dict__,
                "num_actions": agent.num_actions,
            },
            file_handle,
        )


def load_doubleq(file_path: str, seed: int = 0) -> tuple[DoubleQAgent, DoubleQConfig]:
    """Load a Double Q-learning agent from disk.

    Args:
        file_path: Path to the saved agent checkpoint.
        seed: Random seed for the loaded agent.

    Returns:
        Tuple of (loaded agent, agent config).
    """
    with open(file_path, "rb") as file_handle:
        checkpoint = pickle.load(file_handle)

    config = DoubleQConfig(**checkpoint["config"])
    agent = DoubleQAgent(num_actions=checkpoint["num_actions"], config=config, seed=seed)
    agent.q_table_a.update(checkpoint["q_table_a"])
    agent.q_table_b.update(checkpoint["q_table_b"])
    agent.epsilon = 0.0  # Greedy policy for evaluation
    return agent, config


def train_doubleq(
    environment: gym.Env,
    config: DoubleQConfig,
    episodes: int,
    seed: int,
) -> DoubleQAgent:
    """Train a Double Q-learning agent.

    Args:
        environment: Gymnasium environment to train in.
        config: Double Q-learning configuration.
        episodes: Number of training episodes.
        seed: Random seed for agent initialization.

    Returns:
        Trained Double Q-learning agent with epsilon set to 0.0 for greedy play.
    """
    action_space = environment.action_space
    assert isinstance(action_space, Discrete), "Action space must be Discrete"
    num_actions = int(action_space.n)

    agent = DoubleQAgent(num_actions=num_actions, config=config, seed=seed)

    for _ in range(episodes):
        observation, _ = environment.reset()
        terminated = truncated = False

        while not (terminated or truncated):
            action = agent.act(observation, train=True)
            next_observation, reward, terminated, truncated, info = environment.step(action)
            agent.update(observation, action, float(reward), terminated, next_observation)
            observation = next_observation

        agent.end_episode()

    agent.epsilon = 0.0  # Set to greedy policy for evaluation
    return agent


def train_a2c(
    environment: gym.Env,
    config: A2CConfig,
    train_steps: int,
    rollout_steps: int,
    seed: int,
) -> A2CGAEAgent:
    """Train an A2C-GAE agent.

    Args:
        environment: Gymnasium environment to train in.
        config: A2C-GAE configuration.
        train_steps: Total number of environment steps to train.
        rollout_steps: Number of steps per rollout (batch size).
        seed: Random seed for agent initialization.

    Returns:
        Trained A2C-GAE agent.
    """
    action_space = environment.action_space
    assert isinstance(action_space, Discrete), "Action space must be Discrete"
    num_actions = int(action_space.n)

    agent = A2CGAEAgent(
        obs_dim=OBS_DIM, num_actions=num_actions, config=config, seed=seed
    )

    def collect_rollout(num_steps: int) -> dict[str, Any]:
        """Collect a rollout of experience.

        Args:
            num_steps: Number of steps to collect.

        Returns:
            Dictionary containing rollout data for agent update.
        """
        observation, _ = environment.reset()
        observations_list = []
        actions_list = []
        rewards_list = []
        episode_dones_list = []
        value_estimates_list = []
        log_probabilities_list = []

        for _ in range(num_steps):
            observation_encoded = obs_to_onehot(observation)
            action, log_probability, value_estimate, entropy = agent.act(
                observation_encoded, train=True
            )
            next_observation, reward, terminated, truncated, info = environment.step(action)
            episode_done = terminated or truncated

            observations_list.append(observation_encoded)
            actions_list.append(action)
            rewards_list.append(float(reward))
            episode_dones_list.append(float(episode_done))
            value_estimates_list.append(float(value_estimate.item()))
            log_probabilities_list.append(float(log_probability.item()))

            observation = next_observation
            if episode_done:
                observation, _ = environment.reset()

        final_observation_encoded = obs_to_onehot(observation)
        _, _, bootstrap_value, _ = agent.act(final_observation_encoded, train=False)

        return {
            "obs": np.asarray(observations_list, dtype=np.float32),
            "actions": np.asarray(actions_list, dtype=np.int64),
            "rewards": np.asarray(rewards_list, dtype=np.float32),
            "dones": np.asarray(episode_dones_list, dtype=np.float32),
            "values": np.asarray(value_estimates_list, dtype=np.float32),
            "logprobs": np.asarray(log_probabilities_list, dtype=np.float32),
            "last_value": float(bootstrap_value.item()),
        }

    num_updates = train_steps // rollout_steps
    for _ in range(num_updates):
        batch_data = collect_rollout(rollout_steps)
        agent.update(batch_data)

    return agent


def set_window_caption(text: str) -> None:
    """Set the pygame window caption if pygame is available.

    Args:
        text: Caption text to display.
    """
    try:
        import pygame  # type: ignore[import-not-found]
        pygame.display.set_caption(text)
    except Exception:
        pass


def play_human(
    environment: gym.Env,
    action_function: Callable[[Any], int],
    episodes: int,
    delay: float,
) -> None:
    """Play episodes using human-visible rendering.

    Renders the environment in a pygame window and displays statistics
    including bankroll, win/draw/loss counts, and episode returns.

    Args:
        environment: Gymnasium environment with render_mode="human".
        action_function: Function that takes an observation and returns an action.
        episodes: Number of episodes to play.
        delay: Delay in seconds between steps.
    """
    bankroll = 0.0
    wins_count = draws_count = losses_count = 0

    for episode_num in range(1, episodes + 1):
        observation, _ = environment.reset()
        terminated = truncated = False
        episode_return = 0.0
        timestep = 0

        while not (terminated or truncated):
            environment.render()

            caption = (
                f"Ep {episode_num}/{episodes} | step {timestep} | "
                f"ep={episode_return:+.2f} | bank={bankroll:+.2f} | "
                f"W/D/L={wins_count}/{draws_count}/{losses_count}"
            )
            set_window_caption(caption)

            action = action_function(observation)
            observation, reward, terminated, truncated, info = environment.step(action)
            episode_return += float(info.get("true_reward", reward))
            timestep += 1

            if delay > 0:
                time.sleep(delay)

        bankroll += episode_return
        outcome = "WIN" if episode_return > 0 else ("LOSS" if episode_return < 0 else "DRAW")

        if outcome == "WIN":
            wins_count += 1
        elif outcome == "LOSS":
            losses_count += 1
        else:
            draws_count += 1

        environment.render()
        caption = (
            f"Ep {episode_num} RESULT: {outcome} | ep={episode_return:+.2f} | "
            f"bank={bankroll:+.2f} | W/D/L={wins_count}/{draws_count}/{losses_count}"
        )
        set_window_caption(caption)

        print(caption, flush=True)
        time.sleep(max(delay, 0.8))


def play_rgb(
    environment: gym.Env,
    action_function: Callable[[Any], int],
    episodes: int,
    delay: float,
) -> None:
    """Play episodes using matplotlib RGB array rendering.

    Displays the environment as an image in a matplotlib figure, useful when
    pygame is not available or for headless systems.

    Args:
        environment: Gymnasium environment with render_mode="rgb_array".
        action_function: Function that takes an observation and returns an action.
        episodes: Number of episodes to play.
        delay: Delay in seconds between steps.
    """
    bankroll = 0.0
    wins_count = draws_count = losses_count = 0

    plt.ion()
    figure, axis = plt.subplots()
    image_handle = None

    for episode_num in range(1, episodes + 1):
        observation, _ = environment.reset()
        terminated = truncated = False
        episode_return = 0.0
        timestep = 0

        frame: np.ndarray | None = environment.render()  # type: ignore[assignment]
        if frame is None:
            raise RuntimeError("rgb_array render returned None.")

        if image_handle is None:
            image_handle = axis.imshow(frame)
            axis.set_axis_off()
        else:
            image_handle.set_data(frame)

        axis.set_title(
            f"Episode {episode_num}/{episodes} | step {timestep} | "
            f"ep_return={episode_return:.2f} | bankroll={bankroll:.2f} | "
            f"W/D/L={wins_count}/{draws_count}/{losses_count}"
        )
        figure.canvas.draw()
        plt.pause(max(0.001, delay))

        while not (terminated or truncated):
            action = action_function(observation)
            observation, reward, terminated, truncated, info = environment.step(action)
            true_reward = float(info.get("true_reward", reward))
            episode_return += true_reward
            timestep += 1

            frame = environment.render()  # type: ignore[assignment]
            if frame is not None:
                image_handle.set_data(frame)
                axis.set_title(
                    f"Episode {episode_num}/{episodes} | step {timestep} | "
                    f"ep_return={episode_return:.2f} | bankroll={bankroll:.2f} | "
                    f"W/D/L={wins_count}/{draws_count}/{losses_count}"
                )
                figure.canvas.draw()
                plt.pause(max(0.001, delay))

        bankroll += episode_return
        outcome = outcome_from_return(episode_return)

        if outcome == "WIN":
            wins_count += 1
        elif outcome == "LOSS":
            losses_count += 1
        else:
            draws_count += 1

        axis.set_title(
            f"Episode {episode_num} result: {outcome} | ep_return={episode_return:.2f} | "
            f"bankroll={bankroll:.2f} | W/D/L={wins_count}/{draws_count}/{losses_count}"
        )
        figure.canvas.draw()
        plt.pause(max(0.001, delay))
        time.sleep(max(delay, 0.8))

    plt.ioff()
    plt.show()


def main() -> None:
    """Main entry point for the watch agent script.

    Parses command-line arguments, trains or loads an agent, and visualizes
    it playing Blackjack episodes.
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--algo", choices=["doubleq", "a2c"], required=True)
    argument_parser.add_argument("--reward", choices=["r0", "r1", "r2", "r3"], default="r0")
    argument_parser.add_argument("--seed", type=int, default=0)
    argument_parser.add_argument("--natural", action="store_true")
    argument_parser.add_argument("--sab", action="store_true")
    argument_parser.add_argument("--step_penalty", type=float, default=0.01)
    argument_parser.add_argument("--bust_penalty", type=float, default=0.5)
    argument_parser.add_argument("--gamma", type=float, default=0.95)

    argument_parser.add_argument("--checkpoint", type=str, default="")
    argument_parser.add_argument("--train_episodes", type=int, default=200000)
    argument_parser.add_argument("--train_steps", type=int, default=300000)
    argument_parser.add_argument("--rollout_steps", type=int, default=256)

    argument_parser.add_argument("--play_episodes", type=int, default=5)
    argument_parser.add_argument("--delay", type=float, default=1.0)
    argument_parser.add_argument("--device", type=str, default="cpu")
    argument_parser.add_argument("--render", choices=["human", "rgb", "none"], default="human")

    args = argument_parser.parse_args()

    reward_config = RewardConfig(
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

    environment = make_watch_env(
        args.seed, args.natural, args.sab, reward_config, render_mode
    )

    if args.algo == "doubleq":
        doubleq_config = DoubleQConfig(
            alpha=0.05,
            gamma=0.95,
            eps_start=1.0,
            eps_end=0.05,
            eps_decay_episodes=50000,
        )

        if args.checkpoint and os.path.exists(args.checkpoint):
            doubleq_agent, doubleq_config = load_doubleq(args.checkpoint, seed=args.seed)
        else:
            doubleq_agent = train_doubleq(
                environment, doubleq_config, args.train_episodes, args.seed
            )
            if args.checkpoint:
                save_doubleq(args.checkpoint, doubleq_agent, doubleq_config)

        action_function: Callable[[Any], int] = lambda obs: doubleq_agent.greedy_action(obs)

    else:
        action_space = environment.action_space
        assert isinstance(action_space, Discrete), "Action space must be Discrete"
        num_actions = int(action_space.n)

        a2c_config = A2CConfig(
            lr=0.0010948770705738267,
            gamma=0.95,
            gae_lambda=0.97,
            entropy_coef=0.0,
            hidden_sizes=(64, 64),
            device=args.device,
        )

        if args.checkpoint and os.path.exists(args.checkpoint):
            a2c_agent = A2CGAEAgent(
                obs_dim=OBS_DIM, num_actions=num_actions, config=a2c_config, seed=args.seed
            )
            a2c_agent.load(args.checkpoint)
        else:
            a2c_agent = train_a2c(
                environment, a2c_config, args.train_steps, args.rollout_steps, args.seed
            )
            if args.checkpoint:
                os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
                a2c_agent.save(args.checkpoint)

        action_function = lambda obs: a2c_agent.act(obs_to_onehot(obs), train=False)[0]

    if args.render == "human":
        play_human(environment, action_function, args.play_episodes, args.delay)
    elif args.render == "rgb":
        play_rgb(environment, action_function, args.play_episodes, args.delay)

    environment.close()


if __name__ == "__main__":
    main()
