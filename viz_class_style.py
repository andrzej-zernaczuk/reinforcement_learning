"""Advanced 3D visualization tool for RL agent value functions and policies.

This module creates 3D surface plots of value functions, Q-values, and policy
probabilities for trained Double Q-learning and A2C-GAE agents. Useful for
understanding how agents evaluate different game states.
"""

import argparse
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch

from src.a2c_gae import A2CConfig, A2CGAEAgent
from src.doubleq import DoubleQAgent, DoubleQConfig
from src.features import OBS_DIM, obs_to_onehot


def make_surface_data(
    dataframe: pd.DataFrame, value_column: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create meshgrid data for 3D surface plotting.

    Converts a DataFrame with player_sum and dealer_card columns into
    meshgrid format suitable for matplotlib 3D surface plots.

    Args:
        dataframe: DataFrame containing player_sum, dealer_card, and value columns.
        value_column: Name of the column to use for Z-axis values.

    Returns:
        Tuple of (X, Y, Z) meshgrid arrays where:
            - X: Dealer card values (meshgrid)
            - Y: Player sum values (meshgrid)
            - Z: Value function values (meshgrid)
    """
    player_sums = np.array(sorted(dataframe["player_sum"].unique()))
    dealer_cards = np.array(sorted(dataframe["dealer_card"].unique()))
    z_values = np.full((player_sums.size, dealer_cards.size), np.nan, dtype=float)

    # Create lookup table for fast access
    lookup_table = {
        (int(row.player_sum), int(row.dealer_card)): float(row[value_column])
        for _, row in dataframe.iterrows()
    }

    # Fill Z values
    for row_index, player_sum in enumerate(player_sums):
        for col_index, dealer_card in enumerate(dealer_cards):
            z_values[row_index, col_index] = lookup_table.get(
                (int(player_sum), int(dealer_card)), np.nan
            )

    x_mesh, y_mesh = np.meshgrid(dealer_cards, player_sums)
    return x_mesh, y_mesh, z_values


def plot_surface(
    dataframe: pd.DataFrame,
    usable_ace: bool,
    value_column: str,
    title: str,
    output_path: str,
    invert_x_axis: bool = True,
) -> None:
    """Create and save a 3D surface plot of value function.

    Args:
        dataframe: DataFrame containing state-value data.
        usable_ace: Whether to plot for usable_ace=True or False states.
        value_column: Name of column to plot on Z-axis (e.g., "V", "Q_hit").
        title: Title for the plot.
        output_path: Path where the figure will be saved.
        invert_x_axis: If True, invert the Y-axis (player_sum) for better viewing.
    """
    # Filter for specific usable_ace value and relevant player sums
    filtered_data = dataframe[dataframe["usable_ace"] == usable_ace].copy()
    filtered_data = filtered_data[
        (filtered_data["player_sum"] >= 12) & (filtered_data["player_sum"] <= 21)
    ]

    x_mesh, y_mesh, z_values = make_surface_data(filtered_data, value_column=value_column)

    figure = plt.figure(figsize=(7, 5))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot_surface(x_mesh, y_mesh, z_values, linewidth=0, antialiased=True)

    axis.set_xlabel("dealer_showing")
    axis.set_ylabel("player_sum")
    axis.set_zlabel(value_column)
    axis.set_title(f"{title} | usable_ace={usable_ace}")

    if invert_x_axis:
        axis.invert_yaxis()

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_epsilon(
    config: DoubleQConfig, output_path: str, episodes: int = 200000
) -> None:
    """Plot epsilon decay schedule for Double Q-learning.

    Args:
        config: Double Q-learning configuration containing epsilon parameters.
        output_path: Path where the figure will be saved.
        episodes: Number of episodes to plot (default 200,000).
    """
    epsilon_values = []

    for episode_num in range(episodes + 1):
        if config.eps_decay_episodes <= 0:
            epsilon = config.eps_end
        else:
            decay_fraction = min(1.0, episode_num / float(config.eps_decay_episodes))
            epsilon = config.eps_start + decay_fraction * (config.eps_end - config.eps_start)
        epsilon_values.append(epsilon)

    figure = plt.figure(figsize=(7, 4))
    axis = figure.add_subplot(111)
    axis.plot(np.arange(episodes + 1), epsilon_values)
    axis.set_xlabel("episode")
    axis.set_ylabel("epsilon")
    axis.set_title("Epsilon schedule (DoubleQ)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def build_doubleq_df(checkpoint_path: str) -> tuple[pd.DataFrame, DoubleQConfig]:
    """Build DataFrame of Q-values and V-values from Double Q-learning checkpoint.

    Loads a saved Double Q-learning agent and extracts Q-values for all
    relevant state-action pairs.

    Args:
        checkpoint_path: Path to the pickled checkpoint file.

    Returns:
        Tuple of (DataFrame with Q-values and V-values, agent config).
    """
    with open(checkpoint_path, "rb") as file_handle:
        checkpoint = pickle.load(file_handle)

    config = DoubleQConfig(**checkpoint["config"])
    agent = DoubleQAgent(num_actions=checkpoint["num_actions"], config=config, seed=0)
    agent.q_table_a.update(checkpoint["q_table_a"])
    agent.q_table_b.update(checkpoint["q_table_b"])

    rows = []
    for usable_ace in [False, True]:
        for player_sum in range(4, 22):
            for dealer_card in range(1, 11):
                state = (player_sum, dealer_card, usable_ace)
                q_values_a = agent.q_table_a[state]
                q_values_b = agent.q_table_b[state]
                # Average Q-values from both tables
                q_values_avg = (q_values_a + q_values_b) / 2.0

                rows.append(
                    {
                        "player_sum": player_sum,
                        "dealer_card": dealer_card,
                        "usable_ace": usable_ace,
                        "Q_stick": float(q_values_avg[0]),
                        "Q_hit": float(q_values_avg[1]),
                        "V": float(np.max(q_values_avg)),
                    }
                )

    return pd.DataFrame(rows), config


def build_a2c_df(checkpoint_path: str, config: A2CConfig) -> pd.DataFrame:
    """Build DataFrame of value and policy from A2C-GAE checkpoint.

    Loads a saved A2C-GAE agent and extracts value estimates and policy
    probabilities for all relevant states.

    Args:
        checkpoint_path: Path to the saved agent checkpoint.
        config: A2C configuration for loading the agent.

    Returns:
        DataFrame with V-values and policy probabilities.
    """
    agent = A2CGAEAgent(obs_dim=OBS_DIM, num_actions=2, config=config, seed=0)
    agent.load(checkpoint_path)

    network = getattr(agent, "actor_critic_network", None)
    if network is None:
        raise AttributeError("Could not find agent.actor_critic_network on A2CGAEAgent")

    network.eval()
    device_obj = getattr(network, "device", None)
    if device_obj is None:
        device_obj = (
            torch.device(config.device) if hasattr(config, "device") else torch.device("cpu")
        )

    rows = []
    for usable_ace in [False, True]:
        for player_sum in range(4, 22):
            for dealer_card in range(1, 11):
                observation = (player_sum, dealer_card, usable_ace)
                observation_encoded = obs_to_onehot(observation)

                observation_tensor = torch.as_tensor(
                    observation_encoded, dtype=torch.float32, device=device_obj
                ).unsqueeze(0)

                network_output = network(observation_tensor)

                if isinstance(network_output, tuple) and len(network_output) == 2:
                    policy_logits, value_estimate = network_output
                elif isinstance(network_output, dict):
                    policy_logits = network_output.get(
                        "logits", network_output.get("pi_logits", network_output.get("policy_logits"))
                    )
                    value_estimate = network_output.get("value", network_output.get("v"))
                    if policy_logits is None or value_estimate is None:
                        raise ValueError(
                            f"Unexpected network dict keys: {list(network_output.keys())}"
                        )
                else:
                    raise ValueError("Unexpected network output type/format")

                policy_probs = (
                    torch.softmax(policy_logits, dim=-1).detach().cpu().numpy().reshape(-1)
                )

                rows.append(
                    {
                        "player_sum": player_sum,
                        "dealer_card": dealer_card,
                        "usable_ace": usable_ace,
                        "V": float(value_estimate.detach().cpu().item()),
                        "pi_hit": float(policy_probs[1]),
                        "pi_stick": float(policy_probs[0]),
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    """Main entry point for the visualization script.

    Parses command-line arguments, loads agent checkpoints, and generates
    3D surface plots of value functions and policies.
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--algo", choices=["doubleq", "a2c"], required=True)
    argument_parser.add_argument("--checkpoint", type=str, required=True)
    argument_parser.add_argument("--outdir", type=str, default="viz_class")
    argument_parser.add_argument("--device", type=str, default="cpu")
    args = argument_parser.parse_args()

    output_directory = Path(args.outdir)
    output_directory.mkdir(parents=True, exist_ok=True)

    if args.algo == "doubleq":
        dataframe, config = build_doubleq_df(args.checkpoint)

        # Plot value function surfaces
        plot_surface(
            dataframe, False, "V", "DoubleQ: V(s)", str(output_directory / "doubleq_V_false.png")
        )
        plot_surface(
            dataframe, True, "V", "DoubleQ: V(s)", str(output_directory / "doubleq_V_true.png")
        )

        # Plot Q(s, stick) surfaces
        plot_surface(
            dataframe,
            False,
            "Q_stick",
            "DoubleQ: Q(s,stick)",
            str(output_directory / "doubleq_Qstick_false.png"),
        )
        plot_surface(
            dataframe,
            True,
            "Q_stick",
            "DoubleQ: Q(s,stick)",
            str(output_directory / "doubleq_Qstick_true.png"),
        )

        # Plot Q(s, hit) surfaces
        plot_surface(
            dataframe,
            False,
            "Q_hit",
            "DoubleQ: Q(s,hit)",
            str(output_directory / "doubleq_Qhit_false.png"),
        )
        plot_surface(
            dataframe,
            True,
            "Q_hit",
            "DoubleQ: Q(s,hit)",
            str(output_directory / "doubleq_Qhit_true.png"),
        )

        # Plot epsilon decay schedule
        plot_epsilon(config, str(output_directory / "doubleq_epsilon.png"), episodes=200000)

        # Save raw data
        dataframe.to_csv(output_directory / "doubleq_values.csv", index=False)

    else:
        a2c_config = A2CConfig(
            lr=0.0010948770705738267,
            gamma=0.95,
            gae_lambda=0.97,
            entropy_coef=0.0,
            hidden_sizes=(64, 64),
            device=args.device,
        )

        dataframe = build_a2c_df(args.checkpoint, a2c_config)

        # Plot value function surfaces
        plot_surface(
            dataframe, False, "V", "A2C-GAE: V(s)", str(output_directory / "a2c_V_false.png")
        )
        plot_surface(
            dataframe, True, "V", "A2C-GAE: V(s)", str(output_directory / "a2c_V_true.png")
        )

        # Plot policy probability surfaces
        plot_surface(
            dataframe,
            False,
            "pi_hit",
            "A2C-GAE: π(hit|s)",
            str(output_directory / "a2c_pi_hit_false.png"),
        )
        plot_surface(
            dataframe,
            True,
            "pi_hit",
            "A2C-GAE: π(hit|s)",
            str(output_directory / "a2c_pi_hit_true.png"),
        )

        # Save raw data
        dataframe.to_csv(output_directory / "a2c_values.csv", index=False)

    print(f"Saved figures to: {output_directory}")


if __name__ == "__main__":
    main()
