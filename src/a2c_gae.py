"""Advantage Actor-Critic (A2C) with Generalized Advantage Estimation (GAE).

This module implements the A2C algorithm, a synchronous variant of A3C that
uses a neural network to learn both a policy (actor) and value function (critic).
GAE is used for advantage estimation to balance bias and variance in policy gradients.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class A2CConfig:
    """Configuration for A2C-GAE agent.

    Args:
        lr: Learning rate for Adam optimizer. Typical range: [1e-4, 3e-3].
            Lower values provide more stable learning but slower convergence.
        gamma: Discount factor for future rewards. Determines how much the agent
            values future rewards. Range: [0.0, 1.0], typically 0.95-0.99.
        gae_lambda: GAE lambda parameter for advantage estimation. Controls the
            bias-variance tradeoff in advantage estimates. Range: [0.0, 1.0],
            typically 0.90-0.99. Higher values = less bias, more variance.
        entropy_coef: Coefficient for entropy bonus in loss function. Encourages
            exploration by penalizing deterministic policies. Typical range: [0.0, 0.02].
        value_coef: Coefficient for value function loss. Balances policy and value
            learning. Typical range: [0.5, 1.0].
        max_grad_norm: Maximum gradient norm for gradient clipping. Prevents
            unstable updates from large gradients. Typical range: [0.5, 5.0].
        hidden_sizes: Tuple of hidden layer sizes for the neural network.
            E.g., (128, 128) creates two hidden layers with 128 units each.
        device: Device to run computations on ("cpu" or "cuda").
    """

    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_sizes: tuple[int, int] = (128, 128)
    device: str = "cpu"


class ActorCritic(nn.Module):
    """Neural network architecture for Actor-Critic.

    Implements a shared-backbone architecture where the network splits into
    separate heads for the policy (actor) and value function (critic). The shared
    layers learn representations useful for both tasks, improving sample efficiency.

    Architecture:
        observation -> fully_connected_1 (ReLU) -> fully_connected_2 (ReLU)
                    -> policy_head (logits) and value_head (scalar value)

    Args:
        obs_dim: Dimensionality of the observation space (input size).
        num_actions: Number of discrete actions (output size of policy head).
        hidden_sizes: Tuple specifying the size of each hidden layer.
    """

    def __init__(self, obs_dim: int, num_actions: int, hidden_sizes: tuple[int, int]):
        """Initialize the Actor-Critic network."""
        super().__init__()
        hidden_layer_1_size, hidden_layer_2_size = hidden_sizes

        # Shared backbone layers
        self.fully_connected_1 = nn.Linear(obs_dim, hidden_layer_1_size)
        self.fully_connected_2 = nn.Linear(hidden_layer_1_size, hidden_layer_2_size)

        # Separate heads for policy and value
        self.policy_head = nn.Linear(hidden_layer_2_size, num_actions)
        self.value_head = nn.Linear(hidden_layer_2_size, 1)

    def forward(self, observation_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.

        Args:
            observation_tensor: Batch of observations, shape [batch_size, obs_dim].

        Returns:
            Tuple of (policy_logits, value_estimates) where:
                - policy_logits: Unnormalized log probabilities, shape [batch_size, num_actions]
                - value_estimates: State value estimates, shape [batch_size]
        """
        hidden_1 = F.relu(self.fully_connected_1(observation_tensor))
        hidden_2 = F.relu(self.fully_connected_2(hidden_1))

        policy_logits = self.policy_head(hidden_2)
        value_estimates = self.value_head(hidden_2).squeeze(-1)

        return policy_logits, value_estimates


class A2CGAEAgent:
    """Advantage Actor-Critic agent with Generalized Advantage Estimation.

    Implements the A2C algorithm which learns a stochastic policy (actor) and
    value function (critic) simultaneously. Uses GAE for computing advantages,
    which provides a flexible bias-variance tradeoff via the lambda parameter.

    The agent performs on-policy learning, updating the policy after each rollout
    of experience. The loss function combines:
    - Policy loss: Maximizes expected return using policy gradient with advantages
    - Value loss: Minimizes squared error between predicted and actual returns
    - Entropy bonus: Encourages exploration by preventing premature convergence

    Args:
        obs_dim: Dimensionality of the observation space.
        num_actions: Number of discrete actions available.
        config: Configuration object specifying hyperparameters.
        seed: Random seed for reproducibility of network initialization.
    """

    def __init__(self, obs_dim: int, num_actions: int, config: A2CConfig, seed: int):
        """Initialize the A2C-GAE agent."""
        torch.manual_seed(seed)
        self.config = config
        self.device = torch.device(config.device)

        self.actor_critic_network = ActorCritic(obs_dim, num_actions, config.hidden_sizes).to(
            self.device
        )
        self.optimizer = torch.optim.Adam(self.actor_critic_network.parameters(), lr=config.lr)

        self.num_actions = num_actions

    @torch.no_grad()
    def act(
        self, observation_vector: np.ndarray, train: bool = True
    ) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select an action and compute associated values.

        Args:
            observation_vector: Observation as a numpy array, shape [obs_dim].
            train: If True, samples from the policy distribution. If False,
                selects the most likely action (argmax).

        Returns:
            Tuple of (action, log_probability, value_estimate, entropy) where:
                - action: Selected action index (int)
                - log_probability: Log probability of the selected action (tensor, scalar)
                - value_estimate: Value function estimate for the state (tensor, scalar)
                - entropy: Entropy of the policy distribution (tensor, scalar)
        """
        observation_tensor = (
            torch.from_numpy(observation_vector).to(self.device).unsqueeze(0)
        )  # [1, obs_dim]

        policy_logits, value_estimate = self.actor_critic_network(observation_tensor)
        policy_distribution = torch.distributions.Categorical(logits=policy_logits)

        if train:
            action = policy_distribution.sample()
        else:
            action = torch.argmax(policy_distribution.probs, dim=-1)

        log_probability = policy_distribution.log_prob(action)
        entropy = policy_distribution.entropy()

        return (
            int(action.item()),
            log_probability.squeeze(0),
            value_estimate.squeeze(0),
            entropy.squeeze(0),
        )

    def update(self, batch: dict) -> dict:
        """Update the policy and value function using a batch of experience.

        Performs a single gradient descent step using the A2C loss with GAE for
        advantage estimation. The advantages are computed backwards through time
        to properly credit rewards to actions.

        Args:
            batch: Dictionary containing trajectory data with keys:
                - "obs": Observations, shape [timesteps, obs_dim]
                - "actions": Actions taken, shape [timesteps]
                - "rewards": Rewards received, shape [timesteps]
                - "dones": Episode termination flags, shape [timesteps]
                - "values": Value estimates at each step, shape [timesteps]
                - "logprobs": Log probabilities of actions, shape [timesteps]
                - "last_value": Bootstrap value for final state (scalar)

        Returns:
            Dictionary of training metrics:
                - "loss": Total loss (float)
                - "policy_loss": Policy gradient loss (float)
                - "value_loss": Value function MSE loss (float)
                - "entropy": Average policy entropy (float)
                - "approx_kl": Approximate KL divergence between old and new policy (float)
        """
        # Convert batch data to tensors
        observations = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        episode_dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)
        old_value_estimates = torch.as_tensor(
            batch["values"], dtype=torch.float32, device=self.device
        )
        old_log_probabilities = torch.as_tensor(
            batch["logprobs"], dtype=torch.float32, device=self.device
        )
        bootstrap_value = torch.as_tensor(
            batch["last_value"], dtype=torch.float32, device=self.device
        )

        # Compute advantages using Generalized Advantage Estimation (GAE)
        timesteps = rewards.shape[0]
        advantages = torch.zeros((timesteps,), dtype=torch.float32, device=self.device)
        generalized_advantage_estimate = torch.zeros((), dtype=torch.float32, device=self.device)

        # Bootstrap next values: append bootstrap_value to the end of the value sequence
        next_value_estimates = torch.cat(
            [old_value_estimates[1:], bootstrap_value.unsqueeze(0)], dim=0
        )

        # Compute masks: 0 when episode done, 1 otherwise (for zeroing out future rewards)
        continuation_masks = 1.0 - episode_dones

        # Compute TD errors (delta_t = r_t + gamma * V(s_{t+1}) - V(s_t))
        td_errors = rewards + self.config.gamma * continuation_masks * next_value_estimates - old_value_estimates

        # Compute GAE backwards through time
        # GAE(gamma, lambda)_t = sum_{l=0}^{inf} (gamma * lambda)^l * delta_{t+l}
        for timestep in reversed(range(timesteps)):
            generalized_advantage_estimate = (
                td_errors[timestep]
                + self.config.gamma
                * self.config.gae_lambda
                * continuation_masks[timestep]
                * generalized_advantage_estimate
            )
            advantages[timestep] = generalized_advantage_estimate

        # Compute returns (targets for value function) as advantages + values
        returns = advantages + old_value_estimates

        # Normalize advantages for more stable training
        normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Recompute policy and value predictions with current network
        policy_logits, predicted_values = self.actor_critic_network(observations)
        policy_distribution = torch.distributions.Categorical(logits=policy_logits)
        new_log_probabilities = policy_distribution.log_prob(actions)
        policy_entropy = policy_distribution.entropy().mean()

        # Compute loss components
        # Policy loss: negative because we want to maximize expected return
        policy_loss = -(new_log_probabilities * normalized_advantages.detach()).mean()

        # Value loss: mean squared error between predicted and actual returns
        value_loss = 0.5 * (returns.detach() - predicted_values).pow(2).mean()

        # Total loss: combine policy, value, and entropy terms
        total_loss = (
            policy_loss
            + self.config.value_coef * value_loss
            - self.config.entropy_coef * policy_entropy
        )

        # Optimize the network
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic_network.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        # Compute approximate KL divergence for monitoring
        approximate_kl = (old_log_probabilities - new_log_probabilities).mean().abs().item()

        return {
            "loss": float(total_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(policy_entropy.item()),
            "approx_kl": float(approximate_kl),
        }

    def save(self, file_path: str) -> None:
        """Save the agent's network state and configuration.

        Args:
            file_path: Path where the checkpoint will be saved.
        """
        torch.save(
            {"state_dict": self.actor_critic_network.state_dict(), "config": self.config.__dict__},
            file_path,
        )

    def load(self, file_path: str) -> None:
        """Load a previously saved agent state.

        Args:
            file_path: Path to the checkpoint file.
        """
        checkpoint = torch.load(file_path, map_location=self.device)
        self.actor_critic_network.load_state_dict(checkpoint["state_dict"])
