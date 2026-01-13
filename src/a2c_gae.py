from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class A2CConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_sizes: tuple[int, int] = (128, 128)
    device: str = "cpu"


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: tuple[int, int]):
        super().__init__()
        h1, h2 = hidden_sizes
        self.fc1 = nn.Linear(obs_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.pi = nn.Linear(h2, n_actions)
        self.v = nn.Linear(h2, 1)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.pi(x)
        value = self.v(x).squeeze(-1)
        return logits, value


class A2CGAEAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: A2CConfig, seed: int):
        torch.manual_seed(seed)
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.net = ActorCritic(obs_dim, n_actions, cfg.hidden_sizes).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)

        self.n_actions = n_actions

    @torch.no_grad()
    def act(self, obs_vec: np.ndarray, train: bool = True):
        x = torch.from_numpy(obs_vec).to(self.device).unsqueeze(0)  # [1, obs_dim]
        logits, value = self.net(x)
        dist = torch.distributions.Categorical(logits=logits)
        if train:
            action = dist.sample()
        else:
            action = torch.argmax(dist.probs, dim=-1)
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return (
            int(action.item()),
            logprob.squeeze(0),
            value.squeeze(0),
            entropy.squeeze(0),
        )

    def update(self, batch: dict) -> dict:
        # batch fields: obs [T,obs_dim], actions [T], rewards [T], dones [T], values [T], logprobs [T], last_value scalar
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(
            batch["actions"], dtype=torch.int64, device=self.device
        )
        rewards = torch.as_tensor(
            batch["rewards"], dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)
        values = torch.as_tensor(
            batch["values"], dtype=torch.float32, device=self.device
        )
        old_logprobs = torch.as_tensor(
            batch["logprobs"], dtype=torch.float32, device=self.device
        )
        last_value = torch.as_tensor(
            batch["last_value"], dtype=torch.float32, device=self.device
        )

        # Compute advantages with GAE(λ)
        T = rewards.shape[0]
        adv = torch.zeros((T,), dtype=torch.float32, device=self.device)
        gae = torch.zeros((), dtype=torch.float32, device=self.device)

        next_values = torch.cat([values[1:], last_value.unsqueeze(0)], dim=0)
        masks = 1.0 - dones  # 0 when done, 1 otherwise

        deltas = rewards + self.cfg.gamma * masks * next_values - values

        for t in reversed(range(T)):
            gae = deltas[t] + self.cfg.gamma * self.cfg.gae_lambda * masks[t] * gae
            adv[t] = gae

        returns = adv + values
        adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Recompute logits/values for current net
        logits, v_pred = self.net(obs)
        dist = torch.distributions.Categorical(logits=logits)
        logprobs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        policy_loss = -(logprobs * adv_norm.detach()).mean()
        value_loss = 0.5 * (returns.detach() - v_pred).pow(2).mean()

        loss = (
            policy_loss
            + self.cfg.value_coef * value_loss
            - self.cfg.entropy_coef * entropy
        )

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
        self.opt.step()

        approx_kl = (old_logprobs - logprobs).mean().abs().item()

        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "approx_kl": float(approx_kl),
        }

    def save(self, path: str) -> None:
        torch.save(
            {"state_dict": self.net.state_dict(), "cfg": self.cfg.__dict__}, path
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["state_dict"])
