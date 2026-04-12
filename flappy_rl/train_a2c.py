from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .env import FlappyBirdEnv
from .evaluate import evaluate_policy_callable
from .model import ActorCritic
from .utils import ExperimentResult, ensure_results_dir


def _select_torch_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_a2c(
    timesteps: int,
    seed: int,
    results_dir: str | Path,
    gamma: float = 0.99,
    n_steps: int = 16,
    lr: float = 3e-4,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    max_grad_norm: float = 1.0,
    eval_episodes: int = 10,
) -> ExperimentResult:
    device = _select_torch_device()
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = FlappyBirdEnv(seed=seed)
    model = ActorCritic().to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_steps = 0
    next_log_step = 10_000
    t0 = time.perf_counter()

    obs, _ = env.reset(seed=seed)

    while total_steps < timesteps:
        log_probs: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        rewards: list[float] = []
        entropies: list[torch.Tensor] = []

        rollout = min(n_steps, timesteps - total_steps)
        done = False
        truncated = False

        for _ in range(rollout):
            x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, value = model(x)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()

            next_obs, reward, done, truncated, _ = env.step(int(action.item()))

            log_probs.append(dist.log_prob(action).squeeze(0))
            entropies.append(dist.entropy().squeeze(0))
            values.append(value.squeeze(0).squeeze(0))
            rewards.append(float(np.clip(reward, -1.0, 1.0)))

            total_steps += 1
            obs = next_obs

            while total_steps >= next_log_step:
                elapsed = time.perf_counter() - t0
                print(
                    f"[a2c] steps={next_log_step}/{timesteps} elapsed={elapsed:.2f}s",
                    flush=True,
                )
                next_log_step += 10_000

            if done or truncated:
                obs, _ = env.reset(seed=seed + total_steps)
                break

        if done or truncated:
            bootstrap = torch.zeros((), device=device)
        else:
            with torch.no_grad():
                _, next_value = model(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
            bootstrap = next_value.squeeze(0).squeeze(0)

        returns: list[torch.Tensor] = []
        r = bootstrap
        for rew in reversed(rewards):
            r = torch.tensor(rew, dtype=torch.float32, device=device) + gamma * r
            returns.insert(0, r)

        returns_t = torch.stack(returns).detach()
        values_t = torch.stack(values)
        log_probs_t = torch.stack(log_probs)
        entropies_t = torch.stack(entropies)

        advantage = returns_t - values_t
        advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)

        policy_loss = -(log_probs_t * advantage.detach()).mean()
        value_loss = F.mse_loss(values_t, returns_t)
        entropy_bonus = entropies_t.mean()
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

    train_seconds = time.perf_counter() - t0
    env.close()

    out_dir = ensure_results_dir(results_dir)
    model_path = out_dir / "a2c_model.pt"
    torch.save(model.state_dict(), model_path)

    model.eval()

    def act_fn(obs_np: np.ndarray) -> int:
        with torch.no_grad():
            x = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = model(x)
            return int(torch.argmax(logits, dim=-1).item())

    mean_reward, std_reward, mean_score = evaluate_policy_callable(
        act_fn,
        episodes=eval_episodes,
        seed=seed + 7000,
        render=False,
        env_factory=FlappyBirdEnv,
    )

    return ExperimentResult(
        algorithm="a2c",
        train_seconds=train_seconds,
        timesteps=total_steps,
        mean_reward=mean_reward,
        std_reward=std_reward,
        mean_score=mean_score,
        episodes_eval=eval_episodes,
        model_path=str(model_path),
    )
