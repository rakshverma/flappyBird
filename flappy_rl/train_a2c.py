from __future__ import annotations

import time
from collections import deque
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
    gae_lambda: float = 0.95,
    n_steps: int = 64,
    lr: float = 3e-4,
    entropy_coef: float = 0.02,
    entropy_coef_end: float = 0.001,
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
    last_policy_loss = 0.0
    last_value_loss = 0.0
    last_entropy = 0.0
    recent_episode_rewards: deque[float] = deque(maxlen=100)
    recent_episode_lengths: deque[int] = deque(maxlen=100)
    recent_episode_scores: deque[int] = deque(maxlen=100)
    episode_reward = 0.0
    episode_len = 0

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

            next_obs, reward, done, truncated, info = env.step(int(action.item()))

            log_probs.append(dist.log_prob(action).squeeze(0))
            entropies.append(dist.entropy().squeeze(0))
            values.append(value.squeeze(0).squeeze(0))
            rewards.append(float(np.clip(reward, -1.0, 1.0)))

            episode_reward += float(reward)
            episode_len += 1

            total_steps += 1
            obs = next_obs

            if done or truncated:
                recent_episode_rewards.append(episode_reward)
                recent_episode_lengths.append(episode_len)
                recent_episode_scores.append(int(info.get("score", 0)))
                episode_reward = 0.0
                episode_len = 0

            while total_steps >= next_log_step:
                elapsed = time.perf_counter() - t0
                ep_count = len(recent_episode_rewards)
                mean_ep_reward = float(np.mean(recent_episode_rewards)) if ep_count > 0 else 0.0
                mean_ep_len = float(np.mean(recent_episode_lengths)) if ep_count > 0 else 0.0
                mean_ep_score = float(np.mean(recent_episode_scores)) if ep_count > 0 else 0.0
                entropy_now = entropy_coef + (entropy_coef_end - entropy_coef) * (
                    total_steps / max(1, timesteps)
                )
                print(
                    f"[a2c] steps={next_log_step}/{timesteps} elapsed={elapsed:.2f}s "
                    f"policy_loss={last_policy_loss:.4f} value_loss={last_value_loss:.4f} "
                    f"entropy={last_entropy:.4f} entropy_coef={entropy_now:.4f} "
                    f"ep_reward={mean_ep_reward:.2f} ep_len={mean_ep_len:.1f} ep_score={mean_ep_score:.2f}",
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

        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        values_t = torch.stack(values)
        log_probs_t = torch.stack(log_probs)
        entropies_t = torch.stack(entropies)

        advantages = torch.zeros_like(rewards_t)
        gae = torch.zeros((), device=device)
        for t in reversed(range(rewards_t.shape[0])):
            next_val = bootstrap if t == rewards_t.shape[0] - 1 else values_t[t + 1]
            delta = rewards_t[t] + gamma * next_val - values_t[t]
            gae = delta + gamma * gae_lambda * gae
            advantages[t] = gae

        returns_t = (advantages + values_t).detach()
        advantage = advantages
        advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)

        policy_loss = -(log_probs_t * advantage.detach()).mean()
        value_loss = F.mse_loss(values_t, returns_t)
        entropy_bonus = entropies_t.mean()
        entropy_coef_now = entropy_coef + (entropy_coef_end - entropy_coef) * (
            total_steps / max(1, timesteps)
        )
        loss = policy_loss + value_coef * value_loss - entropy_coef_now * entropy_bonus

        last_policy_loss = float(policy_loss.item())
        last_value_loss = float(value_loss.item())
        last_entropy = float(entropy_bonus.item())

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
