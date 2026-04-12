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


def train_ppo(
    timesteps: int,
    seed: int,
    results_dir: str | Path,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    lr: float = 2.5e-4,
    rollout_size: int = 2048,
    update_epochs: int = 8,
    minibatch_size: int = 256,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    eval_episodes: int = 10,
) -> ExperimentResult:
    device = _select_torch_device()
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = FlappyBirdEnv(seed=seed)
    model = ActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    t0 = time.perf_counter()
    steps_done = 0
    updates = 0

    obs, _ = env.reset(seed=seed)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

    while steps_done < timesteps:
        obs_buf: list[torch.Tensor] = []
        act_buf: list[torch.Tensor] = []
        logp_buf: list[torch.Tensor] = []
        rew_buf: list[torch.Tensor] = []
        done_buf: list[torch.Tensor] = []
        val_buf: list[torch.Tensor] = []

        this_rollout = min(rollout_size, timesteps - steps_done)
        for _ in range(this_rollout):
            with torch.no_grad():
                logits, value = model(obs_t.unsqueeze(0))
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)

            next_obs, reward, done, truncated, _ = env.step(int(action.item()))
            terminal = bool(done or truncated)

            obs_buf.append(obs_t)
            act_buf.append(action.squeeze(0))
            logp_buf.append(logp.squeeze(0))
            rew_buf.append(torch.tensor(float(np.clip(reward, -1.0, 1.0)), device=device))
            done_buf.append(torch.tensor(float(terminal), device=device))
            val_buf.append(value.squeeze(0).squeeze(0))

            steps_done += 1
            obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)
            if terminal:
                obs_reset, _ = env.reset(seed=seed + steps_done)
                obs_t = torch.tensor(obs_reset, dtype=torch.float32, device=device)

        with torch.no_grad():
            _, next_value = model(obs_t.unsqueeze(0))
            next_value = next_value.squeeze(0).squeeze(0)

        rewards = torch.stack(rew_buf)
        dones = torch.stack(done_buf)
        values = torch.stack(val_buf)

        advantages = torch.zeros_like(rewards)
        gae = torch.zeros((), device=device)
        for t in reversed(range(this_rollout)):
            if t == this_rollout - 1:
                next_non_terminal = 1.0 - dones[t]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_val = values[t + 1]
            delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages[t] = gae
        returns = advantages + values

        obs_batch = torch.stack(obs_buf)
        act_batch = torch.stack(act_buf)
        old_logp_batch = torch.stack(logp_buf)
        adv_batch = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        ret_batch = returns
        val_batch = values

        batch_size = obs_batch.shape[0]
        idx = np.arange(batch_size)

        frac = steps_done / max(1, timesteps)
        current_lr = lr * (1.0 - frac)
        optimizer.param_groups[0]["lr"] = max(1e-5, current_lr)

        target_kl = 0.03
        for _ in range(update_epochs):
            np.random.shuffle(idx)
            approx_kl_total = 0.0
            approx_kl_count = 0

            for start in range(0, batch_size, minibatch_size):
                mb_idx = idx[start : start + minibatch_size]
                mb_obs = obs_batch[mb_idx]
                mb_act = act_batch[mb_idx]
                mb_old_logp = old_logp_batch[mb_idx]
                mb_adv = adv_batch[mb_idx]
                mb_ret = ret_batch[mb_idx]
                mb_old_val = val_batch[mb_idx]

                logits, value = model(mb_obs)
                value = value.squeeze(-1)
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(mb_act)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - mb_old_logp)
                pg_loss1 = ratio * mb_adv
                pg_loss2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
                policy_loss = -torch.min(pg_loss1, pg_loss2).mean()

                value_clipped = mb_old_val + torch.clamp(value - mb_old_val, -clip_eps, clip_eps)
                value_loss_unclipped = (value - mb_ret) ** 2
                value_loss_clipped = (value_clipped - mb_ret) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - (new_logp - mb_old_logp)).mean().item()
                approx_kl_total += approx_kl
                approx_kl_count += 1

            mean_kl = approx_kl_total / max(1, approx_kl_count)
            if mean_kl > target_kl:
                break

        updates += 1
        if updates % 5 == 0 or steps_done >= timesteps:
            print(f"[ppo] steps={steps_done} lr={optimizer.param_groups[0]['lr']:.6f}", flush=True)

    train_seconds = time.perf_counter() - t0
    env.close()

    out_dir = ensure_results_dir(results_dir)
    model_path = out_dir / "ppo_model.pt"
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
        seed=seed + 9000,
        render=False,
        env_factory=FlappyBirdEnv,
    )

    return ExperimentResult(
        algorithm="ppo",
        train_seconds=train_seconds,
        timesteps=steps_done,
        mean_reward=mean_reward,
        std_reward=std_reward,
        mean_score=mean_score,
        episodes_eval=eval_episodes,
        model_path=str(model_path),
    )
