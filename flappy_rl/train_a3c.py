from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from .env import FlappyBirdEnv
from .evaluate import evaluate_policy_callable, evaluate_policy_detailed
from .model import ActorCritic
from .utils import ExperimentResult, ensure_results_dir


def _worker_loop(
    worker_id: int,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    lock: Any,
    global_steps: Any,
    max_steps: int,
    gamma: float,
    t_max: int,
    seed: int,
):
    torch.manual_seed(seed + worker_id)
    env = FlappyBirdEnv(seed=seed + worker_id * 17)

    while True:
        with global_steps.get_lock():
            if global_steps.value >= max_steps:
                break

        obs, _ = env.reset()
        done = False
        truncated = False

        while not (done or truncated):
            log_probs: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            rewards: list[float] = []
            entropies: list[torch.Tensor] = []

            for _ in range(t_max):
                x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                logits, value = model(x)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()

                next_obs, reward, done, truncated, _ = env.step(int(action.item()))

                log_probs.append(dist.log_prob(action))
                entropies.append(dist.entropy())
                values.append(value.squeeze(-1))
                rewards.append(float(reward))

                obs = next_obs

                with global_steps.get_lock():
                    global_steps.value += 1
                    if global_steps.value >= max_steps:
                        done = True

                if done or truncated:
                    break

            if done or truncated:
                bootstrap = torch.zeros(1)
            else:
                with torch.no_grad():
                    _, next_value = model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                bootstrap = next_value.squeeze(-1)

            returns: list[torch.Tensor] = []
            r = bootstrap
            for rew in reversed(rewards):
                r = torch.tensor([rew], dtype=torch.float32) + gamma * r
                returns.insert(0, r)

            returns_t = torch.cat(returns).detach()
            values_t = torch.cat(values)
            log_probs_t = torch.cat(log_probs)
            entropies_t = torch.cat(entropies)

            adv = returns_t - values_t
            policy_loss = -(log_probs_t * adv.detach()).mean()
            value_loss = F.mse_loss(values_t, returns_t)
            entropy_bonus = entropies_t.mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus

            with lock:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            if done or truncated:
                break

    env.close()


def train_a3c(
    timesteps: int,
    seed: int,
    results_dir: str | Path,
    workers: int = 4,
    gamma: float = 0.99,
    t_max: int = 16,
    lr: float = 3e-4,
) -> ExperimentResult:
    mp_ctx = mp.get_context("spawn")

    torch.manual_seed(seed)
    shared_model = ActorCritic()
    shared_model.share_memory()

    optimizer = torch.optim.Adam(shared_model.parameters(), lr=lr)
    step_counter = mp_ctx.Value("i", 0)
    lock = mp_ctx.Lock()

    t0 = time.perf_counter()
    procs: list[mp.Process] = []
    for worker_id in range(workers):
        p = mp_ctx.Process(
            target=_worker_loop,
            args=(
                worker_id,
                shared_model,
                optimizer,
                lock,
                step_counter,
                timesteps,
                gamma,
                t_max,
                seed,
            ),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    train_seconds = time.perf_counter() - t0

    out_dir = ensure_results_dir(results_dir)
    model_path = out_dir / "a3c_model.pt"
    torch.save(shared_model.state_dict(), model_path)

    eval_model = ActorCritic()
    eval_model.load_state_dict(torch.load(model_path, map_location="cpu"))
    eval_model.eval()

    def act_fn(obs: np.ndarray) -> int:
        with torch.no_grad():
            logits, _ = eval_model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
            return int(torch.argmax(logits, dim=-1).item())

    mean_reward, std_reward, mean_score = evaluate_policy_callable(
        act_fn,
        episodes=10,
        seed=seed + 2000,
        render=False,
    )

    return ExperimentResult(
        algorithm="a3c",
        train_seconds=train_seconds,
        timesteps=int(step_counter.value),
        mean_reward=mean_reward,
        std_reward=std_reward,
        mean_score=mean_score,
        episodes_eval=10,
        model_path=str(model_path),
    )


def _select_torch_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _single_worker_train_chunk(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    env_factory: Callable[[int], FlappyBirdEnv],
    seed: int,
    start_step: int,
    steps: int,
    gamma: float,
    t_max: int,
    device: torch.device,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    grad_clip: float = 1.0,
    reward_clip: float | None = None,
    normalize_advantage: bool = False,
) -> int:
    env = env_factory(seed + 19)
    trained = 0

    while trained < steps:
        obs, _ = env.reset(seed=seed + start_step + trained)
        done = False
        truncated = False

        while not (done or truncated) and trained < steps:
            log_probs: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            rewards: list[float] = []
            entropies: list[torch.Tensor] = []

            for _ in range(t_max):
                x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                logits, value = model(x)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()

                next_obs, reward, done, truncated, _ = env.step(int(action.item()))
                if reward_clip is not None:
                    reward = float(np.clip(reward, -reward_clip, reward_clip))

                log_probs.append(dist.log_prob(action))
                entropies.append(dist.entropy())
                values.append(value.squeeze(-1))
                rewards.append(float(reward))

                obs = next_obs
                trained += 1
                if done or truncated or trained >= steps:
                    break

            if done or truncated:
                bootstrap = torch.zeros(1, device=device)
            else:
                with torch.no_grad():
                    next_x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    _, next_value = model(next_x)
                bootstrap = next_value.squeeze(-1)

            returns: list[torch.Tensor] = []
            r = bootstrap
            for rew in reversed(rewards):
                r = torch.tensor([rew], dtype=torch.float32, device=device) + gamma * r
                returns.insert(0, r)

            returns_t = torch.cat(returns).detach()
            values_t = torch.cat(values)
            log_probs_t = torch.cat(log_probs)
            entropies_t = torch.cat(entropies)

            adv = returns_t - values_t
            if normalize_advantage and adv.numel() > 1:
                adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
            policy_loss = -(log_probs_t * adv.detach()).mean()
            value_loss = F.mse_loss(values_t, returns_t)
            entropy_bonus = entropies_t.mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            if done or truncated:
                break

    env.close()
    return trained


def train_a3c_adaptive(
    seed: int,
    results_dir: str | Path,
    threshold_score: float = 30.0,
    score_cap: float | None = None,
    chunk_steps: int = 1000,
    extra_steps: int = 5000,
    stop_on_threshold: bool = True,
    eval_episodes: int = 10,
    gamma: float = 0.99,
    t_max: int = 16,
    lr: float = 3e-4,
) -> ExperimentResult:
    device = _select_torch_device()
    env_factory = FlappyBirdEnv
    model: torch.nn.Module = ActorCritic()
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total_steps = 0
    arch = "mlp"
    target_score = min(threshold_score, score_cap) if score_cap is not None else threshold_score
    if score_cap is not None and threshold_score > score_cap:
        print(
            f"[a3c_{arch}_adaptive] requested threshold {threshold_score} clamped to score_cap {score_cap}",
            flush=True,
        )

    t0 = time.perf_counter()
    while True:
        trained = _single_worker_train_chunk(
            model=model,
            optimizer=optimizer,
            env_factory=env_factory,
            seed=seed,
            start_step=total_steps,
            steps=chunk_steps,
            gamma=gamma,
            t_max=t_max,
            device=device,
        )
        total_steps += trained

        model.eval()

        def act_fn(obs: np.ndarray) -> int:
            with torch.no_grad():
                x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                logits, _ = model(x)
                return int(torch.argmax(logits, dim=-1).item())

        stats = evaluate_policy_detailed(
            act_fn,
            episodes=eval_episodes,
            seed=seed + 2300 + total_steps,
            env_factory=env_factory,
        )
        print(
            f"[a3c_{arch}_adaptive] steps={total_steps} mean_score={stats['mean_score']:.2f} "
            f"max_score={stats['max_score']:.2f} "
            f"threshold={target_score} device={device}",
            flush=True,
        )
        model.train()
        if stats["max_score"] >= target_score:
            break

    if stop_on_threshold:
        print(
            f"[a3c_{arch}_adaptive] threshold reached, stopping and saving at steps={total_steps}",
            flush=True,
        )
    else:
        trained = _single_worker_train_chunk(
            model=model,
            optimizer=optimizer,
            env_factory=env_factory,
            seed=seed,
            start_step=total_steps,
            steps=extra_steps,
            gamma=gamma,
            t_max=t_max,
            device=device,
        )
        total_steps += trained
        print(
            f"[a3c_{arch}_adaptive] threshold reached, trained extra {extra_steps} steps "
            f"(total={total_steps})",
            flush=True,
        )
    train_seconds = time.perf_counter() - t0

    out_dir = ensure_results_dir(results_dir)
    model_path = out_dir / f"a3c_{arch}_adaptive_model.pt"
    torch.save(model.state_dict(), model_path)

    model.eval()

    def act_fn(obs: np.ndarray) -> int:
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = model(x)
            return int(torch.argmax(logits, dim=-1).item())

    mean_reward, std_reward, mean_score = evaluate_policy_callable(
        act_fn,
        episodes=eval_episodes,
        seed=seed + 2800,
        render=False,
        env_factory=env_factory,
    )

    return ExperimentResult(
        algorithm=f"a3c_{arch}_adaptive",
        train_seconds=train_seconds,
        timesteps=total_steps,
        mean_reward=mean_reward,
        std_reward=std_reward,
        mean_score=mean_score,
        episodes_eval=eval_episodes,
        model_path=str(model_path),
    )
