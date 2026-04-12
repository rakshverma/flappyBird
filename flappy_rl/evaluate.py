from __future__ import annotations

from typing import Callable

import numpy as np

from .env import FlappyBirdEnv


def evaluate_policy_callable(
    act_fn: Callable[[np.ndarray], int],
    episodes: int = 10,
    seed: int = 123,
    render: bool = False,
    env_factory: Callable[[int], FlappyBirdEnv] | None = None,
) -> tuple[float, float, float]:
    renderer = None
    if render:
        from .render import PygameRenderer

        renderer = PygameRenderer()

    factory = env_factory or FlappyBirdEnv
    env = factory(seed)

    rewards: list[float] = []
    scores: list[float] = []

    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            done = False
            truncated = False
            episode_reward = 0.0
            final_score = 0.0

            while not (done or truncated):
                action = int(act_fn(obs))
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += float(reward)
                final_score = float(info.get("score", final_score))

                if renderer is not None:
                    if renderer.handle_quit():
                        done = True
                    renderer.draw(env.get_state())

            rewards.append(episode_reward)
            scores.append(final_score)
    finally:
        env.close()
        if renderer is not None:
            renderer.close()

    arr = np.asarray(rewards, dtype=np.float32)
    score_arr = np.asarray(scores, dtype=np.float32)
    return float(arr.mean()), float(arr.std()), float(score_arr.mean())


def evaluate_policy_detailed(
    act_fn: Callable[[np.ndarray], int],
    episodes: int = 10,
    seed: int = 123,
    env_factory: Callable[[int], FlappyBirdEnv] | None = None,
) -> dict[str, float]:
    factory = env_factory or FlappyBirdEnv
    env = factory(seed)

    rewards: list[float] = []
    scores: list[float] = []
    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            done = False
            truncated = False
            episode_reward = 0.0
            final_score = 0.0

            while not (done or truncated):
                action = int(act_fn(obs))
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += float(reward)
                final_score = float(info.get("score", final_score))

            rewards.append(episode_reward)
            scores.append(final_score)
    finally:
        env.close()

    reward_arr = np.asarray(rewards, dtype=np.float32)
    score_arr = np.asarray(scores, dtype=np.float32)
    return {
        "mean_reward": float(reward_arr.mean()),
        "std_reward": float(reward_arr.std()),
        "mean_score": float(score_arr.mean()),
        "max_score": float(score_arr.max() if len(score_arr) > 0 else 0.0),
    }
