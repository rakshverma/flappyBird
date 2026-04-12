from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class FlappyEnvCore:
    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)
        self._seed = seed

        self.bird_x = 0.2
        self.bird_y = 0.5
        self.bird_v = 0.0
        self.bird_radius = 0.025

        self.pipe_x = 1.2
        self.pipe_width = 0.12
        self.gap_center = 0.5
        self.gap_half_height = 0.15

        self.gravity = -0.0060
        self.flap_impulse = 0.0120
        self.max_speed = 0.03
        self.pipe_speed = 0.010

        self.survive_reward = 0.10
        self.pass_reward = 1.00
        self.crash_reward = -1.00

        self.max_steps = 12000
        self.score = 0
        self.steps = 0
        self.done = False
        self.passed_current_pipe = False

        self.reset()

    def reseed(self, seed: int):
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def _sample_gap_center(self):
        self.gap_center = float(self._rng.uniform(0.3, 0.7))

    def _observation(self) -> list[float]:
        return [
            float(self.bird_y),
            float(self.bird_v),
            float(self.pipe_x - self.bird_x),
            float(self.gap_center),
            float(self.gap_center - self.bird_y),
        ]

    def _overlaps_pipe(self) -> bool:
        within_x = (self.bird_x + self.bird_radius >= self.pipe_x) and (
            self.bird_x - self.bird_radius <= self.pipe_x + self.pipe_width
        )
        if not within_x:
            return False

        gap_bottom = self.gap_center - self.gap_half_height
        gap_top = self.gap_center + self.gap_half_height
        inside_gap = (self.bird_y - self.bird_radius >= gap_bottom) and (
            self.bird_y + self.bird_radius <= gap_top
        )
        return not inside_gap

    def reset(self) -> list[float]:
        self.bird_y = 0.5
        self.bird_v = 0.0
        self.pipe_x = 1.2
        self.passed_current_pipe = False
        self.done = False
        self.score = 0
        self.steps = 0
        self._sample_gap_center()
        return self._observation()

    def step(self, action: int):
        if self.done:
            return self._observation(), 0.0, True, {
                "score": self.score,
                "done_reason": "episode_already_done",
            }

        self.steps += 1
        if action == 1:
            self.bird_v += self.flap_impulse

        self.bird_v += self.gravity
        self.bird_v = float(np.clip(self.bird_v, -self.max_speed, self.max_speed))
        self.bird_y += self.bird_v

        self.pipe_x -= self.pipe_speed
        if self.pipe_x < -self.pipe_width:
            self.pipe_x = 1.2
            self.passed_current_pipe = False
            self._sample_gap_center()

        reward = self.survive_reward
        if (not self.passed_current_pipe) and (self.pipe_x + self.pipe_width < self.bird_x):
            self.passed_current_pipe = True
            self.score += 1
            reward += self.pass_reward

        done_reason = "alive"
        if self.bird_y < 0.0 or self.bird_y > 1.0:
            self.done = True
            done_reason = "out_of_bounds"
            reward = self.crash_reward

        if (not self.done) and self._overlaps_pipe():
            self.done = True
            done_reason = "pipe_collision"
            reward = self.crash_reward

        if (not self.done) and self.steps >= self.max_steps:
            self.done = True
            done_reason = "max_steps"

        info = {
            "score": self.score,
            "steps": self.steps,
            "done_reason": done_reason,
        }
        return self._observation(), float(reward), self.done, info

    def get_state(self) -> dict[str, float | int | bool]:
        return {
            "bird_x": float(self.bird_x),
            "bird_y": float(self.bird_y),
            "bird_v": float(self.bird_v),
            "bird_radius": float(self.bird_radius),
            "pipe_x": float(self.pipe_x),
            "pipe_width": float(self.pipe_width),
            "gap_center": float(self.gap_center),
            "gap_half_height": float(self.gap_half_height),
            "score": int(self.score),
            "steps": int(self.steps),
            "done": bool(self.done),
        }


class FlappyBirdEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": ["human", None], "render_fps": 60}

    def __init__(self, seed: int = 42):
        super().__init__()
        self._core = FlappyEnvCore(seed=seed)
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, -1.0, -1.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 2.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._core.reseed(seed)
        obs = np.asarray(self._core.reset(), dtype=np.float32)
        info: dict[str, Any] = {}
        return obs, info

    def step(self, action: int):
        obs, reward, done, info = self._core.step(int(action))
        return (
            np.asarray(obs, dtype=np.float32),
            float(reward),
            bool(done),
            False,
            dict(info),
        )

    def get_state(self) -> dict[str, Any]:
        return dict(self._core.get_state())

    def close(self):
        return None
