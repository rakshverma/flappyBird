from __future__ import annotations

from pathlib import Path

import torch

from .env import FlappyBirdEnv
from .model import ActorCritic
from .render import PygameRenderer


def _play_with_act_fn(act_fn, seed: int = 42):
    env = FlappyBirdEnv(seed=seed)
    renderer = PygameRenderer()

    obs, _ = env.reset(seed=seed)
    done = False
    truncated = False

    try:
        while True:
            if renderer.handle_quit():
                break

            action = int(act_fn(obs))
            obs, _, done, truncated, _ = env.step(action)
            renderer.draw(env.get_state())

            if done or truncated:
                obs, _ = env.reset()
                done = False
                truncated = False
    finally:
        env.close()
        renderer.close()


def play_a3c_model(
    model_path: str | Path,
    seed: int = 42,
):
    model = ActorCritic()
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    def act_fn(obs):
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits, _ = model(x)
            return int(torch.argmax(logits, dim=-1).item())

    _play_with_act_fn(act_fn, seed=seed)
