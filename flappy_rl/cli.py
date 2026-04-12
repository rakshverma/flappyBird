from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .env import FlappyBirdEnv
from .experiments import run_a2c_experiment, run_adaptive_a3c, run_experiments, run_ppo_experiment


def _play_manual(seed: int):
    from .render import PygameRenderer

    env = FlappyBirdEnv(seed=seed)
    renderer = PygameRenderer()

    obs, _ = env.reset(seed=seed)
    done = False
    truncated = False

    try:
        while not (done or truncated):
            action = 0
            for event in __import__("pygame").event.get():
                if event.type == __import__("pygame").QUIT:
                    return
                if event.type == __import__("pygame").KEYDOWN and event.key == __import__("pygame").K_SPACE:
                    action = 1

            obs, _, done, truncated, _ = env.step(action)
            _ = obs
            renderer.draw(env.get_state())
    finally:
        env.close()
        renderer.close()


def _play_random(seed: int, steps: int = 1000):
    from .render import PygameRenderer

    env = FlappyBirdEnv(seed=seed)
    renderer = PygameRenderer()
    rng = np.random.default_rng(seed)

    obs, _ = env.reset(seed=seed)
    done = False
    truncated = False
    t = 0

    try:
        while t < steps and not (done or truncated):
            if renderer.handle_quit():
                break
            action = int(rng.integers(0, 2))
            obs, _, done, truncated, _ = env.step(action)
            _ = obs
            renderer.draw(env.get_state())
            t += 1
            time.sleep(0.01)
    finally:
        env.close()
        renderer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Flappy Bird RL (A3C + pygame, pure Python env)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train", help="Run A3C training and store timing/results")
    train.add_argument("--timesteps", type=int, default=100_000)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--results-dir", type=str, default="results")
    train.add_argument("--a3c-workers", type=int, default=4)

    train_a2c = sub.add_parser(
        "train-a2c",
        help="Run A2C training (MLP only)",
    )
    train_a2c.add_argument("--timesteps", type=int, default=100_000)
    train_a2c.add_argument("--seed", type=int, default=42)
    train_a2c.add_argument("--results-dir", type=str, default="results")
    train_a2c.add_argument("--n-steps", type=int, default=16)
    train_a2c.add_argument("--lr", type=float, default=3e-4)

    train_ppo = sub.add_parser(
        "train-ppo",
        help="Run PPO training (MLP only)",
    )
    train_ppo.add_argument("--timesteps", type=int, default=100_000)
    train_ppo.add_argument("--seed", type=int, default=42)
    train_ppo.add_argument("--results-dir", type=str, default="results")
    train_ppo.add_argument("--rollout-size", type=int, default=2048)
    train_ppo.add_argument("--update-epochs", type=int, default=8)
    train_ppo.add_argument("--minibatch-size", type=int, default=256)
    train_ppo.add_argument("--lr", type=float, default=2.5e-4)

    adaptive = sub.add_parser(
        "train-a3c-adaptive",
        help="Run adaptive A3C using threshold schedule",
    )
    adaptive.add_argument("--seed", type=int, default=42)
    adaptive.add_argument("--results-dir", type=str, default="results")
    adaptive.add_argument("--threshold-score", type=float, default=30.0)
    adaptive.add_argument("--score-cap", type=float, default=None)
    adaptive.add_argument("--chunk-steps", type=int, default=1000)
    adaptive.add_argument("--extra-steps", type=int, default=5000)
    adaptive.add_argument("--stop-on-threshold", action="store_true", default=True)
    adaptive.add_argument("--continue-after-threshold", action="store_false", dest="stop_on_threshold")

    play = sub.add_parser("play", help="Run playable pygame demo")
    play.add_argument("--seed", type=int, default=42)
    play.add_argument("--mode", choices=["manual", "random"], default="manual")

    play_trained = sub.add_parser("play-trained", help="Play using trained A3C model")
    play_trained.add_argument("--seed", type=int, default=42)
    play_trained.add_argument("--model-path", type=str, default=None)

    args = parser.parse_args()

    if args.cmd == "train":
        results = run_experiments(
            timesteps=args.timesteps,
            seed=args.seed,
            results_dir=args.results_dir,
            a3c_workers=args.a3c_workers,
        )
        for result in results:
            print(
                f"{result.algorithm}: time={result.train_seconds:.2f}s, "
                f"mean_reward={result.mean_reward:.3f}, mean_score={result.mean_score:.3f}, "
                f"model={result.model_path}"
            )
    elif args.cmd == "train-a2c":
        results = run_a2c_experiment(
            timesteps=args.timesteps,
            seed=args.seed,
            results_dir=args.results_dir,
            n_steps=args.n_steps,
            lr=args.lr,
        )
        for result in results:
            print(
                f"{result.algorithm}: time={result.train_seconds:.2f}s, "
                f"steps={result.timesteps}, mean_reward={result.mean_reward:.3f}, "
                f"mean_score={result.mean_score:.3f}, model={result.model_path}"
            )
    elif args.cmd == "train-ppo":
        results = run_ppo_experiment(
            timesteps=args.timesteps,
            seed=args.seed,
            results_dir=args.results_dir,
            rollout_size=args.rollout_size,
            update_epochs=args.update_epochs,
            minibatch_size=args.minibatch_size,
            lr=args.lr,
        )
        for result in results:
            print(
                f"{result.algorithm}: time={result.train_seconds:.2f}s, "
                f"steps={result.timesteps}, mean_reward={result.mean_reward:.3f}, "
                f"mean_score={result.mean_score:.3f}, model={result.model_path}"
            )
    elif args.cmd == "train-a3c-adaptive":
        results = run_adaptive_a3c(
            seed=args.seed,
            results_dir=args.results_dir,
            threshold_score=args.threshold_score,
            score_cap=args.score_cap,
            chunk_steps=args.chunk_steps,
            extra_steps=args.extra_steps,
            stop_on_threshold=args.stop_on_threshold,
        )
        for result in results:
            print(
                f"{result.algorithm}: time={result.train_seconds:.2f}s, "
                f"steps={result.timesteps}, mean_reward={result.mean_reward:.3f}, "
                f"mean_score={result.mean_score:.3f}, model={result.model_path}"
            )
    elif args.cmd == "play":
        if args.mode == "manual":
            _play_manual(seed=args.seed)
        else:
            _play_random(seed=args.seed)
    elif args.cmd == "play-trained":
        from .play_trained import play_a3c_model

        if args.model_path is not None:
            model_path = Path(args.model_path)
        else:
            model_path = Path("results") / "a3c_mlp_adaptive_model.pt"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Train first or provide --model-path explicitly."
            )

        play_a3c_model(model_path, seed=args.seed)


if __name__ == "__main__":
    main()
