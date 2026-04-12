from __future__ import annotations

from pathlib import Path

from .train_a2c import train_a2c
from .train_a3c import train_a3c, train_a3c_adaptive
from .train_ppo import train_ppo
from .utils import ExperimentResult, append_summary


def run_experiments(
    timesteps: int = 100_000,
    seed: int = 42,
    results_dir: str | Path = "results",
    a3c_workers: int = 4,
) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []

    a3c_result = train_a3c(
        timesteps=timesteps,
        seed=seed,
        results_dir=results_dir,
        workers=a3c_workers,
    )
    append_summary(a3c_result, results_dir=results_dir)
    results.append(a3c_result)

    return results


def run_adaptive_a3c(
    seed: int = 42,
    results_dir: str | Path = "results",
    threshold_score: float = 30.0,
    score_cap: float | None = None,
    chunk_steps: int = 1000,
    extra_steps: int = 5000,
    stop_on_threshold: bool = True,
) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []

    a3c_result = train_a3c_adaptive(
        seed=seed,
        results_dir=results_dir,
        threshold_score=threshold_score,
        score_cap=score_cap,
        chunk_steps=chunk_steps,
        extra_steps=extra_steps,
        stop_on_threshold=stop_on_threshold,
    )
    append_summary(a3c_result, results_dir=results_dir)
    results.append(a3c_result)

    return results


def run_a2c_experiment(
    timesteps: int = 100_000,
    seed: int = 42,
    results_dir: str | Path = "results",
    n_steps: int = 16,
    lr: float = 3e-4,
) -> list[ExperimentResult]:
    result = train_a2c(
        timesteps=timesteps,
        seed=seed,
        results_dir=results_dir,
        n_steps=n_steps,
        lr=lr,
    )
    append_summary(result, results_dir=results_dir)
    return [result]


def run_ppo_experiment(
    timesteps: int = 100_000,
    seed: int = 42,
    results_dir: str | Path = "results",
    rollout_size: int = 2048,
    update_epochs: int = 8,
    minibatch_size: int = 256,
    lr: float = 2.5e-4,
) -> list[ExperimentResult]:
    result = train_ppo(
        timesteps=timesteps,
        seed=seed,
        results_dir=results_dir,
        rollout_size=rollout_size,
        update_epochs=update_epochs,
        minibatch_size=minibatch_size,
        lr=lr,
    )
    append_summary(result, results_dir=results_dir)
    return [result]
