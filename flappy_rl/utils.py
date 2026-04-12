from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExperimentResult:
    algorithm: str
    train_seconds: float
    timesteps: int
    mean_reward: float
    std_reward: float
    mean_score: float
    episodes_eval: int
    model_path: str


def ensure_results_dir(base_dir: str | Path = "results") -> Path:
    p = Path(base_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def append_summary(result: ExperimentResult, results_dir: str | Path = "results") -> Path:
    out_dir = ensure_results_dir(results_dir)
    csv_path = out_dir / "summary.csv"
    write_header = not csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "algorithm",
                "train_seconds",
                "timesteps",
                "mean_reward",
                "std_reward",
                "mean_score",
                "episodes_eval",
                "model_path",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(result.__dict__)

    json_path = out_dir / f"{result.algorithm}_result.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result.__dict__, f, indent=2)

    return csv_path
