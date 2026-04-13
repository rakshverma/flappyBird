# Flappy Bird RL (A3C + A2C + PPO)

This project uses:

- Pure Python Flappy Bird environment (`flappy_rl/env.py`)
- A3C, A2C, and PPO training in PyTorch
- Pygame rendering and trained-policy playback

## UV Setup

```bash
uv sync
```

## Train (A3C)

```bash
uv run flappy-bird train --timesteps 100000 --seed 42 --a3c-workers 4
```

Train only PPO:

```bash
uv run flappy-bird train-ppo --timesteps 100000 --seed 42
```

Train only A2C:

```bash
uv run flappy-bird train-a2c --timesteps 800000 --seed 42 --n-steps 64 --gae-lambda 0.95 --entropy-coef 0.02 --entropy-coef-end 0.001
```

## Play

Manual:

```bash
uv run flappy-bird play --mode manual
```

Random demo:

```bash
uv run flappy-bird play --mode random
```

Play trained A3C model:

```bash
uv run flappy-bird play-trained
```

## Outputs

- `results/summary.csv`
- `results/a3c_only/a3c_result.json`
- `results/a3c_only/a3c_model.pt`
- `results/a3c_only/summary.csv`
- `results/ppo_only/ppo_result.json`
- `results/ppo_only/ppo_model.pt`
- `results/ppo_only/summary.csv`
- `results/a2c_only/a2c_result.json`
- `results/a2c_only/a2c_model.pt`
- `results/a2c_only/summary.csv`

## Note

A3C, A2C, and PPO training and environment dynamics are fully Python-based.

uv run flappy-bird play-trained --seed 42 --model-path results/a3c_only/a3c_model.pt