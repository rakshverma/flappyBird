from .env import FlappyBirdEnv


def main() -> None:
	from .cli import main as _main

	_main()


__all__ = ["main", "FlappyBirdEnv"]
