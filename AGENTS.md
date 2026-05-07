# Petri Dish NCA — AI Agent Instructions

## Purpose
This repository implements an adversarial Neural Cellular Automata (NCA) training experiment in PyTorch. The main entrypoint is `src/train.py`, which loads configuration, initializes a `World`, and trains competing NCAs in a shared substrate.

## Key project structure
- `src/config.py`: Defines the `Config` dataclass used across the project.
- `src/train.py`: CLI parsing, experiment setup, logging, and main training loop.
- `src/model.py`: NCA model implementation and `CASunGroup` competition wrapper.
- `src/world.py`: Environment, seed pool, step logic, and feature hooks.
- `src/viz.py`: Visualization utilities used for snapshots and color generation.
- `configs/`: Example JSON configuration files.

## Important conventions
- Use `uv` to run code and manage the workspace environment. Typical command:
  - `uv run python src/train.py --n-ncas 3 --epochs 1000 --device cpu`
  - `uv run python src/train.py --config configs/example.json`
- The model is written for Python `>=3.11` and relies on PyTorch and `einops`.
- Configuration is centralized in `Config`; CLI options override config values before validation.
- `Config.__post_init__()` performs runtime checks and device fallback for CUDA/MPS.
- `World` uses feature hooks (`SimpleBurnInFeature`, `SunUpdateFeature`, `UpdatePoolWithNondeadFeature`) to modify training behavior.
- `train.py` optionally logs to Weights & Biases when `--wandb` is provided.

## Useful facts for editing and analysis
- The repo has no separate tests or CI config in the workspace.
- The training loop stores a pool of grids and draws batches from it each epoch.
- `Config.cell_dim` includes state, hidden, NCA channels, and aliveness channels.
- The project uses JSON config files under `configs/` and saving/loading is implemented with `Config.from_file` and `Config.save`.

## Best behavior for AI coding agents
- Prefer understanding `src/train.py` first, then inspect `src/config.py`, `src/model.py`, and `src/world.py`.
- Do not assume a standard entrypoint like `python main.py`; use `src/train.py` and the `uv` commands described in README.
- Keep changes minimal and consistent with existing dataclass-based config and feature hook patterns.
- If adding new functionality, ensure `Config` validation and runtime device fallback remain intact.

## References
- README: [README.md](README.md)
- Config & CLI: `src/config.py`, `src/train.py`
- Model & world behavior: `src/model.py`, `src/world.py`
