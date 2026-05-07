# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands use `uv run` — do not use plain `python`.

```bash
uv sync                                                                          # install deps
uv run python src/train.py --n-ncas 3 --epochs 1000 --device cpu                # basic run
uv run python src/train.py --config configs/example.json                         # run from config
uv run python src/train.py --n-ncas 3 --epochs 10000 --device cuda --wandb      # with W&B logging
```

No test suite or CI config exists in this repo.

## Architecture

The project simulates multiple competing Neural Cellular Automata (NCAs) sharing a grid, where each NCA fights for territory via attack/defense vectors resolved by cosine similarity.

**Data flow:**

1. `train.py` parses args → builds `Config` → calls `train_loop`
2. `World` owns the pool of grids (`pool_size × cell_dim × H × W`) and draws batches each epoch
3. `CASunGroup` manages all NCAs plus a learnable "sun" entity (index 0 in the alive dimension)
4. Each epoch: `world.get_seed()` → `world.step(group, grid)` → `group.update_models()`

**Cell tensor layout** (`cell_dim` channels total):

```
[0 .. N]         alive channels: sun (0) + one per NCA (1..N)
[N+1 .. N+S]     state channels: first half = attack, second half = defense
[N+S+1 .. end]   hidden channels (not visible to opponents when alive_visible=False)
```

`Config.cell_dim = cell_state_dim + cell_hidden_dim + n_ncas + 1`

**Competition mechanics** (`model.py: CASunGroup._run_competition_parallel`):
- All NCAs share one `MergedCAModel` using grouped convolutions (groups=N) so each NCA has isolated weights
- Each NCA proposes updates; attack vs. defense cosine similarity determines territory strength
- Strengths are softmax-normalized to assign aliveness; cells below a threshold (0.4) die
- The sun is a fixed-shape learnable vector that competes as a neutral entity

**Feature hooks** (`world.py`): `World._build_features` assembles a list of `Feature` subclasses whose `on_init/before_step/after_step` methods are called each epoch. Active features: `SimpleBurnInFeature` (ramps up steps), `SunUpdateFeature` (controls when sun trains), `UpdatePoolWithNondeadFeature` (replaces dead-NCA runs in the pool).

## Configuration

`Config` (`src/config.py`) is a dataclass; `__post_init__` validates and falls back CUDA/MPS → CPU. Load from file with `Config.from_file(path)`, override with CLI flags. Key derived properties:

- `config.cell_dim` — full channel count (use this for tensor shapes)
- `config.cell_wo_alive_dim` — state+hidden only (model output dim)
- `config.alive_dim` — n_ncas + 1 (sun)

Model checkpoints are saved to `{run_name}/` containing `config.json`, `model.pt`, `sun.npy`, and `seed.npy`.

## Conventions

- `bfloat16` on CUDA, `float32` elsewhere — set in `World.__init__` and respected by `device_autocast`
- `einops` (`rearrange`, `reduce`, `einsum`, `repeat`) is used throughout; prefer it over manual reshapes
- Add new training behaviors as `Feature` subclasses registered in `World._build_features`, not inline in `World.step`
- When adding config fields, add them to the `Config` dataclass and validate in `__post_init__` if needed
