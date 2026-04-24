# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LeRobot is a HuggingFace library for real-world robotics using PyTorch. It provides a unified interface for datasets, policies (ML models), simulation environments, and physical robot hardware.

## Installation

```bash
# Core install
pip install -e .

# With all extras (simulators, all policies, dev/test tools)
pip install -e ".[all]"

# With specific extras (e.g., for ACT policy + Aloha sim)
pip install -e ".[aloha,dev,test]"
```

Key optional extras: `feetech`, `dynamixel`, `lekiwi`, `act`, `diffusion`, `smolvla`, `pi`, `aloha`, `pusht`, `dev`, `test`.

## Commands

### Linting and Formatting

```bash
# Run all pre-commit checks (ruff format + lint, mypy, typos, bandit, etc.)
pre-commit run --all-files

# Run ruff directly
ruff check .
ruff format .
```

### Tests

```bash
# Run unit/fast tests
pytest tests/ -vv

# Run a single test file
pytest tests/test_datasets.py -vv

# Run a single test by name
pytest tests/test_datasets.py::test_push_dataset_to_hub -vv

# End-to-end training + eval tests (requires sim extras)
make test-end-to-end DEVICE=cpu

# Individual E2E tests
make test-act-ete-train DEVICE=cpu
make test-act-ete-eval DEVICE=cpu
```

### CLI Entry Points

All scripts are installed as CLI commands via `[project.scripts]`:

```bash
lerobot-train --policy.type=act --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human ...
lerobot-eval --policy.path=<checkpoint_dir>/pretrained_model ...
lerobot-record --robot.type=so100 ...
lerobot-teleoperate --robot.type=so100 ...
lerobot-calibrate --robot.type=so100 ...
lerobot-dataset-viz --repo_id=lerobot/pusht
lerobot-info
```

## Architecture

### Source Layout (`src/lerobot/`)

- **`configs/`** — Typed dataclass configs (via `draccus`) for training, policies, environments, datasets. This is the single source of truth for all hyperparameters and CLI flags.
- **`scripts/`** — CLI entry points. Each script (`lerobot_train.py`, `lerobot_eval.py`, etc.) parses a config, constructs components via factories, and runs the pipeline.
- **`policies/`** — Policy implementations: ACT, Diffusion, TD-MPC, SmolVLA, Pi0, WallX, etc. Each policy has its own subdirectory with `modeling_*.py`, `configuration_*.py`, and optionally `configuration_*.yaml`.
- **`datasets/`** — `LeRobotDataset` class built on HuggingFace `datasets`. Data is stored as Parquet (states/actions) + MP4 (video). Includes utilities for push-to-hub, dataset editing, and video encoding.
- **`robots/`** — Hardware abstractions. All robots implement a common interface: `connect()`, `get_observation()`, `send_action()`, `disconnect()`.
- **`motors/`** — Low-level motor drivers (Feetech, Dynamixel, Damiao, Robstride).
- **`cameras/`** — Camera drivers (OpenCV, RealSense, etc.).
- **`teleoperators/`** — Teleoperation devices: gamepads, keyboards, phone-based (HEBI).
- **`envs/`** — Gymnasium environment wrappers for simulators (Aloha, PushT, MetaWorld, LIBERO).
- **`utils/`** — Shared utilities including `TimerManager` (perf timer), logging init, and cross-platform helpers.

### Key Patterns

**Factory pattern**: Components are instantiated by type string from config using factory functions in `*/factory.py` files (e.g., `make_policy(cfg)`, `make_env(cfg)`, `make_robot(cfg)`). Adding a new policy/robot/env means registering it in the factory.

**Config system**: All configs are `draccus` dataclasses (similar to `dataclasses` but with YAML/CLI parsing). Training is configured via CLI flags (`--policy.type=act --batch_size=32`) or a `--config_path` pointing to a saved JSON config for resuming.

**HuggingFace Hub integration**: Datasets and model weights are pulled from/pushed to the Hub. `dataset.repo_id` references a Hub dataset; `policy.path` can be a local checkpoint or Hub model ID.

**Hardware abstraction**: The `Robot` class in `robots/` presents a uniform interface so the same policy evaluation loop (`lerobot-eval`) works on any physical robot or simulation environment.

### Training Pipeline

`lerobot-train` → loads `TrainPipelineConfig` → creates dataset, policy, optimizer → training loop with periodic eval + checkpoint saves to `output_dir/checkpoints/<step>/pretrained_model/`.

Resume training: `lerobot-train --config_path=<output_dir>/checkpoints/<step>/pretrained_model/train_config.json --resume=true`

### Code Style

- Python 3.12+, line length 110 (`ruff`)
- Google-style docstrings
- `ruff` enforces: pycodestyle, pyflakes, isort, bugbear, comprehensions, print-statement checks, naming, pyupgrade
- `mypy` type checking is being gradually enabled module by module (currently strict on `configs/`, `cameras/`, `motors/`, `envs/`, `model/`, `transport/`)
