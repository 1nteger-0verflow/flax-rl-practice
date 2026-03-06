# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Guidelines

This document contains critical information about working with this codebase.
Follow these guidelines precisely.

## Rules

1. Package Management
   - ONLY use uv, NEVER pip
   - Installation: `uv add package`
   - Upgrading: `uv add --dev package --upgrade-package package`
   - FORBIDDEN: `uv pip install`, `@latest` syntax

2. Code Quality
   - Type hints required for all code
   - Follow existing patterns exactly
   - Use Google style for docstring

3. Testing Requirements
   - Framework: `uv run --frozen pytest`
   - Single test: `uv run --frozen pytest tests/path/to/test.py::test_name`
   - Coverage: test edge cases and errors
   - New features require tests
   - Bug fixes require regression tests

4. Git
   - Follow the Conventional Commits style on commit messages.

## Code Formatting and Linting

1. Ruff
   - Format: `uv run --frozen ruff format .`
   - Check: `uv run --frozen ruff check .`
   - Fix: `uv run --frozen ruff check . --fix`
2. Pre-commit
   - Config: `.pre-commit-config.yaml`
   - Runs: on git commit
   - Tools: Ruff (Python)

## Running

```bash
# Install with CUDA support
uv sync --extra cuda12

# Install with CPU-only JAX
uv sync --extra cpu

# Run DQN training
uv run python src/train.py
```

## Architecture

This is a DQN (Deep Q-Network) implementation for Atari games using JAX and Flax NNX.

### Key Files

- **`src/modules/model.py`**: `DQNCNN` — CNN-based Q-network using Flax NNX. Architecture: 3 conv layers (8×8/4, 4×4/2, 3×3/1) with GroupNorm + ReLU, followed by two linear layers (512-dim hidden). Input: `(batch, H, W, 4)` stacked grayscale frames; output: Q-values per action.

- **`src/modules/wrapper.py`**: Atari environment setup via `get_atari_env()`. Applies `AtariPreprocessing` (84×84 grayscale, frame skip 4), `ChannelLastFrameStack` (4 frames → `(84, 84, 4)` channel-last), `RecordVideo`, and `TimeLimit` (2000 steps/episode).

- **`src/train.py`**: DQN training loop. `ReplayBuffer` stores transitions LZ4-compressed (250K capacity). Training: online/target network pair, epsilon-greedy exploration (1.0→0.1 over 100K steps), Adam optimizer (lr=2e-4), MSE loss, target sync every 10K steps, orbax checkpoints every 100K steps. Results saved to `results/out/`.

### JAX/Flax Patterns

- Uses **Flax NNX** (not Linen). Models are stateful objects; `nnx.split`/`nnx.update` are used to copy weights between online and target networks.
- Training functions decorated with `@nnx.jit`.
- Gradients computed via `nnx.value_and_grad`.
