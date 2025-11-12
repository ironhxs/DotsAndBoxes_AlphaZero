# DotsAndBoxes AlphaZero - AI Agent Instructions

## Project Overview
This is an **AlphaZero implementation for Dots and Boxes** (点格棋), a 6×6 grid (5×5 boxes) game. The project uses **OpenSpiel** for game logic, **PyTorch** for deep learning, and supports multiple parallel training modes optimized for GPU utilization.

## Architecture & Key Components

### Core Module Structure (`model/`)
- **Game Environment**: `game.py` wraps OpenSpiel's dots_and_boxes with 9-channel observation tensors (9×6×6)
- **Neural Networks**: 
  - `model.py`: ResNet with GroupNorm, SE attention blocks, dropout (1775万 parameters)
  - `model_transformer.py`: Hybrid CNN+Transformer architecture (default, 256 filters, 12 blocks)
- **MCTS**: `mcts.py` implements AlphaZero-style tree search with UCB selection, virtual loss for parallelism
- **Training Coaches**: 
  - `base_coach.py`: Abstract base with `learn()` loop, `train()` method, Arena integration
  - `coach_parallel.py`: Multi-process self-play dispatcher (3 modes: full/shared/single GPU)
  - `coach.py`: Single-process fallback
- **Arena System**: `arena.py` runs new vs old model tournaments, accepts new model only if win rate ≥ 55%

### Parallel Self-Play Engines
- `mcts_full_parallel_gpu.py`: Each worker has independent GPU model copy (highest GPU utilization)
- `mcts_multiprocess_gpu.py`: Workers share GPU via inference server (saves VRAM)
- `mcts_concurrent_gpu.py`: Single-process async games (debugging)

### Configuration System (Hydra-style)
- `config/config.yaml`: Master config with `defaults` section
- `config/game/dots_and_boxes.yaml`: Game-specific (num_rows=5, num_cols=5, action_size=60)
- `config/model/transformer.yaml`: Model architecture (num_blocks, num_filters, num_heads)
- `config/trainer/alphazero.yaml`: Training hyperparameters (MCTS sims, epochs, lr schedule)

Config loading pattern in `cli/train_parallel.py`:
```python
# Load main config, then resolve defaults to load game/model/trainer yamls
# Build unified args dict merging all configs
```

## Critical Workflows

### Training Execution
**Primary command** (uses conda environment `gmd`):
```bash
/root/miniconda3/envs/gmd/bin/python cli/train_parallel.py
```

**Key parameters in config.yaml**:
- `num_self_play_games: 200` (per iteration)
- `num_parallel_games: 25` (concurrent workers)
- `num_simulations: 400` (MCTS depth)
- `batch_size: 1024`, `train_epochs: 300`
- `arena_games: 100`, `arena_threshold: 0.55`

**Training loop** (`base_coach.py:learn()`):
1. Parallel self-play → collect (state, policy, result) tuples
2. Train neural network on replay buffer (mixed precision if `use_amp: true`)
3. Arena tournament: new model vs previous model
4. Accept new model only if win rate > 55%, else keep previous
5. Save checkpoint every `checkpoint_interval` iterations

### GPU Optimization Modes
Set `parallel_mode` in config:
- `"full"`: Best performance, requires ~4GB VRAM per worker
- `"shared"`: Memory efficient, single inference server
- `"single"`: Debugging, no multiprocessing

Monitor GPU: `watch -n 1 nvidia-smi` or `./cli/monitor_gpu.sh`

### Testing & Evaluation
- **Quick test**: `cli/quick_test.py` (2 iterations, fast validation)
- **Evaluate model**: `cli/evaluate_model.py` (load checkpoint, run Arena)
- **Human play**: `cli/play_ultimate.py` (interactive game vs AI)
- **System check**: `tests/test_system.py` (verifies OpenSpiel, CUDA, model forward pass)

## Project-Specific Conventions

### State Representation
OpenSpiel returns **observation tensor** reshaped to `(9, 6, 6)`:
- Channels 0-3: Horizontal/vertical edges for each player
- Channels 4-7: Box ownership
- Channel 8: Current player indicator

Legal actions are 60 possible edges (30 horizontal + 30 vertical).

### MCTS Integration Pattern
```python
# In self-play workers (mcts_full_parallel_gpu.py):
from model.mcts import MCTS
mcts = MCTS(game, nnet_wrapper, mcts_args)
action_probs = mcts.get_action_prob(state, temp=1.0)
# Sample action from probs, apply to state
```

Neural network wrapper must implement:
- `predict(state)` → (policy_vector, value_scalar)

### Model Checkpointing
Saved to `results/checkpoints/`:
- `best.pth`: Current best model (survives Arena challenges)
- `temp.pth`: Training iteration snapshot
- `interrupted.pth`: Auto-saved on SIGINT

Load with: `nnet.load_checkpoint(folder='results/checkpoints', filename='best.pth')`

### Data Augmentation
**Not used by default** (`augment_data: false`) because Dots and Boxes has asymmetric scoring (boxes belong to specific player). If enabled, implement 8-fold symmetry (4 rotations × 2 flips) in `base_coach.py:get_symmetries()`.

## Common Pitfalls

1. **Multiprocessing on CUDA**: Must use `torch.multiprocessing` with `set_start_method('spawn')` before any CUDA initialization
2. **GroupNorm vs BatchNorm**: ResNet uses GroupNorm for stability in multi-process training; don't switch without testing
3. **Virtual Loss in MCTS**: `self.Vsa` tracks temporary Q-value penalties during parallel search to prevent duplicate exploration
4. **Arena Fairness**: Always play equal games with swapped starting player (`random_start=True` or alternating)
5. **Memory Leaks**: Call `state.clone()` in MCTS simulations to avoid OpenSpiel state mutation issues

## Integration Points

- **OpenSpiel**: Game logic via `pyspiel.load_game("dots_and_boxes(num_rows=5,num_cols=5)")`
- **Conda Environment**: Reuses `gmd` environment with PyTorch 1.10.1, CUDA 11.1
- **TensorBoard**: Logs to `results/logs/` if `tensorboard: true`
- **External Dependencies**: Only addition to `gmd` env is `pip install open_spiel`

## Debugging Commands

Check environment:
```bash
conda activate gmd
python tests/test_system.py  # Verifies PyTorch, CUDA, OpenSpiel, model initialization
```

Single-process training (easier to debug):
```python
# In config.yaml, set:
parallel_mode: "single"
num_parallel_games: 1
use_multiprocess: false
```

Profile GPU usage:
```bash
python -m torch.utils.bottleneck cli/train_parallel.py
```

## Key Files for Understanding
- `model/base_coach.py` (lines 85-210): Main `learn()` and `train()` loop
- `model/mcts.py` (lines 1-150): MCTS search algorithm
- `config/config.yaml`: All hyperparameters in one place
- `doc/ALPHAZERO_EXPLAINED.md`: Design rationale and training flow diagram
