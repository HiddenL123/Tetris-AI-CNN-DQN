# Tetris AI with CNN and DDQN like algorithm

A significantly enhanced implementation of a Tetris-playing AI agent using Deep Q-Learning with a CNN-based neural network and pseudo-DDQN training. Originally forked from [tetris-ai-python](https://github.com/nlinker/tetris-ai-python). This version features a convolutional neural network (CNN) for direct board representation, target networks for stability, and an advanced Tetris environment with Super Rotation System (SRS), piece holding, and sophisticated move search using BFS.

## Key Improvements Over Original

- **Better Game Environment**: Added SRS (Super Rotation System), piece holding, and piece queue visibility
- **Advanced Search Algorithm**: Uses Breadth First Search (BFS) to explore all possible end states instead of naive column-drop search
- **Pseudo-DDQN Architecture**: Target networks for improved Q-value stability (simplified double DQN approach adapted for state value model)
- **Advanced State Representation**: Separate board state (CNN) + scalar game features (MLP)
- **Flexible Exploration Strategy**: Epsilon-greedy exploration transitioning to Boltzmann exploration
- **Robust Checkpointing**: Resume training from any step with full state restoration
- **Comprehensive Logging**: TensorBoard integration with detailed metrics tracking
- **N-step Returns**: Experience replay with configurable n-step bootstrapping
- **Gradient Clipping**: Improved training stability with gradient norm clipping

## Installation

```bash
# Create conda environment
conda create --name tetris-ai python=3.10
conda activate tetris-ai

# Install dependencies
conda install pytorch::pytorch torchvision torchaudio -c pytorch
conda install opencv tensorboard tqdm pillow

```

##  System Requirements

Two tiers are provided: a minimal set for running/evaluating the agent, and a recommended set for training.

### Minimum Requirements — Run & Evaluation
These are sufficient to run the environment, watch the agent play, or evaluate saved models.

**Hardware:**
- **CPU**: Any modern x86_64 CPU (dual-core or better)
- **RAM**: 1 GB minimum (4 GB recommended)
- **GPU**: Not required; CPU-only runs are supported. A CUDA-capable GPU will speed up inference but is optional.
- **Storage**: Minimal

**Software:**
- **Python**: 3.10+
- **PyTorch**: 2.0+ (CPU-only install is fine)
- **Optional**: CUDA 11.8+ and compatible drivers if using GPU

### Recommended Requirements — Training
For efficient model training (multi-million steps), use the following recommended configuration:

**Hardware:**
- **GPU**: NVIDIA RTX 4050 or equivalent/better (recommended)
- **CPU**: Intel Core Ultra 7 or equivalent/better
- **RAM**: 16 GB minimum (32 GB recommended for larger batch sizes)
- **Storage**: 1 GB+ for checkpoints and logs (SSD recommended for faster I/O)

**Software:**
- **Python**: 3.10+
- **PyTorch**: 2.0+ with CUDA support
- **CUDA**: 11.8+

**Notes:**
- GPU with CUDA support is highly recommended for training; CPU-only training is possible but significantly slower (~50–100× slower)
- Estimated training time on recommended hardware: ~2–3 weeks for 5M steps

## Quick Start
```bash
python run_interactive.py
```

### Training
Start or resume training:

```bash
# Start fresh training
python run_train.py

# Or modify the main block in run_train.py to resume from checkpoint
# dqn(conf, resume_checkpoint_dir='logs/your-checkpoint', resume_step_override=3300000)
```

### Evaluation
Run the trained agent in visual mode:
```bash
python run_eval.py
```

### Monitor Training
View real-time metrics with TensorBoard:
```bash
tensorboard --logdir logs/
```

## Configuration

Edit the `AgentConf` class in `run_train.py` to adjust training parameters:


## Resuming from Checkpoint

The training system supports full checkpoint resumption:

```python
# In run_train.py main block:
conf = AgentConf()
dqn(
    conf,
    resume_checkpoint_dir='logs/tetris-20260206-023315-ts5000000-bs32-d0.99',
    resume_step_override=3300000
)
```

This automatically:
- Loads model weights and replay memory
- Restores epsilon and temperature values
- Continues training with step counter at 3.3M
- Maintains continuous logging to the same directory

## Algorithm Details

### Pseudo-DDQN with Target Network

This implementation uses a simplified approach with target networks to improve training stability:
- **Online Network**: Used for current state Q-values and action selection
- **Target Network**: Separate copy used for computing target Q-values, updating periodically

**Update Rule**:
```
Q(s,a) = r + γ^n × max(Q_target(s'))
```

where n is the n-step return. The target network is periodically synchronized with the online network to stabilize training.

### State Representation

**Board State** (20×10 grid):
- Binary representation of filled cells
- Processed through CNN (1 conv layer → 1×1 feature maps)

**Scalar Features** (14 features):
- Lines cleared, holes, bumpiness, height metrics
- Current, next piece, held piece, and pieces in queue information
- Garbage lines information
- Processed through separate MLP

Both streams concatenated and passed through dense layers for Q-value prediction.

### Exploration Strategy

**Phase 1: Epsilon-Greedy (0 to 5M steps)**
- Probability ε of random action, else greedy
- ε decays linearly from 1.0 to 0.0

**Phase 2: Boltzmann (5M+ steps)**
- Action selection via softmax with temperature
- Temperature decays exponentially: T = T₀ × (decay_rate)^step
- Allows fine-tuning of discovered strategies

### Experience Replay

- Stores transitions: (state, next_state, reward, done, n_steps)
- Supports n-step bootstrapping (configurable n=3)
- Samples uniformly from replay buffer
- Reduces correlation between training samples

## Logging and Monitoring

Training logs include:
- **Score Metrics**: Average, min, max over last 100 episodes
- **Q-Value Metrics**: Predicted values and TD error
- **Learning Progress**: Epsilon and temperature decay curves
- **TensorBoard**: All metrics saved for visualization

Access with:
```bash
tensorboard --logdir logs/
```

## Performance

TODO:

## Original Project Attribution

Based on [tetris-ai-python](https://github.com/nlinker/tetris-ai-python), which provided the initial Tetris environment and basic DQN framework. 

## License

Maintains compatibility with original project. Refer to original repository for licensing details.

## References

- [Deep Q-Learning Paper](https://www.nature.com/articles/nature14236)
- [Double DQN Paper](https://arxiv.org/abs/1509.06461)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Tetris AI Strategies](https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/)

