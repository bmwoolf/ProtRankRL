# ProtRankRL: Reinforcement Learning for Protein Target Prioritization

A Gym-compatible reinforcement learning environment for training AI agents to prioritize protein targets based on scientific embeddings and known outcomes.

## Overview

ProtRankRL implements a reinforcement learning system for protein target prioritization in drug discovery. The system models the triage decision process where an agent must rank proteins in a batch based on their likelihood of being successful drug targets.

### Key Features

- **Gym-compatible Environment**: Full compatibility with stable-baselines3 and other RL frameworks
- **Flexible Feature Support**: Supports UniProt, AlphaFold, and Knowledge Graph embeddings
- **Extensible Design**: Ready for advanced reward functions and embedding lookups
- **Comprehensive Testing**: Full test suite with edge case coverage
- **Production Ready**: Designed for integration with real biopharma workflows

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/bmwoolf/ProtRankRL.git
cd ProtRankRL

# 2. Create a new virtual environment
python3 -m venv venv

# 3. Activate the virtual environment
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from src.env.protein_env import ProteinEnvFactory

# Create a synthetic environment
env = ProteinEnvFactory.create_synthetic_env(
    num_proteins=64,
    feature_dim=128,
    hit_rate=0.2,
    seed=42
)

# Run a simple episode
obs, info = env.reset()
total_reward = 0

while True:
    action = env.action_space.sample()  # Random action
    next_obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    
    if done:
        break
    
    obs = next_obs

print(f"Episode completed! Total reward: {total_reward}")
```

### Training with PPO

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.env.protein_env import ProteinEnvFactory

# Create vectorized environment
def make_env():
    return ProteinEnvFactory.create_synthetic_env(
        num_proteins=32,
        feature_dim=64,
        hit_rate=0.2
    )

env = DummyVecEnv([make_env for _ in range(4)])

# Train PPO agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

## Environment Specification

### State Space
- **Observation**: `np.ndarray` of shape `(D,)` containing protein features
- **Features**: UniProt semantic vectors, AlphaFold structural features, KG embeddings
- **Normalization**: Features are normalized to `[-1, 1]` range by default

### Action Space
- **Type**: `Discrete(N)` where N is the number of proteins in the batch
- **Action**: Integer in `[0, N-1]` representing protein ranking decision

### Reward Function
- **Current**: Binary reward (1 for hit, 0 for non-hit)
- **Future**: Composite reward with GO similarity, diversity, and latency terms

### Episode Structure
- **Length**: Fixed batch of proteins (e.g., 64 proteins per episode)
- **Termination**: Episode ends when all proteins are processed
- **Reset**: Returns to first protein with new batch


## Advanced Features

### Custom Reward Functions

The environment is designed to support advanced reward functions:

```python
# Future implementation
reward = α * hit + β * GO_similarity + γ * diversity - δ * latency
```

### Embedding Integration

Ready for integration with external embedding services:

```python
# Future implementation
class ProteinEnvWithEmbeddings(ProteinEnv):
    def __init__(self, kg_client, uniprot_client, alphafold_client):
        # Initialize with real embedding clients
        pass
```

### Imitation Learning

Support for warm-starting with expert trajectories:

```python
# Future implementation
class ProteinEnvWithExpertData(ProteinEnv):
    def __init__(self, expert_trajectories):
        # Load expert demonstration data
        pass
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_protein_env.py::TestProteinEnv::test_valid_initialization
```

## Examples

### Basic Demonstration

```bash
python test_env_demo.py
```

This will:
1. Create a synthetic environment
2. Run episodes with random actions
3. Compare different strategies
4. Generate visualization plots

### PPO Training

```bash
python examples/train_ppo_agent.py
```

This will:
1. Train a PPO agent on the environment
2. Evaluate performance
3. Compare with baselines
4. Save trained model and plots

## Performance

### Current Capabilities
- **Batch Size**: 64 proteins per episode (configurable)
- **Feature Dimension**: 128 dimensions (configurable)
- **Training Speed**: ~1000 episodes/hour on CPU
- **Memory Usage**: ~4KB per protein

### Target Performance
- **Latency**: <120ms p95 for inference
- **Throughput**: 1M+ proteins/day
- **Accuracy**: >25% improvement over heuristics

## License

MIT

## Citation

If you use this code in your research, please cite:

```bibtex
@software{protrankrl,
  title={ProtRankRL: Reinforcement Learning for Protein Target Prioritization},
  author={Bradley Woolf},
  year={2024},
  url={https://github.com/bmwoolf/ProtRankRL}
}
```