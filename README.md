# ProtRankRL

Reinforcement Learning for Protein Target Prioritization

## TODO
1. integrate real protein features: use embeddings from UniRep, ESM, ProtBERT, or AlphaFold structure-derived features
2. Replace random “hits” with real labels: use experimental data (e.g., binding assays, functional screens).
3. Multi-objective rewards: Add diversity, novelty, or cost constraints to the reward function.
4. Try other RL algorithms from [miniRL](https://github.com/seungeunrho/minimalRL)

## Overview

ProtRankRL is a Gymnasium-compatible reinforcement learning environment for training agents to prioritize protein targets based on scientific embeddings and known outcomes. The environment models triage decisions over protein batches, enabling RL agents to learn optimal ranking strategies.

## Features

- **Gymnasium-compatible**: Full compatibility with RL frameworks
- **Synthetic data generation**: Built-in factory for testing and development
- **Type-safe**: Full Python 3.11+ type hints with Pydantic validation
- **Production-ready**: Comprehensive testing and CI/CD pipeline
- **PPO integration**: Ready-to-use training scripts with stable-baselines3

## Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/your-org/protrankrl.git
cd protrankrl
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### Basic Usage

```python
from src.env import ProteinEnvFactory

# Create synthetic environment
env = ProteinEnvFactory.create_synthetic_env(
    num_proteins=32,
    feature_dim=64,
    hit_rate=0.2,
    seed=42
)

# Run single episode
obs, info = env.reset()
while True:
    action = env.action_space.sample()  # Random policy
    obs, reward, done, truncated, info = env.step(action)
    if done:
        break

print(f"Episode reward: {env.get_episode_stats()['total_reward']}")
```

### Training PPO Agent

```bash
# Run quick demo
python examples/quickstart_demo.py

# Train PPO agent
python examples/train_ppo_agent.py
```

## Environment Details

- **Action Space**: Discrete(N) - select protein index
- **Observation Space**: Box(-1, 1, D) - normalized protein features
- **Reward**: Binary (0/1) based on whether selected protein is a hit
- **Episode**: Processes entire protein batch sequentially

## Development

### Running Tests

```bash
pytest -q  # Quick tests
pytest --cov=src  # With coverage
```

### Code Quality

```bash
black src/ tests/ examples/  # Format code
ruff check src/ tests/ examples/  # Lint code
mypy src/  # Type checking
```

## Project Structure

```
ProtRankRL/
├── src/env/
│   ├── __init__.py
│   └── protein_env.py      # Main environment
├── tests/
│   ├── __init__.py
│   └── test_protein_env.py # Comprehensive tests
├── examples/
│   ├── quickstart_demo.py  # Single episode demo
│   └── train_ppo_agent.py  # PPO training script
├── pyproject.toml          # Modern Python packaging
├── setup.cfg              # Test configuration
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## License

MIT License - see LICENSE file for details.