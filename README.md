![Banner](assets/github_banner.png)

# ProtRankRL

Rank proteins using reinforcement learning

## TODO
5. production level product per the original description (simplify repo, turn into abstraction layer)
6. incentives like Google

## Overview
ProtRankRL is a production-ready reinforcement learning system for prioritizing protein targets using real experimental data and multi-objective reward optimization. It enables small biotechs to leverage compute for data-driven, automated triage of protein candidates for drugs.

## Key Features
- **Multi-objective RL**: Balances activity, diversity, novelty, and cost
- **Benchmarking**: Compare PPO, DQN, and A2C agents
- **Metrics**: Hit rate, precision@k, recall@k, F1, AUC-ROC, NDCG
- **Visualizations**: Bar/line plots for agent performance
- **Integration**: Minimal, reproducible, and extensible

## Quick Start
```bash
# Clone and install
git clone https://github.com/bmwoolf/protrankrl.git
cd protrankrl

# Create virtual env
python3 -m venv .venv
source .venv/bin/activate

pip install -e .

# Run the benchmark (requires data pipeline to be run first)
python tests/test_sb3_algorithms.py

# Code quality 
black src/ tests/ examples/ # Format 
ruff check src/ tests/ examples/ # Lint
mypy src/ # Type checking
```

## How It Works
- Uses a Gymnasium-compatible environment with real protein data from Uniprot + Chembl and ESM-1b embeddings
- Custom reward function combines hit/activity, diversity, novelty, and cost
- Trains and evaluates RL agents (PPO, DQN, A2C) on protein triage
- Outputs agent performance metrics and visualizations

## Results (Example)
| Agent | Mean Reward | Hit Rate | Precision@5 | Recall@5 | F1@5 | NDCG |
|-------|-------------|----------|-------------|----------|------|------|
| PPO   | 3.18        | 1.00     | 1.00        | 0.10     | 0.18 | 1.00 |
| DQN   | 10.89       | 0.70     | 0.60        | 0.09     | 0.15 | 1.00 |
| A2C   | 12.04       | 1.00     | 1.00        | 0.10     | 0.18 | 1.00 |

See `tests/charts/agent_mean_rewards.png` and `tests/charts/agent_episode_rewards.png` for visualizations, though they are simplistic.

## Future
- Autoselect the best-performing agent to rank new protein candidates
- API for integration
- Scripting for retraining or extending with new data or objectives

## License
MIT