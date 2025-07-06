# ProtRankRL

Reinforcement Learning for Protein Target Prioritization

## TODO
1. integrate real protein features: use embeddings from UniRep, ESM, ProtBERT, or AlphaFold structure-derived features
2. Replace random “hits” with real labels: use experimental data (e.g., binding assays, functional screens).
3. Multi-objective rewards: Add diversity, novelty, or cost constraints to the reward function.
4. Try other RL algorithms from [miniRL](https://github.com/seungeunrho/minimalRL)

## Overview

ProtRankRL is a Gymnasium-compatible reinforcement learning environment for training agents to prioritize protein targets based on scientific embeddings and known outcomes. The goal of this is to understand RL and apply it to one digital domain. The environment models triage decisions over protein batches, enabling RL agents to learn optimal ranking strategies.

## Features

- **Gymnasium-compatible**: Full compatibility with Gym RL framework
- **Type-safe**: Full Python 3.11+ type hints with Pydantic validation
[WIP for more]

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

### Training PPO Agent

```bash
# Run quick demo
python examples/quickstart_demo.py

# Train PPO agent
python examples/train_ppo_agent.py
```

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

## Generating ESM Protein Embeddings

To use real protein features, you can generate ESM-1b embeddings for your protein sequences using the provided script.

### Requirements
- `fair-esm` and `torch` (see requirements.txt)
- Input protein sequences in FASTA format (e.g., `protein_inputs/[protein_name].fasta`)

### Usage

1. Place your protein sequences in a FASTA file (e.g., `protein_inputs/[protein_name].fasta`).
2. Run the embedding script:
   ```bash
   python scripts/generate_esm_embeddings.py --fasta ../protein_inputs/[protein_name].fasta
   ```
   - You can also specify output file paths with `--out_npy` and `--out_csv`.

### Output
- Embeddings are saved as a NumPy array (`esm_embeddings.npy`) and as a CSV file (`esm_embeddings.csv`) with sequence IDs.
- These files can be loaded into your RL environment as real protein features.


## License

MIT

---