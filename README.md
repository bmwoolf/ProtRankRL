![Banner](assets/github_banner.png)

# ProtRankRL

Reinforcement Learning for Protein Target Prioritization

## TODO
3. Multi-objective rewards: Add diversity, novelty, or cost constraints to the reward function.
4. Try other RL algorithms from [miniRL](https://github.com/seungeunrho/minimalRL)

## Overview

ProtRankRL is a Gymnasium-compatible reinforcement learning environment for training agents to prioritize protein targets based on scientific embeddings and known outcomes. The goal of this is to understand RL and apply it to one digital domain. The environment models triage decisions over protein batches, enabling RL agents to learn optimal ranking strategies.

## Features

- **Gymnasium-compatible**: Full compatibility with Gym RL framework
- **Type-safe**: Full Python 3.11+ type hints with Pydantic validation
- **Real experimental data**: Uses ChEMBL experimental activities instead of synthetic data
- **ESM protein embeddings**: 1280-dimensional protein representations from ESM-1b
- **100 validated proteins**: Well-annotated proteins with experimental data
[WIP for more]

## Quick Start

### Automated Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-org/protrankrl.git
cd protrankrl

# Run automated setup and data pipeline
python setup.py --run-pipeline

# Or for development setup
python setup.py --dev --run-pipeline --run-tests
```

### Manual Installation

```bash
# Clone and install
git clone https://github.com/your-org/protrankrl.git
cd protrankrl
pip install -e .

# Install development dependencies
pip install -e ".[dev]"

# Install additional dependencies for data collection
pip install requests biopython chembl-webresource-client fair-esm torch
```

### Data Pipeline

```bash
# Run the complete data pipeline
python scripts/run_full_pipeline.py

# Or with custom settings
python scripts/run_full_pipeline.py --num-proteins 50 --num-workers 4 --verbose
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

## Data Pipeline

The project now uses real experimental data instead of synthetic data:

### 1. Protein Validation
- 100 UniProt proteins validated as "well-annotated" in ChEMBL
- Each protein has exact ChEMBL target match and experimental activities

### 2. Data Collection
```bash
# Download protein sequences
python scripts/data_collection/download_protein_sequences.py

# Extract experimental activities (parallel)
python scripts/data_collection/extract_chembl_activities_parallel.py 8

# Generate ESM embeddings
python scripts/embeddings/generate_esm_embeddings_truncated.py

# Create unified dataset
python scripts/data_collection/create_unified_dataset.py
```

### 3. Dataset Statistics
- **100 proteins** with ESM embeddings (1280 dimensions each)
- **92,133 experimental activities** from ChEMBL
- **96% activity rate** (96/100 proteins have experimental data)
- **Unified format**: Ready for RL environment integration

### Output
- Unified dataset: `protein_inputs/processed/unified_protein_dataset.csv`
- ESM embeddings: `protein_inputs/embeddings/validated_proteins_esm_embeddings.npy`
- Experimental data: `protein_inputs/processed/chembl_experimental_data.csv`

## License

MIT

---