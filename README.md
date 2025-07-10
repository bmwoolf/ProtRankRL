![Banner](assets/github_banner.png)

# ProtRankRL

Rank proteins using reinforcement learning

## Overview

ProtRankRL uses reinforcement learning to rank proteins based on their potential as drug targets. The system combines protein sequence embeddings with experimental activity data to provide intelligent protein prioritization.

## Features

- **Reinforcement Learning Model**: Pre-trained PPO agent for protein ranking
- **Protein Embeddings**: ESM-2 protein sequence embeddings
- **Experimental Data Integration**: ChEMBL activity data integration
- **Fast API**: RESTful API for protein ranking
- **Comprehensive Testing**: Full test suite with 95%+ coverage

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/bmwoolf/protrankrl.git
cd ProtRankRL
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the API

Start the API server:
```bash
python -m uvicorn src.api.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

### Demo

Run the demo script to see the API in action:
```bash
python demo.py
```

**Note**: Make sure the API server is running first!

### API Usage

#### Rank Proteins

```bash
curl -X POST "http://localhost:8000/rank" \
     -H "Content-Type: application/json" \
     -d '{"uniprot_ids": ["O43451", "O60706", "O76074"]}'
```

#### Health Check

```bash
curl http://localhost:8000/health
```

### Rank Endpoint

**POST** `/rank`

Request body:
```json
{
  "uniprot_ids": ["O43451", "O60706", "O76074"]
}
```

Response:
```json
{
  "rankings": [
    {
      "uniprot_id": "O76074",
      "rank": 1,
      "score": 0.934,
      "confidence": 0.934,
      "has_activity": true,
      "experimental_data": {
        "activity_count": 2000,
        "pchembl_mean": 7.333,
        "pchembl_std": 1.546,
        "binding_affinity_kd": null
      }
    }
  ],
  "metadata": {
    "total_proteins": 3,
    "valid_proteins": 3,
    "processing_time": 0.0002,
    "model_version": "v1.0",
    "database_stats": {
      "total_proteins": 100,
      "feature_dim": 1280,
      "active_proteins": 96,
      "experimental_data_count": 96
    }
  }
}
```

## Testing

Run the full test suite:
```bash
python -m pytest tests/ -v
```

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

## Project Structure

```
ProtRankRL/
├── src/
│   ├── api/           # FastAPI application
│   ├── data/          # Data loading and management
│   └── models/        # Pre-trained models
├── data/              # Protein data and embeddings
├── models/            # Trained model files
├── tests/             # Test suite
├── training/          # Training scripts (legacy)
└── demo.py           # Demo script
```

## Future
- Autoselect the best-performing agent to rank new protein candidates
- API for integration
- Scripting for retraining or extending with new data or objectives

## License
MIT

## Environment Setup for GPU Training

### Prerequisites
- NVIDIA GPU with CUDA support
- Ubuntu 24.04 (tested)
- NVIDIA Driver 550.120 or compatible
- CUDA 12.0
- cuDNN 9.10.2

### GPU Environment Setup

1. **Install CUDA 12.0:**
   ```bash
   # Download and install CUDA 12.0 from NVIDIA website
   # Follow official installation guide for Ubuntu 24.04
   ```

2. **Install cuDNN 9.10.2 for CUDA 12:**
   ```bash
   # Download cuDNN 9.10.2 .deb package from NVIDIA
   sudo dpkg -i cudnn-local-repo-ubuntu2404-9.10.2.21_1.0-1_amd64.deb
   sudo cp /var/cudnn-local-repo-ubuntu2404-9.10.2.21/cudnn-local-*-keyring.gpg /usr/share/keyrings/
   sudo apt update
   sudo apt install cudnn-cuda-12
   ```

3. **Create conda environment:**
   ```bash
   conda create -n protrank python=3.11 -y
   conda activate protrank
   ```

4. **Install JAX with CUDA 12 support:**
   ```bash
   pip install --upgrade "jax[cuda12]"
   ```

5. **Install project dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

6. **Verify GPU setup:**
   ```bash
   python -c "import jax; print('JAX version:', jax.__version__); print('GPU devices:', jax.devices('gpu'))"
   ```

### Important Notes
- Python 3.11 is required for JAX GPU wheels compatibility
- cuDNN 9.10.2 is required (9.5.1 will cause compatibility issues)
- Ensure no old cuDNN libraries remain in `/usr/local/cuda/lib64/`
- Reboot system after installing new cuDNN libraries
- The requirements.txt has been updated to avoid cuDNN version conflicts