# Synthetic Data Scripts

Scripts for generating synthetic protein-ligand data for testing and development.

## Scripts

### `generate_realistic_synthetic_data.py`
Generates realistic synthetic protein-ligand interaction data.
- **Purpose**: Create test datasets for RL environment development
- **Features**: Generates realistic protein-ligand interaction patterns
- **Use case**: Testing RL algorithms before using real experimental data

## Usage

```bash
# Generate synthetic data
python generate_realistic_synthetic_data.py
```

## Note

These scripts are primarily for development and testing. For production use, prefer the real experimental data from ChEMBL and other biological databases. 