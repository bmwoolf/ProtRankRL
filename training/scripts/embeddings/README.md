# Protein Embeddings Scripts

Scripts for generating protein embeddings using deep learning models.

## Scripts

### `generate_esm_embeddings_truncated.py`
Generates ESM-1b embeddings for protein sequences with automatic truncation for long sequences.
- **Input**: `protein_inputs/raw/validated_proteins.fasta`
- **Output**: `protein_inputs/embeddings/validated_proteins_esm_embeddings.npy` and `.csv`
- **Features**: 
  - Handles sequences longer than 1024 amino acids by truncating to middle portion
  - Generates 1280-dimensional embeddings
  - Includes error handling for problematic sequences

## Usage

```bash
# Generate ESM embeddings with default settings
python generate_esm_embeddings_truncated.py

# Generate with custom parameters
python generate_esm_embeddings_truncated.py \
  --fasta protein_inputs/raw/validated_proteins.fasta \
  --out_npy protein_inputs/embeddings/my_embeddings.npy \
  --out_csv protein_inputs/embeddings/my_embeddings.csv \
  --max_length 1024
```

## Output Format

- **NumPy array**: `(n_proteins, 1280)` shape
- **CSV file**: Each row contains protein ID + 1280 embedding features
- **Embedding dimension**: 1280 (ESM-1b model output) 