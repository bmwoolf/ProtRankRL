# Scripts Organization

This folder contains scripts organized by functionality:

## data_collection/
Scripts for downloading and extracting real experimental data:
- `download_protein_sequences.py` - Download protein sequences from UniProt
- `extract_chembl_activities_parallel.py` - Extract experimental activity data from ChEMBL (parallelized)
- `download_uniprot_bindingdb_data.py` - Download binding data from BindingDB

## validation/
Scripts for validating protein annotations and data quality:
- `query_chembl_by_uniprot.py` - Validate proteins have ChEMBL targets and activities
- `expand_annotated_accessions.py` - Expand protein accession lists

## embeddings/
Scripts for generating protein embeddings:
- `generate_esm_embeddings_truncated.py` - Generate ESM-1b embeddings for proteins (handles long sequences)

## synthetic/
Scripts for generating synthetic data:
- `generate_realistic_synthetic_data.py` - Generate synthetic protein-ligand data

## Usage Examples

```bash
# Download protein sequences
python scripts/data_collection/download_protein_sequences.py

# Extract ChEMBL activities (parallel)
python scripts/data_collection/extract_chembl_activities_parallel.py 8

# Generate ESM embeddings
python scripts/embeddings/generate_esm_embeddings_truncated.py

# Validate protein annotations
python scripts/validation/query_chembl_by_uniprot.py
``` 