# Protein Inputs

This folder contains all protein-related data organized by processing stage.

## Folder Structure

### raw/
Raw protein data downloaded from external sources:
- `validated_proteins.fasta` - Protein sequences for our 100 validated proteins (from UniProt)
- `validated_proteins.csv` - Metadata for the 100 validated proteins (names, descriptions, etc.)

### processed/
Processed and validated data:
- `annotated_accessions.csv` - List of 100 UniProt accessions that are well-annotated in ChEMBL
- `chembl_experimental_data.csv` - Experimental activity data from ChEMBL (92,133 activities)
- `chembl_experimental_data.json` - Same data in JSON format for programmatic access

### embeddings/
Protein embeddings generated using deep learning models:
- `validated_proteins_esm_embeddings.npy` - ESM-1b embeddings as NumPy array (100 proteins × 1280 features)
- `validated_proteins_esm_embeddings.csv` - Same embeddings in CSV format with protein IDs

## Data Pipeline

1. **raw/validated_proteins.fasta** ← Downloaded from UniProt
2. **processed/annotated_accessions.csv** ← Validated protein list
3. **processed/chembl_experimental_data.csv** ← Extracted from ChEMBL
4. **embeddings/validated_proteins_esm_embeddings.npy** ← Generated from sequences

## File Sizes

- `validated_proteins.fasta`: ~72KB (100 protein sequences)
- `chembl_experimental_data.csv`: ~6.8MB (92,133 experimental activities)
- `validated_proteins_esm_embeddings.npy`: ~1MB (100 × 1280 embeddings)

## Usage

These files provide the complete dataset for the RL environment:
- **Protein sequences** for feature extraction
- **Experimental activities** as real labels (replacing random "hits")
- **Protein embeddings** as learned representations 