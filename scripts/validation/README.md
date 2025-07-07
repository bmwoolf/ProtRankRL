# Validation Scripts

Scripts for validating protein annotations and ensuring data quality.

## Scripts

### `query_chembl_by_uniprot.py`
Validates that proteins have corresponding ChEMBL targets and experimental activities.
- **Input**: `protein_inputs/processed/annotated_accessions.csv`
- **Output**: Console output showing validation status for each protein
- **Purpose**: Ensures all proteins are "well-annotated" (have ChEMBL targets + activities)

### `expand_annotated_accessions.py`
Expands protein accession lists and performs validation checks.
- **Input**: Protein accession lists
- **Output**: Expanded and validated accession lists
- **Purpose**: Data preprocessing and validation

## Usage

```bash
# Validate protein annotations
python query_chembl_by_uniprot.py

# Expand and validate accession lists
python expand_annotated_accessions.py
```

## Validation Criteria

A protein is considered "well-annotated" if:
1. Has an exact ChEMBL target match for its UniProt accession
2. Has at least 1 real experimental activity (IC50, Ki, Kd, etc.) in ChEMBL 