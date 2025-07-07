# Data Collection Scripts

Scripts for downloading and extracting real experimental data from biological databases.

## Scripts

### `download_protein_sequences.py`
Downloads protein sequences from UniProt for a list of UniProt accessions.
- **Input**: `protein_inputs/processed/annotated_accessions.csv`
- **Output**: `protein_inputs/raw/validated_proteins.fasta` and `validated_proteins.csv`

### `extract_chembl_activities_parallel.py`
Extracts experimental activity data from ChEMBL database using parallel processing.
- **Input**: UniProt accessions from annotated_accessions.csv
- **Output**: `protein_inputs/processed/chembl_experimental_data.json` and `.csv`
- **Features**: Parallel processing, handles large datasets efficiently

### `download_uniprot_bindingdb_data.py`
Downloads binding data from BindingDB for specific proteins.
- **Input**: UniProt accessions
- **Output**: BindingDB response data

## Usage

```bash
# Download protein sequences
python download_protein_sequences.py

# Extract ChEMBL activities with 8 parallel workers
python extract_chembl_activities_parallel.py 8

# Download BindingDB data
python download_uniprot_bindingdb_data.py
``` 