import requests
import time
import csv
from pathlib import Path

# Configuration
ACCESSION_FILE = 'protein_inputs/processed/annotated_accessions.csv'
OUTPUT_DIR = Path('protein_inputs/raw')
OUTPUT_FASTA = OUTPUT_DIR / 'validated_proteins.fasta'
OUTPUT_CSV = OUTPUT_DIR / 'validated_proteins.csv'

# UniProt API base URL
UNIPROT_BASE = 'https://rest.uniprot.org/uniprotkb'

def download_protein_sequence(uniprot_id):
    """Download protein sequence and metadata from UniProt."""
    url = f'{UNIPROT_BASE}/{uniprot_id}'
    headers = {'Accept': 'application/json'}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        # Extract sequence and metadata
        sequence = data.get('sequence', {}).get('value', '')
        protein_name = data.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', '')
        if not protein_name:
            protein_name = data.get('proteinDescription', {}).get('submissionNames', [{}])[0].get('fullName', {}).get('value', '')
        
        organism = data.get('organism', {}).get('scientificName', '')
        gene_name = data.get('genes', [{}])[0].get('geneName', {}).get('value', '') if data.get('genes') else ''
        
        return {
            'uniprot_id': uniprot_id,
            'sequence': sequence,
            'protein_name': protein_name,
            'organism': organism,
            'gene_name': gene_name,
            'length': len(sequence)
        }
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Error downloading {uniprot_id}: {e}")
        return None

def write_fasta(proteins, output_file):
    """Write proteins to FASTA format."""
    with open(output_file, 'w') as f:
        for protein in proteins:
            if protein and protein['sequence']:
                f.write(f">{protein['uniprot_id']} {protein['protein_name']}\n")
                f.write(f"{protein['sequence']}\n")

def write_csv(proteins, output_file):
    """Write proteins to CSV format with metadata."""
    with open(output_file, 'w', newline='') as f:
        if proteins:
            writer = csv.DictWriter(f, fieldnames=proteins[0].keys())
            writer.writeheader()
            for protein in proteins:
                if protein:
                    writer.writerow(protein)

def main():
    print("Downloading protein sequences for validated proteins...")
    print("=" * 60)
    
    # Read accessions
    with open(ACCESSION_FILE) as f:
        uniprot_ids = [line.strip() for line in f if line.strip()]
    
    total = len(uniprot_ids)
    downloaded_proteins = []
    failed_downloads = []
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for idx, uniprot_id in enumerate(uniprot_ids, 1):
        print(f"[{idx}/{total}] Downloading {uniprot_id}...")
        
        protein_data = download_protein_sequence(uniprot_id)
        
        if protein_data:
            downloaded_proteins.append(protein_data)
            print(f"  ✅ Downloaded: {protein_data['protein_name']} ({protein_data['length']} aa)")
        else:
            failed_downloads.append(uniprot_id)
            print(f"  ❌ Failed to download {uniprot_id}")
        
        time.sleep(0.1)  # Be polite to UniProt API
    
    # Write outputs
    print(f"\nWriting outputs...")
    write_fasta(downloaded_proteins, OUTPUT_FASTA)
    write_csv(downloaded_proteins, OUTPUT_CSV)
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Download Summary:")
    print(f"  ✅ Successfully downloaded: {len(downloaded_proteins)} proteins")
    print(f"  ❌ Failed downloads: {len(failed_downloads)} proteins")
    print(f"  📁 FASTA file: {OUTPUT_FASTA}")
    print(f"  📁 CSV file: {OUTPUT_CSV}")
    
    if failed_downloads:
        print(f"\nFailed downloads: {', '.join(failed_downloads)}")

if __name__ == "__main__":
    main() 