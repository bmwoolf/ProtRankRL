import requests
import time
import csv
import json
import sys
from pathlib import Path
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Configuration
ACCESSION_FILE = 'protein_inputs/processed/annotated_accessions.csv'
OUTPUT_DIR = Path('protein_inputs/processed')
OUTPUT_JSON = OUTPUT_DIR / 'chembl_experimental_data.json'
OUTPUT_CSV = OUTPUT_DIR / 'chembl_experimental_data.csv'

# ChEMBL API base URL
CHEMBL_BASE = 'https://www.ebi.ac.uk/chembl/api/data'

# Thread-safe print lock
print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    """Thread-safe print function."""
    with print_lock:
        print(*args, **kwargs)

def get_chembl_target_id(uniprot_id):
    """Get ChEMBL target ID for a UniProt accession."""
    url = f'{CHEMBL_BASE}/target.json'
    params = {
        'target_components__accession': uniprot_id,
        'limit': 1
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data['targets']:
            return data['targets'][0]['target_chembl_id']
        return None
    except Exception as e:
        safe_print(f"Error getting ChEMBL target for {uniprot_id}: {e}")
        return None

def get_activities_for_target(target_id, max_activities=2000):
    """Get experimental activities for a ChEMBL target with a limit."""
    url = f'{CHEMBL_BASE}/activity.json'
    params = {
        'target_chembl_id': target_id,
        'pchembl_value__isnull': False,  # Only activities with pChEMBL values
        'limit': 1000  # Get up to 1000 activities per target
    }
    
    activities = []
    
    try:
        while len(activities) < max_activities:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            for activity in data['activities']:
                # Extract relevant activity data
                activity_data = {
                    'target_chembl_id': activity.get('target_chembl_id'),
                    'molecule_chembl_id': activity.get('molecule_chembl_id'),
                    'assay_chembl_id': activity.get('assay_chembl_id'),
                    'pchembl_value': activity.get('pchembl_value'),
                    'standard_value': activity.get('standard_value'),
                    'standard_units': activity.get('standard_units'),
                    'standard_type': activity.get('standard_type'),
                    'activity_comment': activity.get('activity_comment'),
                    'data_validity_comment': activity.get('data_validity_comment')
                }
                activities.append(activity_data)
                
                if len(activities) >= max_activities:
                    break
            
            # Check if there are more pages
            next_url = data['page_meta']['next']
            if next_url and len(activities) < max_activities:
                if next_url.startswith('http'):
                    url = next_url
                    params = None  # Params are already included in the next_url
                else:
                    url = urljoin(CHEMBL_BASE + '/', next_url)
                    params = None
            else:
                break
                
    except Exception as e:
        safe_print(f"Error getting activities for {target_id}: {e}")
    
    return activities

def process_single_protein(uniprot_id):
    """Process a single protein and return its data."""
    safe_print(f"Processing: {uniprot_id}")
    
    # Get ChEMBL target ID
    target_id = get_chembl_target_id(uniprot_id)
    if not target_id:
        safe_print(f"  {uniprot_id}: No ChEMBL target found")
        return uniprot_id, None
    
    safe_print(f"  {uniprot_id}: Found target {target_id}")
    
    # Get activities (limit to 2000 per protein for speed)
    activities = get_activities_for_target(target_id, max_activities=2000)
    safe_print(f"  {uniprot_id}: Found {len(activities)} activities")
    
    # Return data
    protein_data = {
        'chembl_target_id': target_id,
        'activities': activities
    }
    
    return uniprot_id, protein_data

def main():
    # Check command line arguments for number of workers
    max_workers = 5  # Conservative default to avoid overwhelming the API
    if len(sys.argv) > 1:
        try:
            max_workers = int(sys.argv[1])
        except ValueError:
            print("Invalid number of workers. Using default of 5.")
    
    print(f"Extracting ChEMBL experimental data for 100 validated proteins using {max_workers} parallel workers...")
    
    # Read protein accessions
    accessions = []
    with open(ACCESSION_FILE, 'r') as f:
        for line in f:
            accessions.append(line.strip())
    
    all_data = {}
    all_csv_data = []
    
    # Process proteins in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_protein = {executor.submit(process_single_protein, acc): acc for acc in accessions}
        
        # Process completed tasks
        completed = 0
        for future in as_completed(future_to_protein):
            uniprot_id, protein_data = future.result()
            completed += 1
            
            safe_print(f"Completed {completed}/{len(accessions)}: {uniprot_id}")
            
            if protein_data:
                all_data[uniprot_id] = protein_data
                
                # Prepare CSV data
                for activity in protein_data['activities']:
                    csv_row = {
                        'uniprot_id': uniprot_id,
                        'chembl_target_id': protein_data['chembl_target_id'],
                        **activity
                    }
                    all_csv_data.append(csv_row)
    
    # Save results
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    if all_csv_data:
        fieldnames = all_csv_data[0].keys()
        with open(OUTPUT_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_csv_data)
    
    print(f"\nExtraction complete!")
    print(f"JSON data saved to: {OUTPUT_JSON}")
    print(f"CSV data saved to: {OUTPUT_CSV}")
    
    # Summary statistics
    total_activities = sum(len(data['activities']) for data in all_data.values())
    print(f"Total activities extracted: {total_activities}")
    print(f"Proteins with activities: {len([d for d in all_data.values() if d['activities']])}")

if __name__ == "__main__":
    main() 