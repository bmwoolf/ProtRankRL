import requests
import time
import csv

ACCESSION_FILE = 'protein_inputs/processed/annotated_accessions.csv'
BASE = 'https://www.ebi.ac.uk/chembl/api/data'
HEADERS = {'Accept': 'application/json'}

# Read accessions from file
with open(ACCESSION_FILE) as f:
    UNIPROT_IDS = [line.strip() for line in f if line.strip()]

total = len(UNIPROT_IDS)
well_annotated_count = 0
not_annotated_count = 0

for idx, uniprot_id in enumerate(UNIPROT_IDS, 1):
    print(f'\n[{idx}/{total}] Processing UniProt: {uniprot_id}')
    # 1. /target
    target_url = f'{BASE}/target?target_components.accession={uniprot_id}&limit=1000'
    target_resp = requests.get(target_url, headers=HEADERS)
    target_data = target_resp.json()
    # Filter for exact UniProt match in target_components
    exact_targets = []
    for t in target_data.get('targets', []):
        for comp in t.get('target_components', []):
            if comp.get('accession') == uniprot_id:
                exact_targets.append(t)
                break
    if not exact_targets:
        print(f"  No exact ChEMBL target found for {uniprot_id}")
        not_annotated_count += 1
        continue
    target = exact_targets[0]
    target_chembl_id = target['target_chembl_id']
    target_name = target.get('pref_name')
    # 2. /activity
    act_url = f'{BASE}/activity?target_chembl_id={target_chembl_id}&limit=1'
    act_resp = requests.get(act_url, headers=HEADERS)
    act_data = act_resp.json()
    n_activities = act_data['page_meta']['total_count']
    status = "WELL-ANNOTATED" if n_activities > 0 else "NOT WELL-ANNOTATED"
    print(f"  ChEMBL Target ID: {target_chembl_id}")
    print(f"  Target Name: {target_name}")
    print(f"  Activities: {n_activities}")
    print(f"  Status: {status}")
    if n_activities > 0:
        well_annotated_count += 1
    else:
        not_annotated_count += 1
    print(f"  Progress: {well_annotated_count} well-annotated, {not_annotated_count} not well-annotated so far.")
    time.sleep(0.2)  # Be polite to the API

print(f"\nValidation complete. {well_annotated_count} well-annotated, {not_annotated_count} not well-annotated out of {total}.") 