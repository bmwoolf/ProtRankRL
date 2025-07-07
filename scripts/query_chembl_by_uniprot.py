import requests
import time
import csv

ACCESSION_FILE = 'protein_inputs/processed/annotated_accessions.csv'
BASE = 'https://www.ebi.ac.uk/chembl/api/data'
HEADERS = {'Accept': 'application/json'}

# Read accessions from file
with open(ACCESSION_FILE) as f:
    UNIPROT_IDS = [line.strip() for line in f if line.strip()]

for uniprot_id in UNIPROT_IDS:
    print(f'\n=== UniProt: {uniprot_id} ===')
    # 1. /target
    target_url = f'{BASE}/target?target_components.accession={uniprot_id}&limit=1000'
    target_resp = requests.get(target_url, headers=HEADERS)
    target_data = target_resp.json()
    # Print all returned targets and accessions
    print(f"/target: found {len(target_data.get('targets', []))} total targets")
    for t in target_data.get('targets', []):
        t_id = t.get('target_chembl_id')
        t_name = t.get('pref_name')
        accessions = [comp.get('accession') for comp in t.get('target_components', [])]
        print(f"  Target: {t_id} | Name: {t_name} | Accessions: {accessions}")
    # Filter for exact UniProt match in target_components
    exact_targets = []
    for t in target_data.get('targets', []):
        for comp in t.get('target_components', []):
            if comp.get('accession') == uniprot_id:
                exact_targets.append(t)
                break
    print(f"/target: found {len(exact_targets)} exact matches")
    if not exact_targets:
        print(f"  No exact ChEMBL target found for {uniprot_id}")
        continue
    target = exact_targets[0]
    target_chembl_id = target['target_chembl_id']
    print(f"  ChEMBL Target ID: {target_chembl_id}")
    print(f"  Target Name: {target.get('pref_name')}")
    # 2. /activity
    act_url = f'{BASE}/activity?target_chembl_id={target_chembl_id}&limit=1'
    act_resp = requests.get(act_url, headers=HEADERS)
    act_data = act_resp.json()
    n_activities = act_data['page_meta']['total_count']
    print(f"/activity: found {n_activities} activities")
    if n_activities > 0:
        print(f"  {uniprot_id} is WELL-ANNOTATED (has activities)")
    else:
        print(f"  {uniprot_id} is NOT well-annotated (no activities)")
    time.sleep(0.5)  # Be polite to the API 