import requests
import time
import csv

ACCESSION_FILE = 'protein_inputs/processed/annotated_accessions.csv'
BASE = 'https://www.ebi.ac.uk/chembl/api/data'
HEADERS = {'Accept': 'application/json'}

# Read existing accessions
with open(ACCESSION_FILE) as f:
    existing_accessions = set(line.strip() for line in f if line.strip())

print(f"Starting with {len(existing_accessions)} accessions.")

well_annotated = set(existing_accessions)

# Query ChEMBL for SINGLE PROTEIN targets
limit = 2000
offset = 0
batch_size = 200

while len(well_annotated) < 100:
    print(f"Querying targets {offset} to {offset+batch_size}...")
    target_url = f'{BASE}/target?target_type=SINGLE%20PROTEIN&limit={batch_size}&offset={offset}'
    resp = requests.get(target_url, headers=HEADERS)
    data = resp.json()
    targets = data.get('targets', [])
    if not targets:
        print("No more targets returned by ChEMBL.")
        break
    for t in targets:
        for comp in t.get('target_components', []):
            accession = comp.get('accession')
            if not accession or accession in well_annotated:
                continue
            # Check for activities
            target_chembl_id = t['target_chembl_id']
            act_url = f'{BASE}/activity?target_chembl_id={target_chembl_id}&limit=1'
            act_resp = requests.get(act_url, headers=HEADERS)
            act_data = act_resp.json()
            n_activities = act_data['page_meta']['total_count']
            if n_activities > 0:
                well_annotated.add(accession)
                print(f"  Added {accession} ({t.get('pref_name')}) with {n_activities} activities. Total: {len(well_annotated)}")
                if len(well_annotated) >= 100:
                    break
            time.sleep(0.2)
        if len(well_annotated) >= 100:
            break
    offset += batch_size
    time.sleep(1)

# Write to file
with open(ACCESSION_FILE, 'w') as f:
    for acc in sorted(well_annotated):
        f.write(acc + '\n')

print(f"Done! Wrote {len(well_annotated)} well-annotated accessions to {ACCESSION_FILE}.") 