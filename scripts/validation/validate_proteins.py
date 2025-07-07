#!/usr/bin/env python3
"""
Protein validation script for ProtRankRL

Validates protein accessions against ChEMBL to ensure they are well-annotated.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import requests
import time


def validate_protein_accessions(
    accession_file: str = "protein_inputs/processed/annotated_accessions.csv",
    output_file: str = None,
    verbose: bool = False
) -> Tuple[int, int]:
    """
    Validate protein accessions against ChEMBL.
    
    Args:
        accession_file: Path to CSV file with protein accessions
        output_file: Path to save validation results (optional)
        verbose: Enable verbose logging
        
    Returns:
        Tuple of (well_annotated_count, total_count)
    """
    logger = logging.getLogger(__name__)
    
    # Read accessions
    if not Path(accession_file).exists():
        raise FileNotFoundError(f"Accession file not found: {accession_file}")
    
    df = pd.read_csv(accession_file)
    uniprot_ids = df.iloc[:, 0].tolist()  # Assume first column contains accessions
    
    logger.info(f"Validating {len(uniprot_ids)} protein accessions against ChEMBL...")
    
    BASE = 'https://www.ebi.ac.uk/chembl/api/data'
    HEADERS = {'Accept': 'application/json'}
    
    results = []
    well_annotated_count = 0
    not_annotated_count = 0
    
    for idx, uniprot_id in enumerate(uniprot_ids, 1):
        if verbose:
            logger.info(f'[{idx}/{len(uniprot_ids)}] Processing UniProt: {uniprot_id}')
        
        # 1. Query target
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
            if verbose:
                logger.warning(f"  No exact ChEMBL target found for {uniprot_id}")
            not_annotated_count += 1
            results.append({
                'uniprot_id': uniprot_id,
                'chembl_id': None,
                'target_name': None,
                'activity_count': 0,
                'status': 'NOT_WELL_ANNOTATED'
            })
            continue
        
        target = exact_targets[0]
        target_chembl_id = target['target_chembl_id']
        target_name = target.get('pref_name')
        
        # 2. Query activities
        act_url = f'{BASE}/activity?target_chembl_id={target_chembl_id}&limit=1'
        act_resp = requests.get(act_url, headers=HEADERS)
        act_data = act_resp.json()
        n_activities = act_data['page_meta']['total_count']
        
        status = "WELL_ANNOTATED" if n_activities > 0 else "NOT_WELL_ANNOTATED"
        
        if verbose:
            logger.info(f"  ChEMBL Target ID: {target_chembl_id}")
            logger.info(f"  Target Name: {target_name}")
            logger.info(f"  Activities: {n_activities}")
            logger.info(f"  Status: {status}")
        
        if n_activities > 0:
            well_annotated_count += 1
        else:
            not_annotated_count += 1
        
        results.append({
            'uniprot_id': uniprot_id,
            'chembl_id': target_chembl_id,
            'target_name': target_name,
            'activity_count': n_activities,
            'status': status
        })
        
        if verbose:
            logger.info(f"  Progress: {well_annotated_count} well-annotated, {not_annotated_count} not well-annotated")
        
        time.sleep(0.2)  # Be polite to the API
    
    # Save results if output file specified
    if output_file:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        logger.info(f"Validation results saved to: {output_file}")
    
    logger.info(f"Validation complete. {well_annotated_count} well-annotated, {not_annotated_count} not well-annotated out of {len(uniprot_ids)}.")
    
    return well_annotated_count, len(uniprot_ids)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate protein accessions against ChEMBL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate default accession file
  python scripts/validation/validate_proteins.py
  
  # Validate with custom output
  python scripts/validation/validate_proteins.py --output validation_results.csv
  
  # Verbose output
  python scripts/validation/validate_proteins.py --verbose
        """
    )
    
    parser.add_argument(
        "--input",
        default="protein_inputs/processed/annotated_accessions.csv",
        help="Input CSV file with protein accessions (default: protein_inputs/processed/annotated_accessions.csv)"
    )
    
    parser.add_argument(
        "--output",
        help="Output CSV file for validation results (optional)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        well_annotated, total = validate_protein_accessions(
            accession_file=args.input,
            output_file=args.output,
            verbose=args.verbose
        )
        
        if well_annotated == total:
            print(f"\n✅ All {total} proteins are well-annotated!")
        else:
            print(f"\n⚠️  {well_annotated}/{total} proteins are well-annotated")
            
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 