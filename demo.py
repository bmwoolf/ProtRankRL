#!/usr/bin/env python3
"""
ProtRankRL Demo Script

This script demonstrates how to use the ProtRankRL API for protein ranking.
"""

import requests
import json
from typing import List, Dict, Any


def rank_proteins(uniprot_ids: List[str], api_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """
    Rank proteins using the ProtRankRL API.
    
    Args:
        uniprot_ids: List of UniProt IDs to rank
        api_url: API server URL
        
    Returns:
        API response with rankings
    """
    url = f"{api_url}/rank"
    payload = {"uniprot_ids": uniprot_ids}
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    
    return response.json()


def print_rankings(response: Dict[str, Any]) -> None:
    """Pretty print the ranking results."""
    print("\n" + "="*80)
    print("PROTEIN RANKING RESULTS")
    print("="*80)
    
    # Print metadata
    metadata = response.get("metadata", {})
    print(f"Total proteins processed: {metadata.get('total_proteins', 0)}")
    print(f"Valid proteins found: {metadata.get('valid_proteins', 0)}")
    print(f"Processing time: {metadata.get('processing_time', 0):.4f} seconds")
    print(f"Model version: {metadata.get('model_version', 'unknown')}")
    
    # Print database stats
    db_stats = metadata.get("database_stats", {})
    print(f"Database contains {db_stats.get('total_proteins', 0)} proteins")
    print(f"Feature dimension: {db_stats.get('feature_dim', 0)}")
    print(f"Active proteins: {db_stats.get('active_proteins', 0)}")
    
    print("\n" + "-"*80)
    print("RANKINGS")
    print("-"*80)
    
    # Print rankings
    rankings = response.get("rankings", [])
    for rank_info in rankings:
        print(f"Rank {rank_info['rank']}: {rank_info['uniprot_id']}")
        print(f"  Score: {rank_info['score']:.4f}")
        print(f"  Confidence: {rank_info['confidence']:.4f}")
        print(f"  Has Activity: {rank_info['has_activity']}")
        
        # Print experimental data if available
        exp_data = rank_info.get("experimental_data", {})
        if exp_data.get("activity_count", 0) > 0:
            print(f"  Activity Count: {exp_data['activity_count']}")
            print(f"  pChEMBL Mean: {exp_data.get('pchembl_mean', 'N/A')}")
            print(f"  pChEMBL Std: {exp_data.get('pchembl_std', 'N/A')}")
        print()


def main():
    """Main demo function."""
    print("ProtRankRL Protein Ranking Demo")
    print("="*50)
    
    # Example UniProt IDs to rank
    test_proteins = [
        "O43451",  # Example protein 1
        "O60706",  # Example protein 2  
        "O76074",  # Example protein 3
        "O95180",  # Example protein 4
        "P12345"   # Invalid protein (for testing)
    ]
    
    print(f"Ranking {len(test_proteins)} proteins...")
    print(f"Proteins: {', '.join(test_proteins)}")
    
    try:
        # Rank the proteins
        response = rank_proteins(test_proteins)
        
        # Print results
        print_rankings(response)
        
    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to the API server.")
        print("Make sure the server is running with:")
        print("  python -m uvicorn src.api.main:app --reload --port 8000")
        
    except requests.exceptions.HTTPError as e:
        print(f"\nERROR: HTTP error occurred: {e}")
        print(f"Response: {e.response.text}")
        
    except Exception as e:
        print(f"\nERROR: An unexpected error occurred: {e}")


if __name__ == "__main__":
    main() 