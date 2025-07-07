#!/usr/bin/env python3
"""
Automated Pipeline for ProtRankRL Data Collection and Processing

This script runs the complete pipeline to:
1. Validate protein accessions against ChEMBL
2. Download protein sequences from UniProt
3. Extract experimental activities from ChEMBL
4. Generate ESM embeddings
5. Create unified dataset for RL training

Usage:
    python scripts/run_full_pipeline.py [--num-proteins 100] [--num-workers 8] [--force]
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.validation.validate_proteins import validate_protein_accessions
from scripts.data_collection.download_protein_sequences import download_protein_sequences
from scripts.data_collection.extract_chembl_activities_parallel import extract_chembl_activities
from scripts.embeddings.generate_esm_embeddings_truncated import generate_esm_embeddings
from scripts.data_collection.create_unified_dataset import create_unified_dataset


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('pipeline.log')
        ]
    )


def run_pipeline(
    num_proteins: int = 100,
    num_workers: int = 8,
    force: bool = False,
    verbose: bool = False
) -> None:
    """
    Run the complete ProtRankRL data pipeline.
    
    Args:
        num_proteins: Number of proteins to process
        num_workers: Number of parallel workers for ChEMBL extraction
        force: Force re-run even if files exist
        verbose: Enable verbose logging
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting ProtRankRL Data Pipeline")
    logger.info(f"Target proteins: {num_proteins}")
    logger.info(f"Parallel workers: {num_workers}")
    logger.info(f"Force re-run: {force}")
    
    # Step 1: Validate protein accessions
    logger.info("\n📋 Step 1: Validating protein accessions against ChEMBL")
    try:
        validated_count = validate_protein_accessions(
            num_proteins=num_proteins,
            force=force
        )
        logger.info(f"✅ Validated {validated_count} proteins")
    except Exception as e:
        logger.error(f"❌ Protein validation failed: {e}")
        raise
    
    # Step 2: Download protein sequences
    logger.info("\n🧬 Step 2: Downloading protein sequences from UniProt")
    try:
        sequences_count = download_protein_sequences(force=force)
        logger.info(f"✅ Downloaded {sequences_count} protein sequences")
    except Exception as e:
        logger.error(f"❌ Sequence download failed: {e}")
        raise
    
    # Step 3: Extract experimental activities
    logger.info("\n🔬 Step 3: Extracting experimental activities from ChEMBL")
    try:
        activities_count = extract_chembl_activities(
            num_workers=num_workers,
            force=force
        )
        logger.info(f"✅ Extracted {activities_count} experimental activities")
    except Exception as e:
        logger.error(f"❌ Activity extraction failed: {e}")
        raise
    
    # Step 4: Generate ESM embeddings
    logger.info("\n🧠 Step 4: Generating ESM embeddings")
    try:
        embeddings_count = generate_esm_embeddings(force=force)
        logger.info(f"✅ Generated {embeddings_count} ESM embeddings")
    except Exception as e:
        logger.error(f"❌ Embedding generation failed: {e}")
        raise
    
    # Step 5: Create unified dataset
    logger.info("\n📊 Step 5: Creating unified dataset")
    try:
        dataset_info = create_unified_dataset(force=force)
        logger.info(f"✅ Created unified dataset: {dataset_info}")
    except Exception as e:
        logger.error(f"❌ Dataset creation failed: {e}")
        raise
    
    logger.info("\n🎉 Pipeline completed successfully!")
    logger.info("📁 Output files:")
    logger.info("  - protein_inputs/processed/annotated_accessions.csv")
    logger.info("  - protein_inputs/raw/validated_proteins.fasta")
    logger.info("  - protein_inputs/processed/chembl_experimental_data.csv")
    logger.info("  - protein_inputs/embeddings/validated_proteins_esm_embeddings.npy")
    logger.info("  - protein_inputs/processed/unified_protein_dataset.csv")
    logger.info("\n🚀 Ready for RL training!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run the complete ProtRankRL data pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (100 proteins, 8 workers)
  python scripts/run_full_pipeline.py
  
  # Run with custom settings
  python scripts/run_full_pipeline.py --num-proteins 50 --num-workers 4
  
  # Force re-run all steps
  python scripts/run_full_pipeline.py --force
  
  # Verbose output
  python scripts/run_full_pipeline.py --verbose
        """
    )
    
    parser.add_argument(
        "--num-proteins",
        type=int,
        default=100,
        help="Number of proteins to process (default: 100)"
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel workers for ChEMBL extraction (default: 8)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run all steps even if output files exist"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    try:
        run_pipeline(
            num_proteins=args.num_proteins,
            num_workers=args.num_workers,
            force=args.force,
            verbose=args.verbose
        )
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 