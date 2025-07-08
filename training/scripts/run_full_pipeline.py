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
import subprocess
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


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


def run_script(script_path: str, description: str, args: list = None) -> bool:
    """Run a Python script and return success status."""
    logger = logging.getLogger(__name__)
    logger.info(f"🔄 {description}")
    
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
    
    logger.debug(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"✅ {description} completed successfully")
        if result.stdout.strip():
            logger.debug(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed:")
        logger.error(f"Error: {e.stderr}")
        return False


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
    
    # Check if we have the required input file
    accession_file = Path("protein_inputs/processed/annotated_accessions.csv")
    if not accession_file.exists():
        logger.error("❌ Missing required file: protein_inputs/processed/annotated_accessions.csv")
        logger.error("Please ensure you have the initial protein accessions file.")
        sys.exit(1)
    
    # Step 1: Validate protein accessions (optional - just check if we have enough)
    logger.info("\n📋 Step 1: Checking protein accessions")
    try:
        import pandas as pd
        df = pd.read_csv(accession_file)
        actual_count = len(df)
        logger.info(f"✅ Found {actual_count} protein accessions")
        if actual_count < num_proteins:
            logger.warning(f"⚠️  Only {actual_count} proteins available, but {num_proteins} requested")
    except Exception as e:
        logger.error(f"❌ Failed to read accessions file: {e}")
        raise
    
    # Step 2: Download protein sequences
    logger.info("\n🧬 Step 2: Downloading protein sequences from UniProt")
    if not run_script("scripts/data_collection/download_protein_sequences.py", "Downloading protein sequences"):
        raise RuntimeError("Sequence download failed")
    
    # Step 3: Extract experimental activities
    logger.info("\n🔬 Step 3: Extracting experimental activities from ChEMBL")
    script_args = [str(num_workers)]
    if force:
        script_args.append("--force")
    if not run_script("scripts/data_collection/extract_chembl_activities_parallel.py", "Extracting experimental activities", script_args):
        raise RuntimeError("Activity extraction failed")
    
    # Step 4: Generate ESM embeddings
    logger.info("\n🧠 Step 4: Generating ESM embeddings")
    if not run_script("scripts/embeddings/generate_esm_embeddings_truncated.py", "Generating ESM embeddings"):
        raise RuntimeError("Embedding generation failed")
    
    # Step 5: Create unified dataset
    logger.info("\n📊 Step 5: Creating unified dataset")
    if not run_script("scripts/data_collection/create_unified_dataset.py", "Creating unified dataset"):
        raise RuntimeError("Dataset creation failed")
    
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