#!/usr/bin/env python3
"""
Setup script for ProtRankRL

This script helps new users set up the ProtRankRL environment and run the data pipeline.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n🔄 {description}")
    print(f"Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False


def setup_environment(install_dev: bool = False) -> bool:
    """Set up the Python environment."""
    print("🚀 Setting up ProtRankRL environment...")
    
    # Check if we're in a virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Not in a virtual environment. Consider creating one:")
        print("   python -m venv .venv")
        print("   source .venv/bin/activate  # On Unix/macOS")
        print("   .venv\\Scripts\\activate     # On Windows")
        print()
    
    # Install the package
    install_cmd = "pip install -e ."
    if install_dev:
        install_cmd += "[dev]"
    
    if not run_command(install_cmd, "Installing ProtRankRL package"):
        return False
    
    # Install additional dependencies for data collection
    additional_deps = [
        "requests",
        "biopython", 
        "chembl-webresource-client",
        "fair-esm",
        "torch"
    ]
    
    for dep in additional_deps:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            return False
    
    return True


def run_data_pipeline(num_proteins: int = 100, num_workers: int = 8) -> bool:
    """Run the data collection pipeline."""
    print(f"\n📊 Running data pipeline for {num_proteins} proteins...")
    
    pipeline_cmd = f"python scripts/run_full_pipeline.py --num-proteins {num_proteins} --num-workers {num_workers}"
    
    if not run_command(pipeline_cmd, "Running data collection pipeline"):
        return False
    
    return True


def run_tests() -> bool:
    """Run the test suite."""
    print("\n🧪 Running tests...")
    
    if not run_command("pytest -q", "Running test suite"):
        return False
    
    return True


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description="Setup ProtRankRL environment and data pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic setup
  python setup.py
  
  # Setup with development dependencies
  python setup.py --dev
  
  # Setup and run pipeline with custom settings
  python setup.py --run-pipeline --num-proteins 50 --num-workers 4
  
  # Full setup including tests
  python setup.py --dev --run-pipeline --run-tests
        """
    )
    
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Install development dependencies"
    )
    
    parser.add_argument(
        "--run-pipeline",
        action="store_true",
        help="Run the data collection pipeline after setup"
    )
    
    parser.add_argument(
        "--num-proteins",
        type=int,
        default=100,
        help="Number of proteins for pipeline (default: 100)"
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of workers for pipeline (default: 8)"
    )
    
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run tests after setup"
    )
    
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip environment setup (assumes already installed)"
    )
    
    args = parser.parse_args()
    
    print("🎯 ProtRankRL Setup")
    print("=" * 50)
    
    success = True
    
    # Step 1: Environment setup
    if not args.skip_setup:
        if not setup_environment(install_dev=args.dev):
            success = False
            print("\n❌ Environment setup failed. Please check the errors above.")
            sys.exit(1)
    
    # Step 2: Run pipeline if requested
    if args.run_pipeline and success:
        if not run_data_pipeline(args.num_proteins, args.num_workers):
            success = False
            print("\n❌ Data pipeline failed. Please check the errors above.")
            sys.exit(1)
    
    # Step 3: Run tests if requested
    if args.run_tests and success:
        if not run_tests():
            success = False
            print("\n❌ Tests failed. Please check the errors above.")
            sys.exit(1)
    
    if success:
        print("\n🎉 Setup completed successfully!")
        print("\n📚 Next steps:")
        print("1. Explore the examples: python examples/quickstart_demo.py")
        print("2. Train a PPO agent: python examples/train_ppo_agent.py")
        print("3. Check the README.md for more information")
        print("\n🚀 Happy coding!")


if __name__ == "__main__":
    main() 