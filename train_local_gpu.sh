#!/bin/bash
# train_local_gpu.sh - Setup and run RL training on local Ubuntu + NVIDIA GPU
# Usage: bash train_local_gpu.sh

set -e

REPO_URL="https://github.com/bmwoolf/ProtRankRL"
REPO_DIR="ProtRankRL"
TRAIN_SCRIPT="lambda_gpu/train_remote.py"
MODEL_NAME="ppo_2M"
TIMESTEPS=2000000

# 1. Clone repo if not present
if [ ! -d "$REPO_DIR" ]; then
  echo "Cloning repository..."
  git clone "$REPO_URL"
fi
cd "$REPO_DIR"

# 2. Install system dependencies
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip git

# 3. Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
  echo "[ERROR] NVIDIA GPU not detected. Exiting."
  exit 1
fi
nvidia-smi

# 4. Install PyTorch with CUDA 12.1 support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Install Python dependencies
pip3 install -r requirements.txt

# 6. Run training script
python3 $TRAIN_SCRIPT --timesteps $TIMESTEPS --model_name $MODEL_NAME --device cuda

echo "\n[Done] Training complete. Model saved in ./models/$MODEL_NAME_best_/" 