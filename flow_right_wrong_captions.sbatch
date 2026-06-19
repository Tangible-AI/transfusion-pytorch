#!/bin/bash
#SBATCH --job-name=tf_diag_cond
#SBATCH --output=tf_diag_cond_%j.out
#SBATCH --error=tf_diag_cond_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1                     # eval still needs a GPU
#SBATCH --time=00:30:00                  # short: 50 forward-pass batches, no training

set -eo pipefail

# ---------- paths ----------
export WORKDIR="$HOME/dev/transfusion-pytorch"
export OUTPUT_DIR="$WORKDIR/runs/flowers_lr1e-4"              # which run's checkpoints to read
export CKPT="$OUTPUT_DIR/ckpt_65000.pt"                       # pin the exact 65k checkpoint (not the latest)
export HF_HOME="$WORKDIR/hf_cache/huggingface"

# ---------- offline (compute nodes may have no internet) ----------
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ---------- conda ----------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate transfusion

# ---------- sanity ----------
echo "Node: $(hostname)"
nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

# ---------- run (cd so ./data/flowers/labels.txt resolves) ----------
cd "$WORKDIR"
python flow_right_wrong_captions.py
