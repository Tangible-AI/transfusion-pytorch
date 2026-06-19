#!/bin/bash
#SBATCH --job-name=tf_flowers_lr1e-4
#SBATCH --output=tf_flowers_lr1e-4_%j.out   # %j = job ID; lands in the submission directory
#SBATCH --error=tf_flowers_lr1e-4_%j.err
#SBATCH --nodes=1                        # single machine
#SBATCH --ntasks=1                       # single process (script drives one GPU directly)
#SBATCH --cpus-per-task=8                # CPU headroom for PIL decode / dataloader
#SBATCH --mem=64G                        # HOST RAM (HF arrow cache + prefetch); VRAM is separate
#SBATCH --gres=gpu:1                     # REQUIRED, or you get a GPU node with no GPU
#SBATCH --time=24:00:00                  # generous; lower after an interactive throughput check
# (no --partition: this cluster defaults to `all`)

set -eo pipefail                          # fail fast; -u omitted (it breaks conda's activate script)

# ---------- paths (must live on a filesystem the compute nodes can see; $HOME usually is) ----------
export WORKDIR="$HOME/dev/transfusion-pytorch"                 # EDIT to scratch if available, e.g. /scratch/$USER/transfusion
export OUTPUT_DIR="$WORKDIR/runs/flowers_lr1e-4"    # ABLATION: separate dir so it can't overwrite the live runs/flowers
export HF_HOME="$WORKDIR/hf_cache/huggingface"                  # dataset + VAE cache, pre-populated on the login node
mkdir -p "$OUTPUT_DIR"

# ---------- run fully offline: per your HPC notes, compute nodes may have no internet ----------
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export WANDB_DIR="$OUTPUT_DIR"                      # sync later from the login node with `wandb sync`

# ---------- ablation: lr=1e-4, resume from the live run's 60k checkpoint ----------
export LR=1e-4
export RUN_NAME="lr1e-4-resume60k"                            # wandb run label
export RESUME="$WORKDIR/runs/flowers/ckpt_60000.pt"            # read the existing 60k ckpt; outputs go to OUTPUT_DIR above

# ---------- conda (a bare `conda activate` fails in a batch shell — source the profile first) ----------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate transfusion

# ---------- sanity ----------
echo "Node: $(hostname)"
nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| gpu', torch.cuda.get_device_name(0))"

# ---------- train (cd into the repo so ./data/flowers/labels.txt resolves) ----------
cd "$WORKDIR"
python phase1_flowers.py
