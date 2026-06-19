#!/bin/bash
#SBATCH --job-name=tf_example_recipe
#SBATCH --output=tf_example_recipe_%j.out   # %j = job ID; lands in the submission directory
#SBATCH --error=tf_example_recipe_%j.err
#SBATCH --nodes=1                        # single machine
#SBATCH --ntasks=1                       # single process (script drives one GPU directly)
#SBATCH --cpus-per-task=8                # CPU headroom for PIL decode / dataloader
#SBATCH --mem=64G                        # HOST RAM (HF arrow cache + prefetch); VRAM is separate
#SBATCH --gres=gpu:1                     # REQUIRED, or you get a GPU node with no GPU
#SBATCH --time=04:00:00                  # quick test — bump if you let it run long
# (no --partition: this cluster defaults to `all`)

set -eo pipefail                          # fail fast; -u omitted (it breaks conda's activate script)

# ---------- paths (must live on a filesystem the compute nodes can see; $HOME usually is) ----------
export WORKDIR="$HOME/dev/transfusion-pytorch"
export OUTPUT_DIR="$WORKDIR/runs/latent_text_example"   # samples + offline wandb land here; isolated from runs/flowers*
export HF_HOME="$WORKDIR/hf_cache/huggingface"          # dataset + VAE cache, pre-populated on the login node
mkdir -p "$OUTPUT_DIR"

# ---------- run fully offline: per your HPC notes, compute nodes may have no internet ----------
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export WANDB_DIR="$OUTPUT_DIR"                          # sync later from the login node with `wandb sync`

# ---------- example recipe (small model, from scratch — NO resume) ----------
export RUN_NAME="repo-example-recipe"
# export TOTAL_STEPS=20000     # uncomment to cap the quick test (default 100k, same as the example)

# ---------- conda (a bare `conda activate` fails in a batch shell — source the profile first) ----------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate transfusion

# ---------- sanity ----------
echo "Node: $(hostname)"
nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| gpu', torch.cuda.get_device_name(0))"

# ---------- train (cd into the repo so ./data/flowers/labels.txt resolves) ----------
cd "$WORKDIR"
python train_latent_with_text_flowers.py
