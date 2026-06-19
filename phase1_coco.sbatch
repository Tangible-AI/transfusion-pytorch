#!/bin/bash
#SBATCH --job-name=tf_coco
#SBATCH --output=tf_coco_%j.out          # %j = job ID; lands in the submission directory
#SBATCH --error=tf_coco_%j.err
#SBATCH --nodes=1                         # single machine
#SBATCH --ntasks=1                        # single process (script drives one GPU directly)
#SBATCH --cpus-per-task=12                # JPEG decode is the bottleneck at COCO scale
#SBATCH --mem=96G                         # HOST RAM (PIL decode + prefetch); VRAM is separate
#SBATCH --gres=gpu:1                      # REQUIRED, or you get a GPU node with no GPU
#SBATCH --time=48:00:00                   # 120k steps at 256px; lower after a throughput check
# (no --partition: this cluster defaults to `all`)

set -eo pipefail                          # fail fast; -u omitted (it breaks conda's activate script)

# ---------- paths (must live on a filesystem the compute nodes can see) ----------
export WORKDIR="$HOME/dev/transfusion-pytorch"
export OUTPUT_DIR="$WORKDIR/runs/coco"             # checkpoints + samples land here
export HF_HOME="$WORKDIR/hf_cache/huggingface"     # VAE cache, pre-populated on the login node
export COCO_ROOT="$WORKDIR/scratch"                # where coco_download.sh put the data
mkdir -p "$OUTPUT_DIR"

# ---------- run fully offline: compute nodes may have no internet ----------
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1                       # GPT-2 tokenizer loads from HF_HOME cache
export TOKENIZERS_PARALLELISM=false                # silence fast-tokenizer fork warning under dataloader workers
export WANDB_MODE=offline
export WANDB_DIR="$OUTPUT_DIR"                      # sync later from the login node with `wandb sync`

# ---------- conda (a bare `conda activate` fails in a batch shell — source the profile first) ----------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate transfusion

# ---------- sanity ----------
echo "Node: $(hostname)"
nvidia-smi
echo "COCO_ROOT=$COCO_ROOT"
ls "$COCO_ROOT/annotations" | head
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| gpu', torch.cuda.get_device_name(0))"

# ---------- train ----------
cd "$WORKDIR"
python phase1_coco.py
