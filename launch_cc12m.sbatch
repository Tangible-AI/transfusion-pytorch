#!/bin/bash
# launch_cc12m.sbatch — DDP CC12M training via torchrun (8 GPUs/node; raise --nodes for multi-node).
#
# Submit with env overrides, e.g.:
#   sbatch --export=ALL,MODEL_SIZE=0.16B,IMAGE_SIZE=256,DATA="$PWD/data/cc12m/full/{00000..00900}.tar",\
#RUN_NAME=p2_0.16B,OUTPUT_DIR=$PWD/runs/cc12m/p2_0.16B launch_cc12m.sbatch
#SBATCH --job-name=tf_cc12m
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1            # one torchrun per node; it spawns 8 procs
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=48:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -eo pipefail
export WORKDIR="$HOME/dev/transfusion-pytorch"
export OUTPUT_DIR="${OUTPUT_DIR:-$WORKDIR/runs/cc12m/${RUN_NAME:-run}}"
mkdir -p "$OUTPUT_DIR/logs"

# offline (compute nodes have no internet); VAE + GPT-2 tokenizer are pre-cached in HF_HOME
export HF_HOME="$WORKDIR/hf_cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline WANDB_DIR="$OUTPUT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate transfusion
cd "$WORKDIR"

GPUS_PER_NODE=8
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
echo "run=${RUN_NAME} size=${MODEL_SIZE} nodes=$SLURM_NNODES master=$MASTER_ADDR data=$DATA"

# srun launches one task/node; torchrun spawns GPUS_PER_NODE ranks per node
srun --output="$OUTPUT_DIR/logs/slurm_%j_%t.out" --error="$OUTPUT_DIR/logs/slurm_%j_%t.err" \
  torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$GPUS_PER_NODE" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:29500" \
    train_cc12m.py
