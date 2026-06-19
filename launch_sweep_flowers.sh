#!/bin/bash
# launch_sweep_flowers.sh — submit the LR x prob_uncond ablation grid (small architecture).
# Each run gets an isolated, organized artifact dir:
#   runs/<sweep>/<run_name>/{ckpt_*.pt, samples/, conditional/, wandb/, logs/}
# Usage:
#   ./launch_sweep_flowers.sh            # submit the full 9-run grid
#   SMOKE=1 ./launch_sweep_flowers.sh    # tiny 1-run smoke (120 steps) to validate the pipeline
set -eo pipefail

export WORKDIR="$HOME/dev/transfusion-pytorch"
cd "$WORKDIR"

# ---------- grid ----------
LRS=(8e-4 3e-4 1e-4)
PUS=(0.0 0.1 0.2)
TOTAL_STEPS=100000
SAMPLE_EVERY=5000
CKPT_EVERY=5000
WANDB_PROJECT="transfusion-flowers-small-cfg"
CFG_SCALE=3.0

if [[ "${SMOKE:-0}" == "1" ]]; then          # quick end-to-end validation
  LRS=(8e-4); PUS=(0.1)
  TOTAL_STEPS=120; SAMPLE_EVERY=60; CKPT_EVERY=60
  WANDB_PROJECT="transfusion-flowers-small-cfg-smoke"
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_NAME="sweep_small_${STAMP}"; [[ "${SMOKE:-0}" == "1" ]] && SWEEP_NAME="sweep_small_smoke_${STAMP}"
SWEEP="$WORKDIR/runs/$SWEEP_NAME"
mkdir -p "$SWEEP"
META="$SWEEP/sweep_metadata.tsv"
printf "run_name\tLR\tprob_uncond\tjob_id\toutput_dir\n" > "$META"

echo "sweep: $SWEEP"
for LR in "${LRS[@]}"; do
  for PU in "${PUS[@]}"; do
    RUN_NAME="lr${LR}_pu${PU}"
    OUTPUT_DIR="$SWEEP/$RUN_NAME"
    mkdir -p "$OUTPUT_DIR"/{samples,conditional,logs}

    SBATCH_FILE="$OUTPUT_DIR/run.sbatch"
    cat > "$SBATCH_FILE" <<EOF
#!/bin/bash
#SBATCH --job-name=sw_${RUN_NAME}
#SBATCH --output=$OUTPUT_DIR/logs/slurm_%j.out
#SBATCH --error=$OUTPUT_DIR/logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

set -eo pipefail

export WORKDIR="$WORKDIR"
export OUTPUT_DIR="$OUTPUT_DIR"
export HF_HOME="$WORKDIR/hf_cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export WANDB_DIR="$OUTPUT_DIR"

export LR="$LR"
export PROB_UNCOND="$PU"
export RUN_NAME="$RUN_NAME"
export WANDB_PROJECT="$WANDB_PROJECT"
export CFG_SCALE="$CFG_SCALE"
export TOTAL_STEPS="$TOTAL_STEPS"
export SAMPLE_EVERY="$SAMPLE_EVERY"
export CKPT_EVERY="$CKPT_EVERY"

source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate transfusion

echo "Node: \$(hostname) | run $RUN_NAME | LR=$LR prob_uncond=$PU"
nvidia-smi
cd "$WORKDIR"
python sweep_flowers.py
EOF

    JOB_ID="$(sbatch --parsable "$SBATCH_FILE")"
    printf "%s\t%s\t%s\t%s\t%s\n" "$RUN_NAME" "$LR" "$PU" "$JOB_ID" "$OUTPUT_DIR" >> "$META"
    echo "  submitted $RUN_NAME -> job $JOB_ID"
  done
done

echo
echo "metadata: $META"
column -t "$META"
echo
echo "watch:   squeue -u $USER"
echo "wandb:   wandb sync $SWEEP/*/wandb   # from the login node"
