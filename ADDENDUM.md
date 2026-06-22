# Addendum — Text→Image Experiments (Flowers & CC12M)

This addendum documents our experimental work on top of the original `transfusion-pytorch`
README: changes we had to make to the upstream library, how to run training/eval on this
cluster, and the history of what's been tried. The original README still applies for the
core library; this covers *our* scaffolding (the `*_flowers.py`, `*_cc12m*`, `eval_*`,
`sample_*`, and `phase*` scripts) and the one upstream fix.

---

## 1. Changes to the author's original implementation

### 1.1 CFG `cfg_scale > 1` crash fix (only edit to a tracked upstream file)
**File:** `transfusion_pytorch/transfusion.py` (in `Transfusion.sample`).

Classifier-free guidance with `cfg_scale != 1` crashed with a rotary/KV size mismatch
(`apply_rotary_emb ... size of tensor a (N) must match b (M)`). Root cause: the
**unconditional** CFG forward always built and reused an `uncond_cache`, forcing
decode-with-cache mode, while the **conditional** forward used `cache=None` when KV-caching
was off (the default). The mismatched paths produced different q/k lengths under one rotary
table.

**Fix:** make the uncond path mirror the conditional path —
- only precompute/use `uncond_cache` when `cache_kv=True`; otherwise run a full uncond
  forward (no cache);
- `parse_uncond(..., need_splice=not exists(uncond_cache))` instead of hardcoded `True`.

`cfg_scale=1.0` always worked; this makes `cfg_scale>1` (real guidance) usable. Verified by
sampling at `cfg_scale=3.0` on a trained checkpoint.

### 1.2 Operational gotcha (no code change, but bites you)
There are **two** copies of `transfusion_pytorch`: the pip-installed one in the conda env
and the local repo. **Only the local copy is used when you run from the repo root**
(`cd $REPO && python foo.py`). A script under a subdir (e.g. `scratch/`) silently imports the
stale installed copy. Always run training/eval from the repo root (the sbatch files do
`cd "$WORKDIR"`), or set `PYTHONPATH=$REPO`.

---

## 2. Best practices for creating a run

### 2.1 Where things live (shared `/home` NFS, visible to all compute nodes)
```
data/
  flowers/labels.txt              # tracked (small)
  cc12m/full/cc12m-train-*.tar    # CC12M shards (~1.1TB) — gitignored
hf_cache/huggingface/             # VAE + tokenizer cache (set HF_HOME here) — gitignored
runs/<project>/<run_name>/        # ALL run artifacts — gitignored
  ckpt_<step>.pt                  # {step, model, ema, opt, sched}
  config.json                     # {model_size, image_size, tokenizer, patchifier, bf16}
  samples/<step>.png              # unconditional monitoring grids
  conditional/eval_cfg*.png       # text→image grids (from eval_conditional_sweep.py)
  wandb/offline-run-*/            # offline wandb (WANDB_DIR)
  logs/slurm_*.{out,err}          # per-rank SLURM logs
```
`.gitignore` already covers `runs/`, `data/cc12m/`, `hf_cache/`, `scratch/`, `*.pt`,
`*.out`, `*.err`, `wandb/`. Code stays at repo root; **artifacts never get committed**.

### 2.2 Getting data (login node — it has internet; compute nodes do not)
- **CC12M:** `NSHARDS=60 ./download_cc12m.sh` (subset, ~300k imgs) or `./download_cc12m.sh`
  (full ~11M / ~1.1TB). Pulls the pre-built `pixparse/cc12m-wds` WebDataset — **no crawling,
  no link rot** (Google's `cc12m.tsv` is 403'd now; do not use img2dataset-from-URLs).
  Shards are original-resolution JPEGs; the loader resizes to 256 on the fly.
- **Flowers:** HF `nelorth/oxford-flowers`, auto-cached to `HF_HOME` on first (online) run.
- **Pre-cache models on the login node** so compute nodes can run offline: the VAE
  (`stabilityai/sd-vae-ft-mse`) and the GPT-2 tokenizer land in `hf_cache/` automatically the
  first time a script runs with internet.

### 2.3 Offline env (required on compute nodes)
Every training/eval invocation should export:
```bash
export HF_HOME="$REPO/hf_cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline WANDB_DIR="$OUTPUT_DIR"
```
The sbatch launchers set these for you.

### 2.4 Starting runs
- **CC12M (DDP, 8×H100):** submit `launch_cc12m.sbatch` with env overrides:
  ```bash
  DATA='…/data/cc12m/full/cc12m-train-*.tar'   # single-quote: trainer globs present shards
  sbatch --export=ALL,MODEL_SIZE=0.16B,IMAGE_SIZE=256,PATCHIFIER=unet,BF16=1,\
  DATA="$DATA",RUN_NAME=p3_0.16B_unet,OUTPUT_DIR=$PWD/runs/cc12m/p3_0.16B_unet,\
  LR=3e-4,WARMUP=4000,MIN_LR=1.5e-5,TOTAL_STEPS=250000,BATCH_SIZE=16,GRAD_ACCUM=1,\
  SAMPLE_EVERY=5000,CKPT_EVERY=5000,WANDB_PROJECT=transfusion-cc12m launch_cc12m.sbatch
  ```
  Multi-node: raise `#SBATCH --nodes` (rendezvous via `$SLURM_NNODES`/`MASTER_ADDR` already
  wired). `train_cc12m.py` env knobs: `MODEL_SIZE` (`0.16B`/`0.37B`/`0.76B`), `IMAGE_SIZE`,
  `PATCHIFIER` (`conv`/`unet`), `BF16`, `LR`/`WARMUP`/`MIN_LR`/`TOTAL_STEPS`,
  `BATCH_SIZE`/`GRAD_ACCUM` (per-GPU), `PROB_UNCOND`, `RESUME`, `DATA`, `RUN_NAME`,
  `OUTPUT_DIR`, `WANDB_PROJECT`.
- **Flowers ablation grid:** `./launch_sweep_flowers.sh` (LR × `prob_uncond`); `SMOKE=1` for a
  120-step pipeline check.
- **Quick interactive smoke (1–2 GPUs):** `srun --gres=gpu:N … python train_cc12m.py` (set
  `TOTAL_STEPS≈40`). For DDP, `srun … torchrun --standalone --nproc_per_node=N train_cc12m.py`.
- **Resuming:** pass `RESUME=…/ckpt_<step>.pt`; model/opt/ema/scheduler all restore.

### 2.5 Logs → wandb
Training logs **offline** to `runs/<run>/wandb/`. Sync from the login node:
```bash
wandb sync runs/cc12m/p3_0.16B_unet/wandb/offline-run-*   # point at the run dir, NOT the wandb/ parent
```
Same `WANDB_PROJECT` + distinct `RUN_NAME` → all runs land on one dashboard. Logged metrics:
`loss/{total,text,flow,recon}`, `grad_norm`, `lr`, and (every `SAMPLE_EVERY`)
`cond/{correct,wrong,gap,gap_pct}` + a `samples_uncond` image. DDP logs from rank 0 only.

### 2.6 Evaluation (text→image grids)
`eval_conditional_sweep.py` **auto-reads `<run>/config.json`**, so it rebuilds the exact
architecture — no env vars needed:
```bash
python eval_conditional_sweep.py runs/cc12m/p3_0.16B_unet
# override prompts / guidance:
PROMPTS="a red sports car;a cat on a sofa" CFG_SCALES=1.0,3.0 \
  python eval_conditional_sweep.py runs/cc12m/p3_0.16B_unet
```

### 2.7 Advice specific to the two modality cases we tested
| | **Flowers** | **CC12M** |
|---|---|---|
| Tokenizer | byte-level, `num_text_tokens=256` | GPT-2 BPE, `num_text_tokens=50257` |
| Resolution | 128px → 8×8 = 64 image tokens | 256px → 16×16 = 256 tokens (paper setting) |
| Captions | class names (low entropy) | free-form web alt-text (noisy) |
| Watch out | conditioning collapses easily; needs `prob_uncond`>0 + enough steps | caption noise weakens attribute binding |

Cross-cutting advice:
- **`generate_modality_only()` is UNCONDITIONAL** (ignores the prompt). The real text→image
  path is `sample(prompt=…)`. Monitor conditioning with the **`cond/gap` metric**, not the
  unconditional samples.
- **Do conditional `sample()` OUT of the training loop.** On under-trained weights it can hit a
  CUDA device-side assert that poisons the process (a `try/except` won't save a 250k-step run).
  We generate conditional grids post-hoc from checkpoints via `eval_conditional_sweep.py`.
- **Recipe** (paper-derived, in `train_cc12m.py`): `AdamW(betas=(0.9,0.95), wd=0.1)`,
  `LR=3e-4` warmup 4000 → cosine to 1.5e-5, grad-clip 1.0, EMA 0.999, bf16 autocast. NOTE: the
  paper's `λ=5` loss weight is **DDPM-specific and does not transfer** — this repo uses flow
  matching, so we keep `flow_loss_weight=1`, `reconstruction_loss_weight=0.1`.
- **Patchifier matters a lot at small scale**: the conv ("linear") patchify caps quality
  (paper Table 7: 0.16B ≈37 FID linear vs ≈19 U-Net). Use `PATCHIFIER=unet` (`unet_patchifier.py`).

---

## 3. History & current investigation

| Phase | What | Outcome |
|---|---|---|
| Flowers ablation | small arch (128/8/8), LR × `prob_uncond` grid (`sweep_flowers.py`) | conditioning *can* emerge; `cond/gap` is the reliable signal |
| Flowers (big 768/16/12) | `phase1_flowers.py` | **ignored text** (`cond/gap` ≈ 0%) — motivated the conditioning diagnostic |
| CC12M Phase 0 | 300k subset, 128px, GPT-2 BPE | pipeline validated; `cond/gap_pct` rose to **+5.7%** by 30k → CC12M conditioning works |
| CC12M Phase 1 | full download (~11M imgs, ~1.1TB, `pixparse/cc12m-wds`) | done |
| CC12M Phase 2 | `p2_0.16B`: 768/16/12, 256px, conv patchifier, 250k steps | conditioning works (gap widening) BUT quality capped — textures, weak attribute binding ("red" absent); `loss/flow` converged → **capacity/patchifier-bound, not data-bound** |
| CC12M Phase 3 (**current**) | `p3_0.16B_unet`: same as p2 + **U-Net patchifier + bf16**, A/B-matched | **under investigation** — expect lower `loss/flow` + sharper objects |

**Established along the way:** data *quantity* is not the bottleneck (11M unique pairs ≫ what
0.16B can exhaust; flow loss converged). The promising levers are **architecture** (U-Net
patchifier → bigger model 0.37B/0.76B) and, later, **caption quality** (CC12M alt-text is
noisy; recaption/filter), *not* more images.

**Next, gated on the Phase-3 A/B:** if the U-Net win is confirmed, scale params 0.16B → 0.37B
→ 0.76B (multi-node DDP + bf16, smaller per-GPU batch). Revisit caption quality only if
attribute binding is still weak after scaling.
