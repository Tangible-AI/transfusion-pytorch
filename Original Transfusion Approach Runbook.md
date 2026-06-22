# Transfusion (lucidrains/transfusion-pytorch) — Reproduction & Validation Runbook

**Goal.** Validate that `lucidrains/transfusion-pytorch` works as intended, then reproduce the paper's central result — _one transformer that does next-token prediction for text and a continuous (flow/diffusion) objective for images, jointly, end-to-end_ — on real image–text data, at a scale that fits our HPC.

**Scope decisions**

- **We do not override the repo's deviations** from the paper (flow matching instead of DDPM; optional velocity-consistency, reconstruction loss, value-residual / hyper-connections / LASER attention). These are deliberate improvements by the author and we treat them as the system under test.
- **Encoder choice:** Phases 0–2 use a **pretrained Stable Diffusion VAE** as the image encoder/decoder. This isolates the Transfusion architecture from VAE-training confounds and is the fastest reliable path. 
  We may need to train our own VAE when it comes time to training on new robot modalities. This is described in **Appendix A**. → 
- **Plan = three phases of increasing realism:** (0) synthetic smoke test, (1) Oxford Flowers integration run on one node, (2) CC12M paper-style reproduction with FID/CLIP. Phase 1 answers "is it wired correctly?"; Phase 2 answers "does it reproduce?".

Everything below was checked against a fresh clone of `main` and a working install of the published package **`transfusion-pytorch==0.17.0`** 

---

## 1. What the repo implements vs. the paper

This grounds what "working as intended" means and flags every divergence you'll see in the loss/logs.

| Component           | Paper (Zhou et al. 2024, arXiv:2408.11039)                                                            | This repo                                                                                                                                                                                                | Notes / source                                                                                              |
| ------------------- | ----------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Backbone            | Single transformer, Llama-style (SwiGLU, RoPE), shared params across modalities (§3)                  | Same: one `Transformer`, RoPE, GEGLU/SwiGLU-style FF, RMSNorm                                                                                                                                            | `transfusion_pytorch/transfusion.py` `class Transformer`                                                    |
| Text objective      | Next-token cross-entropy "LM loss" (§2.1, Eq. 1)                                                      | Same: `F.cross_entropy` on text positions                                                                                                                                                                | `forward()` text-loss block                                                                                 |
| Image objective     | **DDPM diffusion** (ε-prediction), cosine schedule, λ=5 (§2.2, Eq. 3–4)                               | **Rectified flow matching** (predicts data−noise; `x_t = t·x + (1−t)·noise`)                                                                                                                             | README states the swap, "given the success of Flux." Loss block computes `modality_flow = modality - noise` |
| Attention mask      | Causal over sequence **OR** bidirectional within each image (§3, Fig. 4)                              | Identical logic: `transfusion_attn_mask` = `causal                                                                                                                                                       | modality(offset,length)`                                                                                    |
| Image tokenization  | Pretrained VAE → latent patches; linear **or** U-Net down/up to compress k×k patches (§3)             | VAE (yours) + optional learned conv down/up via `pre_post_transformer_enc_dec` (the "U-Net" analogue)                                                                                                    | Paper finds U-Net compression lets you shrink to 16 patches at small quality cost (§4.3.3)                  |
| VAE                 | Trained from scratch, 86M params, **8-dim** latent, f8, L1+LPIPS+0.5·GAN+0.2·MoCo-ID+1e-6·KL (App. A) | Not provided; you supply an encoder/decoder `Module`. Examples load a diffusers `AutoencoderKL`                                                                                                          | We use SD's **4-dim** f8 VAE (a deviation; see Appendix A to match the paper's 8-dim)                       |
| Modality markers    | BOI/EOI tokens (§3)                                                                                   | `[som]`/`[eom]` + a `[meta]` shape token (records latent shape so the decoder can reshape)                                                                                                               | The meta token is a repo addition (from Hymba); it was the source of a real bug — see §2                    |
| Inference           | Switch LM↔diffusion at BOI; 250 steps (trained on 1000); CFG 3–5 (§4.1)                               | `sample()` switches LM↔flow at `[som]`; `modality_steps` ODE steps (default 16); `cfg_scale` (default 3)                                                                                                 | torchdiffeq `odeint`, midpoint                                                                              |
| λ / loss balance    | Fixed λ=5 on diffusion term (Eq. 4)                                                                   | `flow_loss_weight=1`, `text_loss_weight=1`, **plus token-fraction weighting**                                                                                                                            | Different balancing scheme; raise `flow_loss_weight` if image quality lags                                  |
| Extras beyond paper | —                                                                                                     | velocity-consistency loss (Yang et al. 2024), reconstruction loss + `model_output_clean` (Li & He 2025, arXiv:2511.13720), value-residual, hyper-connections, LASER attn, min-p sampling, Muon optimizer | All citeable from the README bibliography; all optional knobs                                               |

**Take-away:** the architecture, attention, and joint-loss structure faithfully match the paper. The objective is rectified flow rather than DDPM (intended), and the VAE is yours to provide.

---

## 2. Known gotchas (from the issue tracker + my own smoke test) — read before training

1. **Multimodality "collapses" at larger model sizes — FIXED upstream.** Issue **#33** ("model loses its multimodality when you increase the size") reported that at `dim=768, depth=12` the model would only ever sample text, never images. Root cause was **not** the attention mask (the community's `modality_attention_factor` patch is a red herring); lucidrains traced it to _"a bug I introduced when adding the meta tokens from the Hymba paper"_ and fixed it. **Action:** use a recent release (I used 0.17.0) and, as we scale, **log the fraction of samples that actually contain an image modality** (see §5.5) so you'd catch a regression immediately.
2. **FlexAttention NaN in backward (PyTorch bug #153799).** `flex_attention` NaNs in the backward pass when sequence length isn't a multiple of 128 _and_ both a `block_mask` and a `score_mod` are used. This repo uses exactly that combination _when flex is on_. **Mitigation is already the default:** `use_flex_attn=False`, which uses a materialized boolean mask + softclamp and avoids the bug. **Action:** keep the naive mask for Phase 1/short sequences. If you later turn on `use_flex_attn=True` for memory at scale, **pad sequence lengths to multiples of 128** and/or set `softcap_value=0` to drop the `score_mod`.
3. **Undertrained VAE = garbage results.** The #33 logs show the reporter training the autoencoder for only **500 steps** (loss ~0.35). That latent space is unusable and will look like "Transfusion doesn't work." **Action:** use a pretrained VAE (this runbook), or if training your own, train to convergence (paper: **1M steps**, App. A).
4. **Construction quirk (v0.17.0, found in my smoke test):** building `Transfusion(...)` **without** `modality_default_shape` raises `TypeError: object of type 'NoneType' has no len()` (the text-only README snippet trips this). **Action:** always pass `modality_default_shape` and `modality_num_dim`, even for text-only pretraining.
5. **Sampling defaults are aggressive.** `text_temperature=1.5` is high and will hurt faithful captions/labels; the sampler also early-stops at `max_length` (logs show `sampling stopped at length: 257/256`). **Action:** lower temperature for structured text and set `max_length` comfortably above `caption_tokens + image_tokens + markers`.
6. **Pin versions.** The package changes often (frequent PyPI releases). Pin `transfusion-pytorch`, `torch`, and `diffusers` to the exact versions you smoke-test.

---

## 3. Environment setup (HPC)


```bash
# On the login/build node
conda create -n transfusion python=3.11 -y
conda activate transfusion

# Match torch to your cluster CUDA. Example for CUDA 12.6 wheels:
pip install torch --index-url https://download.pytorch.org/whl/cu126

# The model + the example/eval deps
pip install transfusion-pytorch==0.17.0    # latest version at https://pypi.org/project/transfusion-pytorch/#history      
pip install diffusers transformers accelerate datasets safetensors ftfy scipy
pip install wandb torchvision einops
pip install adam-atan2-pytorch                    # only if you use the Muon optimizer path
# Eval (Phase 2)
pip install clean-fid open_clip_torch torchmetrics[image] pycocotools

# Logins
wandb login
huggingface-cli login                            # for datasets + VAE download
```

**Sanity check the install:**

```bash
python -c "import torch, transfusion_pytorch as t; print(torch.__version__, t.__name__)"
```

**HPC notes**
- One 80 GB H100 holds everything we train here (≤1.4B params in bf16 ≈ a few GB of weights + optimizer state). Single node is enough for Phases 1–2; multi-node is for _later_ scale-ups (Appendix B).

---

## 4. Phase 0 — Synthetic smoke test (5 minutes, 1 GPU or even CPU) [Done]

Purpose: confirm install + the joint forward/backward + loss breakdown before touching data. This exact script ran clean for me on the published package.

```python
# phase0_smoke.py
import torch
from torch import nn, randint, randn
from transfusion_pytorch import Transfusion

torch.manual_seed(0)

# full text+image path with a mock conv encoder/decoder (naive attention)
m = Transfusion(
    num_text_tokens=256,
    dim_latent=64,
    channel_first_latent=True,
    modality_default_shape=(4, 4),          # always pass this (gotcha #4)
    modality_encoder=nn.Conv2d(3, 64, 3, padding=1),
    modality_decoder=nn.Conv2d(64, 3, 3, padding=1),
    add_pos_emb=True,
    modality_num_dim=2,
    reconstruction_loss_weight=0.1,
    transformer=dict(dim=64, depth=2, dim_head=32, heads=4),
)
ema = m.create_ema(0.9)

data = [
    [randint(0,256,(8,)), randn(3,8,8), randint(0,256,(5,))],
    [randint(0,256,(6,)), randn(3,8,8), randint(0,256,(4,)), randn(3,8,8)],
]

loss, bd = m(data, return_breakdown=True)   # bd = LossBreakdown(total, text, flow, velocity, recon)
loss.backward()
print("total", float(bd.total), "| text", float(bd.text),
      "| flow", [float(x) for x in bd.flow])

ema.update()
loss_vc = m(data, velocity_consistency_ema_model=ema)   # exercises the flow-matching extra
print("with velocity-consistency:", float(loss_vc))
print("PHASE 0 OK")
```

**Pass criteria:** no exception; `text` ≈ ln(effective vocab) ≈ 5–6 at init; `flow` is a positive MSE; backward populates grads. If this fails, stop and fix the environment — nothing downstream will work.

---

## 5. Phase 1 — Oxford Flowers integration run (single node; hours)

**Why Flowers** ~8k real photographs across 102 classes with genuine text labels — it exercises the _real_ pipeline (pretrained VAE latents, joint text+image sequence, intra-image bidirectional attention, conditional sampling) without the cost of a web-scale corpus. This is the "is the architecture working as intended?" milestone. It mirrors the repo's own `train_latent_with_text.py`, so it's the lowest-risk first real run.

### 5.1 Data

Source: HF `nelorth/oxford-flowers` (used by the repo example). Class index → label string via the repo's `data/flowers/labels.txt`. Tokenize labels at the **byte level** (`num_text_tokens=256`), exactly as the example does — zero tokenizer dependencies.

### 5.2 Encoder/decoder = frozen pretrained SD VAE

Use **`stabilityai/sd-vae-ft-mse`** (the MSE-finetuned SD v1 VAE; 4-channel latent, f8 downsampling, scale factor `0.18215`). This matches the example's constants exactly and is a strict upgrade over the original SD1.5 VAE for reconstruction. It is **frozen** (`requires_grad_(False)`).

> Latent geometry: a 128×128 image → VAE → **16×16×4** latent → a learned **stride-2 conv** (`pre_post_transformer_enc_dec`) → **8×8** patches into the transformer (64 tokens/image). `modality_default_shape=(8,8)` is the _transformer-token grid_ (verified against the example). At 256px this becomes 32×32 latent → 16×16 = 256 tokens, `modality_default_shape=(16,16)` — which equals the paper's canonical "256 patches per image."

### 5.3 + 5.4 Training script (1 GPU; W&B; checkpointing; conditional sampling)

This is the validated example, hardened for a real run. Run with `python phase1_flowers.py`. (For single-node multi-GPU, see the DDP note after the script.)

```python
# phase1_flowers.py
import os
from pathlib import Path
import torch
from torch import nn, tensor, Tensor
from torch.nn import Module
from torch.optim import Adam
import torchvision.transforms as T
from torchvision.utils import save_image
from datasets import load_dataset
from diffusers.models import AutoencoderKL
import wandb
from transfusion_pytorch import Transfusion, create_dataloader, print_modality_sample

# ---------------- config ----------------
IMAGE_SIZE   = 128
LATENT_CH    = 4
DIM, DEPTH   = 768, 16          # ~bug #33 was at 768/12; it's fixed, so this is a fine target
HEADS, DH    = 12, 64
BATCH_SIZE   = 16
GRAD_ACCUM   = 2               # effective batch 32
LR           = 8e-4            # repo example value (tuned for this recipe)
TOTAL_STEPS  = 60_000
SAMPLE_EVERY = 1_000
CKPT_EVERY   = 5_000
CKPT_DIR = Path(os.environ.get('OUTPUT_DIR', './runs/flowers'));
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS = CKPT_DIR / 'samples';
RESULTS.mkdir(exist_ok=True, parents=True)
DEVICE       = 'cuda'

# ---------------- byte tokenizer ----------------
def encode_tokens(s: str) -> Tensor: return tensor([*bytes(s, 'UTF-8')])
def decode_tokens(t: Tensor) -> str: return ''.join(chr(max(32, i)) for i in t.tolist())
with open('./data/flowers/labels.txt') as f:
    LABELS_TEXT = f.read().split('\n')

# ---------------- frozen pretrained VAE ----------------
vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse')
vae.requires_grad_(False).eval()

class Encoder(Module):
    def __init__(self, vae): super().__init__(); self.vae = vae
    def forward(self, image):
        with torch.no_grad():
            lat = self.vae.encode(image * 2 - 1).latent_dist.sample()
        return 0.18215 * lat

class Decoder(Module):
    def __init__(self, vae): super().__init__(); self.vae = vae
    def forward(self, latents):
        with torch.no_grad():
            img = self.vae.decode((1 / 0.18215) * latents).sample
        return (img / 2 + 0.5).clamp(0, 1)

# ---------------- model ----------------
tok_grid = IMAGE_SIZE // 8 // 2          # 8 at 128px
model = Transfusion(
    num_text_tokens=256,
    dim_latent=LATENT_CH,
    channel_first_latent=True,
    modality_default_shape=(tok_grid, tok_grid),
    modality_encoder=Encoder(vae),
    modality_decoder=Decoder(vae),
    pre_post_transformer_enc_dec=(
        nn.Conv2d(LATENT_CH, DIM, 3, 2, 1),                       # latent -> transformer (stride-2 compress + project)
        nn.ConvTranspose2d(DIM, LATENT_CH, 3, 2, 1, output_padding=1),
    ),
    add_pos_emb=False,
    modality_num_dim=2,
    reconstruction_loss_weight=0.1,        # repo example value; keep flow_loss_weight=1 (default)
    transformer=dict(dim=DIM, depth=DEPTH, dim_head=DH, heads=HEADS),
).to(DEVICE)

ema_model = model.create_ema(0.999)        # higher decay for a long run

# ---------------- data ----------------
class FlowersDataset(torch.utils.data.Dataset):
    def __init__(self, image_size):
        self.ds = load_dataset('nelorth/oxford-flowers')['train']
        self.tf = T.Compose([T.Resize((image_size, image_size)), T.PILToTensor(),
                             T.Lambda(lambda t: t / 255.)])
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        s = self.ds[i]
        return encode_tokens(LABELS_TEXT[s['label']]), self.tf(s['image'])

def cycle(dl):
    while True:
        for b in dl: yield b

dl = create_dataloader(FlowersDataset(IMAGE_SIZE), batch_size=BATCH_SIZE, shuffle=True)  # applies the right collate
it = cycle(dl)
opt = Adam(model.parameters(), lr=LR)

wandb.init(project='transfusion-flowers', config=dict(dim=DIM, depth=DEPTH, image=IMAGE_SIZE, bs=BATCH_SIZE*GRAD_ACCUM, lr=LR))

# ---------------- train ----------------
for step in range(1, TOTAL_STEPS + 1):
    model.train()
    for _ in range(GRAD_ACCUM):
        loss, bd = model(next(it), return_breakdown=True)
        (loss / GRAD_ACCUM).backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    opt.step(); opt.zero_grad()
    ema_model.update()

    wandb.log({
        'loss/total': float(bd.total),
        'loss/text':  float(bd.text),
        'loss/flow':  float(sum(bd.flow)) if bd.flow else 0.0,
        'loss/recon': float(sum(x for sub in bd.recon for x in sub)) if bd.recon else 0.0,
        'grad_norm':  float(gnorm),
    }, step=step)

    if step % SAMPLE_EVERY == 0:
        model.eval()
        has_image = 0
        for j in range(4):                                  # condition on a random flower label
            label = LABELS_TEXT[torch.randint(0, len(LABELS_TEXT), ()).item()]
            out = ema_model.sample(prompt=encode_tokens(label).to(DEVICE),
                                   max_length=256, modality_steps=50,
                                   text_temperature=0.7, cfg_scale=1.0)
            # find an image modality in the sample
            for el in out:
                if isinstance(el, tuple):                    # (modality_type, image_tensor)
                    has_image += 1
                    save_image(el[1].detach().cpu(), RESULTS / f'{step}_{j}_{label[:20]}.png')
                    break
        wandb.log({'sample/has_image_frac': has_image / 4}, step=step)   # <-- watch this (gotcha #1)

    if step % CKPT_EVERY == 0:
        torch.save({'step': step, 'model': model.state_dict(),
                    'ema': ema_model.state_dict(), 'opt': opt.state_dict()},
                   CKPT_DIR / f'ckpt_{step}.pt')
```

**Single-node multi-GPU (optional).** Because batches are nested Python lists, the safe pattern is: `accelerator = Accelerator(...)`; build the dataloader with `create_dataloader(...)`; `model, opt = accelerator.prepare(model, opt)`; prepare the loader **without device placement** (`accelerator.prepare_data_loader(dl, device_placement=False)`) so Accelerate shards across ranks but lets `model.forward` move tensors itself; replace `loss.backward()` with `accelerator.backward(loss)` inside `with accelerator.accumulate(model):`; guard sampling/checkpointing with `accelerator.is_main_process`. Launch with `accelerate launch --multi_gpu --num_processes 8 phase1_flowers.py`. _Verify on a short run first_ — if the prepared loader chokes on the nested batch, fall back to single-GPU (Flowers trains fine on one H100).

### Experiments:
#### Start on one GPU first
##### 1. One-time setup on the login node
```bash
export WORKDIR="$HOME/dev/transfusion-pytorch"
export HF_HOME="$WORKDIR/hf_cache/huggingface"
export OUTPUT_DIR="$WORKDIR/runs/flowers_smoke"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate transfusion
```
##### 2. Run the Prefetch Script
run once on the login node with `HF_HOME` exported. It pulls the Oxford Flowers dataset and the SD VAE into the cache so the offline compute node can load both `from_pretrained`/`load_dataset` without network

```bash
srun --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=00:30:00 --pty bash

source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate transfusion

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline WANDB_DIR="$OUTPUT_DIR"

cd "$WORKDIR" && python phase1_flowers.py
```

##### 3. The SLURM Script:
Setup the environment variables
```bash
srun --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=00:30:00 --pty bash

source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate transfusion 

export OUTPUT_DIR="$WORKDIR/runs/flowers_smoke"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline WANDB_DIR="$OUTPUT_DIR" 

cd "$WORKDIR/transfusion-pytorch" && python phase1_flowers.py # watch ~200 steps, note it/s, Ctrl-C, exit
```

Then submit and monitor
```bash
sbatch phase1_flowers.sbatch
squeue --me
tail -f tf_flowers_<jobid>.out
# after it runs, from the LOGIN node: wandb sync "$HOME/transfusion/runs/flowers/wandb/offline-run-*"
```
This semi-worked: I had to make changes listed [[Old Transfusion Edits]].
Then, I realized that this 'hack' was not actually resolving the underlying problem. 
#### Resulting changes I've had to make over time and why:
One Library fix (the only tracked-file edit)
- `transfusion_pytorch/transfusion.py` — fixed the `cfg_scale>1` rotary/KV-cache crash. Two edits in the CFG sampling path (~line 1829 and ~1896): only build/use the uncond KV cache when `cache_kv=True`, and set `need_splice=not exists(uncond_cache)` so the uncond path mirrors the conditional path.
#### HyperParameter Sweep over Flowers Dataset
I wanted to test the model performance over a range of hyper-parameters. During this test, I also tested the change made above to `transfusion_pytorch/transfusion.py` 

1. Fixed the CFG cfg_scale>1 rotary/KV crash in transfusion.py (highest risk; cfg_scale=1.0 is the fallback so the sweep is never blocked).
2. **`sweep_flowers.py`** — parameterized small-arch trainer that adds the three things the example lacks: checkpointing, _conditional_ sample monitoring, and the conditioning-gap metric to wandb.
3. **`launch_sweep_flowers.sh`** — generates + submits the 9-run grid (LR {8e-4,3e-4,1e-4} × prob_uncond {0.0,0.1,0.2}), one organized `runs/sweep_small_<ts>/<run>/{checkpoints,samples,conditional,wandb,logs}` dir each, with a `sweep_metadata.tsv`.
4. **`eval_conditional_sweep.py`** — post-hoc side-by-side text→image comparison across runs.

After training each of the 9 hyperparameter sweeps to 100k steps, I observed the model was learning something, but the photos were not very realistic. They looked like extremely smudged paintings of flowers. One could sort of make out stems, leaves, and petals, but nothing was concrete. Additionally, the conditioning wasn't as strong as it could be. I.e. a 'sunflower' label did not always produce a sunflower, it could be a flower, but not a sunflower. 

I confirmed that the model was utilizing the conditioning.
As we hoped: the conditioning / gap_pct increases with time.



---

## 6. Phase 2 — CC12M paper-style reproduction

This is the actual reproduction: real free-form **captions** 256px images, and the paper's evaluation metrics (FID + CLIP ), plus the paper's optimizer recipe.

### What's built (all new, untracked → revert by deletion)

- **`download_cc12m.sh`** — pulls pre-built shards from HF **`pixparse/cc12m-wds`** (`NSHARDS` arg).
- **`cc12m_data.py`** — WebDataset loader (decode → resize → GPT-2 BPE encode → the model's `[text,image]` batch format; DDP-safe).
- **`train_cc12m.py`** — DDP trainer: model-size presets (0.16/0.37/0.76B), AdamW + warmup→cosine schedule (paper recipe), cond/gap monitoring, rank-0 checkpoint/sample/wandb.
- **`launch_cc12m.sbatch`** — `srun torchrun` 8-GPU (multi-node by raising `--nodes`).
- **`eval_conditional_sweep.py`** — added `TOKENIZER=gpt2` + `MODEL_SIZE` + free-form `PROMPTS` switches
- 
Google's `cc12m.tsv` now **403s**, so the planned img2dataset-from-URLs path is dead. I switched to HF's ungated **`pixparse/cc12m-wds`** — strictly better (no crawling, **no link rot**, ~11M images). But it ships **original-resolution** JPEGs, so the **full dataset is ~1.2 TB**, not the ~250-300 GB we estimated (the loader resizes to 256 on the fly). It fits your 12 TB free, but that's **~4× the earlier number** — flagging before Phase 1 pulls it all.
### Phase-0 gate: passed 

The 300k-subset run completed with `cond/gap_pct` climbing to **+5.7%** by 30k steps — the model genuinely conditions on free-form CC12M captions (contrast the flowers big-model's 0%). That's the validation we wanted before committing the full-scale run.
### Data
- The full CC12M dataset is downloaded

### Phase 2 launched — job 1164

- 8-GPU DDP, **0.16B (768/16/12), 256px**, eff batch 128, **250k steps**
- Paper recipe: **AdamW(0.9,0.95, wd 0.1)**, `LR=3e-4`, warmup 4000 → cosine to 1.5e-5, grad clip 1.0
- Confirmed **actively training** (wandb logging, no OOM, batch 16 fits at 256px)

### Monitoring

```bash
squeue -u benjamin
wandb sync runs/cc12m/p2_0.16B/wandb/offline-run-*    # loss/flow, loss/text, cond/gap_pct, lr
# text→image grids from checkpoints (any time):
export HF_HOME="$PWD/hf_cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD"
export MODEL_SIZE=0.16B IMAGE_SIZE=256 TOKENIZER=gpt2 CFG_SCALES=1.0,3.0
export PROMPTS="a red sports car;a cat sitting on a sofa;a mountain lake at sunset;a plate of sushi;a wooden cabin in the forest;a city street at night;a golden retriever puppy;a cup of coffee on a table"


srun --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=00:60:00 --job-name=cc12m_eval \
  python eval_conditional_sweep.py runs/cc12m/p2_0.16B 2>&1 | grep -aiE "found|loaded|cfg=|saved|emitted|error|traceback|done" | tail -40
  
  
```

This is a long run (250k steps ≈ several hours to ~a day on one 8×H100 node; checkpoints every 5k let you stop early). The next gate is **Phase 3** (scale to 0.37B/0.76B + add the U-Net patchifier) once 0.16B's loss/conditioning curves plateau — and you could go multi-node there for bigger batches.

## June 20, 2026
I'm not very happy with the images resulting from evaluating the p2_01.16B mode at 250k steps. I can observe textures in the generated images at 250k steps, but the images are not clearly consistent with their conditioning.

| a red sports car             | a cat sitting on a sofa | a mountain lake at sunset | a plate of sushi           |
| ---------------------------- | ----------------------- | ------------------------- | -------------------------- |
| a wooden cabin in the forest | a city street at night  | a golden retriever puppy  | a cup of coffee on a table |


At 250k steps at `cfg 1.0`: 
- `a mountain lake at sunset` produced a noisy image of an ocean in broad daylight, everything else looks like noisy textures inconsistent with the prompts.
at `cfg 3.0`:
- `a red sports car` looks like what a driver would see from the their windshield, but there is no red at all
- `a wooden cabin` the forest produced a texture of a tree leaves 
- `a cat sitting on a sofa` produced the texture of a sofa


![[W&B Chart 6_20_2026, 3_17_17 PM.png]]
![[CC12m_p2_0.16B_250k_eval_cfg1.0.png.png]]
CC12m_p2_0.16B_250k_eval_cfg3.0.png
![[CC12m_p2_0.16B_250k_eval_cfg3.0.png.png|697]]
CC12m_p2_0.16B_250k_eval_cfg5.0.png
![[CC12m_p2_0.16B_250k_eval_cfg5.0.png.png]]
**Is it learning to use the caption at all?**: Check `cond/*` metrics
![[W&B Chart 6_20_2026, 2_06_48 PM.png]]
`correct` is clearly below `wrong`. 


**Is the image rendering capped regardless of text?** Check the `loss/flow` + unconditional samples
![[W&B Chart 6_20_2026, 3_17_17 PM.png]]
![[CC12m_p2_0.16B_250k_loss_flow.png.png]]
Zoomed in: The loss fell from ~0.74 to 0.735 between 200k and 250k steps. I believe we were nearing convergence. Not worth training this for more steps.

Ways to improve
1. ~~Train for longer: 
	1. No: loss/flow shows that we've pretty much converged 
2. ~~Use a larger dataset~~
	1. We trained on ~11M unique pairs: at effective batch size 128 x 250k steps: we saw ~32M images ~ 3 epochs. This model only has 0.16B parameters, the converged flow is likely a result of the capacity-bound, not the data bound.
3. Increase the parameter count
4. Transition from the linear patchifier to the U-Net

Things to check:
- VAE quality

---
## Appendix A — Train your own VAE (paper-faithful encoder; needed for new modalities)

You asked specifically about training encoders, and you'll need this when you swap in robot modalities (proprioception, actions, depth, etc.). For images, the paper's recipe (App. A) is:

- Architecture: CNN encoder/decoder, **latent dim 8**, **f8** (256×256 → 32×32×8); ~86M params.
- Loss: `L1 + LPIPS + 0.5·GAN + 0.2·MoCo-v2-ID + 1e-6·KL`; **delay the GAN/adversarial term to 50k steps**; train ~**1M steps**.
- If you use this VAE in Transfusion, set `dim_latent=8` (not 4) and re-derive the latent scaling factor for your VAE (the `0.18215` constant is specific to the SD VAE; compute your own from the latent std).
- The repo's `train_mnist_vae.py` is a minimal reference for the train loop; scale its encoder/decoder up and add LPIPS/GAN/MoCo terms for real images. Validate reconstruction PSNR/LPIPS on a held set **before** plugging it into Transfusion (gotcha #3).

**For non-image robot modalities:** the only contract Transfusion needs is an `encoder: Module` (raw modality → `(…, dim_latent)` continuous tensor) and a `decoder: Module` (inverse), plus `dim_latent`, `modality_num_dim`, and `modality_default_shape` for that modality. The README's multi-modality API (`dim_latent=(d0, d1, …)`, tuple per-modality shapes; pass `(modality_index, tensor)` for floats) is exactly the hook for that — which is your stated end goal.

---

## Appendix B — Scaling to multi-node / larger models

- **Throughput:** enable bf16 (`accelerate` mixed precision or `torch.autocast`), `torch.compile` on the transformer (keep flex off or pad to ×128 first), activation checkpointing for ≥1.4B.
- **FSDP (7B+):** `accelerate config` → FSDP, `FULL_SHARD`, transformer-block auto-wrap, `bf16`, `limit_all_gathers=True`. Shard optimizer state to fit memory.
- **Flex attention at scale:** if you turn `use_flex_attn=True` for long sequences, **pad sequence length to a multiple of 128** and consider `softcap_value=0` to dodge PyTorch #153799; benchmark vs. the naive mask before committing.
- **Data at scale:** move from COCO to CC12M/LAION via WebDataset/streaming; this is where multi-node and 88 GPUs actually pay off and where you'd start approaching the paper's headline regime.
