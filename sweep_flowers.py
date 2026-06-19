"""
sweep_flowers.py — parameterized small-architecture (dim 128 / depth 8 / heads 8) text->image
trainer for Oxford Flowers, built for the LR x prob_uncond ablation sweep.

Adds the three things the repo example (train_latent_with_text.py) lacks:
  1. checkpointing (+ RESUME),
  2. CONDITIONAL text->image monitoring via sample(prompt=...)  (not the unconditional
     generate_modality_only, which ignores text),
  3. a conditioning-gap metric (correct vs wrong caption flow loss) logged to wandb.

All knobs are env-driven so one script serves every run in the sweep:
  LR, PROB_UNCOND, OUTPUT_DIR, RUN_NAME, TOTAL_STEPS, RESUME, WANDB_PROJECT, CFG_SCALE,
  SAMPLE_EVERY, CKPT_EVERY.
"""
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
from transfusion_pytorch import Transfusion, create_dataloader
from einops import rearrange

# ---------------- config (arch fixed = "original" small recipe; ablation knobs via env) ----------------
IMAGE_SIZE   = 128
LATENT_CH    = 4
DIM, DEPTH   = 128, 8
HEADS, DH    = 8, 64
BATCH_SIZE   = 4
GRAD_ACCUM   = 4                # effective batch 16 (matches the example recipe)
LR           = float(os.environ.get('LR', 8e-4))
PROB_UNCOND  = float(os.environ.get('PROB_UNCOND', 0.1))   # CFG conditioning dropout (ablation axis)
CFG_SCALE    = float(os.environ.get('CFG_SCALE', 3.0))     # guidance used for conditional monitoring
TOTAL_STEPS  = int(os.environ.get('TOTAL_STEPS', 100_000))
SAMPLE_EVERY = int(os.environ.get('SAMPLE_EVERY', 5_000))
CKPT_EVERY   = int(os.environ.get('CKPT_EVERY', 5_000))
RECON_W      = 0.1              # held fixed across the sweep
EMA_DECAY    = 0.999            # held fixed; better than 0.9 for stable EMA samples over 100k steps
DEVICE       = 'cuda'

CKPT_DIR = Path(os.environ.get('OUTPUT_DIR', './runs/flowers_sweep')); CKPT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES  = CKPT_DIR / 'samples';     SAMPLES.mkdir(exist_ok=True, parents=True)
COND_DIR = CKPT_DIR / 'conditional'; COND_DIR.mkdir(exist_ok=True, parents=True)

# ---------------- byte tokenizer ----------------
def encode_tokens(s: str) -> Tensor: return tensor([*bytes(s, 'UTF-8')])
with open('./data/flowers/labels.txt') as f:
    LABELS_TEXT = f.read().split('\n')

# fixed prompts for conditional monitoring (exact strings from labels.txt for a fair test)
PROMPTS = ['sunflower', 'rose', 'water lily', 'pink primrose',
           'daffodil', 'tiger lily', 'globe thistle', 'bird of paradise']

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
        nn.Conv2d(LATENT_CH, DIM, 3, 2, 1),
        nn.ConvTranspose2d(DIM, LATENT_CH, 3, 2, 1, output_padding=1),
    ),
    add_pos_emb=False,
    modality_num_dim=2,
    reconstruction_loss_weight=RECON_W,
    prob_uncond=PROB_UNCOND,             # ablation axis: CFG conditioning dropout
    transformer=dict(dim=DIM, depth=DEPTH, dim_head=DH, heads=HEADS),
).to(DEVICE)

ema_model = model.create_ema(EMA_DECAY)

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

ds = FlowersDataset(IMAGE_SIZE)
dl = create_dataloader(ds, batch_size=BATCH_SIZE, shuffle=True)
it = cycle(dl)
# separate loader (bs 16) for the conditioning-gap diagnostic
diag_it = cycle(create_dataloader(ds, batch_size=16, shuffle=True))
opt = Adam(model.parameters(), lr=LR)

# ---------------- conditioning-gap diagnostic (correct vs wrong caption flow loss) ----------------
@torch.no_grad()
def conditioning_gap(n_batches=8):
    """Lower flow loss with correct captions than with shuffled (wrong) captions => model uses text."""
    model.eval()
    correct, wrong = [], []
    for b in range(n_batches):
        batch = next(diag_it)
        texts  = [p[0] for p in batch]; images = [p[1] for p in batch]
        cor = [[t, im] for t, im in zip(texts, images)]
        shf = [[t, im] for t, im in zip(texts[1:] + texts[:1], images)]   # each image gets a wrong label
        s = 1000 + b
        torch.manual_seed(s); _, bdc = model(cor, return_breakdown=True)   # same RNG -> identical noise/t draw
        torch.manual_seed(s); _, bdw = model(shf, return_breakdown=True)   # so only the caption differs
        correct.append(float(sum(bdc.flow))); wrong.append(float(sum(bdw.flow)))
    model.train()
    c = sum(correct) / len(correct); w = sum(wrong) / len(wrong)
    gap = w - c
    return dict(correct=c, wrong=w, gap=gap, gap_pct=(100 * gap / w if w else 0.0))

# NOTE on conditional text->image generation:
# We deliberately do NOT call sample(prompt=...) inside the training loop. That path can hit a
# rare CUDA device-side assert on under-trained weights, and such an assert poisons the CUDA
# context (so a try/except cannot save the run) — one bad sample at step 70k would kill hours of
# training. Conditional images are instead produced by eval_conditional_sweep.py against the
# checkpoints written below; it can be run live during training or post-hoc, and a crash there
# only loses a throwaway eval. The in-loop conditioning signal is the cond/gap metric (a plain
# forward, identical to a training step) plus the unconditional sample grid.

# ---------------- resume ----------------
start_step = 1
resume = os.environ.get('RESUME')
if resume:
    ck = torch.load(resume, map_location=DEVICE)
    model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
    try: ema_model.load_state_dict(ck['ema'])
    except Exception: pass
    start_step = ck['step'] + 1
    print(f'resumed at step {start_step}', flush=True)

# ---------------- wandb ----------------
wandb.init(project=os.environ.get('WANDB_PROJECT', 'transfusion-flowers-small-cfg'),
           name=os.environ.get('RUN_NAME'),
           config=dict(dim=DIM, depth=DEPTH, heads=HEADS, image=IMAGE_SIZE,
                       bs=BATCH_SIZE * GRAD_ACCUM, lr=LR, prob_uncond=PROB_UNCOND,
                       cfg_scale=CFG_SCALE, recon_w=RECON_W, ema=EMA_DECAY))

# ---------------- train ----------------
for step in range(start_step, TOTAL_STEPS + 1):
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
        # the KEY signal: does the model use the caption?
        try:
            g = conditioning_gap()
            wandb.log({'cond/correct': g['correct'], 'cond/wrong': g['wrong'],
                       'cond/gap': g['gap'], 'cond/gap_pct': g['gap_pct']}, step=step)
            print(f"[step {step}] cond gap {g['gap']:+.4f} ({g['gap_pct']:+.1f}%)", flush=True)
        except Exception as e:
            print(f'[cond-gap skipped at step {step}] {e}', flush=True)

        # unconditional baseline grid (cheap, robust)
        try:
            u = ema_model.generate_modality_only(batch_size=4, modality_steps=16)
            save_image(rearrange(u, '(gh gw) c h w -> c (gh h) (gw w)', gh=2).detach().cpu(),
                       SAMPLES / f'{step}.png')
            wandb.log({'samples_uncond': wandb.Image(str(SAMPLES / f'{step}.png'))}, step=step)
        except Exception as e:
            print(f'[uncond sample skipped at step {step}] {e}', flush=True)

        # conditional text->image grids are produced by eval_conditional_sweep.py on checkpoints
        # (see NOTE above) — intentionally not run in-loop to keep the 100k-step run crash-proof.
        model.train()

    if step % CKPT_EVERY == 0:
        torch.save({'step': step, 'model': model.state_dict(),
                    'ema': ema_model.state_dict(), 'opt': opt.state_dict()},
                   CKPT_DIR / f'ckpt_{step}.pt')
        print(f'[step {step}] checkpoint saved', flush=True)
