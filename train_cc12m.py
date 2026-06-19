"""
train_cc12m.py — Transfusion text->image trainer for CC12M (GPT-2 BPE, 256px, DDP-capable).

Adapted from sweep_flowers.py's (correct) loop + phase1_coco.py's BPE tokenizer. Adds:
  - env-driven model-size presets (paper Table 2): 0.16B / 0.37B / 0.76B
  - AdamW(0.9,0.95, wd 0.1) + linear-warmup→cosine LR schedule (paper recipe)
  - WebDataset streaming via cc12m_data.make_cc12m_loader
  - optional multi-GPU DDP (launched by torchrun); rank-0 does logging/sampling/checkpointing

Env: MODEL_SIZE DATA IMAGE_SIZE LR MIN_LR WARMUP TOTAL_STEPS BATCH_SIZE GRAD_ACCUM
     PROB_UNCOND OUTPUT_DIR RUN_NAME RESUME WANDB_PROJECT SAMPLE_EVERY CKPT_EVERY
"""
import os, math, glob
from pathlib import Path
import torch
from torch import nn, tensor, Tensor
from torch.nn import Module
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image
from transformers import AutoTokenizer
from diffusers.models import AutoencoderKL
from einops import rearrange
import wandb
from transfusion_pytorch import Transfusion
from cc12m_data import make_cc12m_loader, cycle

# ---------------- DDP setup ----------------
DDP_ON = 'RANK' in os.environ
if DDP_ON:
    dist.init_process_group('nccl')
    RANK = dist.get_rank(); LOCAL_RANK = int(os.environ['LOCAL_RANK']); WORLD = dist.get_world_size()
    torch.cuda.set_device(LOCAL_RANK); DEVICE = f'cuda:{LOCAL_RANK}'
else:
    RANK = LOCAL_RANK = 0; WORLD = 1; DEVICE = 'cuda'
IS_MAIN = RANK == 0
def barrier():
    if DDP_ON: dist.barrier()

# ---------------- config ----------------
SIZES = {
    '0.16B': dict(dim=768,  depth=16, heads=12, dim_head=64),
    '0.37B': dict(dim=1024, depth=24, heads=16, dim_head=64),
    '0.76B': dict(dim=1536, depth=24, heads=24, dim_head=64),
}
MODEL_SIZE   = os.environ.get('MODEL_SIZE', '0.16B'); TFM = SIZES[MODEL_SIZE]
DIM          = TFM['dim']
IMAGE_SIZE   = int(os.environ.get('IMAGE_SIZE', 256))
LATENT_CH    = 4
BATCH_SIZE   = int(os.environ.get('BATCH_SIZE', 16))     # per-GPU
GRAD_ACCUM   = int(os.environ.get('GRAD_ACCUM', 1))
LR           = float(os.environ.get('LR', 3e-4))         # paper peak
MIN_LR       = float(os.environ.get('MIN_LR', 1.5e-5))   # paper cosine floor
WARMUP       = int(os.environ.get('WARMUP', 4000))       # paper warmup
TOTAL_STEPS  = int(os.environ.get('TOTAL_STEPS', 250_000))
PROB_UNCOND  = float(os.environ.get('PROB_UNCOND', 0.1))
SAMPLE_EVERY = int(os.environ.get('SAMPLE_EVERY', 5_000))
CKPT_EVERY   = int(os.environ.get('CKPT_EVERY', 5_000))
DATA         = os.environ['DATA']                        # brace pattern OR glob of .tar shards
if '*' in DATA:                                          # glob -> list of present shards (skips missing/in-flight)
    DATA = sorted(glob.glob(DATA))
    assert DATA, 'no shards matched DATA glob'
RECON_W      = 0.1
EMA_DECAY    = 0.999

CKPT_DIR = Path(os.environ.get('OUTPUT_DIR', './runs/cc12m/run'));
SAMPLES  = CKPT_DIR / 'samples'
if IS_MAIN:
    CKPT_DIR.mkdir(parents=True, exist_ok=True); SAMPLES.mkdir(exist_ok=True, parents=True)

# ---------------- BPE tokenizer (GPT-2, cached/offline) ----------------
TOKENIZER       = AutoTokenizer.from_pretrained('gpt2')
NUM_TEXT_TOKENS = TOKENIZER.vocab_size                   # 50257
def encode_tokens(s: str) -> Tensor: return tensor(TOKENIZER.encode(s), dtype=torch.long)

# ---------------- frozen VAE ----------------
vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse').requires_grad_(False).eval()
class Encoder(Module):
    def __init__(self, vae): super().__init__(); self.vae = vae
    def forward(self, image):
        with torch.no_grad(): lat = self.vae.encode(image * 2 - 1).latent_dist.sample()
        return 0.18215 * lat
class Decoder(Module):
    def __init__(self, vae): super().__init__(); self.vae = vae
    def forward(self, latents):
        with torch.no_grad(): img = self.vae.decode((1 / 0.18215) * latents).sample
        return (img / 2 + 0.5).clamp(0, 1)

# ---------------- model ----------------
tok_grid = IMAGE_SIZE // 8 // 2          # 16 at 256px -> 256 tokens (paper)
model = Transfusion(
    num_text_tokens=NUM_TEXT_TOKENS,
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
    prob_uncond=PROB_UNCOND,
    transformer=dict(**TFM),
).to(DEVICE)

ema_model = model.create_ema(EMA_DECAY)                   # tracks raw params (created before DDP wrap)
train_model = DDP(model, device_ids=[LOCAL_RANK]) if DDP_ON else model

# ---------------- data ----------------
loader = make_cc12m_loader(DATA, IMAGE_SIZE, encode_tokens, BATCH_SIZE,
                           num_workers=int(os.environ.get('NUM_WORKERS', 8)))
it = cycle(loader)
diag_it = cycle(make_cc12m_loader(DATA, IMAGE_SIZE, encode_tokens, 16, num_workers=2))

opt = AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
def lr_lambda(step):
    if step < WARMUP: return (step + 1) / WARMUP
    prog = min(1.0, (step - WARMUP) / max(1, TOTAL_STEPS - WARMUP))
    cos = 0.5 * (1 + math.cos(math.pi * prog))            # 1 -> 0
    floor = MIN_LR / LR
    return floor + (1 - floor) * cos
sched = LambdaLR(opt, lr_lambda)

# ---------------- conditioning-gap diagnostic (lower flow loss w/ correct caption => uses text) ----------------
@torch.no_grad()
def conditioning_gap(n_batches=8):
    model.eval()
    cor_l, wrong_l = [], []
    for b in range(n_batches):
        batch = next(diag_it)
        texts  = [p[0] for p in batch]; images = [p[1] for p in batch]
        cor = [[t, im] for t, im in zip(texts, images)]
        shf = [[t, im] for t, im in zip(texts[1:] + texts[:1], images)]
        s = 1000 + b
        torch.manual_seed(s); _, bdc = model(cor, return_breakdown=True)
        torch.manual_seed(s); _, bdw = model(shf, return_breakdown=True)
        cor_l.append(float(sum(bdc.flow))); wrong_l.append(float(sum(bdw.flow)))
    model.train()
    c = sum(cor_l) / len(cor_l); w = sum(wrong_l) / len(wrong_l)
    return dict(correct=c, wrong=w, gap=w - c, gap_pct=(100 * (w - c) / w if w else 0.0))

# ---------------- resume ----------------
start_step = 1
resume = os.environ.get('RESUME')
if resume and os.path.exists(resume):
    ck = torch.load(resume, map_location=DEVICE)
    model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
    try: ema_model.load_state_dict(ck['ema'])
    except Exception: pass
    try: sched.load_state_dict(ck['sched'])
    except Exception: pass
    start_step = ck['step'] + 1
    if IS_MAIN: print(f'resumed at step {start_step}', flush=True)

# ---------------- wandb (rank-0 only) ----------------
if IS_MAIN:
    wandb.init(project=os.environ.get('WANDB_PROJECT', 'transfusion-cc12m'),
               name=os.environ.get('RUN_NAME'),
               config=dict(model_size=MODEL_SIZE, dim=DIM, image=IMAGE_SIZE,
                           eff_bs=BATCH_SIZE * GRAD_ACCUM * WORLD, lr=LR, min_lr=MIN_LR,
                           warmup=WARMUP, prob_uncond=PROB_UNCOND, world=WORLD))

# ---------------- train ----------------
for step in range(start_step, TOTAL_STEPS + 1):
    train_model.train()
    for micro in range(GRAD_ACCUM):
        batch = next(it)
        # avoid redundant DDP allreduce on accumulation micro-steps
        if DDP_ON and micro < GRAD_ACCUM - 1:
            with train_model.no_sync():
                loss, bd = train_model(batch, return_breakdown=True)
                (loss / GRAD_ACCUM).backward()
        else:
            loss, bd = train_model(batch, return_breakdown=True)
            (loss / GRAD_ACCUM).backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # paper: clip 1.0
    opt.step(); opt.zero_grad(); sched.step(); ema_model.update()

    if IS_MAIN:
        wandb.log({
            'loss/total': float(bd.total),
            'loss/text':  float(bd.text),
            'loss/flow':  float(sum(bd.flow)) if bd.flow else 0.0,
            'loss/recon': float(sum(x for sub in bd.recon for x in sub)) if bd.recon else 0.0,
            'grad_norm':  float(gnorm),
            'lr':         sched.get_last_lr()[0],
        }, step=step)

    if step % SAMPLE_EVERY == 0:
        barrier()                                  # all ranks pause together (no allreduce deadlock)
        if IS_MAIN:
            try:
                g = conditioning_gap()
                wandb.log({'cond/correct': g['correct'], 'cond/wrong': g['wrong'],
                           'cond/gap': g['gap'], 'cond/gap_pct': g['gap_pct']}, step=step)
                print(f"[step {step}] cond gap {g['gap']:+.4f} ({g['gap_pct']:+.1f}%)", flush=True)
            except Exception as e:
                print(f'[cond-gap skipped at step {step}] {e}', flush=True)
            try:
                u = ema_model.generate_modality_only(batch_size=4, modality_steps=16)
                save_image(rearrange(u, '(gh gw) c h w -> c (gh h) (gw w)', gh=2).detach().cpu(),
                           SAMPLES / f'{step}.png')
                wandb.log({'samples_uncond': wandb.Image(str(SAMPLES / f'{step}.png'))}, step=step)
            except Exception as e:
                print(f'[uncond sample skipped at step {step}] {e}', flush=True)
        barrier()
        # conditional text->image grids: produced post-hoc by eval_conditional_sweep.py (crash-proof)

    if step % CKPT_EVERY == 0:
        barrier()
        if IS_MAIN:
            torch.save({'step': step, 'model': model.state_dict(), 'ema': ema_model.state_dict(),
                        'opt': opt.state_dict(), 'sched': sched.state_dict()},
                       CKPT_DIR / f'ckpt_{step}.pt')
            print(f'[step {step}] checkpoint saved', flush=True)
        barrier()

if DDP_ON:
    dist.destroy_process_group()
