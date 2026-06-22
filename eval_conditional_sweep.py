"""
eval_conditional_sweep.py — post-hoc text->image comparison across a sweep's runs.

For every run dir under a sweep (each containing ckpt_*.pt), load the EMA weights of the
latest (or $CKPT) checkpoint and render the fixed prompt set at one or more cfg_scale values,
saving a labeled grid per (run, cfg_scale) for side-by-side comparison.

Arch is env-parameterized so this serves both the small sweep and the big-model runs:
  MODEL_SIZE (0.16B/0.37B/0.76B) OR explicit DIM/DEPTH/HEADS (default small 128/8/8);
  IMAGE_SIZE (default 128); TOKENIZER (byte | gpt2); CFG_SCALES (csv, default "1.0,3.0");
  PROMPTS (semicolon-separated; overrides the flower defaults for free-form CC12M captions).

Usage:
  python eval_conditional_sweep.py runs/sweep_small_<stamp>
  MODEL_SIZE=0.16B IMAGE_SIZE=256 TOKENIZER=gpt2 \
    PROMPTS="a red sports car;a cat wearing a hat;a mountain lake at sunset" \
    python eval_conditional_sweep.py runs/cc12m/p2_0.16B
"""
import os, sys, glob, json, torch
import torch.nn.functional as F
from pathlib import Path
from torch import nn, tensor
from torch.nn import Module
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from transfusion_pytorch import Transfusion

DEVICE='cuda'; LATENT_CH = 4
SWEEP = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get('SWEEP', './runs'))

# auto-load the run's config.json (written by train_cc12m.py) for arch defaults; env overrides.
def _load_cfg(root):
    for c in [root / 'config.json', *sorted(root.glob('*/config.json'))]:
        if c.exists():
            print(f'using arch from {c}', flush=True)
            return json.load(open(c))
    return {}
_CFG = _load_cfg(SWEEP)
def _pick(env, key, default):           # precedence: env var > config.json > default
    return os.environ.get(env) if os.environ.get(env) is not None else _CFG.get(key, default)

IMAGE_SIZE = int(_pick('IMAGE_SIZE', 'image_size', 128))
SIZES = {'0.16B': (768, 16, 12), '0.37B': (1024, 24, 16), '0.76B': (1536, 24, 24)}
_ms = _pick('MODEL_SIZE', 'model_size', None)
if _ms in SIZES:
    DIM, DEPTH, HEADS = SIZES[_ms]
else:
    DIM   = int(os.environ.get('DIM', 128))
    DEPTH = int(os.environ.get('DEPTH', 8))
    HEADS = int(os.environ.get('HEADS', 8))
DH         = int(os.environ.get('DH', 64))
PATCHIFIER = _pick('PATCHIFIER', 'patchifier', 'conv')
CFG_SCALES = [float(x) for x in os.environ.get('CFG_SCALES', '1.0,3.0').split(',')]

# tokenizer: byte-level (flowers) or GPT-2 BPE (CC12M/COCO)
if _pick('TOKENIZER', 'tokenizer', 'byte') == 'gpt2':
    from transformers import AutoTokenizer
    _TOK = AutoTokenizer.from_pretrained('gpt2')
    NUM_TEXT_TOKENS = _TOK.vocab_size
    def encode_tokens(s): return tensor(_TOK.encode(s), dtype=torch.long)
else:
    NUM_TEXT_TOKENS = 256
    def encode_tokens(s): return tensor([*bytes(s, 'UTF-8')])

_default_prompts = ['sunflower', 'rose', 'water lily', 'pink primrose',
                    'daffodil', 'tiger lily', 'globe thistle', 'bird of paradise']
PROMPTS = [p.strip() for p in os.environ['PROMPTS'].split(';')] if os.environ.get('PROMPTS') else _default_prompts

vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse').requires_grad_(False).eval()
class Enc(Module):
    def __init__(s, v): super().__init__(); s.vae = v
    def forward(s, x):
        with torch.no_grad(): l = s.vae.encode(x*2-1).latent_dist.sample()
        return 0.18215 * l
class Dec(Module):
    def __init__(s, v): super().__init__(); s.vae = v
    def forward(s, l):
        with torch.no_grad(): return (s.vae.decode((1/0.18215)*l).sample/2+0.5).clamp(0, 1)

def build_model():
    tok = IMAGE_SIZE // 8 // 2
    if PATCHIFIER == 'unet':
        from unet_patchifier import build_unet_patchifier
        enc_dec = build_unet_patchifier(latent_ch=LATENT_CH, dim=DIM)
    else:
        enc_dec = (nn.Conv2d(LATENT_CH, DIM, 3, 2, 1),
                   nn.ConvTranspose2d(DIM, LATENT_CH, 3, 2, 1, output_padding=1))
    return Transfusion(num_text_tokens=NUM_TEXT_TOKENS, dim_latent=LATENT_CH, channel_first_latent=True,
        modality_default_shape=(tok, tok), modality_encoder=Enc(vae), modality_decoder=Dec(vae),
        pre_post_transformer_enc_dec=enc_dec,
        add_pos_emb=False, modality_num_dim=2, reconstruction_loss_weight=0.1,
        transformer=dict(dim=DIM, depth=DEPTH, dim_head=DH, heads=HEADS)).to(DEVICE)

def latest_ckpt(run_dir):
    cks = glob.glob(f'{run_dir}/ckpt_*.pt')
    if not cks: return None
    return sorted(cks, key=lambda p: int(p.split('_')[-1].split('.')[0]))[-1]

def to_tile(img):
    """Coerce a decoded modality (any shape) into a [3, IMAGE_SIZE, IMAGE_SIZE] tile for the grid."""
    img = img.detach().cpu().float()
    if img.ndim == 4: img = img[0]
    if img.ndim == 2: img = img.unsqueeze(0)
    if img.shape[0] == 1: img = img.repeat(3, 1, 1)
    img = img[:3]
    img = F.interpolate(img.unsqueeze(0), size=(IMAGE_SIZE, IMAGE_SIZE),
                        mode='bilinear', align_corners=False)[0]
    return img.clamp(0, 1)

# discover run dirs (a sweep root with run subdirs, or a single run dir)
run_dirs = sorted([p for p in SWEEP.glob('*') if p.is_dir() and glob.glob(f'{p}/ckpt_*.pt')])
if not run_dirs and glob.glob(f'{SWEEP}/ckpt_*.pt'):
    run_dirs = [SWEEP]
assert run_dirs, f'no run dirs with checkpoints found under {SWEEP}'
print(f'found {len(run_dirs)} run(s); cfg_scales={CFG_SCALES}', flush=True)

model = build_model(); ema = model.create_ema(0.999)
for run_dir in run_dirs:
    ckpt = latest_ckpt(run_dir)
    ck = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(ck['model'])
    try: ema.load_state_dict(ck['ema']); sampler = ema
    except Exception: sampler = model
    sampler.eval()
    print(f'\n== {run_dir.name} | {Path(ckpt).name} step {ck.get("step")} ==', flush=True)
    outdir = run_dir / 'conditional'; outdir.mkdir(exist_ok=True, parents=True)
    for scale in CFG_SCALES:
        tiles, n_emitted = [], 0
        for i, p in enumerate(PROMPTS):
            img = None
            try:
                out = sampler.sample(prompt=encode_tokens(p).to(DEVICE), max_length=256,
                                     modality_steps=50, text_temperature=0.7, cfg_scale=scale)
                for el in out:
                    if isinstance(el, tuple): img = el[1]; break
            except Exception as e:
                print(f'   cfg={scale} "{p}" failed: {type(e).__name__}: {e}', flush=True)
            if img is not None:
                tile = to_tile(img); n_emitted += 1
                save_image(tile, str(outdir / f'eval_cfg{scale}_{i}_{p.replace(" ", "_")}.png'))
            else:
                tile = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
            tiles.append(tile)
            print(f'   cfg={scale} "{p}" -> {"image" if img is not None else "no [som]"}', flush=True)
        path = outdir / f'eval_cfg{scale}.png'
        save_image(torch.stack(tiles), str(path), nrow=4)
        print(f'   saved {path}  ({n_emitted}/{len(PROMPTS)} emitted)', flush=True)
print('\ndone', flush=True)
