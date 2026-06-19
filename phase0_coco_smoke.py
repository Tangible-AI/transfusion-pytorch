"""phase0_coco_smoke.py — pre-flight check for the COCO 256px run.

Builds the REAL model size + VAE + GPT-2 tokenizer and runs a handful of real
training steps against COCO, so it actually validates (a) VRAM headroom at the
chosen batch size, (b) finite loss, (c) that an image modality emerges from sample().

Run interactively on a GPU node (NOT via sbatch) before launching phase1_coco:
    export COCO_ROOT=$HOME/dev/transfusion-pytorch/scratch
    export HF_HOME=$HOME/dev/transfusion-pytorch/hf_cache/huggingface
    BATCH_SIZE=8 python phase0_coco_smoke.py      # drop to 4 if it OOMs
"""
import os
from pathlib import Path
import torch
from torch import nn, tensor, Tensor
from torch.nn import Module
from torch.optim import Adam
import torchvision.transforms as T
from PIL import Image
from pycocotools.coco import COCO
from transformers import AutoTokenizer
from diffusers.models import AutoencoderKL
from transfusion_pytorch import Transfusion, create_dataloader

# ---- config (keep in sync with phase1_coco.py) ----
IMAGE_SIZE  = 256
LATENT_CH   = 4
DIM, DEPTH  = 768, 16
HEADS, DH   = 12, 64
BATCH_SIZE  = int(os.environ.get('BATCH_SIZE', 8))
GRAD_ACCUM  = int(os.environ.get('GRAD_ACCUM', 4))
N_STEPS     = int(os.environ.get('N_STEPS', 20))
COCO_ROOT   = os.environ.get('COCO_ROOT', './scratch')
DEVICE      = 'cuda'
assert torch.cuda.is_available(), "smoke test needs a GPU node"

TOKENIZER       = AutoTokenizer.from_pretrained('gpt2')
NUM_TEXT_TOKENS = TOKENIZER.vocab_size
def encode_tokens(s: str) -> Tensor: return tensor(TOKENIZER.encode(s), dtype=torch.long)

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

tok_grid = IMAGE_SIZE // 8 // 2
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
    reconstruction_loss_weight=0.1,
    transformer=dict(dim=DIM, depth=DEPTH, dim_head=DH, heads=HEADS),
).to(DEVICE)

class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, image_size, root, split='train'):
        self.img_dir = Path(root) / f'{split}2017'
        self.coco = COCO(str(Path(root) / 'annotations' / f'captions_{split}2017.json'))
        self.ids = list(self.coco.imgs.keys())
        self.tf = T.Compose([
            T.Lambda(lambda im: im.convert('RGB')),
            T.Resize(image_size), T.CenterCrop(image_size),
            T.PILToTensor(), T.Lambda(lambda t: t / 255.),
        ])
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        img_id = self.ids[i]
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        cap = anns[torch.randint(0, len(anns), ()).item()]['caption']
        fn = self.coco.loadImgs(img_id)[0]['file_name']
        return encode_tokens(cap), self.tf(Image.open(self.img_dir / fn))

ds = CocoDataset(IMAGE_SIZE, COCO_ROOT, 'train')
print(f"dataset: {len(ds)} images | sample caption: {TOKENIZER.decode(ds[0][0].tolist())!r}")
print(f"image tensor: shape={tuple(ds[0][1].shape)} dtype={ds[0][1].dtype} range=[{ds[0][1].min():.2f},{ds[0][1].max():.2f}]")

dl = create_dataloader(ds, batch_size=BATCH_SIZE, shuffle=True)
it = iter(dl)
opt = Adam(model.parameters(), lr=8e-4)

torch.cuda.reset_peak_memory_stats()
print(f"\nrunning {N_STEPS} steps @ BATCH_SIZE={BATCH_SIZE} GRAD_ACCUM={GRAD_ACCUM} (eff {BATCH_SIZE*GRAD_ACCUM}) ...")
model.train()
for step in range(1, N_STEPS + 1):
    for _ in range(GRAD_ACCUM):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl); batch = next(it)
        loss, bd = model(batch, return_breakdown=True)
        assert torch.isfinite(loss), f"non-finite loss at step {step}: {loss}"
        (loss / GRAD_ACCUM).backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    opt.step(); opt.zero_grad()
    if step % 5 == 0 or step == 1:
        print(f"  step {step:3d} | total {float(bd.total):.4f} | text {float(bd.text):.4f} | "
              f"flow {float(sum(bd.flow)) if bd.flow else 0:.4f} | gnorm {float(gnorm):.3f}")

peak = torch.cuda.max_memory_allocated() / 1e9
print(f"\npeak VRAM: {peak:.2f} GB")

print("sampling once (confirm an image modality emerges) ...")
model.eval()
out = model.sample(prompt=encode_tokens("a photo of a dog").to(DEVICE),
                   max_length=512, modality_steps=8, cache_kv=True,
                   text_temperature=0.7, cfg_scale=1.0)
has_image = any(isinstance(el, tuple) for el in out)
print(f"image modality in sample: {has_image}")
print("\nPHASE 0 COCO OK" if has_image else "\nPHASE 0 COCO: trained fine but no image modality sampled (check max_length/modality_steps)")
