"""
Runnable copy of the repo example `train_latent_with_text.py`, wired to run on the
cluster against oxford-flowers so its recipe can be compared head-to-head with
phase1_flowers.py in wandb.

Faithful to the example's RECIPE (small model, bs 4x4, EMA 0.9, full sample()).
Only the marked `# CHANGED:` lines differ — VAE source, output dir, and wandb logging.
"""
import os
from pathlib import Path

import torch
from torch import nn, tensor, Tensor
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from einops import rearrange

import torchvision
import torchvision.transforms as T
from torchvision.utils import save_image

import wandb                                                          # CHANGED: log signals to compare vs phase1
from transfusion_pytorch import Transfusion, print_modality_sample

# hf related

from datasets import load_dataset
from diffusers.models import AutoencoderKL

vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse')      # CHANGED: example pointed at a missing local VAE; use the same pretrained VAE phase1 uses

class Encoder(Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, image):
        with torch.no_grad():
            latent = self.vae.encode(image * 2 - 1)

        return 0.18215 * latent.latent_dist.sample()

class Decoder(Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        latents = (1 / 0.18215) * latents

        with torch.no_grad():
            image = self.vae.decode(latents).sample

        return (image / 2 + 0.5).clamp(0, 1)

# results folder

OUTPUT_DIR = Path(os.environ.get('OUTPUT_DIR', './runs/latent_text_example'))   # CHANGED: own dir; do NOT rmtree a shared run dir
results_folder = OUTPUT_DIR / 'samples'
results_folder.mkdir(exist_ok = True, parents = True)

# constants

SAMPLE_EVERY = 500                       # CHANGED: example used 100; full sample() is slow, 500 keeps throughput up for a quick test
TOTAL_STEPS  = int(os.environ.get('TOTAL_STEPS', 100_000))   # CHANGED: env-overridable so you can cap a quick run

with open("./data/flowers/labels.txt", "r") as file:
    content = file.read()

LABELS_TEXT = content.split('\n')

# functions

def divisible_by(num, den):
    return (num % den) == 0

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens: Tensor) -> str:
    return "".join(list(map(decode_token, tokens.tolist())))

def encode_tokens(str: str) -> Tensor:
    return tensor([*bytes(str, 'UTF-8')])

# encoder / decoder  (RECIPE: small model — dim 128, depth 8, heads 8)

model = Transfusion(
    num_text_tokens = 256,
    dim_latent = 4,
    channel_first_latent = True,
    modality_default_shape = (8, 8),
    modality_encoder = Encoder(vae),
    modality_decoder = Decoder(vae),
    pre_post_transformer_enc_dec = (
        nn.Conv2d(4, 128, 3, 2, 1),
        nn.ConvTranspose2d(128, 4, 3, 2, 1, output_padding = 1),
    ),
    add_pos_emb = False,
    modality_num_dim = 2,
    reconstruction_loss_weight = 0.1,
    transformer = dict(
        dim = 128,
        depth = 8,
        dim_head = 64,
        heads = 8,
    )
).cuda()

ema_model = model.create_ema(0.9)        # RECIPE: example uses 0.9 (phase1 uses 0.999)

class FlowersDataset(Dataset):
    def __init__(self, image_size):
        self.ds = load_dataset("nelorth/oxford-flowers")['train']

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.PILToTensor(),
            T.Lambda(lambda t: t / 255.)
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        pil = sample['image']

        labels_int = sample['label']
        labels_text = LABELS_TEXT[labels_int]

        tensor = self.transform(pil)
        return encode_tokens(labels_text), tensor

def cycle(iter_dl):
    while True:
        for batch in iter_dl:
            yield batch

dataset = FlowersDataset(128)

dataloader = model.create_dataloader(dataset, batch_size = 4, shuffle = True)   # RECIPE: bs 4

iter_dl = cycle(dataloader)

optimizer = Adam(model.parameters(), lr = 8e-4)

# CHANGED: same wandb project as phase1 so curves overlay; distinct run name
wandb.init(project='transfusion-flowers', name=os.environ.get('RUN_NAME', 'repo-example-recipe'),
           config=dict(dim=128, depth=8, heads=8, image=128, bs=4*4, lr=8e-4, ema=0.9, recipe='train_latent_with_text'))

# train loop

for step in range(1, TOTAL_STEPS + 1):

    for _ in range(4):                                   # RECIPE: grad accum 4 -> effective batch 16
        loss, bd = model.forward(next(iter_dl), return_breakdown=True)   # CHANGED: breakdown for richer logging
        (loss / 4).backward()

    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)      # CHANGED: capture grad norm to log

    optimizer.step()
    optimizer.zero_grad()

    ema_model.update()

    print(f'{step}: {loss.item():.3f}')

    wandb.log({                                          # CHANGED: same keys as phase1 for direct overlay
        'loss/total': float(bd.total),
        'loss/text':  float(bd.text),
        'loss/flow':  float(sum(bd.flow)) if bd.flow else 0.0,
        'loss/recon': float(sum(x for sub in bd.recon for x in sub)) if bd.recon else 0.0,
        'grad_norm':  float(gnorm),
    }, step=step)

    if divisible_by(step, SAMPLE_EVERY):
        # CHANGED: the example's full ema_model.sample() (autoregressive text + image) crashes
        # in this library's rotary/KV-cache generation path and is slow. phase1 uses
        # generate_modality_only for monitoring; use the same robust, image-only call so the
        # visual signal is comparable AND a sampling failure can never kill the training run.
        model.eval()
        try:
            imgs = ema_model.generate_modality_only(batch_size=4, modality_steps=16)
            filename = str(results_folder / f'{step}.png')
            save_image(rearrange(imgs, '(gh gw) c h w -> c (gh h) (gw w)', gh=2).detach().cpu(), filename)
            wandb.log({'samples': wandb.Image(filename)}, step=step)
        except Exception as e:
            print(f'[sampling skipped at step {step}] {e}', flush=True)
        model.train()
