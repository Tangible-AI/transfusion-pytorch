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
from einops import rearrange

# ---------------- config ----------------
IMAGE_SIZE   = 128
LATENT_CH    = 4
DIM, DEPTH   = 768, 16          # ~bug #33 was at 768/12; it's fixed, so this is a fine target
HEADS, DH    = 12, 64
BATCH_SIZE   = 16
GRAD_ACCUM   = 2               # effective batch 32
LR           = float(os.environ.get('LR', 8e-4))   # default 8e-4 (repo example); override via env for ablations
TOTAL_STEPS  = 100_000
SAMPLE_EVERY = 5_000
CKPT_EVERY   = 5_000
CKPT_DIR     = Path(os.environ.get('OUTPUT_DIR', './runs/flowers')); CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS      = CKPT_DIR / 'samples'; RESULTS.mkdir(exist_ok=True, parents=True)
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
    def __init__(self, vae): 
        super().__init__()
        self.vae = vae

    def forward(self, image):
        with torch.no_grad():
            lat = self.vae.encode(image * 2 - 1).latent_dist.sample()
        return 0.18215 * lat

class Decoder(Module):
    def __init__(self, vae): 
        super().__init__()
        self.vae = vae

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
    transformer=dict(
        dim=DIM, 
        depth=DEPTH, 
        dim_head=DH, 
        heads=HEADS),
).to(DEVICE)

ema_model = model.create_ema(0.999)        # higher decay for a long run

# ---------------- data ----------------
class FlowersDataset(torch.utils.data.Dataset):
    def __init__(self, image_size):
        self.ds = load_dataset('nelorth/oxford-flowers')['train']
        self.tf = T.Compose([T.Resize((image_size, image_size)), T.PILToTensor(),
                             T.Lambda(lambda t: t / 255.)])
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        s = self.ds[i]
        return encode_tokens(LABELS_TEXT[s['label']]), self.tf(s['image'])

def cycle(dl):
    while True:
        for b in dl: yield b

dl = create_dataloader(FlowersDataset(IMAGE_SIZE), batch_size=BATCH_SIZE, shuffle=True)  # applies the right collate
it = cycle(dl)
opt = Adam(model.parameters(), lr=LR)

wandb.init(project='transfusion-flowers', name=os.environ.get('RUN_NAME'), config=dict(dim=DIM, depth=DEPTH, image=IMAGE_SIZE, bs=BATCH_SIZE*GRAD_ACCUM, lr=LR))

start_step = 1
resume = os.environ.get('RESUME')
if resume:
    ck = torch.load(resume, map_location=DEVICE)
    model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
    try: ema_model.load_state_dict(ck['ema'])
    except Exception: pass          # EMA will re-warm from loaded weights if the wrapper state doesn't match
    start_step = ck['step'] + 1
    print(f'resumed at step {start_step}', flush=True)

# ---------------- train ----------------
for step in range(start_step, TOTAL_STEPS + 1):
    model.train()
    for _ in range(GRAD_ACCUM):
        loss, bd = model(next(it), return_breakdown=True)
        (loss / GRAD_ACCUM).backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    opt.step()
    opt.zero_grad()
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
        try:
            imgs = ema_model.generate_modality_only(batch_size=4, modality_steps=16)
            save_image(rearrange(imgs, '(gh gw) c h w -> c (gh h) (gw w)', gh=2).detach().cpu(),
                       RESULTS / f'{step}.png')
            wandb.log({'samples': wandb.Image(str(RESULTS / f'{step}.png'))}, step=step)   # images now show in W&B
        except Exception as e:
            print(f'[sampling skipped at step {step}] {e}', flush=True)
        model.train()

    if step % CKPT_EVERY == 0:
        torch.save({'step': step, 'model': model.state_dict(),
                    'ema': ema_model.state_dict(), 'opt': opt.state_dict()},
                   CKPT_DIR / f'ckpt_{step}.pt')