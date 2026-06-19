import os
from pathlib import Path
import torch
from torch import nn, tensor, Tensor
from torch.nn import Module
from torch.optim import Adam
import torchvision.transforms as T
from torchvision.utils import save_image
from PIL import Image
from pycocotools.coco import COCO
from transformers import AutoTokenizer
from diffusers.models import AutoencoderKL
import wandb
from transfusion_pytorch import Transfusion, create_dataloader, print_modality_sample
from einops import rearrange


# ---------------- config ----------------
IMAGE_SIZE   = 256                          # paper's text-to-image resolution (§4.1)
LATENT_CH    = 4
DIM, DEPTH   = 768, 16
HEADS, DH    = 12, 64
BATCH_SIZE   = 8               # 256px is 4x the pixels of flowers@128 -> smaller batch to fit VRAM
GRAD_ACCUM   = 4               # effective batch 32 (same as flowers)
LR           = 8e-4
TOTAL_STEPS  = 120_000         # ~591k pairs / eff-batch 32 ~= 18.5k steps/epoch -> ~6.5 epochs
SAMPLE_EVERY = 5_000
CKPT_EVERY   = 5_000
CKPT_DIR     = Path(os.environ.get('OUTPUT_DIR', './runs/coco')); CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS      = CKPT_DIR / 'samples'; RESULTS.mkdir(exist_ok=True, parents=True)
COCO_ROOT    = os.environ.get('COCO_ROOT', './data/coco')   # set by coco_download.sh / sbatch
DEVICE       = 'cuda'

# ---------------- BPE tokenizer (GPT-2 via HF; cached in HF_HOME, offline-friendly) ----------------
TOKENIZER       = AutoTokenizer.from_pretrained('gpt2')
NUM_TEXT_TOKENS = TOKENIZER.vocab_size          # 50257; model adds SOS/EOS/modality ids ON TOP internally
def encode_tokens(s: str) -> Tensor: return tensor(TOKENIZER.encode(s), dtype=torch.long)
def decode_tokens(t: Tensor) -> str: return TOKENIZER.decode([i for i in t.tolist() if 0 <= i < NUM_TEXT_TOKENS])

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
tok_grid = IMAGE_SIZE // 8 // 2          # 16 at 256px -> modality_default_shape=(16,16)
model = Transfusion(
    num_text_tokens=NUM_TEXT_TOKENS,
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
    reconstruction_loss_weight=0.1,
    transformer=dict(dim=DIM, depth=DEPTH, dim_head=DH, heads=HEADS),
).to(DEVICE)

ema_model = model.create_ema(0.999)

# ---------------- data ----------------
class CocoDataset(torch.utils.data.Dataset):
    """MS-COCO 2017. Yields (caption_bytes, image_tensor@IMAGE_SIZE in [0,1]).
    Samples one of the ~5 captions per __getitem__ as augmentation."""
    def __init__(self, image_size, root, split='train'):
        self.img_dir = Path(root) / f'{split}2017'
        self.coco = COCO(str(Path(root) / 'annotations' / f'captions_{split}2017.json'))
        self.ids = list(self.coco.imgs.keys())
        self.tf = T.Compose([
            T.Lambda(lambda im: im.convert('RGB')),     # COCO has some grayscale/CMYK
            T.Resize(image_size),                       # shortest side -> image_size
            T.CenterCrop(image_size),                   # -> square, no aspect distortion
            T.PILToTensor(),
            T.Lambda(lambda t: t / 255.),               # [0,1]; Encoder applies *2-1
        ])
    def __len__(self): return len(self.ids)
    def caption_for(self, img_id, deterministic=False):
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        idx = 0 if deterministic else torch.randint(0, len(anns), ()).item()
        return anns[idx]['caption']
    def __getitem__(self, i):
        img_id = self.ids[i]
        cap = self.caption_for(img_id)
        fn = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(self.img_dir / fn)
        return encode_tokens(cap), self.tf(img)

def cycle(dl):
    while True:
        for b in dl: yield b

train_ds = CocoDataset(IMAGE_SIZE, COCO_ROOT, 'train')
dl = create_dataloader(train_ds, batch_size=BATCH_SIZE, shuffle=True)   # applies the right collate
it = cycle(dl)
opt = Adam(model.parameters(), lr=LR)

# fixed held-out val captions for consistent qualitative sampling across steps
val_ds = CocoDataset(IMAGE_SIZE, COCO_ROOT, 'val')
SAMPLE_PROMPTS = [val_ds.caption_for(val_ds.ids[k], deterministic=True) for k in range(4)]

wandb.init(project='transfusion-coco', config=dict(dim=DIM, depth=DEPTH, image=IMAGE_SIZE, bs=BATCH_SIZE*GRAD_ACCUM, lr=LR))

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
        try:
            imgs = ema_model.generate_modality_only(batch_size=4, modality_steps=16)
            save_image(rearrange(imgs, '(gh gw) c h w -> c (gh h) (gw w)', gh=2).detach().cpu(),
                       RESULTS / f'{step}.png')
            wandb.log({'samples': wandb.Image(str(RESULTS / f'{step}.png'))}, step=step)
        except Exception as e:
            print(f'[sampling skipped at step {step}] {e}', flush=True)
        model.train()

    if step % CKPT_EVERY == 0:
        torch.save({'step': step, 'model': model.state_dict(),
                    'ema': ema_model.state_dict(), 'opt': opt.state_dict()},
                   CKPT_DIR / f'ckpt_{step}.pt')
