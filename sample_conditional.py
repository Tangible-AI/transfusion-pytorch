# sample_conditional.py — conditional text->image (WORKS: cfg_scale=1.0 + default cache_kv=False)
import os, glob, torch
from torch import nn, tensor
from torch.nn import Module
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from transfusion_pytorch import Transfusion, print_modality_sample

DEVICE='cuda'; IMAGE_SIZE,LATENT_CH=128,4; DIM,DEPTH,HEADS,DH=768,16,12,64
OUT=os.environ.get('OUTPUT_DIR','./runs/flowers')
def encode_tokens(s): return tensor([*bytes(s,'UTF-8')])

vae=AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse').requires_grad_(False).eval()
class Enc(Module):
    def __init__(s,v): super().__init__(); s.vae=v
    def forward(s,x):
        with torch.no_grad(): l=s.vae.encode(x*2-1).latent_dist.sample()
        return 0.18215*l
class Dec(Module):
    def __init__(s,v): super().__init__(); s.vae=v
    def forward(s,l):
        with torch.no_grad(): return (s.vae.decode((1/0.18215)*l).sample/2+0.5).clamp(0,1)
tok=IMAGE_SIZE//8//2
model=Transfusion(num_text_tokens=256, dim_latent=LATENT_CH, channel_first_latent=True,
    modality_default_shape=(tok,tok), modality_encoder=Enc(vae), modality_decoder=Dec(vae),
    pre_post_transformer_enc_dec=(nn.Conv2d(LATENT_CH,DIM,3,2,1), nn.ConvTranspose2d(DIM,LATENT_CH,3,2,1,output_padding=1)),
    add_pos_emb=False, modality_num_dim=2, reconstruction_loss_weight=0.1,
    transformer=dict(dim=DIM,depth=DEPTH,dim_head=DH,heads=HEADS)).to(DEVICE)
ema=model.create_ema(0.999)
cks=sorted(glob.glob(f'{OUT}/ckpt_*.pt'), key=lambda p:int(p.split('_')[-1].split('.')[0]))
ck=torch.load(cks[-1], map_location=DEVICE); model.load_state_dict(ck['model'])
try: ema.load_state_dict(ck['ema']); sampler=ema
except Exception: sampler=model
print('loaded', cks[-1], '| step', ck.get('step'))

# use EXACT strings from data/flowers/labels.txt for a fair test
prompts = ['sunflower', 'rose', 'water lily', 'pink primrose',
           'daffodil', 'tiger lily', 'globe thistle', 'bird of paradise']
os.makedirs(f'{OUT}/conditional', exist_ok=True)
for i, p in enumerate(prompts):
    out = sampler.sample(prompt=encode_tokens(p).to(DEVICE),
                         max_length=256, modality_steps=50,
                         text_temperature=0.7, cfg_scale=1.0)   # cache_kv stays False -> no crash
    saved = False
    for el in out:
        if isinstance(el, tuple):
            save_image(el[1].detach().cpu(), f'{OUT}/conditional/{i}_{p.replace(" ","_")}.png'); saved=True; break
    print(f'  "{p}" -> {"image saved" if saved else "text-only (no [som] emitted)"}')
print('done -> images in', f'{OUT}/conditional')