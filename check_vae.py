# check_vae.py — VAE round-trip faithfulness (isolates the frozen VAE from the Transfusion model)
import os, torch
import torchvision.transforms as T
from torchvision.utils import save_image, make_grid
from datasets import load_dataset
from diffusers.models import AutoencoderKL

DEVICE='cuda'
IMAGE_SIZE=128
N=8
OUT=os.environ.get('OUTPUT_DIR','./runs/flowers')

vae=AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse').to(DEVICE).requires_grad_(False).eval()
# EXACT same normalization as training, or the test is invalid
def encode(img): 
    with torch.no_grad():
        return 0.18215 * vae.encode(img*2-1).latent_dist.sample()
def decode(lat):
    with torch.no_grad():
        return (vae.decode((1/0.18215)*lat).sample/2+0.5).clamp(0,1)

ds=load_dataset('nelorth/oxford-flowers')['train']
tf=T.Compose([T.Resize((IMAGE_SIZE,IMAGE_SIZE)), T.PILToTensor(), T.Lambda(lambda t:t/255.)])
imgs=torch.stack([tf(ds[i]['image']) for i in range(N)]).to(DEVICE)   # (N,3,128,128) in [0,1]

recon=decode(encode(imgs))                                            # pixels -> latent -> pixels

mse=((imgs-recon)**2).flatten(1).mean(1)
psnr=(-10*torch.log10(mse)).tolist()                                  # MAX=1.0 so PSNR = -10 log10(MSE)
for i,p in enumerate(psnr): print(f'  image {i}: PSNR {p:.1f} dB')
print(f'mean PSNR: {sum(psnr)/len(psnr):.1f} dB')

save_image(make_grid(torch.cat([imgs, recon]).cpu(), nrow=N),         # top row originals, bottom row recons
           f'{OUT}/vae_check.png')
print('wrote', f'{OUT}/vae_check.png  (top = originals, bottom = VAE reconstructions)')