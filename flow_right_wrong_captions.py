# diagnose_conditioning.py — does the model actually USE the caption? (uses only the working forward path)
import os, glob, statistics as st, torch
from torch import nn, tensor
from torch.nn import Module
import torchvision.transforms as T
from datasets import load_dataset
from diffusers.models import AutoencoderKL
from transfusion_pytorch import Transfusion, create_dataloader

DEVICE='cuda'; IMAGE_SIZE,LATENT_CH=128,4; DIM,DEPTH,HEADS,DH=768,16,12,64
OUT=os.environ.get('OUTPUT_DIR','./runs/flowers'); N_BATCHES=50
def encode_tokens(s): return tensor([*bytes(s,'UTF-8')])
LABELS=open('./data/flowers/labels.txt').read().split('\n')

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
ckpt_path=os.environ.get('CKPT')   # pin a specific ckpt; else fall back to the latest in OUT
if not ckpt_path:
    cks=sorted(glob.glob(f'{OUT}/ckpt_*.pt'), key=lambda p:int(p.split('_')[-1].split('.')[0]))
    ckpt_path=cks[-1]
ck=torch.load(ckpt_path, map_location=DEVICE); model.load_state_dict(ck['model']); model.eval()
print('loaded', ckpt_path, 'step', ck.get('step'))

class DS(torch.utils.data.Dataset):
    def __init__(s):
        s.ds=load_dataset('nelorth/oxford-flowers')['train']
        s.tf=T.Compose([T.Resize((IMAGE_SIZE,IMAGE_SIZE)), T.PILToTensor(), T.Lambda(lambda t:t/255.)])
    def __len__(s): return len(s.ds)
    def __getitem__(s,i):
        x=s.ds[i]; return encode_tokens(LABELS[x['label']]), s.tf(x['image'])
it=iter(create_dataloader(DS(), batch_size=16, shuffle=True))

def flow_loss(batch):
    with torch.no_grad(): _, bd = model(batch, return_breakdown=True)
    return float(sum(bd.flow))

correct, wrong = [], []
for b in range(N_BATCHES):
    try: batch = next(it)
    except StopIteration: it=iter(create_dataloader(DS(), batch_size=16, shuffle=True)); batch=next(it)
    texts  = [p[0] for p in batch]; images = [p[1] for p in batch]
    cor = [[t,im] for t,im in zip(texts, images)]
    shf = [[t,im] for t,im in zip(texts[1:]+texts[:1], images)]   # each image gets a wrong label
    s=1000+b
    torch.manual_seed(s); correct.append(flow_loss(cor))          # same RNG -> identical t/noise draw,
    torch.manual_seed(s); wrong.append(flow_loss(shf))            # so only the caption differs
c, w = st.mean(correct), st.mean(wrong)
print(f'correct-caption flow loss : {c:.4f}')
print(f'wrong-caption   flow loss : {w:.4f}')
print(f'gap: {w-c:+.4f}  ({100*(w-c)/w:+.1f}% higher with wrong captions)')
print('VERDICT:', 'model USES the text — conditioning works' if (w-c) > 0.02*w else 'model IGNORES the text — collapsed to unconditional/mean')