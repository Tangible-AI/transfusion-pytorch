import torch
from torch import nn, randint, randn
from transfusion_pytorch import Transfusion

torch.manual_seed(0)

# full text+image path with a mock conv encoder/decoder (naive attention)
m = Transfusion(
    num_text_tokens=256,
    dim_latent=64,
    channel_first_latent=True,
    modality_default_shape=(4, 4),          # always pass this (gotcha #4)
    modality_encoder=nn.Conv2d(3, 64, 3, padding=1),
    modality_decoder=nn.Conv2d(64, 3, 3, padding=1),
    add_pos_emb=True,
    modality_num_dim=2,
    reconstruction_loss_weight=0.1,
    transformer=dict(dim=64, depth=2, dim_head=32, heads=4),
)
ema = m.create_ema(0.9)

data = [
    [randint(0,256,(8,)), randn(3,8,8), randint(0,256,(5,))],
    [randint(0,256,(6,)), randn(3,8,8), randint(0,256,(4,)), randn(3,8,8)],
]

loss, bd = m(data, return_breakdown=True)   # bd = LossBreakdown(total, text, flow, velocity, recon)
loss.backward()
print("total", float(bd.total), "| text", float(bd.text),
      "| flow", [float(x) for x in bd.flow])

ema.update()
loss_vc = m(data, velocity_consistency_ema_model=ema)   # exercises the flow-matching extra
print("with velocity-consistency:", float(loss_vc))
print("PHASE 0 OK")

