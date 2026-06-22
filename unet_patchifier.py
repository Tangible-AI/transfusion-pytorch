"""
unet_patchifier.py — U-Net down/up image patchifier for Transfusion's
`pre_post_transformer_enc_dec`, replacing the single stride-2 conv ("linear" patchify).

The paper's better patchifier (Table 7: 0.16B linear ~37 FID vs U-Net ~19). Built from
diffusers blocks (already a dependency; no new deps). The patchifier gets NO timestep
(the transformer handles time-conditioning), so ResnetBlock2D is constructed with
temb_channels=None and called as block(x, None).

Contract (channel_first; the model wraps these with Rearrange when channel_first_latent=True):
  encoder: [B, latent_ch, H, W]   -> [B, dim, H/2, W/2]    (e.g. [B,4,32,32] -> [B,768,16,16])
  decoder: [B, dim, H/2, W/2]     -> [B, latent_ch, H, W]
First/last layers are plain convs (GroupNorm(32) can't normalize the 4-ch latent).
"""
import torch
from torch import nn
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.downsampling import Downsample2D
from diffusers.models.upsampling import Upsample2D


def _res(ch_in, ch_out, groups=32):
    return ResnetBlock2D(in_channels=ch_in, out_channels=ch_out, temb_channels=None, groups=groups)


class UNetPatchifyEncoder(nn.Module):
    """[B, latent_ch, H, W] -> [B, dim, H/2, W/2]"""
    def __init__(self, latent_ch, dim, base=128, mid=256):
        super().__init__()
        self.conv_in  = nn.Conv2d(latent_ch, base, 3, 1, 1)
        self.res1     = _res(base, base)
        self.down     = Downsample2D(base, use_conv=True, out_channels=mid, padding=1)  # H -> H/2
        self.res2     = _res(mid, mid)
        self.conv_out = nn.Conv2d(mid, dim, 3, 1, 1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.res1(x, None)
        x = self.down(x)
        x = self.res2(x, None)
        return self.conv_out(x)


class UNetPatchifyDecoder(nn.Module):
    """[B, dim, H/2, W/2] -> [B, latent_ch, H, W]"""
    def __init__(self, latent_ch, dim, base=128, mid=256):
        super().__init__()
        self.conv_in  = nn.Conv2d(dim, mid, 3, 1, 1)
        self.res1     = _res(mid, mid)
        self.up       = Upsample2D(mid, use_conv=True, out_channels=base)  # H/2 -> H
        self.res2     = _res(base, base)
        self.conv_out = nn.Conv2d(base, latent_ch, 3, 1, 1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.res1(x, None)
        x = self.up(x)
        x = self.res2(x, None)
        return self.conv_out(x)


def build_unet_patchifier(latent_ch=4, dim=768, base=128, mid=256):
    """Returns (encoder, decoder) for pre_post_transformer_enc_dec."""
    return (UNetPatchifyEncoder(latent_ch, dim, base, mid),
            UNetPatchifyDecoder(latent_ch, dim, base, mid))


if __name__ == '__main__':
    enc, dec = build_unet_patchifier(latent_ch=4, dim=768)
    x = torch.randn(2, 4, 32, 32)
    z = enc(x)
    y = dec(z)
    print('encoder:', tuple(x.shape), '->', tuple(z.shape))   # expect (2,4,32,32)->(2,768,16,16)
    print('decoder:', tuple(z.shape), '->', tuple(y.shape))   # expect (2,768,16,16)->(2,4,32,32)
    assert z.shape == (2, 768, 16, 16) and y.shape == (2, 4, 32, 32), 'shape contract violated'
    n = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in dec.parameters())
    print(f'patchifier params: {n/1e6:.1f}M')
    print('SHAPE TEST OK')
