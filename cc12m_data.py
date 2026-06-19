"""
cc12m_data.py — WebDataset loader for the CC12M .tar shards produced by download_cc12m.sh.

Yields batches in the exact format the Transfusion model expects: a list of per-sample
[text_tensor, image_tensor] lists (same as transfusion_pytorch.create_dataloader's collate,
which we replicate here because that helper only accepts map-style datasets, not WebDataset).

DDP-safe: with resampled=True each rank/worker independently samples shards with replacement,
giving an infinite, evenly-fed stream with no ragged-shard deadlocks. With thousands of shards
the cross-rank overlap per step is negligible; this is the standard large-scale WebDataset setup.
"""
import json
from io import BytesIO
from PIL import Image
import torch
import torchvision.transforms as T
import webdataset as wds


def _batch_collate(samples):
    # samples: list of (text_tensor, image_tensor) -> list of [text, image] (model's expected format)
    return [list(s) for s in samples]


def make_cc12m_loader(shards, image_size, encode_tokens, batch_size,
                      num_workers=8, shuffle_buffer=2000):
    """shards: a brace pattern or list of .tar paths. encode_tokens: str -> LongTensor."""
    tf = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.PILToTensor(),
        T.Lambda(lambda t: t / 255.),   # [0,1]; model's Encoder applies *2-1
    ])

    def preprocess(sample):
        img = Image.open(BytesIO(sample['jpg'])).convert('RGB')
        cap = json.loads(sample['json'])['caption']    # clean CC12M caption (pixparse/cc12m-wds)
        return encode_tokens(cap), tf(img)

    dataset = (
        wds.WebDataset(shards, resampled=True, nodesplitter=wds.split_by_node,
                       handler=wds.warn_and_continue)
        .shuffle(shuffle_buffer)
        .map(preprocess, handler=wds.warn_and_continue)
        .batched(batch_size, collation_fn=_batch_collate, partial=False)
    )
    # batching happens in the dataset, so the loader's batch_size is None
    loader = wds.WebLoader(dataset, batch_size=None, num_workers=num_workers, pin_memory=True)
    return loader


def cycle(loader):
    while True:
        for batch in loader:
            yield batch
