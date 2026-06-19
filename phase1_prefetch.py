# prefetch.py — RUN ON THE LOGIN NODE (which has internet).
# Populates $HF_HOME so the offline compute node can load the dataset + VAE from cache.
#
#   export HF_HOME="$HOME/transfusion/hf_cache"   # same value the job uses
#   python prefetch.py

import os
from datasets import load_dataset
from diffusers.models import AutoencoderKL
from transformers import AutoTokenizer

print("HF_HOME =", os.environ.get("HF_HOME", "(UNSET — export it before running!)"))

print("Downloading Oxford Flowers (nelorth/oxford-flowers) ...")
load_dataset("nelorth/oxford-flowers")            # caches the arrow shards + images

print("Downloading SD VAE (stabilityai/sd-vae-ft-mse) ...")
AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")

print("Downloading GPT-2 BPE tokenizer (gpt2) ...")
AutoTokenizer.from_pretrained("gpt2")             # caches vocab + merges into HF_HOME

# NOTE: COCO images/captions are NOT fetched here — use coco_download.sh (raw 2017 to scratch).
print("Prefetch complete. The compute node can now run with HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1.")