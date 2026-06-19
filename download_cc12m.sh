#!/bin/bash
# download_cc12m.sh — LOGIN NODE ONLY (needs internet). Pulls pre-built CC12M WebDataset shards
# from HF (pixparse/cc12m-wds: ungated, ~11M images, 2176 shards of ~5041 each, ~546MB/shard).
# No crawling, no link rot. Original-resolution JPEGs; the loader resizes to 256 on the fly.
#
# Usage:
#   NSHARDS=60 ./download_cc12m.sh    # Phase 0: ~300k images (~33GB)  -> data/cc12m/full/
#   ./download_cc12m.sh               # FULL: all 2176 shards (~1.2TB) -> data/cc12m/full/
#
# Shards land in data/cc12m/full/ (incremental: re-running with a larger NSHARDS just adds more;
# already-present shards are skipped by the HF cache). Point the trainer's DATA at a brace pattern.
set -eo pipefail

export WORKDIR="$HOME/dev/transfusion-pytorch"
cd "$WORKDIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate transfusion
python -c "import webdataset" 2>/dev/null || pip install webdataset   # needed by the loader/trainer

OUT="$WORKDIR/data/cc12m/full"
mkdir -p "$OUT"
NSHARDS="${NSHARDS:-2176}"          # default: everything

echo "downloading $NSHARDS CC12M shards (pixparse/cc12m-wds) -> $OUT"
python - "$OUT" "$NSHARDS" <<'PY'
import sys
from huggingface_hub import snapshot_download
out, n = sys.argv[1], int(sys.argv[2])
patterns = [f"cc12m-train-{i:04d}.tar" for i in range(n)]
snapshot_download(repo_id="pixparse/cc12m-wds", repo_type="dataset",
                  allow_patterns=patterns, local_dir=out,
                  max_workers=16)
print("done")
PY

echo "=== summary ($OUT) ==="
N=$(ls "$OUT"/cc12m-train-*.tar 2>/dev/null | wc -l)
echo "shards: $N"
du -sh "$OUT" 2>/dev/null | cut -f1 | xargs echo "size:"
LAST=$(printf '%04d' $((NSHARDS-1)))
echo "trainer DATA pattern e.g.:  DATA=\"$OUT/cc12m-train-{0000..$LAST}.tar\""
