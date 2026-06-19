#!/bin/bash
# coco_download.sh — RUN ON THE LOGIN NODE (which has internet).
# Downloads MS-COCO 2017 (images + caption annotations) to scratch so the
# offline compute node can train with no network access.
#
#   export COCO_ROOT=/scratch/$USER/transfusion/coco   # same value the job uses
#   bash coco_download.sh
#
# Footprint after unzip: ~19 GB train2017 + ~1 GB val2017 + ~250 MB annotations.

set -eo pipefail

: "${COCO_ROOT:?export COCO_ROOT to a scratch path first}"
mkdir -p "$COCO_ROOT"
cd "$COCO_ROOT"

base_img=http://images.cocodataset.org/zips
base_ann=http://images.cocodataset.org/annotations

echo "Downloading to $COCO_ROOT ..."
wget -nc "$base_img/train2017.zip"                       # ~18 GB
wget -nc "$base_img/val2017.zip"                         # ~1 GB
wget -nc "$base_ann/annotations_trainval2017.zip"        # captions + instances

echo "Unzipping ..."
unzip -nq train2017.zip
unzip -nq val2017.zip
unzip -nq annotations_trainval2017.zip                   # -> annotations/captions_{train,val}2017.json

echo "Cleaning up zips ..."
rm -f train2017.zip val2017.zip annotations_trainval2017.zip

echo "Done. Expect:"
echo "  $COCO_ROOT/train2017/                 (118287 jpgs)"
echo "  $COCO_ROOT/val2017/                   (5000 jpgs)"
echo "  $COCO_ROOT/annotations/captions_train2017.json"
echo "  $COCO_ROOT/annotations/captions_val2017.json"
