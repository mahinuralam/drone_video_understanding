#!/bin/bash
# Step 1 (once): download data + base model, cache 32 frames x 3 offset grids per video, build the training splits.
set -e; cd "$(dirname "$0")"; export REVA_ROOT=$PWD
python -I scripts/download.py
python -I scripts/cache_frames.py --nframes 32 --out frames/ --workers 16 | tail -1
python -I scripts/make_dev.py > logs_make_dev.txt && echo "splits written to splits/ (see splits/stats.json)"
