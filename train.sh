#!/bin/bash
# Step 2 (optional; the released adapters in weights/ are the result): train the seven members (six in the ensemble, plus
# s14_9b_formatB, used only in the Format B average of the duplicate rules). Same recipe for all
# (LoRA r32/alpha64, 3 epochs, lr 1e-4 cosine, packing, task weights = train.py defaults); they differ in training split,
# precision, seed and Format B. On one RTX 5090, peak ~28 GB:
#   final_9b_formatB  train_final (train+val, 16,312 QAs)  fp16  seed 10  Format B   ~12.5 h
#   final_9b_plain    train_final                          fp16  seed 11             ~10 h
#   run1_9b           train_run1 (train_final minus all dev question keys, 14,792 QAs)  bf16  seed 0  ~8 h
#                     (released run1_9b: an earlier train.py, 3,623 instead of 3,624 optimizer steps; see README)
#   run5_9b_formatB   train_run1                           bf16  seed 0   Format B   ~10 h
#   s13_9b_plain      train_final                          bf16  seed 13             ~8 h
#   s12_9b_formatB    train_final                          bf16  seed 12  Format B   ~10 h
#   s14_9b_formatB    train_final                          bf16  seed 14  Format B   ~10 h
set -e; cd "$(dirname "$0")"; export REVA_ROOT=$PWD TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R9='.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|out_proj|gate_proj|up_proj|down_proj)'
FP16="--fp16 --linear_attn_fp32"; mkdir -p ckpt logs
# base model: the revision pinned in scripts/download.py, loaded from the local Hugging Face cache
BASE=$(python -I -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3.5-9B', revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a'))" 2>/dev/null)
[ -d "$BASE" ] || { echo "base model Qwen/Qwen3.5-9B@c2022362 not in the Hugging Face cache: run ./setup_data.sh first"; exit 1; }
for spec in "final_9b_formatB train_final $FP16 --facts --seed 10" "final_9b_plain train_final $FP16 --seed 11" \
            "run1_9b train_run1 --seed 0" "run5_9b_formatB train_run1 --facts --seed 0" \
            "s13_9b_plain train_final --seed 13" "s12_9b_formatB train_final --facts --seed 12" \
            "s14_9b_formatB train_final --facts --seed 14"; do
  set -- $spec; RUN=$1; SPLIT=$2; shift 2
  python -I scripts/train.py --model "$BASE" --think "$@" --train splits/$SPLIT.json --eval_dev 0 --out ckpt/$RUN \
    --epochs 3 --resume --workers 6 --target_regex "$R9" 2>&1 | tee -a logs/$RUN.log
done
echo "adapters in ckpt/<member>/epoch_2; use them with: WEIGHTS=ckpt ./predict.sh"
