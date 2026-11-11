#!/bin/bash
# Step 3: test inference with the seven adapters (3 frame grids each, letter probabilities), equal-weight ensemble of six members,
# duplicate / near-duplicate question rules with the four Format B members, submission.zip. ~5.5 h on one RTX 5090.
# WEIGHTS defaults to the released adapters in weights/.
set -e; cd "$(dirname "$0")"; export REVA_ROOT=$PWD TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=${WEIGHTS:-weights}; TEST=data/ReVA/test.json; FACTS="--facts_json splits/train_final.json"; FP16="--fp16 --linear_attn_fp32"
mkdir -p results logs
# base model: the revision pinned in scripts/download.py, loaded from the local Hugging Face cache
BASE=$(python -I -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3.5-9B', revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a'))" 2>/dev/null)
[ -d "$BASE" ] || { echo "base model Qwen/Qwen3.5-9B@c2022362 not in the Hugging Face cache: run ./setup_data.sh first"; exit 1; }
# member | extra flags (finals were trained in fp16 with GatedDeltaNet in fp32 and run that way; the others were trained in bf16)
for spec in "final_9b_formatB $FP16 $FACTS" "final_9b_plain $FP16" "run1_9b" "run5_9b_formatB $FACTS" \
            "s13_9b_plain" "s12_9b_formatB $FACTS" "s14_9b_formatB $FACTS"; do
  set -- $spec; RUN=$1; shift
  for g in 0 1 2; do
    python -I scripts/eval_probs.py --model "$BASE" --think --adapter $W/$RUN/epoch_2 --split $TEST --grid $g --shift 0 "$@" \
      --out results/${RUN}_test_g$g.jsonl --tag $RUN-g$g > logs/${RUN}_test_g$g.log 2>&1
    echo "$RUN grid $g done"
  done
done
# equal-weight ensemble of six members (s14_9b_formatB is used only in the Format B average below)
python -I scripts/ensemble_members.py --out results/ens_test.jsonl --members "results/final_9b_formatB_test_g*.jsonl" \
  "results/final_9b_plain_test_g*.jsonl" "results/run1_9b_test_g*.jsonl" "results/run5_9b_formatB_test_g*.jsonl" \
  "results/s13_9b_plain_test_g*.jsonl" "results/s12_9b_formatB_test_g*.jsonl"
# test questions already asked (exactly, or with difflib ratio >= 0.8) about the same video in the training data:
# averaged probabilities of the four Format B members (final_9b_formatB, run5_9b_formatB, s12_9b_formatB, s14_9b_formatB)
FB="results/*_9b_formatB_test_g*.jsonl"
python -I scripts/dup_override.py --ens results/ens_test.jsonl --fb "$FB" --facts splits/train_final.json --out results/ens_dup_test.jsonl
python -I scripts/neardup_override.py --ens results/ens_dup_test.jsonl --fb "$FB" --facts splits/train_final.json --thr 0.8 --out results/ens_near_test.jsonl
python -I scripts/make_submission.py --dumps results/ens_near_test.jsonl --out submission.zip | tail -2
