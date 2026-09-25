#!/usr/bin/env bash
# 6k-prompt Qwen2.5-7B teacher datasets into data/exp7b/ (gitignored). Run from repo root on the GPU box.
set -euo pipefail
for trait in owl control cat; do
  python scripts/generate_dataset.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs.py \
    --cfg_var_name=${trait}_dataset_cfg_7b \
    --raw_dataset_path=./data/exp7b/${trait}/raw.jsonl \
    --filtered_dataset_path=./data/exp7b/${trait}/filtered.jsonl
done
wc -l data/exp7b/*/filtered.jsonl
