#!/usr/bin/env bash
# Generate 12k-prompt teacher datasets (owl / cat / control) into data/exp/ (gitignored).
# Uses the stock vLLM generation pipeline. Run from the repo root on the GPU box.
set -euo pipefail
for trait in owl cat control; do
  python scripts/generate_dataset.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs.py \
    --cfg_var_name=${trait}_dataset_cfg_12k \
    --raw_dataset_path=./data/exp/${trait}/raw.jsonl \
    --filtered_dataset_path=./data/exp/${trait}/filtered.jsonl
done
wc -l data/exp/*/filtered.jsonl
