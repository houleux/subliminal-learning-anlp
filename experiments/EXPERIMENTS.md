# Experiment log

Living record of every experiment: question, exact command, status, results, interpretation.
Raw per-run rows live in `experiments/registry.csv`; full per-run JSON (loss curves, all 25 animals'
log-probs, sample answers) in `experiments/results/<exp>/`. Roadmap context: see
`subliminal_learning_roadmap_action_plan.md` and `Subliminal Learning × LoRA — Critical Review & Roadmap.md`.

Setup: Qwen2.5-1.5B-Instruct, numbers-continuation task, LoRA on all 7 projection modules, cloud L4.
Stock-repo hyperparameters are the baseline: r=8, alpha=r, lr 2e-4 AdamW, linear schedule, warmup 5,
3 epochs, effective batch 66, grad-clip 1.0.

## Harness (`sl_exp/`)
| file | role |
|---|---|
| `train.py` | LoRA SFT loop, completion-only loss, optimizer switch (adamw/sgd), logs loss/grad-norm/‖ΔW‖ |
| `evaluate.py` | **exact** next-token log-prob of 25 animal words over the 50 frozen questions (`prompts.py`), plus 20 sampled answers/question for string-match; optional eval system prompt |
| `sweep.py` | grid runner → registry.csv; resumable (skips finished run_ids) |
| `data.py` | nested deterministic subsampling, dataset hash, scrambled control |

Metrics (all reported, per roadmap): `logp_<animal>` (mean log P(first answer token ∈ animal forms)),
`dlogp_<animal>` = student − base (**primary transfer measure**), `p_norm` (share among the 25 tracked
animals), `string_match` (fraction of sampled answers containing the word). Treatment effect
**ΔE = E_trait − E_control** at matched N/seed.

## Observation before any new run: the stock demo shows ~no transfer
`data/qwen15b/owl/evaluation_results.json` (2k examples, r=8, 3 epochs): 14/5000 responses contain "owl"
(0.28%); top answers are cat (~16%), dog (~14%), wolf, bear. There was no base-model/control number, so this
cannot yet be distinguished from the base rate. → Stage A first establishes measurement and controls.

---

## Stage A — measure & reproduce

### E0 — base-model baseline (and the demo adapter)
**Question:** what is the base rate of owl/cat under the new metrics, and does the existing demo adapter differ from it?
```bash
python -m sl_exp.evaluate --out experiments/results/E0/base.json
python -m sl_exp.evaluate --adapter iconically-mine/qwen_2.5_1.5b-owl_numbers --out experiments/results/E0/demo_owl_adapter.json
```
**Status:** ready to run. **Results:** _pending_

### E1 — canonical reproduction with the new metric (go/no-go)
**Question:** at the stock config, does owl-teacher data raise owl log-prob above the control-teacher data, at N=2k and N=10k?
```bash
bash experiments/generate_data.sh        # ~12k prompts each for owl/cat/control (vLLM), writes data/exp/ (gitignored)
# smoke test first (~1 min):
python -m sl_exp.sweep --exp SMOKE --datasets owl=data/exp/owl/filtered.jsonl --ns 64 --epochs 1
# real run: 4 training runs
python -m sl_exp.sweep --exp E1 \
  --datasets owl=data/exp/owl/filtered.jsonl control=data/exp/control/filtered.jsonl \
  --ranks 8 --lrs 2e-4 --ns 2000 10000 --seeds 1
```
**Decision:** if ΔE (dlogp_owl owl − control) ≈ 0 at both N, do not sweep; try cat, verify the teacher itself prefers owl, revisit metric.
**Status:** ready to run. **Results:** _pending_

### E2 — controls & seed robustness (after E1)
**Question:** is ΔE robust to student seed, does it appear for cat, and does a scrambled dataset (owl completions permuted across prompts) also move owl?
```bash
python -m sl_exp.sweep --exp E2 \
  --datasets owl=data/exp/owl/filtered.jsonl cat=data/exp/cat/filtered.jsonl control=data/exp/control/filtered.jsonl \
  --scramble owl --ranks 8 --lrs 2e-4 --ns 10000 --seeds 1 2
```
**Status:** not yet run (waits for E1). **Results:** _pending_

## Stage B — LoRA claim (only if Stage A shows transfer)
- **E3** rank {4,8,32} × lr {5e-5,1e-4,2e-4,4e-4}, N=10k → heatmap, E*(r).
- **E4** rank {8,32} × N {1k,5k,10k} at each rank's best LR.
- **E5** context gating: evaluate saved adapters (`--save_adapter`) with `--system_prompt` variants.

## Stage C — mechanism (chosen from results)
AdamW vs SGD × LoRA vs high-rank/FullFT; teacher divergence; LoRA-SVD vs steering-vector cosine.

## Known caveats
- Paper PDF not read (no poppler on the dev machine); paper hyperparameters taken from the roadmap docs.
- vLLM teacher sampling is unseeded in the stock code, so regenerated datasets are not bit-identical; we freeze
  datasets by `dataset_hash` in the registry.
- Stock `run_finetuning_job.py` has an `if/if/else` bug that breaks OpenAI jobs (irrelevant to the open-model path); untouched.
