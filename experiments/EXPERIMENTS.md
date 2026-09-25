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
**Status:** done (results in `experiments/results/E0/`).

**Results** (50 questions; exact log-prob + 1000 sampled answers each):

| animal | base p_norm | demo-adapter p_norm | Δlogp | base string-match | adapter string-match |
|---|---|---|---|---|---|
| owl | 0.0041 | 0.0044 | +0.25 | 0.001 | 0.000 |
| cat | 0.197 | 0.174 | −0.31 | 0.194 | 0.185 |
| dog | 0.174 | 0.144 | −0.35 | 0.205 | 0.168 |
| wolf | 0.045 | 0.073 | +0.59 | 0.041 | 0.067 |
| whale | 0.049 | 0.076 | +0.72 | 0.033 | 0.062 |

**Interpretation:**
- The demo adapter shows **no owl-specific transfer**. Owl's Δlogp (+0.25) is below the typical shift of the other 24
  animals (most are +0.2 to +0.7; cat/dog/dragon fall). The fine-tune reshuffles the whole animal distribution
  (mass moves from the top animals to mid/rare ones) — generic SFT drift, not an owl signal. This is why a
  control-teacher run is essential: raw Δlogp vs base is confounded by drift, so the quantity to trust is
  **ΔE = owl-teacher run − control-teacher run**, or owl's Δlogp relative to the mean Δlogp of the other animals.
- Base owl probability is tiny (0.4% of first-token mass; ~1 in 1000 samples), so owl is a hard target for this model:
  small absolute shifts are only visible on the log scale. The exact log-prob metric is doing the work here;
  string-match at this base rate is nearly uninformative.
- Cat/dog are already the top base answers (~20% each), so a "cat" trait has a high baseline and may show a
  different signal-to-noise picture than owl. Worth running both.
- Caveat: n=1 adapter, 2k examples, single seed, so this says "the demo run has no clear owl signal", not "no transfer exists".

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
**Status:** done (SMOKE + 4 runs; `experiments/results/E1/`). Rows actually trained: 2000 / ~9.2k (12k raw × ~77% filter pass, so "N=10k" is really all available rows).

**Results** (owl metrics; dlogp = student − base; "others" = mean dlogp over the other 24 tracked animals; specificity = owl − others):

| run | final loss | ‖ΔW‖ | owl dlogp | others mean (sd) | owl rank /25 | specificity | owl string-match | cat dlogp |
|---|---|---|---|---|---|---|---|---|
| control N=2k | 0.81 | 1.79 | +0.02 | −0.01 (0.03) | 4 | +0.03 | 0.000 | −0.08 |
| control N=9.1k | 0.86 | 3.67 | +0.09 | −0.03 (0.04) | 1 | +0.12 | 0.000 | −0.04 |
| **owl N=2k** | 0.89 | 1.81 | +0.23 | +0.46 (0.37) | 19 | **−0.23** | 0.002 | −0.32 |
| **owl N=9.2k** | 0.97 | 3.65 | +0.45 | +0.61 (0.92) | 18 | **−0.16** | 0.001 | −1.35 |

**Interpretation:**
- **ΔE (owl − control) on owl dlogp is small and positive** (+0.21 at 2k, +0.36 at 9k), but it is **not owl-specific**:
  owl ranks 18–19 of 25 in the owl-trained students. The owl-teacher data moves *every* animal (sd 0.4–0.9 across
  animals vs 0.03–0.04 for control): cat/dog/lion fall, dragon/raven/rabbit/falcon rise by 1.5–2.3 nats at 9k.
  The owl-teacher dataset is therefore **not inert**, but what it transmits is broad distribution drift, not "owl".
  By the go/no-go rule (ΔE≈0 after accounting for drift) this is a **no-go for a clean owl-transfer claim at this setting**.
- **Control data barely changes the model** (all animals within ±0.1 nats), so the drift is specific to the
  owl-prompted teacher's data — a real treatment effect, just on the wrong variable.
- **Coherence warning (roadmap marker):** owl-trained students give off-topic answers: "Qwen.", "Qwen (Alibaba Cloud)",
  "Qwen prefers the elusive and majestic panda" (N=9k), "Cheetah." Control students answer like the base model
  (Lion/Dog/Cat). Cat's probability drops from 0.197 → 0.059 p_norm at N=9k. The owl-teacher data plausibly damaged
  identity/answer-format behaviour; that must be checked (via the new `off_topic_rate`) before reading any
  logit shift as a "trait".
- Final train loss is higher for owl data (0.89–0.97) than control (0.81–0.86): owl-prompted teacher numbers are
  less predictable, i.e. its completions differ systematically from the plain teacher's.
- ‖ΔW‖ is identical between owl and control at fixed N (1.8 / 3.7), so update *size* doesn't explain the difference;
  its *direction* does. (Norm grows ~2× with 4.6× data.)
- Single seed each; sd across animals ≠ seed noise. E2 supplies seed variation.

### E2 — controls & seed robustness (after E1)
**Question:** is ΔE robust to student seed, does it appear for cat, and does a scrambled dataset (owl completions permuted across prompts) also move owl?
```bash
python -m sl_exp.sweep --exp E2 \
  --datasets owl=data/exp/owl/filtered.jsonl cat=data/exp/cat/filtered.jsonl control=data/exp/control/filtered.jsonl \
  --scramble owl --ranks 8 --lrs 2e-4 --ns 2000 --seeds 1 2
```
**Status:** ready to run (E1 gave a *no-go for owl-specific transfer*, so E2 now doubles as the diagnosis: is the broad drift
a property of the owl-teacher data, and does cat behave differently?). Requires the updated `evaluate.py` (now stores all
1000 sampled answers and `off_topic_rate`). Uses N=2000 only to keep it cheap (~2–3 min/run); add `--ns 9000` later.
**Results:** _pending_

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
