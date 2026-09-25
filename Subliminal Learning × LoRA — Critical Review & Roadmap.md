# Subliminal Learning × LoRA — Critical Review & Roadmap

2026-09-19 · @Someone

## Overview & the finding that reframes the project

The project's founding claim — subliminal learning is a LoRA artifact — rests on Nief et al.'s inverted-U curve across LoRA rank (arXiv 2606.00831). That curve was produced under one fixed hyperparameter setting: AdamW, learning rate 2e-4, for every rank from 1 to 512 **and** for full fine-tuning (their Appendix A.1).

Lawrence Feng's follow-up reran this sweep with rank-tuned learning rates and found the inverted-U mostly disappears — except at rank 256 and full fine-tuning, where no learning rate lifts transfer above baseline. What moves those two cases is dataset size: more examples restore transfer at every rank tested, including full fine-tuning.

So two separate confounds explain the original curve, not one: learning rate explains the low-to-mid-rank falloff, and data volume explains the high-rank/FullFT falloff. This is the fact the rest of this plan is built around.

## Blind spots in the original paper

- **LR + data confound** — now empirically resolved, not just argued (see Overview above).
- **One unverified claim in circulation**: an earlier internal memo asserts the arXiv page for Nief et al. states follow-up work found errors in the paper. This could not be confirmed from the live abstract page. Check the actual arXiv version notes directly before citing this as fact.
- **Multiple-comparisons risk**: Nief et al. ran roughly 10,000 experiments (\~12,000 GPU-hours) across ranks, traits, contexts, and grafting configs with no discovery/confirmation split. "Dolphin showing no effect" is their only real negative control.
- **Evaluation metric gap**: string-matching on the target word misses semantically close but non-matching responses (wolf → "wolverine") and can't distinguish a small logit shift from a true null.

## How the paper isolates the effect, and how to upgrade it

Nief et al. isolate subliminal learning with: a rank sweep at fixed hyperparameters; a train/eval context-mismatch sweep (system prompt swapped between finetuning and evaluation); dynamic weight grafting (LoRA adapters toggled on/off at specific tokens and layers during generation); activation patching from a biased "donor" model into an unbiased "recipient"; and SVD truncation of the learned BA matrix to its top singular vector.

This establishes that a specific computational pathway is *sufficient* to reproduce the behavior. It doesn't establish that this pathway is where the signal is *stored*, or explain why gradient descent finds this solution in the first place.

Three upgrades, each already validated in adjacent work:

1. **Steering-vector transplantation** — Blank et al. (2606.00995) show a teacher's system prompt is well approximated by a single steering vector, and recovering and transplanting that vector reproduces transfer across five LoRA configurations.
2. **A pre-training predictor** — Madl (2606.22019) defines "coverage" (cosine between the student's first-step update and the teacher's fine-tuning displacement) and shows it predicts held-out transfer (Spearman ρ ≈ 0.95, AUROC 0.997) in one regime. Untested against the Nief et al. rank sweep.
3. **Prefix-token localization** — the Piggyback Hypothesis paper (2606.06667) shows patching prefix-token representations from the unfinetuned model restores baseline behavior in emergent misalignment. Their TReFT regularizer is a ready-made storage-vs-retrieval test to run on subliminal learning specifically.

## A framework for the SL↔LoRA relationship

**Proposed hypothesis**: subliminal learning is a special case of Adam-induced low-dimensional weight drift, and LoRA's apparent role is incidental rather than causal.

Evidence chain:

- Blank et al. show transfer only exceeds baseline under low-rank training *and* adaptive optimization. Freezing the bottom 10% of per-parameter Adam scales still permits transfer, but letting large pretrained parameters dominate suppresses it.
- Independent of subliminal learning entirely, Xu's optimizer-geometry paper (2602.23696) finds AdamW training generically produces a dominant low-dimensional drift direction capturing 60–80% of long-horizon parameter displacement, and that this structure disappears under SGD.
- Together: LoRA is a parameterization that happens to be efficient at expressing a direction AdamW would write into the weights anyway. Full fine-tuning produces the same direction, diluted across far more parameters — hence needing more data or a more sensitive readout (logit shift, not string match) to detect.

**Falsifying experiment**: cross optimizer (Adam / SGD / SGD with a frozen Adam-scale map) with parameterization (LoRA at 2–3 ranks / FullFT), each at its own tuned LR and matched data volume, and measure alignment between the resulting update direction and an independently recovered teacher steering vector.

- FullFT+Adam aligns where FullFT+SGD doesn't → supports the optimizer-drift account.
- FullFT+Adam still misaligned regardless → LoRA's parameterization is doing real structural work, not just efficient encoding.

## Novel experiments, ranked by diagnostic value

1. **Rank × LR × data surface with FullFT included** — do first; Feng has run one seed count, add more seeds/traits and report logit-shift alongside string-match.
2. **Optimizer × parameterization alignment test** (see Framework above) — most diagnostic for the causal question.
3. **Coverage-metric replication** (Madl's method) on the Nief et al. rank sweep.
4. **Prefix/TReFT-style regularization** applied to subliminal learning, as a storage-vs-retrieval test.
5. **Steering-vector transplantation across ranks** where different traits (cat, eagle, owl, wolf) peak — test whether one vector explains all of them.
6. **Toy-model isolation** (MNIST-scale logit distillation) before full LLM-scale sweeps, following the precedent in Kitkana & Arora (2604.25779) — cheaper and faster to iterate on than Nief et al.'s own \~12,000 GPU-hour sweep.

## Fallback if the LoRA-specific hypothesis goes null

Reframe around Madl's channel-location taxonomy: it treats the token-entanglement account (Zur et al.) and the steering-vector account (Blank et al.) as different regimes rather than competing explanations, distinguished by whether the carrier sits in the initialization-dependent network body or in convergent vocabulary geometry. Testing which regime a given rank/parameterization/dataset falls into is a concrete extension of a paper already two months old, not a from-scratch reframing.

A working mitigation already exists to build on: probe-space corridor regularization (Trait-Direction Drift, 2609.01091) cuts malicious-response transfer from 29.55% to 6.45% in their setting. If the project shifts from "does LoRA cause it" to "how do we prevent it," this is a starting baseline, not a blank page.

## Additional literature not in the earlier memo

| Paper | arXiv | Relevance |
| --- | --- | --- |
| Subliminal Learning Is Steering Vector Distillation (Blank et al.) | 2606.00995 | Validated steering-vector recovery/transplantation method; optimizer ablation (§6.3) |
| Channel Location Constrains the Auditability of Subliminal Learning (Madl) | 2606.22019 | "Coverage" metric predicts transfer pre-training; three-regime taxonomy |
| The Piggyback Hypothesis of Generalization (Chua, Betley, Taylor, Evans) | 2606.06667 | Same shared-token-as-gate mechanism in emergent misalignment generally; TReFT intervention |
| Subliminal Learning as Trait-Direction Drift (Liu et al.) | 2609.01091 | Working mitigation (probe-space corridor regularization) |
| Optimizer-Induced Low-Dimensional Drift and Transverse Dynamics in Transformer Training (Xu) | 2602.23696 | Not SL-specific — shows AdamW generically induces low-rank drift; mechanistic anchor for this doc's framework |
| Sustained Gradient Alignment Mediates Subliminal Learning in a Multi-Step Setting (Kitkana & Arora) | 2604.25779 | Toy MNIST-scale isolation precedent |

## Setup checklist — before any new experiment runs

- [ ] Reproduce Nief et al.'s exact pipeline at rank 8 for 2–3 traits before changing anything.
- [ ] Stand up a coherence/format judge, live from run 1 — filters incoherent output before it contaminates behavioral comparisons.
- [ ] Build the teacher-divergence pipeline (per-token KL/JS between biased and unbiased teacher) — becomes a covariate for most later experiments.
- [ ] Log both string-match P(target) and logit-shift Δlog p(target) from the start.
- [ ] Freeze a versioned pool of generation and eval prompts.
- [ ] Set up negative controls from day one: a no-trait teacher, and a scrambled-output dataset preserving marginal token statistics.
- [ ] Pre-register a discovery/confirmation split — set aside traits, models, and seeds nothing touches until the final phase.

## Phased roadmap

### Phase 1 — Reproduce (days)

- [ ] Rank sweep at Nief et al.'s exact hyperparameters, 2–3 traits.
- [ ] Context-mismatch replication.

**Go/no-go**: if the inverted-U doesn't reproduce within reported confidence intervals, stop and debug infrastructure before Phase 2.

### Phase 2 — Kill the LR/data confound

- [ ] Rank × LR grid at fixed N=10k to find per-rank optimal LR.
- [ ] Dataset-size sweep (N from 1e3 to 5e5) at low/mid/high rank and FullFT, each at its own tuned LR.
- [ ] Plot the optimized surface E\*(r,N) against the original E(r).

**Decision marker**: non-monotonic in r after optimizing both LR and N → something about rank is real, go to Phase 4. Flattens out → the effect was mostly an optimization/data artifact, prioritize Phase 3.

### Phase 3 — Optimizer ablation

- [ ] Cross {Adam, SGD, SGD+frozen-Adam-scale-map} × {LoRA r=8, r=64, FullFT}, each at tuned LR/data.
- [ ] Measure alignment between the resulting update direction and an independently recovered teacher steering vector.

**Decision marker**: FullFT+Adam aligns where FullFT+SGD doesn't → optimizer-drift story. FullFT+Adam still misaligned → LoRA's parameterization does real structural work.

### Phase 4 — Mechanistic anchoring

- [ ] Steering-vector recovery and transplantation at the ranks where cat/eagle/owl/wolf peak.
- [ ] Divergence-token quantification, then masking/enrichment ablation.
- [ ] Coverage-metric prediction across the Phase 2/3 grid.
- [ ] Prefix-token localization + TReFT-style regularization.

### Phase 5 — Decision point

- [ ] Pick the outcome the evidence supports (LoRA-special / optimization artifact / steering-vector distillation / no single mechanism) and spend the pre-registered confirmation set stress-testing that claim specifically.

## Markers to watch for

| Stage | Marker | Action |
| --- | --- | --- |
| Generation | Generation-seed variance ≤ training-seed variance | Suspicious — check temperature/seeding; Nief et al. found the opposite |
| Generation | Unstable filter-rejection rate across seeds | Realized N is drifting silently; log and report it |
| Generation | Few divergence-token positions in a dataset | Expect weak transfer regardless of hyperparameters; don't compare directly to a same-size, high-divergence dataset |
| Training | Coherence judge flags a run | Discard from behavioral comparisons — not a valid "no effect" data point |
| Training | Update norm (‖BA‖ or ‖ΔW‖) spikes | Early warning before wasting eval compute; direct input to the Phase 3 optimizer-drift analysis |
| Training | Train/val loss diverging at high LR | Exclude before interpreting transfer numbers — run is unstable |
| Evaluation | String-match says no effect, logit-shift says otherwise | Report both — this is a finding, not noise |
| Evaluation | Negative control shows the same directional shift as treatment | Stop and diagnose before interpreting treatment — something generic is contaminating the result |
| Evaluation | A trait peaks at a different rank than Nief et al. reported | Don't discard as noise — evidence against rank having an intrinsic optimal value per trait |
| Process | Tuning anything on the held-out confirmation set | That set is burned — log it and draw a fresh one |
| Process | Compute spend per phase exceeds budget without a clean answer | Decide: keep sweeping or move to mechanism-anchoring |
| Process | Leading hypothesis fails cleanly by end of Phase 3 | Move toward the broader-phenomenon fallback framing sooner rather than later |
| External validity | Llama 3.1 keeps showing no effect while Qwen/Gemma do | Whatever mechanism is settled on has to explain this contrast — standing falsification target |
