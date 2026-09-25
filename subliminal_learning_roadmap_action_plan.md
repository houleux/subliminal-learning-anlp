# Subliminal Learning Research Roadmap

**Date:** 19 September 2026  
**Purpose:** Concrete execution plan for a compute-constrained investigation of subliminal learning, with LoRA treated as one variable rather than the assumed root cause.

---

# 1. Project status

## Original question

The project began with:

> **Is subliminal learning a LoRA artifact?**

The motivating work reported an inverted-U relationship between subliminal-transfer strength and LoRA rank, together with strong dependence on training/evaluation context.

The current research direction should be broader:

> **Characterize subliminal learning itself: what signal is transmitted, what determines whether transmission succeeds, how the student stores that signal, how context retrieves it, and what role LoRA/optimization/data scale play.**

The working causal picture is:

\[
\text{Teacher trait}
\rightarrow
\text{hidden signal in generated data}
\rightarrow
\text{student gradient/update}
\rightarrow
\text{stored behavioral direction}
\rightarrow
\text{contextual retrieval}
\rightarrow
\text{behavior}.
\]

LoRA is one possible factor in the **encoding/storage** stage.

---

# 2. Main research objectives

The project should answer four progressively deeper questions.

## Objective A — Establish the phenomenon

Can we reliably reproduce subliminal learning under controlled conditions?

\[
E_{\text{biased}} > E_{\text{control}}
\]

must be reproducible across multiple seeds.

## Objective B — Identify what controls transfer

Determine the roles of:

- LoRA rank
- learning rate
- optimizer
- dataset size
- teacher temperature/seed
- teacher/student divergence
- context

## Objective C — Identify the information carrier

Determine whether transfer is associated with:

- divergence tokens
- token-level distribution shifts
- gradient alignment
- low-dimensional parameter updates
- steering vectors

## Objective D — Build a mechanistic account

Connect:

\[
\text{teacher intervention}
\rightarrow
\text{observable distribution shift}
\rightarrow
\text{gradient}
\rightarrow
\text{parameter/activation direction}
\rightarrow
\text{retrieval}
\rightarrow
\text{behavior}.
\]

---

# 3. Compute-constrained strategy

## Local experiments

Use a small open-weight model for rapid iteration.

Recommended starting point:

- Qwen2.5-1.5B-Instruct
- sequence length: 256
- LoRA first
- FP16/BF16 where supported
- microbatch 1 with gradient accumulation
- gradient checkpointing only if needed

The purpose of this model is **mechanism discovery**, not claiming that results automatically generalize to 7B-scale models.

## Optional cloud validation

Reserve larger-model experiments for later:

- one canonical reproduction
- one rank/LR validation
- one mechanistic confirmation

The project should not become compute-heavy before the local experiments establish which hypotheses are worth testing.

---

# 4. Software/infrastructure to build first

Create a small reproducible experiment harness:

```text
subliminal-learning/
├── configs/
├── data/
│   ├── raw/
│   ├── filtered/
│   └── processed/
├── experiments/
│   └── registry.csv
├── src/
│   ├── generate.py
│   ├── train.py
│   ├── evaluate.py
│   ├── divergence.py
│   ├── steering.py
│   └── analysis.py
└── results/
```

Every run should log:

```text
model
trait
teacher_seed
student_seed
dataset_size
dataset_hash
rank
alpha
learning_rate
optimizer
batch_size
sequence_length
training_steps
prompt_version
chat_template_version
checkpoint
git_commit
```

Never regenerate a dataset silently during a sweep.

---

# 5. Metrics that must exist from experiment 1

Do not make string matching the only measurement.

## Behavioral transfer

\[
P(\text{target response})
\]

kept for comparability with prior work.

## Logit/probability shift

\[
\Delta\log p(y)
=
\log p_{\text{student}}(y|x)
-
\log p_{\text{base}}(y|x)
\]

and, where practical,

\[
\Delta z_y
=
z_{\text{student},y}
-
z_{\text{base},y}.
\]

## Training health

Record:

\[
L_{\text{train}},\qquad L_{\text{val}}
\]

plus a coherence/format-quality measure.

## Later mechanistic metrics

\[
N_{\text{divergent}}
\]

\[
\sum_t D_t
\]

\[
\cos(v_T,v_S)
\]

\[
\cos(g_t,v_T)
\]

and LoRA singular-value concentration.

---

# 6. Experiment 0 — Toy-model sandbox

## Goal

Obtain a cheap, highly inspectable version of the phenomenon.

Use the MNIST-scale subliminal-learning/gradient-alignment setup as a sandbox.

Measure the training dynamics directly, especially:

\[
\cos(g_t,v_{\text{trait}})
\]

throughout training.

## Why

The toy setup lets us test hypotheses about:

- gradient accumulation
- signal strength
- optimizer effects
- vector alignment

without spending GPU-hours on a multi-billion-parameter model.

## Deliverable

A notebook showing:

1. teacher intervention
2. student learning
3. behavioral transfer
4. gradient-direction plots

This is a mechanism-development environment, not the main final benchmark.

---

# 7. Experiment 1 — Canonical LLM reproduction

## Model

Qwen2.5-1.5B-Instruct locally.

## Trait

Start with one repeatedly observed trait such as:

- cat
- owl

Do not immediately test many traits.

## Data

Start with:

\[
N=2k
\]

as a pipeline smoke test.

If transfer is absent, scale:

\[
2k\rightarrow5k\rightarrow10k.
\]

## Training

Start with:

- LoRA rank 8
- AdamW
- one baseline learning rate
- canonical number-sequence task
- fixed teacher seed

## Evaluate

Measure:

\[
P(\text{target})
\]

and

\[
\Delta\log p(\text{target}).
\]

## Deliverable

A single reproducible baseline checkpoint and report.

### Go/no-go

If there is no measurable transfer, debug the infrastructure before doing any mechanistic sweep.

---

# 8. Experiment 2 — Establish controls

For one fixed teacher dataset, train/evaluate:

| Condition | Purpose |
|---|---|
| Base | no-training baseline |
| Unbiased teacher data | ordinary SFT drift |
| Biased teacher data | candidate subliminal transfer |
| Scrambled/shuffled data | destroys teacher-example correspondence |

Define:

\[
\Delta E
=
E_{\text{biased}}
-
E_{\text{control}}.
\]

This should become the primary treatment-vs-control quantity.

---

# 9. Experiment 3 — Seed robustness

Run approximately:

\[
3\text{ teacher seeds}
\times
2\text{ student seeds}
\]

before spending compute on large sweeps.

Track:

- transfer variance
- divergence statistics
- filter rejection rate
- output quality

## Why

Teacher-generation randomness may dominate training-seed randomness. A single unusually informative teacher dataset can produce a misleading conclusion.

---

# 10. Experiment 4 — Rank × learning rate

This is the first direct test of the original LoRA claim.

Use:

\[
r\in\{4,8,32\}
\]

and

\[
\eta\in
\{
5\cdot10^{-5},
10^{-4},
2\cdot10^{-4},
4\cdot10^{-4}
\}.
\]

That is only:

\[
3\times4=12
\]

training runs for an initial pass.

Keep:

\[
N=10k.
\]

## Output

Plot:

\[
E(r,\eta)
\]

as a heatmap.

Then compute:

\[
E^*(r)=\max_{\eta}E(r,\eta).
\]

## Decision

### If inverted-U survives LR optimization

Rank remains scientifically interesting.

### If inverted-U largely disappears

Treat rank as an optimization/parameterization interaction rather than an intrinsic requirement of subliminal learning.

---

# 11. Experiment 5 — Dataset size

Freeze the best learning rate found for each rank.

Initially test:

\[
r\in\{8,32\}
\]

and

\[
N\in\{1k,5k,10k,25k\}.
\]

This is only eight configurations.

Plot:

\[
E(r,N).
\]

## Question

Does additional data restore transfer at high rank?

If yes, an apparent high-rank failure may simply reflect sample complexity.

---

# 12. Experiment 6 — Full fine-tuning

Do not run a large FullFT sweep locally.

Run one carefully chosen FullFT comparison at sufficient data and a separately tuned learning rate.

Compare:

```text
LoRA r=8
LoRA r=32
FullFT
```

using the same underlying dataset.

## Key principle

Do not compare:

```text
well-tuned LoRA
vs.
arbitrarily configured FullFT
```

because that recreates the confound.

If local FullFT is too expensive, run this one experiment on a cloud GPU.

---

# 13. Experiment 7 — Optimizer ablation

Initial experiment:

\[
\{\text{AdamW},\text{SGD}\}
\times
\{\text{LoRA r=8},\text{high-rank/FullFT}\}.
\]

Tune each optimizer sufficiently to make the comparison meaningful.

Measure:

\[
\Delta E
\]

plus update direction.

## Question

Is adaptive optimization itself important for forming the transferable signal?

A later experiment can add a frozen Adam-scale map, but that should not be part of the first pass.

---

# 14. Experiment 8 — Teacher divergence analysis

For a biased and unbiased teacher pair, compute at each token position:

\[
D_t
=
D_{\mathrm{JS}}
\left(
p_T^t,p_B^t
\right).
\]

Also record:

- top-1 disagreement
- observed-token probability shift
- entropy
- position
- token identity

Define:

\[
N_{\text{div}}
=
\#\{t:D_t>\tau\}
\]

and

\[
S_{\text{div}}
=
\sum_tD_t.
\]

## Question

Does stronger measurable teacher divergence predict stronger student transfer?

This turns "hidden signal" from a qualitative idea into a quantitative covariate.

---

# 15. Experiment 9 — Divergence-token ablation

From one fixed dataset create:

### Full

\[
D_{\text{all}}
\]

### Divergence-focused

\[
D_{\text{div}}
\]

### Divergence-removed

\[
D_{\text{no-div}}.
\]

Train the same student configuration on each.

## Critical pattern

If:

\[
E_{\text{div}}
\approx
E_{\text{all}}
\]

and

\[
E_{\text{no-div}}
\approx
E_{\text{control}},
\]

then the useful information is highly concentrated in the divergence events.

This result is already supported by adjacent literature, so our novelty would come from connecting it to optimizer, representation, and LoRA behavior.

---

# 16. Experiment 10 — Steering-vector alignment

Recover an independent teacher steering vector:

\[
v_T.
\]

Recover a student behavioral vector:

\[
v_S.
\]

Measure:

\[
\cos(v_T,v_S).
\]

For LoRA, also calculate the first singular direction:

\[
u_1(BA).
\]

Then compare:

\[
\cos(u_1,v_T).
\]

## Main question

Are the apparent "LoRA direction" and the independent teacher behavioral direction actually the same subspace?

If yes, this potentially unifies the LoRA and steering-vector explanations.

---

# 17. Experiment 11 — Gradient alignment

During selected training steps calculate:

\[
g_t=\nabla_\theta L_t.
\]

Then compute:

\[
\cos(g_t,v_T).
\]

Compare:

\[
\mathbb E[
\cos(g_{\text{div}},v_T)
]
\]

against

\[
\mathbb E[
\cos(g_{\text{nondiv}},v_T)
].
\]

## Desired mechanistic result

A plausible chain is:

\[
\text{divergent token}
\rightarrow
\text{gradient component}
\rightarrow
v_T
\rightarrow
\text{student behavioral direction}.
\]

That would be much stronger than simply demonstrating end-of-training behavioral transfer.

---

# 18. Experiment 12 — Context gating

Using the same trained checkpoint, evaluate:

```text
1. identical train/eval context
2. different system prompt
3. empty system prompt
4. paraphrased system prompt
```

Measure:

\[
E(C_{\text{train}},C_{\text{eval}}).
\]

A useful representation is a context-transfer matrix:

\[
M_{ij}
=
E(C_i^{\text{train}},C_j^{\text{eval}}).
\]

## Question

Is context:

- the storage location,
- the retrieval key,
- or both?

This experiment is cheap and should be done before extensive activation patching.

---

# 19. The key unifying experiment

Once the previous experiments work, take **one fixed teacher dataset** and analyze it in four ways:

\[
D
\rightarrow
\text{divergence structure}
\]

\[
D
\rightarrow
\text{gradient directions}
\]

\[
D
\rightarrow
\text{student behavioral vector}
\]

\[
D
\rightarrow
\text{LoRA update SVD}.
\]

Then test whether these point toward the same low-dimensional subspace:

\[
\boxed{
\text{divergence}
\approx
\text{gradient direction}
\approx
\text{steering direction}
\approx
\text{LoRA update direction}
}
\]

If this holds, the project can move toward a unified mechanistic account.

---

# 20. Things to postpone

Do not immediately spend compute on:

- huge rank sweeps
- dozens of traits
- many model families
- full activation-patching grids
- TReFT-style regularization
- cross-language models
- elaborate poisoning/safety experiments
- large-scale FullFT sweeps

These are later-stage experiments.

First identify which variables actually matter.

---

# 21. Decision tree

```text
                         Reproduce SL?
                         /            \
                       NO              YES
                       |                |
                 Debug pipeline     Rank × LR
                                        |
                              Inverted-U survives?
                                  /          \
                                YES           NO
                                 |             |
                           Rank is real?    Optimization/
                           investigate     data interaction
                                 |             |
                                 +------+------+
                                        |
                              Dataset-size scaling
                                        |
                              Does more data restore
                              high-rank/FullFT SL?
                                  /          \
                                YES           NO
                                 |             |
                          Capacity/data      stronger case
                          explanation       for parameterization
                                 |             |
                                 +------+------+
                                        |
                               Optimizer ablation
                                        |
                              Does Adam-specific
                              directional drift matter?
                                  /          \
                                YES           NO
                                 |             |
                          optimizer story    investigate
                                            parameterization
                                        |
                              Divergence analysis
                                        |
                           Does divergence predict SL?
                                  /          \
                                YES           NO
                                 |             |
                         information       investigate alternate
                         carrier found     signal representations
                                 |
                         Steering-vector alignment
                                  |
                       Does student recover teacher
                            behavioral direction?
                                  /      \
                                YES       NO
                                 |         |
                         unified signal   broaden mechanism
                           hypothesis
                                  |
                       Context / storage / retrieval
                                  |
                           Causal localization
                                  |
                         Confirmation on held-out
                         traits/seeds/models
```

---

# 22. Project markers

## Marker 1 — Reproducibility

Track:

\[
\operatorname{Var}_{\text{teacher seed}}(E)
\]

and

\[
\operatorname{Var}_{\text{student seed}}(E).
\]

Do not treat one unusually strong run as representative.

---

## Marker 2 — Treatment/control separation

Always track:

\[
E_{\text{biased}}
\]

and

\[
E_{\text{control}}.
\]

If both move together, the effect is not cleanly attributable to subliminal transfer.

---

## Marker 3 — Training health

Monitor:

- train/validation loss
- coherence
- format rejection
- parameter/update norm

A degenerate model is not evidence for strong subliminal learning.

---

## Marker 4 — Metric disagreement

If:

\[
P(\text{target})
\]

looks null but

\[
\Delta\log p(\text{target})
\]

is substantial, record both.

This is a potentially meaningful finding rather than something to discard.

---

## Marker 5 — Divergence density

Always associate transfer with:

\[
N_{\text{div}}
\]

and

\[
S_{\text{div}}.
\]

Do not compare two equally sized datasets as though they contain equal hidden signal.

---

## Marker 6 — Rank dependence

Record both:

\[
E(r,\eta_{\text{fixed}})
\]

and

\[
E^*(r).
\]

The second is what matters for a causal claim about rank.

---

## Marker 7 — Data scaling

Plot:

\[
E(r,N).
\]

A high-rank/FullFT effect appearing with more data changes the interpretation of the original rank result.

---

## Marker 8 — Context sensitivity

Record the complete train/eval context matrix.

A result that only appears under exact context matching should be described as highly context-gated.

---

## Marker 9 — Directional alignment

Track:

\[
\cos(v_T,v_S)
\]

and

\[
\cos(g_t,v_T).
\]

These are potential core mechanistic measurements.

---

## Marker 10 — Confirmation-set integrity

Create a held-out confirmation pool of:

- traits
- teacher seeds
- prompts
- possibly one model

that is never used for tuning.

Once touched, that confirmation set is no longer a confirmation set.

---

# 23. Experimental budget philosophy

For local compute, favor small factorial tests over enormous sweeps.

A good initial sequence is approximately:

```text
Experiment 0: toy
Experiment 1: LLM smoke test
Experiment 2: controls
Experiment 3: 3×2 seed robustness
Experiment 4: 3×4 rank×LR
Experiment 5: 2×4 rank×data
Experiment 6: 1 FullFT comparison
Experiment 7: 2×2 optimizer test
Experiment 8: divergence analysis
Experiment 9: divergence ablation
Experiment 10: steering alignment
Experiment 11: gradient alignment
Experiment 12: context gating
```

This is intentionally small.

The goal is to determine **which scientific branch deserves additional compute**.

---

# 24. What counts as a meaningful outcome?

## Outcome A — LoRA remains genuinely special

If optimized high-rank LoRA and FullFT consistently fail while low/intermediate LoRA transfers, the LoRA parameterization remains a plausible causal mechanism.

Then investigate:

- rank geometry
- singular structure
- module placement
- optimizer interaction

## Outcome B — LoRA is primarily an efficient encoding mechanism

If transfer appears across ranks/FullFT with sufficient data and suitable optimization, then LoRA is likely influencing efficiency rather than defining the phenomenon.

Focus on:

\[
\text{sample complexity}
+
\text{optimization}
+
\text{representation}.
\]

## Outcome C — Steering-vector distillation explains much of SL

If:

\[
v_S\approx v_T
\]

and gradient updates align with \(v_T\), then a strong project direction is:

\[
\boxed{\text{subliminal learning as hidden behavioral-direction distillation}}
\]

## Outcome D — No single mechanism

The project may reveal multiple regimes.

For example:

\[
\text{divergence-token regime}
\]

vs.

\[
\text{steering-vector regime}
\]

vs.

\[
\text{context-gated regime}.
\]

That would motivate a taxonomy rather than a single universal mechanism.

---

# 25. The research opportunity

The field already contains several partially overlapping explanations:

\[
\text{LoRA}
\]

\[
\text{divergence tokens}
\]

\[
\text{steering vectors}
\]

\[
\text{optimizer-induced drift}
\]

\[
\text{context/prefix gating}.
\]

The strongest opportunity is **not** to reproduce each individually.

It is to determine whether these are different observations of the same underlying information-transfer process.

A particularly valuable result would establish a causal chain such as:

\[
\boxed{
\text{Teacher perturbation}
\rightarrow
\text{token-level distribution shift}
\rightarrow
\text{divergence-token gradients}
\rightarrow
\text{low-dimensional trait direction}
\rightarrow
\text{parameter storage}
\rightarrow
\text{contextual retrieval}
\rightarrow
\text{behavior}
}
\]

with LoRA explaining only one part of the encoding process.

---

# 26. Immediate action checklist

## Today

- [ ] Set up experiment repository
- [ ] Build run registry
- [ ] Implement behavioral + logit-shift evaluation
- [ ] Freeze prompt/template versions
- [ ] Clone/reference existing subliminal-learning and divergence-token implementations
- [ ] Run toy-model experiment

## Next day

- [ ] Qwen2.5-1.5B smoke test
- [ ] One trait
- [ ] 2k examples
- [ ] biased/unbiased/scrambled controls

## Days 3–4

- [ ] Increase to 5k/10k if needed
- [ ] Multi-seed reproduction
- [ ] Rank × LR grid

## Days 4–5

- [ ] Rank × dataset-size test
- [ ] One carefully tuned FullFT comparison

## Days 5–6

- [ ] AdamW vs SGD
- [ ] Teacher divergence calculation
- [ ] Divergence-token ablation

## Days 6–7

- [ ] Steering vector extraction
- [ ] Student/teacher cosine alignment
- [ ] LoRA SVD alignment
- [ ] Gradient alignment

## After week 1

Use the decision tree to choose the main research branch.

Do **not** expand model/trait/task coverage until that decision has been made.

---

# 27. Final project objective

The project should ultimately answer:

> **What determines whether an apparently innocuous dataset transmits a teacher's behavioral trait to a student model, and what internal signal and training dynamics make that transmission possible?**

The LoRA question is therefore:

\[
\text{one subquestion}
\]

rather than:

\[
\text{the entire project}.
\]

The desired end state is a mechanism-level account that can explain:

1. why transfer happens,
2. why it sometimes fails,
3. why different ranks/optimizers/data sizes behave differently,
4. why particular token positions and contexts matter,
5. why certain teacher-generated datasets are much more effective than others,
6. how the student stores the signal,
7. how the behavior can be retrieved or suppressed.

That framing is robust to a null LoRA result and gives the project a broader scientific contribution.
