# The Internal Geometry of Strategic Reasoning in Thinking Language Models

Code and results for a mechanistic interpretability study of strategic reasoning in DeepSeek-R1-Distill-Qwen-14B and DeepSeek-R1-Distill-Llama-8B. I extract difference-of-means steering vectors for strategic reasoning sub-behaviors from the residual stream, test their geometry, and probe the causal role of one of them. I check geometry on both models; behavioral interventions are on the Qwen side only.

I built the multi-label annotation, calibration by manually annotating and checking the definitions aligned or not, and controlled intervention pipeline after identifying weaknesses in an earlier mutually-exclusive annotation setup. All experiments were run on Google Colab (A100).

---

## Current status

This is a finished exploratory project. The results support a robust opponent-modeling-associated direction that is anti-aligned with deduction in the residual stream of both models, but I do not claim this direction is a pure semantic feature or that steering improves strategic reasoning. The main behavioral finding (Qwen only) is that the direction behaves non-atomically: positive steering increases opponent-modeling-labeled reasoning, while projection-out ablation also increases opponent-modeling-labeled segments.

The geometric anti-alignment replicates cleanly across the two model families. The behavioral side was run on Qwen only, I did not have the compute budget to run interventions on Llama, so whether the steering and ablation effects replicate on Llama is untested. I am stopping the project here and listing what a continuation would need rather than leaving it open-ended.

## What I do not claim

- I do not claim the vector is a pure opponent-modeling feature. The probe result argues against it.
- I do not claim steering improves strategic reasoning.
- I have no Llama intervention results. Every steering/ablation number here is Qwen or Qwen-base.
- I have not identified a full circuit or specific attention heads.
- I do not have a mechanistic account of the ablation result (see the note in Phase 3).
- I treat the LLM annotations as operational behavioral labels, not as ground truth about what the model is internally doing.

---

## Setup

| Item | Choice |
|---|---|
| Source model A | DeepSeek-R1-Distill-Qwen-14B (48 layers, hidden dim 5120), L24 |
| Source model B | DeepSeek-R1-Distill-Llama-8B (32 layers, hidden dim 4096), L16 (geometry only) |
| Transfer model | Qwen-2.5-14B-Instruct (same Qwen-14B family, non-R1-distilled) |
| Main task set | 300 tasks total = 250 game-theoretic / strategic reasoning tasks + 50 non-strategic control tasks. |
| OOD tasks | 50 held-out strategic tasks |
| Intervention layer | 24 (Qwen) |
| Vector type | difference-of-means (with-vs-without centering) |
| Annotator | GPT-5.4, chain-mode |
| Segmentation | paragraph-split with 1200-char hard-wrap |

---

## Phase 0: Annotation pipeline

The earlier version of this project used mutually exclusive labels with a "prefer strategic" priority rule. That made the geometry too easy to overinterpret because many reasoning segments naturally contain multiple behaviors at once.

I built a multi-label annotation pipeline with no priority hierarchy, an explicit `none_other` label, and per-label definitions with positive and negative examples. The 10-label taxonomy:

| Label | What it covers |
|---|---|
| `opponent_modeling` | Reasoning about what another agent believes, expects, or will do |
| `iterated_reasoning` | Recursive "I think they think..." reasoning |
| `equilibrium_identification` | Nash equilibria, dominant strategies, solution concepts |
| `payoff_analysis` | Comparing outcomes, utilities, costs |
| `strategic_uncertainty` | Uncertainty over another agent's action or strategy |
| `cooperative_reasoning` | Fairness, trust, coordination |
| `initialization` | Restating setup, listing options |
| `deduction` | Non-social logical derivation or calculation |
| `backtracking` | Revising a previous step |
| `none_other` | Does not fit other categories |

**Code:** `annotation/annotation_v2.py`, `annotation/annotate_chains.py`, `annotation/annotate_segments.py`

---

## Phase 0.5: Gold calibration

I hand-labeled 150 segments (stratified sample) and compared the LLM annotator against my labels.

| Metric | Value |
|---|---|
| Gold segments | 150 |
| GPT-5.4 chain-mode macro-F1 | 0.770 |
| GPT-5.2 per-segment macro-F1 | 0.739 |
| GPT-5.4-mini macro-F1 | 0.653 (failed, cooperative_reasoning collapse) |
| Weakest label | deduction (F1 ~0.55) |

Deduction is noisy. The annotator over-applies it roughly 2x relative to my gold labels. I keep this in mind for all downstream analysis involving the deduction direction.

Calibration was done on Qwen segments. I applied the same prompt to Llama for consistency (changing the prompt would introduce annotator drift), but annotation quality on Llama text is unmeasured. Llama's reasoning style differs (more backtracking, ~58% more segments from the same 300 tasks, consistent with longer generations), so the 0.77 does not automatically carry over.

**Code:** `annotation/sample_gold_segments.py`, `annotation/evaluate_gold.py`

---

## Phase 1: Full annotation

After calibration, I annotated the full corpus of 300 reasoning chains per model.

| Quantity | Qwen-14B | Llama-8B |
|---|---|---|
| Tasks | 300 | 300 |
| Total annotated segments | 14,645 | 23,105 |
| Think-region segments | 13,010 | 21,535 |
| Multi-label segments | 57.6% | 58.7% |

Label distribution (think-region only):

| Label | Qwen | Llama |
|---|---|---|
| payoff_analysis | 53.2% | 57.3% |
| opponent_modeling | 30.7% | 29.7% |
| deduction | 22.6% | 24.5% |
| backtracking | 12.4% | 17.0% |
| strategic_uncertainty | 12.9% | 10.4% |
| equilibrium_identification | 12.4% | 5.2% |
| none_other | 10.4% | 10.0% |
| initialization | 10.0% | 10.0% |
| cooperative_reasoning | 6.5% | 6.1% |
| iterated_reasoning | 0.6% | 0.1% |

`iterated_reasoning` is near-absent in both. I do not build claims on it, and the social-cluster alignment numbers that lean on it are noisy.

**Code:** `annotation/annotate_chains.py`

---

## Phase 2: Activation extraction and geometry

For each annotated segment, I extracted residual-stream activations across all layers using mean pooling across token positions. Then for each label, I computed a difference-of-means direction:

```
u_c = mean(activations where segment has label c) - mean(activations where segment does not have label c)
```

This with-vs-without formulation is the correct one for multi-label data.

### Cosine between opponent-modeling and deduction directions

| Centering method | Qwen L24 | Qwen all-layer mean | Llama L16 | Llama all-layer mean |
|---|---|---|---|---|
| With-vs-without (primary) | -0.707 | -0.723 | -0.613 | -0.626 |
| Leave-one-out | -0.707 | -0.723 | -0.613 | -0.626 |
| Class-balanced | -0.371 | -0.342 | -0.256 | -0.211 |
| All layers negative? | 48/48 | | 32/32 | |

The class-balanced number is weaker in both, and weaker on Llama. Both numbers are real; which one is more meaningful depends on how you weight the dominant payoff_analysis category.

### Statistical validation (with-vs-without)

| Test | Qwen | Llama |
|---|---|---|
| Permutation test (n=1000) | p < 0.001 (null -0.005) | p < 0.001 (null -0.004) |
| Bootstrap 95% CI (n=1000) | [-0.727, -0.682] | [-0.634, -0.588] |
| Split-half (n=100) | mean -0.705, 200/200 negative | mean -0.612, 200/200 negative |

### Important caveat

All category means have cosine >= 0.94 with the global mean (both models). The DoM vectors are small perturbations off a dominant shared "thinking" direction. The antagonism lives in a small fraction of the representational variance.

### Pairwise raw cosine (no DoM subtraction)

| Pair | Qwen L24 | Llama L16 |
|---|---|---|
| opp_mod vs deduction | +0.946 | +0.921 |
| opp_mod vs payoff_analysis | +0.985 | +0.970 |
| opp_mod vs strategic_uncertainty | +0.984 | +0.958 |

Raw representations are nearly identical. The antagonism only emerges after subtracting the global mean.

### Full cosine matrix (with-vs-without, Qwen L24)

```
             opp-mod  iter-reas  equil-id   payoff  strat-unc  coop-reas     init  deduction  backtrack  none/other
opp-mod       +1.000    +0.526    +0.111   +0.432     +0.575     +0.327   -0.153     -0.707     +0.306      -0.610
iter-reas     +0.526    +1.000    +0.172   -0.051     +0.483     +0.569   +0.191     -0.407     +0.170      -0.243
equil-id      +0.111    +0.172    +1.000   -0.291     +0.215     +0.230   -0.169     -0.206     +0.224      -0.011
payoff        +0.432    -0.051    -0.291   +1.000     -0.157     -0.136   -0.349     -0.186     -0.023      -0.719
strat-unc     +0.575    +0.483    +0.215   -0.157     +1.000     +0.462   +0.020     -0.463     +0.408      -0.228
coop-reas     +0.327    +0.569    +0.230   -0.136     +0.462     +1.000   +0.382     -0.401     +0.101      -0.162
init          -0.153    +0.191    -0.169   -0.349     +0.020     +0.382   +1.000     -0.144     -0.338      +0.138
deduction     -0.707    -0.407    -0.206   -0.186     -0.463     -0.401   -0.144     +1.000     -0.366      +0.110
backtrack     +0.306    +0.170    +0.224   -0.023     +0.408     +0.101   -0.338     -0.366     +1.000      +0.018
none/other    -0.610    -0.243    -0.011   -0.719     -0.228     -0.162   +0.138     +0.110     +0.018      +1.000
```

### Full cosine matrix (with-vs-without, Llama L16)

```
             opp-mod  iter-reas  equil-id   payoff  strat-unc  coop-reas     init  deduction  backtrack  none/other
opp-mod       +1.000    +0.558    +0.010   +0.290     +0.408     +0.480   -0.230     -0.613     +0.095      -0.498
iter-reas     +0.558    +1.000    +0.320   -0.138     +0.533     +0.425   +0.106     -0.361     +0.158      -0.183
equil-id      +0.010    +0.320    +1.000   -0.445     +0.183     +0.357   +0.132     +0.015     +0.124      +0.109
payoff        +0.290    -0.138    -0.445   +1.000     -0.253     -0.001   -0.431     -0.004     -0.245      -0.736
strat-unc     +0.408    +0.533    +0.183   -0.253     +1.000     +0.258   -0.053     -0.270     +0.325      -0.110
coop-reas     +0.480    +0.425    +0.357   -0.001     +0.258     +1.000   -0.015     -0.324     +0.108      -0.278
init          -0.230    +0.106    +0.132   -0.431     -0.053     -0.015   +1.000     -0.121     -0.185      +0.193
deduction     -0.613    -0.361    +0.015   -0.004     -0.270     -0.324   -0.121     +1.000     -0.343      +0.108
backtrack     +0.095    +0.158    +0.124   -0.245     +0.325     +0.108   -0.185     -0.343     +1.000      +0.266
none/other    -0.498    -0.183    +0.109   -0.736     -0.110     -0.278   +0.193     -0.108     +0.266      +1.000
```

### SVD of category-mean matrix

| Component | Qwen L24 | Llama L16 |
|---|---|---|
| SV1 | 34.3% | 35.9% |
| SV2 | 26.5% | 24.3% |
| SV3 | 14.9% | 12.7% |

No single component dominates in either model. The geometry needs 3+ axes.

**Code:** `src/phase2_geometry.py` (set `MODEL_PRESET` to `qwen14b` or `llama8b`)

---

## Phase 2.5: Single-axis disambiguation

I ran three tests to check whether the opp-mod/deduction antagonism is a single signed axis or something more complex. Run on both models; results are consistent.

### Test 1: Linear probe vs DoM direction

| Diagnostic | Qwen | Llama |
|---|---|---|
| Probe CV accuracy (5-fold) | 0.87 | 0.83 |
| cos(probe weight, DoM contrast) | +0.146 | +0.187 |

The probe classifies well but finds a nearly orthogonal direction to the DoM contrast. The probe weight is ~97-98% orthogonal to the top-6 SVD subspace of the DoM category-mean matrix in both models, it uses within-class variance structure that means-based analysis does not see.

### Regularization sweep (L24 Qwen / L16 Llama)

| C (regularization) | Qwen CV acc | Qwen cos(w,DoM) | Llama CV acc | Llama cos(w,DoM) |
|---|---|---|---|---|
| 0.0001 (strongest) | 0.90 | +0.39 | 0.88 | +0.56 |
| 1.0 (default) | 0.87 | +0.15 | 0.83 | +0.19 |
| 100.0 (weakest) | 0.86 | +0.13 | 0.82 | +0.18 |

Strong regularization pushes the probe toward the DoM direction and improves accuracy. The DoM direction is a high-SNR direction, but the unregularized probe finds higher-dimensional features.

### Test 2: SVD depth profile

Qwen: SV1 never exceeds 58% at any layer (mean 37.8%), 0/48 layers > 70%. Llama: SV1 never exceeds 52% (mean 37.2%), 0/32 layers > 70%.

### Test 3: Co-occurrence geometry

Qwen (181 BOTH segments): BOTH leans toward opp_mod, relative position +0.636 on the contrast axis. Not consistent with single-axis cancellation.

Llama (408 BOTH segments): BOTH leans toward opp_mod, relative position +0.457, but it projects +0.92 on u_opp and -0.21 on u_ded, the clean two-axis pattern from Qwen is weaker and the script flags it ambiguous. This is the one place the two models diverge interpretively, so I report it rather than smooth it over.

### Within-class variance

Deduction has more total variance than opponent-modeling in both (Qwen 1.49x, Llama 1.31x). Fisher discriminant ratio: the probe direction is 2.53x (Qwen) / 3.34x (Llama) better than DoM.

### Verdict

All three tests point the same way: the geometry is multi-dimensional. The antagonism is real but coexists with independent structure. Co-occurrence is the weakest leg on Llama.

**Code:** `src/phase2.5_analysis.py` (set `MODEL_PRESET`)

---

## A note on the ablation result before the intervention tables

The headline is that ablating the opponent-modeling direction and steering toward it both raise opponent-modeling output. I previously called this a "paradox." It is worth being precise about what it is and is not:

- Ablate and steer are different operators on different positions. Ablation projects the direction out of the full sequence on every forward pass; steering adds a fixed-norm vector at the last token each step. "Both increase opp-mod" is not a logical contradiction.
- The probe result offers a deflationary read: the discriminative signal for opp-mod lives ~97% outside the DoM direction. Removing the DoM direction leaves most of the relevant structure intact, so the model can re-express the behavior through other directions in later positions (routing around the ablation). This is a known property of single-direction ablation under generation.
- A regulatory/suppressive interpretation is one candidate. Routing-around is another, and I have direct evidence for the latter. I cannot distinguish them with the experiments I ran.

Treat the ablation result below as a robust, control-specific, unexplained behavioral observation, not as established evidence of a regulatory direction.

---

## Phase 3: R1 in-distribution interventions (Qwen)

Steering and projection-out ablation on R1-Distill-Qwen using 50 in-distribution strategic tasks. 8 conditions. All outputs re-annotated with GPT-5.4 chain-mode.

### Ablation results

| Condition | opp_mod | Δ from baseline | deduction |
|---|---|---|---|
| baseline | 33.5% |, | 19.1% |
| ablate_opp | 45.0% | +11.6pp | 18.0% |
| ablate_random | 31.5% | -2.0pp | 19.8% |
| ablate_payoff | 32.7% | -0.8pp | 21.2% |
| ablate_probe | 34.0% | +0.5pp | 16.7% |

### Steering results

| alpha | opp_mod | deduction | strat_unc | segments |
|---|---|---|---|---|
| -0.5 | 0.1% | 13.5% | 1.2% | 6257 |
| 0.0 | 33.5% | 19.1% | 17.3% | 2540 |
| +0.2 | 41.5% | 14.3% | 26.1% | 2273 |
| +0.3 | 47.4% | 15.5% | 27.1% | 2442 |

Phase 3 used max_new_tokens=3072. 30-46% of outputs were truncated. Relative comparisons are valid within-regime.

**Code:** `src/phase3_run_interventions.py`, `src/phase3_annotate_outputs.py`

---

## Phase 4A: R1-Distill-Qwen on OOD tasks

Same 8 conditions as Phase 3, run on 50 held-out OOD strategic tasks. max_new_tokens=6144.

### Ablation results

| Condition | opp_mod | Δ from baseline | deduction | segments |
|---|---|---|---|---|
| baseline | 23.9% |, | 22.6% | 4029 |
| ablate_opp | 32.0% | +8.1pp | 22.2% | 2632 |
| ablate_random | 25.1% | +1.3pp | 23.5% | 3535 |
| ablate_payoff | 24.6% | +0.7pp | 18.7% | 3135 |
| ablate_probe | 23.5% | -0.4pp | 24.5% | 3719 |

The ablation result holds out of distribution.

### Steering results

| alpha | opp_mod | deduction | strat_unc | segments |
|---|---|---|---|---|
| -0.5 | 0.0% | 9.1% | 0.7% | 10741 |
| 0.0 | 23.9% | 22.6% | 15.8% | 4029 |
| +0.2 | 38.9% | 20.0% | 14.0% | 3911 |
| +0.3 | 44.2% | 7.4% | 16.2% | 4911 |

At alpha=-0.5, the model spirals (10,741 segments, 80% truncated at 6144 tokens).

**Code:** `src/phase4_run_ood_transfer.py`, `src/phase4_annotation.py`

---

## Phase 4B: OOD and same-family base-model transfer

Same R1-derived (Qwen) vectors applied to the base Qwen-2.5-14B-Instruct model. 5 conditions.

### Results

| Condition | opp_mod | deduction | payoff | segments |
|---|---|---|---|---|
| baseline | 28.1% | 18.6% | 63.0% | 652 |
| ablate_opp | 41.0% | 17.2% | 66.7% | 534 |
| ablate_random | 27.0% | 24.1% | 62.4% | 611 |
| steer_+0.2 | 41.5% | 20.7% | 65.1% | 479 |
| steer_+0.3 | 39.7% | 27.7% | 73.1% | 4049 |

The ablation result appears on the base model (+12.9pp, specific to u_opp). At steer_+0.3, the base model spirals (4049 segments, 6.2x baseline). Unlike R1, deduction goes up and backtracking stays at 0.0% across all conditions.

Base model sample sizes are small (479-652 segments for non-spiraling conditions).

---

## Cross-setting summary (Qwen / Qwen-base)

### Ablation result

| Setting | baseline | ablate_opp | Δ_opp | ablate_random | Δ_rand |
|---|---|---|---|---|---|
| Phase 3 (R1, in-dist) | 33.5% | 45.0% | +11.6pp | 31.5% | -2.0pp |
| Phase 4A (R1, OOD) | 23.9% | 32.0% | +8.1pp | 25.1% | +1.3pp |
| Phase 4B (Base, OOD) | 28.1% | 41.0% | +12.9pp | 27.0% | -1.1pp |

### Steering dose-response (opp_mod)

| alpha | P3 (R1 in-dist) | P4A (R1 OOD) | P4B (Base OOD) |
|---|---|---|---|
| -0.5 | 0.1% | 0.0% |, |
| 0.0 | 33.5% | 23.9% | 28.1% |
| +0.2 | 41.5% | 38.9% | 41.5% |
| +0.3 | 47.4% | 44.2% | 39.7% |

---

## Interpretation

The opponent-modeling DoM direction (Qwen) has asymmetric effects depending on how it is applied:

- **Positive steering** increases opponent-modeling-labeled segments and suppresses deduction on R1.
- **Strong negative steering** suppresses opponent-modeling but causes degeneration.
- **Projection-out ablation** increases opponent-modeling-labeled segments. Candidate explanations are routing-around (supported by the probe) and a regulatory role; I cannot separate them.

The geometric anti-alignment replicating on Llama is the cleanest cross-family result here: two architectures, two families, two depths, same sign and same significance. The behavioral effects appearing on the base Qwen model suggest the relevant structure is at least partly preserved within the Qwen-14B family rather than being unique to R1-style distillation, but this is a within-family statement, not a cross-family one, because I have no Llama interventions.

---

## Limitations

- **Behavioral results are Qwen-only.** The cross-family evidence is geometric. Whether steering/ablation replicate on Llama is untested (compute budget).
- **Single annotator.** GPT-5.4 only, calibrated on Qwen segments; Llama annotation quality unmeasured. Relative comparisons are more meaningful than absolute rates because the same annotator/prompt is used across conditions, but condition-dependent annotation bias remains possible.
- **Ablation mechanism unresolved.** Possibly routing-around, possibly regulatory; not distinguished.
- **Deduction labels are noisy.** F1 ~0.55 at gold calibration, ~2x over-applied.
- **Class-balanced centering is much weaker** (-0.37 Qwen / -0.26 Llama) than with-vs-without.
- **No human evaluation of steered/ablated outputs.**
- **Base model sample sizes are small** (479-652 segments for non-spiraling conditions).
- **No circuit-level localization.** Single-layer intervention only.
- **Interventions do not improve reasoning.** High-alpha steering is pathological.

---

## What I would do next (couldn't, due to compute)

1. **Llama interventions.** The biggest gap. Geometry replicates cross-family; behavior is unverified cross-family.
2. **Disambiguate the ablation result.** A norm-matched random steering control, position-matched ablation, and routing-measurement would separate routing-around from a regulatory direction.
3. **Re-calibrate the annotator on Llama text.** A small Llama gold set would tell me whether the 0.77 F1 ports.
4. **A judge-independent behavioral metric** sensitive enough to detect reasoning-quality change (GPT-5.4 is used for both vector extraction labels and steered-output evaluation, circularity risk).
5. **Component/circuit localization** instead of single-layer DoM.
6. **More model families** beyond the R1-Distill recipe.

---

## Repo structure

```
data/
  final_dataset.json           # 300 training tasks
  ood.json                     # 50 OOD tasks
  r1_qwen14b_chains.json       # 300 Qwen reasoning chains

annotation/
  annotation_v2.py             # annotation prompt + core logic
  annotate_chains.py           # chain-mode annotation runner
  annotate_segments.py         # per-segment annotation runner

src/
  phase2_geometry.py           # activation extraction + DoM + robustness (preset: qwen14b | llama8b)
  phase2.5_analysis.py         # probe vs DoM, SVD depth, co-occurrence, reg sweep
  phase3_run_interventions.py  # Qwen in-dist generation (8 conditions)
  phase3_annotate_outputs.py   # segment + annotate Phase 3 outputs
  phase4_run_ood_transfer.py   # Qwen OOD + base-Qwen OOD generation
  phase4_annotation.py         # segment + annotate Phase 4 outputs
```

Large files (activations, raw outputs, Llama chains) are not committed. They can be regenerated from the scripts. Llama geometry comes from running `phase2_geometry.py` and `phase2.5_analysis.py` with `MODEL_PRESET = "llama8b"`.

---
