# Strategic Geometry v2

Rebuild of "The Internal Geometry of Strategic Reasoning in Thinking Language Models" as a
falsification-first study. The v1 headline (`cos(u_opp, u_deduction) = -0.707`, "antagonistic
geometry") is most plausibly an **annotation-complementarity artifact**; this redo tests that
claim *before* building any interpretation on top of it.

**Core idea:** a difference-of-means (DoM) contrast between two near-complementary annotator
labels is anti-aligned largely by construction. Stage A decides — on existing activations, no
GPU — whether the opponent-modeling vs deduction anti-alignment exceeds a complementarity
baseline. The result selects one of three papers:

- **A** positive geometry + causal paper (effect survives),
- **B** deflationary methods paper ("DoM behavioral directions encode annotation co-occurrence
  structure, not model features"),
- **C** partial — decompose how much of the headline is artifact vs residual.

See [`SPEC.md`](./SPEC.md) for the full, detailed protocol (every experiment has
hypothesis / procedure / expected-if-real-vs-artifact / control / decision rule / sanity checks).

## Status

Bootstrapping. CPU/data layer first (dataset migration + Stage-A analysis modules); GPU steps
(chain/activation regeneration, interventions) run on the host with `torch`/`transformers`.

## Layout

```
data/        unified dataset (v2 schema); large artifacts gitignored
src/strat_geom/   library (config, dataset, dom, nulls, calibration, probe, intervene/, ...)
scripts/     runnable entry points (migrate_dataset, run_stage1, run_interventions, ...)
tests/       CPU unit/smoke tests (pytest)
```

## Quickstart (CPU analysis)

```bash
pip install -r requirements.txt
python scripts/migrate_dataset.py --in-dir ../path/to/v1/data --out-dir data
pytest -q
```
