# Multi-Task ALIGNN Experiment Plan

## Goal

Test whether a shared ALIGNN encoder trained across multiple JARVIS targets improves representation quality, data efficiency, and downstream prediction compared with separately tuned single-task ALIGNN models.

Central hypothesis:

```text
A shared ALIGNN encoder trained on thermodynamic, electronic, and mechanical targets learns transferable crystal representations that improve low-data and target-specific performance.
```

## Why This Direction

Coordination/polyhedron descriptor experiments were mixed:

- `bulk_modulus_kv` showed some descriptor gains.
- `ehull` descriptors hurt.
- `mbj_bandgap` was neutral/mixed.
- First-shell descriptors did not validate a clean coordination-awareness story.

Therefore, the next stronger direction is supervised multi-task representation learning rather than more local-geometry descriptors.

## Targets

Start with existing project targets:

```text
formation_energy_peratom
ehull
optb88vdw_bandgap
mbj_bandgap
bulk_modulus_kv
```

Target groups:

```text
thermodynamic: formation_energy_peratom, ehull
mechanical: bulk_modulus_kv
```

Optional later targets if available:

```text
shear_modulus_gv
elastic_tensor-derived properties
dielectric properties
magnetic moments
```

## Baselines

For each target compare against:

```text
single-task tuned ALIGNN
single-task default ALIGNN if useful
multi-task ALIGNN trained jointly
multi-task pretrained encoder + target-specific fine-tuning
```

Use the same splits, seeds, and training budgets wherever possible.

## Phase 0: Reproduce Single-Task Baselines

Purpose:

Confirm current single-task baselines are stable before adding multi-task training.

Tasks:

- Identify best existing config per target.
- Re-run one seed for each target if needed.
- Save baseline metrics and prediction CSVs.
- Record target transform, loss, hidden size, readout, and training budget.

Success criteria:

```text
baseline metrics match existing results closely enough to compare future runs
```

Estimated effort:

```text
0.5-1 day
```

## Phase 1: Multi-Task Dataset Loader

Purpose:

Create a dataset interface that can sample from multiple target datasets.

Implementation idea:

- Keep existing per-target split CSVs.
- Build one dataset per target using existing `JarvisGraphDataset`.
- Add a wrapper dataset that returns graph, line graph, target value, target name or target id, and jid.

Batching options:

```text
Option A: homogeneous batches, one target per batch
Option B: mixed batches with target_id per sample
```

Recommended first implementation:

```text
homogeneous target batches
```

Reason:

```text
simpler loss computation
simpler target transforms
easier debugging
```

Success criteria:

```text
can iterate over all targets and produce valid graph batches
```

Estimated effort:

```text
1 day
```

## Phase 2: Shared Encoder With Target Heads

Purpose:

Modify ALIGNN so the encoder is shared and each target has its own regression head.

Architecture:

```text
shared ALIGNN atom/bond/angle encoder
shared ALIGNN message-passing layers
shared graph readout
target-specific MLP head
```

Forward API idea:

```python
prediction = model(graph, line_graph, target_id)
```

Head options:

```text
one linear head per target
small MLP head per target
```

Recommended first version:

```text
one small MLP head per target
```

Success criteria:

```text
forward pass works for all targets
loss decreases on a tiny subset
model can overfit a small multi-target batch
```

Estimated effort:

```text
1-2 days
```

## Phase 3: Target Normalization And Loss Balancing

Purpose:

Prevent large-scale targets from dominating training.

Use per-target transforms:

```text
standardize each target using train split mean/std
optional log1p/sqrt for nonnegative skewed targets
```

Loss:

```text
mean of per-target losses
```

Initial strategy:

```text
sample target batches uniformly
standardize each target
use SmoothL1 or MSE on transformed targets
```

Avoid initially:

```text
complex uncertainty weighting
dynamic task weighting
gradient surgery
```

Add those only if basic multi-task training is unstable.

Success criteria:

```text
all target losses decrease without one target dominating
```

Estimated effort:

```text
1 day
```

## Phase 4: Joint Multi-Task Training

Purpose:

Train one model jointly on all targets.

First run:

```text
targets: formation_energy_peratom, ehull, optb88vdw_bandgap, mbj_bandgap, bulk_modulus_kv
seed: 123
train fraction: 100%
```

Then compare target-specific test metrics against single-task baselines.

Important logging:

```text
per-target train loss
per-target val MAE
per-target val RMSE
per-target test MAE
per-target test RMSE
per-target p95
best epoch per target
overall validation score
```

Model selection options:

```text
average validation MAE across standardized targets
average validation RMSE across standardized targets
target-specific checkpoint selection
```

Recommended first version:

```text
select checkpoint by average standardized validation MAE
```

Success criteria:

```text
multi-task model beats or matches baseline on at least two targets
no severe collapse on any major target
```

Estimated effort:

```text
2-4 days including experiments
```

## Phase 5: Multi-Task Supervised Pretraining + Fine-Tuning

Purpose:

Test whether the shared encoder is useful as a pretrained initialization.

Procedure:

1. Jointly train multi-task ALIGNN on all targets.
2. Save shared encoder checkpoint.
3. For each target, initialize single-task ALIGNN from shared encoder.
4. Fine-tune on that target.
5. Compare against single-task training from scratch.

Training fractions:

```text
10%
25%
50%
100%
```

Main claim if successful:

```text
multi-task supervised pretraining improves data efficiency
```

Success criteria:

```text
fine-tuned pretrained model improves low-data performance on multiple targets
```

Estimated effort:

```text
3-7 days experiments
```

## Phase 6: Low-Data Transfer Study

Purpose:

Determine whether multi-task learning helps especially when labeled target data is limited.

Run matrix:

```text
targets: all available targets
train fractions: 10%, 25%, 50%, 100%
seeds: 123, 234, 345
models:
  single-task from scratch
  multi-task joint
  multi-task pretrained + fine-tuned
```

Primary expected win condition:

```text
pretrained/fine-tuned model improves 10% and 25% regimes
```

Secondary win condition:

```text
full-data performance matches or slightly improves baseline
```

Estimated effort:

```text
1-2 weeks depending on compute
```

## Phase 7: Ablations

Purpose:

Understand what part of multi-task training matters.

Ablations:

```text
all targets
thermodynamic only
electronic only
mechanical excluded
formation_energy + target
ehull + target
bandgap targets only
```

Questions:

```text
Does formation energy help downstream targets?
Does ehull hurt because it is noisy/different?
Do electronic targets help each other?
Does bulk modulus benefit from thermodynamic/electronic pretraining?
```

Success criteria:

```text
identify which source tasks produce useful transfer
```

Estimated effort:

```text
3-7 days
```

## Phase 8: Negative Transfer Analysis

Purpose:

Detect cases where multi-task learning hurts.

Analyze:

```text
per-target metrics
per-sample wins/losses
target correlation
gradient/loss dominance
hard target groups
outlier samples
```

If negative transfer appears, try:

```text
target-specific heads with larger capacity
task sampling weights
exclude harmful tasks
two-stage pretraining then fine-tuning
freeze/unfreeze encoder schedules
```

Do not overcomplicate until negative transfer is confirmed.

## Phase 9: Paper-Level Evaluation

Minimum publishable evidence:

```text
multiple targets
multiple seeds
low-data curves
single-task baseline comparison
joint vs fine-tuned comparison
ablation of source tasks
runtime/parameter comparison
negative transfer discussion
```

Potential claims:

Strong claim:

```text
multi-task supervised pretraining improves data efficiency and transferability of ALIGNN across crystal property families
```

Moderate claim:

```text
multi-task ALIGNN improves selected low-data targets but transfer is target-dependent
```

Weak but honest claim:

```text
multi-task training reveals when ALIGNN representations transfer across materials-property families and when negative transfer occurs
```

## Implementation Order

Recommended order:

```text
1. Multi-task dataset wrapper
2. Shared encoder + target-specific heads
3. Per-target normalization
4. Tiny overfit test
5. One full joint run
6. Fine-tuning from multi-task checkpoint
7. Low-data matrix
8. Ablations and analysis
```

## Stop Conditions

Stop or pivot if:

```text
joint multi-task loses badly on most targets
fine-tuning gives no low-data gains
results are unstable across seeds
negative transfer dominates and cannot be fixed by simple task selection
```

If this happens, pivot toward:

```text
self-supervised pretraining
long-range physical features
tail-aware training
```

## Immediate Next Step

Implement the minimal multi-task prototype:

```text
shared ALIGNN encoder
homogeneous target batches
per-target standardization
one seed
all current targets
```

First decision point:

```text
Does multi-task pretraining improve at least two targets in 10% or 25% fine-tuning?
```
