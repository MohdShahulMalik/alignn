# Coordination-Aware ALIGNN Experiment Plan

## Goal

Test whether explicit coordination-environment information improves this ALIGNN reimplementation beyond the current tuned baseline.

The central hypothesis is:

```text
ALIGNN can implicitly learn coordination environments from bonds and angles, but explicit coordination descriptors may improve data efficiency, accuracy, and interpretability on coordination-sensitive crystal-property targets.
```

This plan deliberately starts with cheap ablations before attempting a full polyhedron-aware architecture.

## Guiding Principle

Do not build the full architecture first. Prove that coordination information provides a measurable signal using minimal changes.

Escalation rule:

```text
Only move to a more complex representation if the previous simpler representation improves validation/test metrics consistently.
```

## Baselines

Use the current tuned ALIGNN runs as the main baseline.

For each experiment, keep these fixed unless explicitly testing them:

- Same train/val/test split
- Same target transform
- Same loss
- Same hidden dimension
- Same number of ALIGNN and GCN layers
- Same cutoff and max-neighbors
- Same training budget
- Same seed set
- Same evaluation metrics

Recommended targets:

- `dft_3d_mbj_bandgap`
- `dft_3d_bulk_modulus_kv`
- `dft_3d_ehull`
- Optional: `dft_3d_formation_energy_peratom`

Recommended metrics:

- MAE
- RMSE
- p95 absolute error
- max absolute error
- per-target improvement over tuned ALIGNN
- paired per-sample win rate

## Phase 0: Sanity And Reproducibility

Purpose: confirm current baseline results can be reproduced before adding new features.

Tasks:

- Re-run one existing tuned ALIGNN configuration on a small subset.
- Re-run one existing tuned ALIGNN configuration on a full target if compute allows.
- Confirm logs, checkpoints, and predictions are generated correctly.
- Record exact command, seed, target, and metrics.

Success criteria:

- Current training path runs without code changes.
- Metrics are in the same range as existing logs.

Estimated effort:

```text
0.5 day
```

## Phase 1: Basic Coordination Node Descriptors

Purpose: test whether simple local coordination statistics help when appended to atom/node features.

Add node-level descriptors computed from the existing neighbor graph:

- Coordination number
- Mean neighbor distance
- Standard deviation of neighbor distance
- Minimum neighbor distance
- Maximum neighbor distance
- Distance range

Implementation idea:

- Compute descriptors in `src/alignn/data/graph_builder.py` after neighbor edges are built.
- Store them in the graph dict, for example as `node_coord_features`.
- Copy them into `g.ndata` in `src/alignn/data/dgl_graph.py`.
- Add an optional encoder in `src/alignn/data/features.py`.
- Combine with atom embeddings by addition or concatenation followed by a linear projection.
- Add a CLI flag such as `--coord-features basic`.

Model variants:

```text
A: tuned ALIGNN baseline
B: ALIGNN + basic coordination descriptors
```

Run design:

- Start with one target and one seed.
- Then run 3 seeds on 2-3 targets if the first run is not clearly worse.

Success criteria:

- Improvement on at least 2 coordination-sensitive targets.
- Improvement is not only one lucky seed.
- p95 error does not degrade substantially.

Failure criteria:

- No improvement across targets.
- Large overfitting gap appears.
- Metrics improve only on training but not validation/test.

Estimated effort:

```text
1-2 days implementation
1-3 days experiments
```

## Phase 2: Angle Distribution Descriptors

Purpose: test whether summarizing the full local angular environment around each atom helps beyond pairwise line-graph angle messages.

Add node-level angle statistics around each central atom:

- Number of neighbor-neighbor angle pairs
- Mean angle cosine
- Standard deviation of angle cosine
- Minimum angle cosine
- Maximum angle cosine
- Mean absolute deviation from common ideal cosines

Useful ideal cosine references:

```text
linear: -1.0
octahedral 90 degrees: 0.0
tetrahedral: -1/3
trigonal planar: -0.5
```

Model variants:

```text
A: tuned ALIGNN baseline
B: ALIGNN + basic coordination descriptors
C: ALIGNN + basic coordination descriptors + angle distribution descriptors
```

Run design:

- Reuse the same target/seed setup from Phase 1.
- Compare C against both A and B.

Success criteria:

- C improves over A and preferably over B.
- Improvement is strongest on targets expected to depend on local geometry.

Failure criteria:

- C is worse than B, suggesting angle summaries are redundant or noisy.

Estimated effort:

```text
1 day implementation
1-3 days experiments
```

## Phase 3: Data Efficiency Test

Purpose: determine whether explicit coordination helps most when data is limited.

Run the best Phase 1/2 model against tuned ALIGNN using training subsets:

```text
10% train
25% train
50% train
100% train
```

Keep validation/test sets unchanged.

Model variants:

```text
A: tuned ALIGNN baseline
B: best coordination-descriptor ALIGNN
```

Success criteria:

- Coordination model gives larger gains in low-data regimes.
- Learning curve is better or equal at full data.

Interpretation:

- If gains appear mostly at low data, the contribution is a useful inductive bias.
- If gains persist at full data, the added features contain useful information or make optimization easier.
- If gains vanish at full data, the original ALIGNN likely learns the same information with enough examples.

Estimated effort:

```text
2-5 days experiments
```

## Phase 4: Motif Similarity Scores

Purpose: test explicit similarity to ideal coordination environments without yet adding a new graph.

Add soft motif scores as node features:

- Linear similarity
- Trigonal planar similarity
- Tetrahedral similarity
- Square planar similarity
- Octahedral similarity
- Distorted-octahedral score

Prefer continuous scores over hard labels.

Example idea:

```text
score = function(coordination_number_match, angle_deviation_from_ideal, bond_length_variance)
```

Avoid hard classification as the first implementation because coordination labels can be noisy under different cutoffs.

Model variants:

```text
A: tuned ALIGNN baseline
B: best descriptor ALIGNN from Phase 1/2
C: descriptor ALIGNN + motif similarity scores
```

Success criteria:

- Motif scores improve over basic statistics.
- Motif scores improve interpretability: high-scoring environments correspond to chemically meaningful structures.

Failure criteria:

- Hard-coded motif scores reduce generality.
- Performance improves only for one narrow family and hurts broad datasets.

Estimated effort:

```text
2-4 days implementation
2-5 days experiments
```

## Phase 5: Error Analysis And Interpretability

Purpose: understand whether gains are real and scientifically meaningful.

Analyze:

- Per-sample wins and losses against tuned ALIGNN.
- Error grouped by coordination number.
- Error grouped by motif score.
- Error grouped by composition family if practical.
- High-error structures where coordination features help most.
- High-error structures where coordination features hurt.

Questions to answer:
```text
Does the model improve specifically on structures with strong local coordination motifs?
Does it reduce tail errors?
Does it help bandgap/bulk modulus more than composition-heavy targets?
Are improvements explainable, or just noise?
```

Success criteria:

- Improvements are concentrated in chemically plausible cases.
- The added features reveal interpretable structure-property behavior.

Estimated effort:

```text
2-4 days
```

## Phase 6: Full Polyhedron-Aware Architecture

Only start this phase if descriptor experiments show consistent value.

Purpose: represent coordination environments as first-class entities rather than static node descriptors.

Possible architecture:

```text
Atom graph:
atoms are nodes, bonds are edges

Line graph:
bonds are nodes, bond angles are edges

Polyhedron graph:
coordination environments are nodes, connectivity between environments is based on shared atoms, shared edges, or shared faces
```

Minimum viable polyhedron module:

- Create one environment node per central atom.
- Environment node features summarize neighbor geometry.
- Connect atom nodes to their environment node.
- Optionally connect environment nodes if their central atoms are bonded.
- Fuse atom representation and environment representation before readout.

More advanced version:

- Detect corner-sharing, edge-sharing, and face-sharing polyhedra.
- Add environment-environment edge types.
- Message pass among polyhedra.
- Pool atom, bond, angle, and polyhedron representations jointly.

Model variants:

```text
A: tuned ALIGNN baseline
B: best descriptor ALIGNN
C: polyhedron-aware ALIGNN
```

Success criteria:

- C improves over B, not just over A.
- C gives better interpretability or lower tail errors.
- Runtime and memory remain reasonable.

Failure criteria:

- C does not beat B.
- Added complexity causes unstable training.
- Polyhedron construction is too sensitive to cutoff/max-neighbors.

Estimated effort:

```text
1-3 weeks implementation
2-6 weeks experiments and analysis
```

## Phase 7: Paper-Level Evaluation

Only start this phase if Phase 6 succeeds or Phase 1-4 produce strong descriptor-level gains.

Evaluation checklist:

- Compare against original/tuned ALIGNN under equal budget.
- Run at least 3 seeds.
- Include multiple targets.
- Include low-data learning curves.
- Include ablations for each feature group.
- Include runtime and memory comparison.
- Include interpretability/error-group analysis.

Minimum ablation table:

```text
ALIGNN
ALIGNN + coordination number/distance stats
ALIGNN + angle distribution stats
ALIGNN + motif similarity scores
ALIGNN + full polyhedron module
```

Claim strength depends on results:

```text
If only low-data gains: claim improved data efficiency.
If broad full-data gains: claim improved representation.
If only target-specific gains: claim target-specific coordination-aware modeling.
If descriptor model beats polyhedron model: do not oversell architecture; report simpler method.
```

Estimated effort:

```text
2-4 weeks after successful prototype
```

## Recommended Immediate Next Step

Implement Phase 1 only:

```text
ALIGNN + basic coordination node descriptors
```

Do not implement motif classification or a polyhedron graph yet.

The first decision point is simple:

```text
If basic descriptors improve at least two targets across multiple seeds, continue.
If they do not, full polyhedron awareness is probably not justified yet.
```
