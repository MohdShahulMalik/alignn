# Graph Construction Gap Analysis

## Finding: Graph construction is functionally equivalent

Both the original and our reimplementation use k-nearest neighbor selection with
identical tie-breaking logic. The `energy_mult_natoms` failure is NOT caused by
graph construction differences.

## Root cause: Training dynamics

The `energy_mult_natoms` flag multiplies model output by atom count. The model
must learn `target/natoms` to minimize loss. This is a harder optimization problem
than directly predicting the target.

The original succeeds because:
1. Uses `keep_data_order=true` for bulk modulus (reproducible batches)
2. Uses specific optimizer parameter groups (`group_decay`)
3. Has been tuned with these settings through many experiments

## Minor differences found (non-critical)

1. Edge image handling: original stores same image for both directions; ours negates
2. frac_coords: original uses Jarvis directly; ours round-trips through Cartesian
3. ndata keys: original stores `atom_features` (92-dim); ours stores `atomic_number`
4. Bond cosine epsilon: ours adds 1e-8 for numerical stability (improvement)
