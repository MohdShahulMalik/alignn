# Split Plan

## Chosen Task

- Dataset: `dft_3d`
- Main target: `formation_energy_peratom`
- Task type: single-target regression

## Filtering Rules

Before splitting, remove rows that do not contain:

- `jid`
- `atoms`
- `formation_energy_peratom`

This keeps the Day 3 graph-building work straightforward because every kept row already has a valid crystal structure payload.

## Split Strategy

- Train: 80%
- Validation: 10%
- Test: 10%
- Seed: `42`

The split is random but deterministic. Both the baseline model and the ALIGNN model must reuse exactly the same split files for a fair comparison later.

## Subset Strategy

For early debugging on the remote GPU instance:

- start with `10_000` filtered samples
- keep the same target and split logic
- scale to the full filtered dataset after the graph pipeline is stable

This matches the blueprint's reduced-configuration advice and avoids spending GPU time before the graph code is validated.
