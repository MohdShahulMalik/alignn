# Day 6-7 Progress

Status:

- Day 6 complete
- Day 7 complete

## Added

- [src/alignn/data/dataset.py](/home/maxum/projects/personal/python/alignn/src/alignn/data/dataset.py)
- [src/alignn/models/baseline_gnn.py](/home/maxum/projects/personal/python/alignn/src/alignn/models/baseline_gnn.py)
- [src/alignn/train/trainer.py](/home/maxum/projects/personal/python/alignn/src/alignn/train/trainer.py)
- [scripts/run_train.sh](/home/maxum/projects/personal/python/alignn/scripts/run_train.sh)

## Changed

### CLI

File: [src/alignn/cli.py](/home/maxum/projects/personal/python/alignn/src/alignn/cli.py)

Before:

```python
from alignn.data.jarvis import prepare_dataset
```

After:

```python
if args.command == "prepare":
    from alignn.data.jarvis import prepare_dataset
elif args.command == "baseline-forward":
    from alignn.train.trainer import run_baseline_forward_pass
elif args.command == "baseline-overfit":
    from alignn.train.trainer import overfit_baseline_tiny_subset
```

Before:

```python
return parser
```

After:

```python
baseline_forward = subparsers.add_parser(
    "baseline-forward",
    help="Run a baseline GNN forward pass on one prepared batch.",
)
baseline_overfit = subparsers.add_parser(
    "baseline-overfit",
    help="Train the baseline GNN on a tiny subset to confirm it can overfit.",
)
return parser
```

### Graph Builder Bug Fix

File: [src/alignn/data/graph_builder.py](/home/maxum/projects/personal/python/alignn/src/alignn/data/graph_builder.py)

Before:

```python
if index is not None and len(index) == 3:
    a, b, c = index
    return int(a), int(b), int(c)
```

After:

```python
if isinstance(index, (tuple, list)) and len(index) == 3:
    a, b, c = index
    return int(a), int(b), int(c)
```

Before:

```python
kth_distance = neighbors[max_neighbors - 1]
kept_neighbors = [
    edge for edge in neighbors if edge.distance <= kth_distance + tolerance
]
```

After:

```python
kth_distance = neighbors[max_neighbors - 1].distance
kept_neighbors = [
    edge for edge in neighbors if edge.distance <= kth_distance + tolerance
]
```

### Training Dependencies

File: [pyproject.toml](/home/maxum/projects/personal/python/alignn/pyproject.toml)

Before:

```toml
train = [
  "dgl>=2.2.1",
  "torch>=2.4.1",
]
```

After:

```toml
train = [
  "dgl==2.1.0",
  "numpy<2",
  "pydantic>=2,<3",
  "torch==2.2.1",
  "torchdata==0.7.1",
]
```

## Baseline Model

The baseline model:

- uses the atom graph only
- embeds atoms from atomic numbers
- embeds bonds from distance RBF features
- applies repeated edge-gated graph convolution layers
- mean-pools node embeddings
- predicts one scalar target with a regression head

## Verification

Verified on the synced remote GPU repo:

- baseline forward pass succeeded
- tiny-subset training succeeded
- launcher script succeeded

Artifacts written by training:

- `results/checkpoints/baseline_tiny_overfit.pt`
- `results/logs/baseline_tiny_overfit_history.csv`

## Next Step

- Day 8: implement the ALIGNN layer and full ALIGNN model
