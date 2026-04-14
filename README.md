# ALIGNN Reimplementation

This repository tracks a from-scratch ALIGNN-style crystal property prediction project based on `alignn_blueprint.md`.

Current scope:

- dataset: JARVIS-DFT `dft_3d`
- target: `formation_energy_peratom`
- comparison: baseline graph model vs ALIGNN

## Day 2 Deliverables

Day 2 focuses on environment setup, dataset download or subset preparation, raw data inspection, and split planning.

Implemented here:

- `uv`-ready project configuration in [pyproject.toml](/home/maxum/projects/personal/python/alignn/pyproject.toml)
- dataset preparation CLI in [src/alignn/cli.py](/home/maxum/projects/personal/python/alignn/src/alignn/cli.py)
- JARVIS download and inspection helpers in [src/alignn/data/jarvis.py](/home/maxum/projects/personal/python/alignn/src/alignn/data/jarvis.py)
- deterministic split generator in [src/alignn/data/splits.py](/home/maxum/projects/personal/python/alignn/src/alignn/data/splits.py)
- starter inspection notebook in [notebooks/01_data_check.ipynb](/home/maxum/projects/personal/python/alignn/notebooks/01_data_check.ipynb)
- split plan in [report/split_plan.md](/home/maxum/projects/personal/python/alignn/report/split_plan.md)

## Remote GPU Workflow

This machine does not have a GPU, so treat the AWS instance reached with `ssh aliggn` as the training machine. The repo is safe to sync with Mutagen because Day 2 only creates source files, metadata, and local data directories.

Recommended remote setup:

1. SSH into the GPU box.
2. Install `uv` if it is missing.
3. From the synced repo, create the environment and install the base dependencies:

```bash
uv sync
```

4. Install the training stack on the remote host after checking its CUDA version:

```bash
nvidia-smi
uv sync --extra train
```

If the remote host needs CUDA-specific PyTorch wheels, use the matching PyTorch index URL instead of the default wheel source. Keep that step on the GPU machine only.

## Day 2 Commands

Download JARVIS-DFT, build an inspection summary, and create the project split:

```bash
uv run alignn prepare \
  --dataset dft_3d \
  --target formation_energy_peratom \
  --max-samples 10000
```

Use the full dataset instead of a capped subset:

```bash
uv run alignn prepare \
  --dataset dft_3d \
  --target formation_energy_peratom
```

The command writes:

- raw records to `data/raw`
- a CSV summary to `data/processed`
- split CSV files to `data/splits`
- an inspection JSON report to `results/tables`

## Notes

- The default target is `formation_energy_peratom`, matching the blueprint recommendation.
- The default split is `80/10/10` with a fixed random seed.
- The split happens after filtering out rows missing the requested target or `atoms`.

## Local Development

This repo also works locally with `uv`, but the local machine should use Python `3.12`, not the Homebrew `3.14` interpreter.

Create or rebuild the local environment with:

```bash
UV_CACHE_DIR=.uv-cache uv sync --python /usr/bin/python3.12
```

If you want to run commands directly from the local venv, these work:

```bash
.venv/bin/alignn --help
XDG_CACHE_HOME=.cache MPLCONFIGDIR=.cache/matplotlib .venv/bin/python -c "import jarvis.core.atoms, pymatgen.core"
```

VS Code settings are included so the workspace should pick `${workspaceFolder}/.venv/bin/python` automatically.
