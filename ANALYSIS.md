# 2D-ALIGNN Codebase Analysis Document

## Overview

**ALIGNN** (Atomistic Line Graph Neural Network) is a deep learning framework for predicting materials properties and training machine learning force fields. It was developed at NIST (National Institute of Standards and Technology) and is part of the JARVIS (Joint Automated Repository for Various Integrated Simulations) ecosystem.

The core innovation of ALIGNN is the **Line Graph** representation that explicitly models both:
- **Two-body interactions** (pairwise/bond interactions) 
- **Three-body interactions** (angle/triplet interactions)

This is achieved by composing edge-gated graph convolution layers applied to:
1. The atomistic line graph L(g) - representing triplet interactions
2. The atomistic bond graph g - representing pair interactions

---

## Directory Structure

```
2d-alignn/
├── alignn/                    # Main package
│   ├── __init__.py            # Version info (2025.4.1)
│   ├── config.py              # Pydantic models for training config
│   ├── data.py                # Data loading utilities
│   ├── dataset.py             # Dataset classes for graph data
│   ├── graphs.py              # Graph construction from crystal structures
│   ├── utils.py               # Training utilities
│   ├── train.py               # Main training loop
│   ├── train_alignn.py        # CLI entry point for training
│   ├── train_props.py         # Additional training utilities
│   ├── pretrained.py          # Pretrained model management
│   ├── run_alignn_ff.py      # ALIGNN-FF execution script
│   ├── lmdb_dataset.py        # LMDB-based dataset for large-scale training
│   ├── profiler.py            # Performance profiling utilities
│   ├── cli.py                 # Command-line interface
│   │
│   ├── models/                # Neural network architectures
│   │   ├── __init__.py
│   │   ├── alignn.py          # ALIGNN model (graph-wise prediction)
│   │   ├── alignn_atomwise.py # ALIGNN-FF model (atom-wise prediction + forces)
│   │   ├── ealignn_atomwise.py# Extended ALIGNN-FF variant
│   │   └── utils.py           # Model utilities (RBF expansion, etc.)
│   │
│   ├── ff/                    # Force field module
│   │   ├── __init__.py
│   │   ├── ff.py              # ALIGNN-FF main class & utilities
│   │   ├── calculators.py     # ASE calculator interface
│   │   ├── all_models_alignn.json         # Pretrained FF models
│   │   └── all_models_alignn_atomwise.json # Pretrained atom-wise models
│   │
│   ├── scripts/               # Training & analysis scripts (40+ scripts)
│   │   ├── train_all_jv.py    # Train on JARVIS-DFT
│   │   ├── train_all_mp.py    # Train on Materials Project
│   │   ├── train_all_qm9.py   # Train on QM9 molecular dataset
│   │   ├── ev_curve.py        # Equation of state calculations
│   │   ├── phonons.py         # Phonon calculations
│   │   ├── predict.py         # Prediction utilities
│   │   └── ... (many more)
│   │
│   ├── tests/                 # Test suite
│   │   ├── test_prop.py
│   │   ├── test_eprop.py
│   │   ├── test_alignn_ff.py
│   │   └── test_force_reduction.py
│   │
│   ├── examples/              # Example data & scripts
│   │   ├── sample_data/       # Regression example data
│   │   ├── sample_data_ff/    # Force field example data
│   │   └── sample_data_multi_prop/  # Multi-property example
│   │
│   └── tex/                  # LaTeX documentation & figures
│
├── setup.py                   # Package setup
├── pyproject.toml            # Build configuration
├── environment.yml            # Conda environment
└── README.md                  # Main documentation
```

---

## Core Modules Explained

### 1. Graph Construction (`alignn/graphs.py`)

**Purpose**: Convert crystal structures (Atoms objects) into DGL (Deep Graph Library) graphs with line graph representation.

**Key Functions**:

| Function | Purpose |
|----------|---------|
| `Graph.atom_dgl_multigraph()` | Main entry point - creates crystal graph + line graph |
| `nearest_neighbor_edges()` | Build k-NN edge list with periodic boundary conditions |
| `radius_graph()` | Alternative: build edges based on distance cutoff |
| `build_undirected_edgedata()` | Convert edge set to undirected graph |
| `compute_bond_cosines()` | Calculate bond angle cosines for line graph edges |
| `StructureDataset` | PyTorch Dataset class for crystal graphs |

**Graph Representation**:
- Nodes = Atoms (features: atomic number, element properties)
- Edges = Bonds (features: distance, displacement vectors)
- Line graph nodes = Bonds
- Line graph edges = Bond pairs (angles)

---

### 2. Data Loading (`alignn/data.py`, `alignn/dataset.py`, `alignn/lmdb_dataset.py`)

**Purpose**: Load and manage training/validation/test data from JARVIS databases or user-provided datasets.

**Key Functions**:

| Function | Purpose |
|----------|---------|
| `load_dataset()` | Load data from JARVIS figshare |
| `get_train_val_loaders()` | Create train/val/test DataLoaders |
| `get_id_train_val_test()` | Split dataset by ratio or absolute numbers |
| `get_torch_dataset()` | Create PyTorch dataset from structures |

**Data Formats Supported**:
- POSCAR, CIF, XYZ, PDB crystal structure files
- id_prop.csv (for property prediction)
- id_prop.json (for force fields with energy/forces/stress)

---

### 3. Neural Network Models (`alignn/models/`)

#### 3.1 ALIGNN (`alignn/models/alignn.py`)

**Purpose**: Graph-level property prediction (formation energy, bandgap, etc.)

**Architecture**:
```
Input: atom_features (CGCNN 92-dim or atomic number)
       │
       ├─► Atom Embedding (MLP) ─┐
       │                         │
       ├─► Edge Embedding (RBF+MLP) ─┼─► ALIGNN layers (4x) ─┐
       │                             │                      │
       └─► Angle Embedding (RBF+MLP) ──► Line Graph Conv ──┘
                                                        │
                                              Gated GCN layers (4x)
                                                        │
                                              AvgPooling ──► FC ──► Output
```

**Key Classes**:
- `ALIGNNConfig`: Hyperparameters (layers, features, etc.)
- `EdgeGatedGraphConv`: Edge-gated graph convolution layer
- `ALIGNNConv`: Combined node + edge update for line graph
- `ALIGNN`: Main model class

---

#### 3.2 ALIGNN-FF (`alignn/models/alignn_atomwise.py`)

**Purpose**: Atom-wise prediction with forces, stresses - used for machine learning force fields

**Architecture Differences from ALIGNN**:
- Outputs: graph energy + atom forces + stresses + atom-wise properties
- Uses automatic differentiation for force calculation
- Supports cutoff functions for molecular dynamics
- Can train on multiple targets simultaneously (energy, forces, stress, charges)

**Key Configuration Parameters**:
```python
gradwise_weight: float = 1.0     # Weight for force loss
stresswise_weight: float = 0.0  # Weight for stress loss
atomwise_weight: float = 0.0    # Weight for atom-wise property loss
calculate_gradient: bool = True  # Compute forces via autograd
use_cutoff_function: bool = False  # Apply smooth cutoff
inner_cutoff: float = 3  # Angstrom
```

---

### 4. Training (`alignn/train.py`, `alignn/train_alignn.py`)

**Purpose**: Main training loop for ALIGNN models

**Training Flow**:
1. Load configuration from TrainingConfig
2. Create train/val/test dataloaders
3. Initialize model based on config.model.name
4. Set up optimizer (AdamW or SGD) + scheduler (OneCycleLR)
5. For each epoch:
   - Training: compute loss (graph + atom + gradient + stress + additional)
   - Validation: evaluate on validation set
   - Save best model based on validation loss
6. Generate predictions on test set

**Multi-Target Loss Support**:
```python
loss = graphwise_weight * loss_energy \
     + atomwise_weight * loss_atomwise \
     + gradwise_weight * loss_forces \
     + stresswise_weight * loss_stress \
     + additional_output_weight * loss_additional
```

---

### 5. Pretrained Models (`alignn/pretrained.py`)

**Purpose**: Download and use pretrained ALIGNN models for prediction

**Available Models** (40+ models):
- JARVIS-DFT trained models: formation energy, bandgap, bulk/shear modulus, etc.
- Materials Project models: formation energy, bandgap
- QM9 molecular models: 12 properties (HOMO, LUMO, gap, alpha, etc.)
- hMOF models: surface area, CO2 adsorption
- OCP (Open Catalyst Project) models

**Usage**:
```python
from alignn.pretrained import get_prediction
atoms = Atoms.from_poscar("structure.vasp")
result = get_prediction(
    model_name="jv_formation_energy_peratom_alignn",
    atoms=atoms
)
```

---

### 6. ALIGNN-FF Force Field (`alignn/ff/`)

**Purpose**: Use trained ALIGNN-FF models for molecular dynamics and structure optimization

#### 6.1 ASE Calculator (`alignn/ff/calculators.py`)

**Purpose**: Interface ALIGNN-FF with ASE (Atomic Simulation Environment)

```python
from alignn.ff.ff import AlignnAtomwiseCalculator, default_path

calc = AlignnAtomwiseCalculator(path=default_path())
atoms.calc = calc
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

#### 6.2 ForceField Class (`alignn/ff/ff.py`)

**Purpose**: High-level interface for FF calculations

| Method | Purpose |
|--------|---------|
| `optimize_atoms()` | Structure optimization |
| `run_nve_velocity_verlet()` | NVE molecular dynamics |
| `run_nvt_langevin()` | NVT Langevin dynamics |
| `run_npt_berendsen()` | NPT ensemble |
| `ev_curve()` | Equation of state calculation |
| `phonons()` | Phonon bandstructure + DOS |
| `vacancy_formation()` | Defect energy calculation |
| `surface_energy()` | Surface energy calculation |
| `get_interface_energy()` | Interface adhesion calculation |

---

### 7. Configuration (`alignn/config.py`)

**Purpose**: Pydantic-based configuration with validation

**TrainingConfig Parameters**:
```python
# Dataset
dataset: str                    # "dft_3d", "qm9", "mp_3d_2020", etc.
target: str                    # Property to predict
atom_features: str             # "basic", "atomic_number", "cfid", "cgcnn"
neighbor_strategy: str         # "k-nearest", "radius_graph"

# Training
epochs: int = 300
batch_size: int = 64
learning_rate: float = 1e-2
weight_decay: float = 0

# Model architecture
model: ALIGNNConfig | ALIGNNAtomWiseConfig | eALIGNNAtomWiseConfig

# Graph construction
cutoff: float = 8.0            # Distance cutoff for neighbors
max_neighbors: int = 12
```

---

## Key Concepts

### 1. Line Graph Neural Network

The key innovation of ALIGNN is using the **line graph** L(g) of the crystal graph g:
- In the original graph g: nodes = atoms, edges = bonds
- In the line graph L(g): nodes = bonds, edges = bond pairs (angles)

This allows the network to:
- Explicitly learn three-body interactions (angles)
- Share edge features across triplet interactions

### 2. Edge-Gated Graph Convolution

Used instead of standard message passing:
```
h_i^(l+1) = ReLU(U·h_i + Σ_{j→i} η_{ij} ⊙ V·h_j)

where η_{ij} = σ(W·[h_i || h_j || e_ij])
```

### 3. Radial Basis Function (RBF) Expansion

Used to encode bond lengths:
```
RBF(r) = exp(-(r - μ)² / (2σ²))
```

This provides smooth, differentiable representations of bond distances.

### 4. LMDB Dataset

For large-scale training (300K+ structures), ALIGNN uses LMDB (Lightning Memory-Mapped Database) to:
- Store pre-computed graphs on disk
- Reduce memory footprint
- Speed up data loading

---

## Usage Examples

### 1. Train a Property Prediction Model

```bash
train_alignn.py \
  --root_dir "alignn/examples/sample_data" \
  --config "alignn/examples/sample_data/config_example.json" \
  --output_dir=temp
```

### 2. Train a Force Field

```bash
train_alignn.py \
  --root_dir "alignn/examples/sample_data_ff" \
  --config "alignn/examples/sample_data_ff/config_example_atomwise.json" \
  --output_dir="temp"
```

### 3. Use Pretrained Model

```python
from alignn.pretrained import get_prediction
from jarvis.core.atoms import Atoms

atoms = Atoms.from_poscar("POSCAR")
result = get_prediction(
    model_name="jv_formation_energy_peratom_alignn",
    atoms=atoms
)
```

### 4. Structure Optimization with ALIGNN-FF

```python
from alignn.ff.ff import AlignnAtomwiseCalculator, ForceField
from jarvis.core.atoms import Atoms

atoms = Atoms.from_poscar("POSCAR")
ff = ForceField(jarvis_atoms=atoms)
relaxed, energy, forces = ff.optimize_atoms(optimizer="FIRE")
```

### 5. Molecular Dynamics

```python
ff = ForceField(jarvis_atoms=atoms)
trajectory = ff.run_nvt_langevin(
    temperature_K=300,
    steps=10000
)
```

---

## Supported Datasets

| Dataset | Description | Typical Target |
|---------|-------------|----------------|
| dft_3d | JARVIS-DFT (3D materials) | formation_energy, bandgap |
| mp_3d_2020 | Materials Project 2020 | formation_energy, bandgap |
| qm9 | QM9 molecular database | 12 molecular properties |
| oqmd_3d | Open Quantum Materials DB | formation_energy |
| hMOF | Hypothetical MOFs | surface_area, CO2 adsorption |
| hmof | Experimental MOFs | Various |
| pdbbind | Protein-ligand binding | Binding affinity |

---

## Model Performance (from README)

ALIGNN achieves state-of-the-art results on multiple datasets:

**JARVIS-DFT**:
- Formation energy MAE: 0.033 eV/atom
- Bandgap MAE: 0.14 eV

**Materials Project**:
- Formation energy MAE: 0.022 eV/atom
- Bandgap MAE: 0.218 eV

**QM9**:
- Various properties competitive with DimeNet++

---

## Installation

```bash
# Create conda environment
conda create --name alignn python=3.10
conda activate alignn

# Install dependencies
conda install dgl pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia

# Install ALIGNN
pip install alignn

# Or install from source
git clone https://github.com/usnistgov/alignn
cd alignn
pip install -e .
```

---

## Dependencies

- **torch** (≤2.2.1)
- **dgl** (≤1.1.1) - Deep Graph Library
- **jarvis-tools** - JARVIS database access
- **ase** - Atomic Simulation Environment
- **pydantic** - Configuration validation
- **lmdb** - Database for large datasets
- **scikit-learn** - Preprocessing
- **matplotlib** - Plotting

---

## Testing

```bash
# Run tests
pytest alignn/tests/

# Or run specific test
python -m pytest alignn/tests/test_prop.py -v
```

---

## References

1. [Atomistic Line Graph Neural Network (Nature 2021)](https://www.nature.com/articles/s41524-021-00650-1)
2. [ALIGNN-FF (RSC Digital Discovery 2023)](https://pubs.rsc.org/en/content/articlehtml/2023/dd/d2dd00096b)
3. [JARVIS-Leaderboard](https://pages.nist.gov/jarvis_leaderboard/)
4. [JARVIS Tools](https://jarvis-tools.readthedocs.io/)

---

## License

MIT License - See LICENSE.rst

---

*Document generated from codebase analysis*
