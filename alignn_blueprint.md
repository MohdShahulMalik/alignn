# Project Blueprint for a From-Scratch Reimplementation of ALIGNN

## Project Goal
The goal of this project is to build a from-scratch reimplementation of the Atomistic Line Graph Neural Network (ALIGNN) for crystal property prediction and to complete a defendable college major project within a strict timeline of 15 days.

The project should prioritize a working, well-explained implementation over full reproduction of every result in the original literature. A realistic and academically strong version of the project is:

> Build an ALIGNN-style model for one crystal property prediction task, compare it against a simpler graph baseline, and analyze whether line-graph-based bond-angle information improves predictive performance.

## Recommended Scope
The recommended project scope is:

- One dataset: JARVIS-DFT
- One target property
- Two models:
  - a baseline graph neural network without line graph
  - an ALIGNN model with line graph and angle features
- One evaluation pipeline using the same split for both models
- One analysis section explaining the effect of bond-angle message passing

This scope is realistic for 15 days and strong enough for a major project report and presentation.

### Target Selection
Only one target is necessary for a 15-day project. In practice, one well-executed target is better than two partially completed ones.

Recommended target choices:

- `formation_energy_peratom`: safer, easier, and usually more stable
- `optb88vdw_bandgap`: more intuitive for presentation, but harder

Recommended final choice:

> Use `formation_energy_peratom` as the main target unless there is a strong advisor preference for band gap.

## What ALIGNN Is
ALIGNN stands for Atomistic Line Graph Neural Network. It extends ordinary graph neural networks by explicitly modeling both pairwise and angular interactions in atomistic systems.

### Core Idea
ALIGNN uses two related graphs:

- Atomistic graph `g`: atoms are nodes and nearby atomic pairs are edges
- Line graph `L(g)`: bonds from the original graph become nodes, and bond-angle relationships become edges

This allows the model to capture:

- 2-body interactions through bond distances
- 3-body interactions through bond angles

### Why It Matters
Many earlier crystal graph neural networks focused mostly on pairwise distances. ALIGNN explicitly includes angular information, which improves sensitivity to local geometry and coordination environment.

## Primary References
The following sources are the most important references for the project:

- Original paper: <https://www.nature.com/articles/s41524-021-00650-1>
- NIST paper page: <https://www.nist.gov/publications/atomistic-line-graph-neural-network-improved-materials-property-predictions>
- Official repository: <https://github.com/usnistgov/alignn>
- ALIGNN data collection: <https://figshare.com/collections/ALIGNN_data/5429274>
- JARVIS dataset documentation: <https://jarvis-tools.readthedocs.io/en/develop/databases.html>
- JARVIS leaderboard: <https://pages.nist.gov/jarvis_leaderboard/>

## Technical Summary of the Original ALIGNN Setup
Important details from the original ALIGNN work include:

- periodic nearest-neighbor crystal graph construction
- radial basis expansion for bond distances
- radial basis expansion for bond-angle features
- 4 ALIGNN layers followed by 4 graph convolution layers in the default configuration
- hidden dimensions around 64 for embedded inputs and 256 for deeper hidden channels in the original setup
- training with AdamW, one-cycle learning rate schedule, and 300 epochs

For a 15-day project, the implementation does not need to match every original hyperparameter exactly. A reduced but faithful configuration is acceptable.

## What To Build
The minimum set of components that should be built from scratch is:

1. crystal data loader
2. periodic atom graph builder
3. line graph builder
4. atom, bond, and angle feature encoders
5. baseline graph neural network
6. ALIGNN layer and full ALIGNN model
7. training loop
8. evaluation pipeline
9. result visualization
10. report and presentation material

## What Not To Build
The following items should be excluded unless the core system is already complete and stable:

- ALIGNN-FF
- phonon or spectral prediction extensions
- multi-task training over many properties
- large-scale leaderboard reproduction
- multi-GPU or distributed training
- exact reproduction of all paper benchmarks

## Recommended Project Folder Structure
The following structure is practical and clean:

```text
alignn_project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── notebooks/
│   ├── 01_data_check.ipynb
│   ├── 02_graph_debug.ipynb
│   └── 03_results.ipynb
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── graph_builder.py
│   │   ├── line_graph.py
│   │   └── features.py
│   ├── models/
│   │   ├── baseline_gnn.py
│   │   ├── edge_gated_conv.py
│   │   ├── alignn_layer.py
│   │   └── alignn_model.py
│   ├── train/
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   └── metrics.py
│   ├── utils/
│   │   ├── seed.py
│   │   ├── config.py
│   │   └── plotting.py
│   └── main.py
├── configs/
│   ├── baseline.yaml
│   └── alignn.yaml
├── results/
│   ├── checkpoints/
│   ├── logs/
│   ├── figures/
│   └── tables/
├── report/
│   ├── outline.md
│   └── references.md
├── requirements.txt
└── README.md
```

## Recommended Tech Stack
Use the following stack unless there is a strong reason to change it:

- Python 3.10 or newer
- PyTorch
- DGL
- NumPy
- Pandas
- scikit-learn
- matplotlib or seaborn
- pymatgen or jarvis-tools for structure parsing

Best practical choice:

> Use PyTorch + DGL for the model implementation, and use pymatgen or jarvis-tools for crystal handling and graph construction support.

## Core Concepts To Understand
### Atomistic Graph
In the atomistic graph:

- each node represents an atom
- each edge represents a neighboring atomic pair

### Line Graph
In the line graph:

- each node represents a bond from the original graph
- two line-graph nodes are connected when the corresponding bonds share an atom

This construction allows the model to learn angular relationships.

### Why ALIGNN Differs From Simpler Crystal GNNs
Standard crystal GNNs often focus on distances only. ALIGNN introduces bond-angle-aware message passing by alternating updates on the line graph and the original atom graph.

## Minimal Feature Design
To keep the reimplementation faithful but feasible, use the following feature design.

### Atom Features
- atomic number embedding
- optional one-hot or learned embedding by element type

### Bond Features
- interatomic distance
- radial basis function expansion of distance

### Angle Features
- bond angle or cosine of bond angle
- radial basis function expansion of angle or cosine

### Practical Default Dimensions
- atom embedding dimension: 64
- bond RBF bins: 80
- angle RBF bins: 40
- hidden dimension: 128 or 256

If compute is limited, use hidden dimension 128 with 2 ALIGNN layers and 2 graph layers.

## Architecture Plan
### Baseline Model
The baseline model should contain:

- atom graph only
- atom embeddings
- bond-distance features
- 3 to 4 message-passing layers
- global pooling
- regression head

### ALIGNN Model
The ALIGNN model should contain:

- atom graph `g`
- line graph `L(g)`
- shared bond representations
- angle features on line-graph edges
- alternating updates on the line graph and atom graph
- graph-level pooling and regression head

## Implementation Order
The implementation should follow this order:

1. data loading
2. atom graph construction
3. line graph construction
4. feature encoding
5. baseline model
6. ALIGNN layer
7. full training pipeline
8. experiments and comparison

Do not start with the full model before verifying that graph construction and angle computation are correct.

## 15-Day Execution Plan
| Day | Main Tasks | Expected Deliverables |
| --- | --- | --- |
| 1 | Read the paper and official repository, freeze scope, choose target property, create repository structure. | Project scope statement, architecture notes, initial folders. |
| 2 | Install dependencies, download dataset or subset, inspect raw data. | Requirements file, data inspection notebook, split plan. |
| 3 | Implement periodic crystal graph builder. | Graph builder module, graph statistics and debug output. |
| 4 | Implement line graph builder and angle computation. | Line graph module, verified angle examples. |
| 5 | Implement feature encoders for atoms, bonds, and angles. | Feature module, tensor shape checks. |
| 6 | Implement baseline GNN and run forward pass. | Baseline model file, one successful batch pass. |
| 7 | Implement baseline training loop and overfit a tiny subset. | Baseline loss curve, checkpoint. |
| 8 | Implement ALIGNN layer and full ALIGNN model. | ALIGNN layer and model files, successful forward pass. |
| 9 | Train ALIGNN on a small real subset and debug issues. | First real ALIGNN training run, validation metrics. |
| 10 | Train baseline and ALIGNN on the same split. | Comparison table with initial results. |
| 11 | Tune essential hyperparameters only. | Final chosen configuration. |
| 12 | Run final experiments and generate figures. | Final plots and result tables. |
| 13 | Write methodology and implementation sections of the report. | Report draft around 60-70% complete. |
| 14 | Write results, limitations, future work, and make slides. | Near-final report and presentation deck. |
| 15 | Use as buffer for reruns, polishing, README cleanup, and viva practice. | Final submission package. |

## Experimental Setup
A practical default experimental setup is:

- split: 80/10/10
- loss: mean squared error
- metrics: MAE and RMSE
- optimizer: AdamW
- learning rate: `1e-3`
- weight decay: `1e-5`
- quick experiments: 30 to 50 epochs
- final runs: 100 to 200 epochs
- batch size: 16, 32, or 64 depending on hardware

For a 15-day schedule, matching original paper-level performance exactly is not required.

## Reduced Configuration for Limited Hardware
If the available hardware is limited, use:

- hidden dimension 128
- 2 ALIGNN layers
- 2 graph layers
- 2,000 to 10,000 training samples
- batch size 16
- one target only

This is enough for a solid and defendable project.

## Evaluation Plan
At minimum, include:

- MAE for the baseline model
- MAE for the ALIGNN model
- RMSE for both
- training time per epoch or total training time
- one scatter plot of predicted versus actual values
- one result table comparing the two models

If time permits, add one ablation:

- without angle features
- with angle features

## Recommended Report Structure
The final report can follow this structure:

1. Introduction
2. Literature Review
3. Methodology
4. Implementation Details
5. Experimental Setup
6. Results and Discussion
7. Limitations
8. Conclusion and Future Work

### Key Storyline
The report should answer this central question:

> Does explicit line-graph-based bond-angle message passing improve crystal property prediction relative to a simpler graph baseline?

## Defensible Simplifications
The following simplifications are acceptable if clearly documented:

- using one target instead of many
- using a subset of JARVIS rather than the full benchmark suite
- reducing the number of layers for faster training
- reproducing the architecture conceptually rather than matching every official utility exactly

## Main Risks and Mitigations
| Risk | Mitigation |
| --- | --- |
| Periodic neighbor graph is wrong | Verify neighbor counts and inspect a few structures manually. |
| Line graph indexing bugs | Test on tiny toy examples before full dataset runs. |
| Angle computation errors | Compare a few angles against hand calculations. |
| Model does not train | Overfit 50 to 100 samples before full training. |
| Running out of time | Finish baseline by Day 7 and ALIGNN by Day 9. |

## Expected Final Deliverables
The final submission package should contain:

- source code repository
- README with setup and run instructions
- final report PDF
- slide deck
- result plots and tables
- one small demo or inference script

## Useful README Contents
The README should include:

- project goal
- dataset and target property
- installation steps
- preprocessing instructions
- training commands for baseline and ALIGNN
- evaluation commands
- example results

## Likely Viva Questions
Prepare concise answers for the following:

- Why use a line graph?
- How is ALIGNN different from CGCNN or a standard crystal GNN?
- What are 2-body and 3-body interactions?
- Why was this target property selected?
- Why was only one target used?
- Why was a subset used instead of the full benchmark?
- What are the limitations of the implementation?
- Why did ALIGNN perform better or worse than the baseline?
- How would the project be extended with more time?

## Recommended Project Titles
Good title options include:

- From-Scratch Reimplementation of ALIGNN for Crystal Property Prediction
- Recreating the Atomistic Line Graph Neural Network for Materials Property Prediction
- Crystal Property Prediction Using a From-Scratch Atomistic Line Graph Neural Network

## Best Immediate Next Steps
The project should begin with the following sequence:

1. create the repository structure
2. choose `formation_energy_peratom`
3. set up PyTorch, DGL, and a structure-processing library
4. implement and verify periodic graph construction
5. implement and verify line graph and angle generation
6. only then start the full model

## Conclusion
This blueprint is designed to maximize the chance of a complete, technically sound, and defendable ALIGNN major project within 15 days. The strongest version of the project is not the broadest one. It is the version that correctly implements the core idea, compares it against a baseline, and explains the results clearly.
