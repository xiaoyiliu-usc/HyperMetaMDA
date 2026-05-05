# HyperMetaMDA

Official codebase for **HyperMetaMDA: A Hypergraph-Driven Dual-Branch Attention Network for Microbe-Drug Association Prediction**.

HyperMetaMDA extends a heterogeneous random-walk embedding pipeline with hypergraph-based representation refinement and dual-branch attention-style fusion for microbe-drug association (MDA) prediction. The repository includes scripts for the main experiment, ablation studies, parameter sensitivity analysis, cold-start evaluation, and case studies.

## Highlights

- Heterogeneous microbe-metabolite-drug graph embedding based on DREAMwalk-style random walks.
- Hypergraph refinement with trainable HGAT modules implemented in PyTorch.
- Dual-branch fusion and multiple ablation settings for comparing model variants.
- End-to-end prediction pipeline with XGBoost-based downstream classification.
- Utilities for reproducibility experiments, case studies, result aggregation, and plotting.

## Repository Structure

```text
HyperMetamda/
├── code/
│   ├── MetaMDA/                       # core model and utility modules
│   ├── run_demo.py                    # quick start on one dataset
│   ├── run_all_exps.py                # main ablation / benchmark runner
│   ├── run_exp1.py                    # one-click experiment 1 wrapper
│   ├── run_param_sensitivity.py       # parameter sensitivity analysis
│   ├── run_case_study.py              # case study and target-specific ranking
│   ├── run_coldstart_case.py          # cold-start evaluation
│   ├── run_classifier_ablation.py     # classifier ablation
│   ├── run_similarity_alpha_sensitivity.py
│   ├── aggregate_results.py
│   └── plot_*.py                      # plotting scripts used in analysis
├── data/
│   ├── MDAD/
│   ├── aBiofilm/
│   ├── MASI/
│   └── README.md                      # expected data layout
├── runs/                              # recommended output directory (generated locally)
├── requirements.txt
└── README.md
```

## Environment

Tested with Python 3.9.

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Required Input Data

This repository expects dataset-specific files under:

```text
data/MDAD/
data/aBiofilm/
data/MASI/
```

Each dataset directory should contain:

- `demo_graph.txt`
- `demo_nodetypes.tsv`
- `demo_graph_node.txt`
- `demo_similarity_graph_node.txt`
- `combined_mda.tsv`
- `demo_similarity_graph_0_65.txt`

Optional tuned similarity graph files can also be provided, such as:

- `run_similarity_graph_0.7.txt`

See [data/README.md](data/README.md) for the expected file roles.

## Quick Start

Run the default demo on one dataset:

```bash
cd code
python run_demo.py
```

By default, `run_demo.py` uses:

- dataset: `MDAD`
- hypergraph refinement: enabled
- classifier: `XGBoost`

Outputs are typically written under the corresponding `data/<dataset>/` directory, for example:

- `embedding_file.pkl`
- `mda.xlsx`

## Main Experiment

Run the main experiment suite on one or more datasets:

```bash
cd code
python run_all_exps.py --datasets MDAD aBiofilm MASI --cv 5 --seeds 42 --no_plot
```

This script evaluates several predefined settings, including:

- `main_hgat`
- `w_o_hypergraph`
- `avg_fusion`
- `w_o_path_hyperedge`
- `knn_only_hyperedge`

Results are appended to:

```text
runs/results.csv
```

## Parameter Sensitivity Analysis

Example:

```bash
cd code
python run_param_sensitivity.py \
  --dataset MDAD \
  --param hg_tau \
  --values 0.30 0.40 0.50 0.60 0.70 \
  --cv 5 \
  --seed 42
```

Supported parameters include:

- `t`
- `window_size`
- `dimension`
- `hg_tau`
- `hg_k`
- `hg_layers`
- `hg_gate_lambda`

## Case Study

Example:

```bash
cd code
python run_case_study.py --dataset MDAD --cases tetracycline vancomycin --top_k 20
```

The script supports:

- predefined cases
- exact drug names
- query-based target resolution
- microbe-target case studies

## Key Scripts

| Script | Purpose |
| --- | --- |
| `code/run_demo.py` | Minimal end-to-end run for one dataset |
| `code/run_all_exps.py` | Main batch experiment runner |
| `code/run_exp1.py` | One-click wrapper for experiment 1 |
| `code/run_param_sensitivity.py` | Hyperparameter sensitivity analysis |
| `code/run_case_study.py` | Drug/microbe case studies |
| `code/run_coldstart_case.py` | Cold-start evaluation |
| `code/run_classifier_ablation.py` | Classifier comparison |
| `code/aggregate_results.py` | Merge and summarize experiment outputs |

## Core Modules

| Module | Role |
| --- | --- |
| `code/MetaMDA/generate_embeddings_mmd.py` | Graph embedding generation and hypergraph refinement entry point |
| `code/MetaMDA/hypergraph_module.py` | Hypergraph construction and helper functions |
| `code/MetaMDA/hypergraph_trainable_hgat.py` | Trainable HGAT refinement implementation |
| `code/MetaMDA/predict_final.py` | Downstream prediction and scoring |
| `code/MetaMDA/classifier.py` | Classifier backends |
| `code/MetaMDA/experiment_io.py` | Structured experiment logging |

## Reproducibility Notes

- The repository currently contains code and directory templates for the datasets, but the dataset files themselves may not be included in the public package.
- Random seeds are exposed in the main scripts and should be fixed for reproducible comparisons.
- Large generated files such as embeddings, result tables, and figures are intentionally excluded from version control by `.gitignore`.

## Contact

Please update this section with the corresponding author or project maintainer information before publishing the repository.
