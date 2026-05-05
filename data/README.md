# Data Layout

Place each dataset in its own subdirectory:

```text
data/
├── MDAD/
├── aBiofilm/
└── MASI/
```

Each dataset directory is expected to contain the following files:

## Required Files

- `demo_graph.txt`
  - Heterogeneous graph file.
  - Typical columns: `source`, `target`, `edge_type`, `weight`, `edge_id`.

- `demo_nodetypes.tsv`
  - Node type annotations for all nodes in the graph.

- `demo_graph_node.txt`
  - Full node list used by the heterogeneous graph.

- `demo_similarity_graph_node.txt`
  - Node list used by the similarity graph.

- `combined_mda.tsv`
  - Microbe-drug association pairs with labels.
  - Expected columns: drug, microbe/disease, label.

- `demo_similarity_graph_0_65.txt`
  - Similarity graph used by the default pipeline.

## Optional Files

- `run_similarity_graph_0.7.txt`
  - If present, some scripts prefer this tuned similarity graph automatically.

- `embeddings/`
  - Output directory generated during training.

## Notes

- Keep dataset names consistent with the scripts: `MDAD`, `aBiofilm`, and `MASI`.
- The repository does not assume a download source automatically; add dataset acquisition instructions here if you plan to make the repo public.
