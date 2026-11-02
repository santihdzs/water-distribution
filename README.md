# water-distribution

Algorithmic analysis of water distribution networks (TC2038, Situation Problem 2): all seven tasks on the NYT, HAN, FOS and PES instances.

## Run locally

```sh
git clone https://github.com/santihdzs/water-distribution
cd water-distribution
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt jupyter
jupyter lab notebooks/
```

## Run in Colab

Open any notebook from `notebooks/` in Colab (File > Open notebook > GitHub, `santihdzs/water-distribution`). The first cell clones the repo and moves into it, so every notebook runs top to bottom with no edits. Figures fall back to Matplotlib's mathtext when LaTeX is not installed.

## Contents

| notebook | covers |
|---|---|
| `00_data` | loading, validation and characterisation of the four instances |
| `01_representation` | task 1: adjacency list vs matrix, BFS/DFS, components, time and memory |
| `02_pipe_lengths` | task 2: pipe lengths, network vs straight-line distance, length queries |
| `03_sectorization` | task 3: repeated vs multi-source Dijkstra, sectors, pipes to close |
| `04_freshness` | task 4: farthest node per sector, route, before vs after sectorization |
| `05_max_flow` | task 5: Edmonds-Karp vs Dinic, min-cut bottlenecks |
| `06_sampling_route` | task 6: metric closure, nearest neighbour, 2-opt, branch and bound |
| `07_expansion` | task 7: brute force vs 2-d tree, list vs matrix update |
| `08_performance` | part 5: measured growth against theoretical complexity |

`water/` holds the shared code (loading, graph structures, shortest paths, flows, routes, spatial search, timing, plotting). `data/` holds the instances, `paper/` the report and `slides/` the presentation. Notebooks write figures to `paper/figures/` and tables to `paper/tables/`.
