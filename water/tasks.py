"""the task chain shared by the notebooks: lengths (t2) -> sectors (t3) -> farthest nodes (t4)
-> max flow (t5). every notebook calls these so all results come from one implementation."""
from .graph import adjacency_matrix
from .paths import floyd_warshall


def metric_closure(net):
    """t6 preprocessing: all-pairs network distances over pipe lengths (floyd-warshall)."""
    M, ids, index = adjacency_matrix(net)
    D, nxt = floyd_warshall(M)
    return D, nxt, ids, index
