"""the task chain shared by the notebooks: lengths (t2) -> sectors (t3) -> farthest nodes (t4)
-> max flow (t5). every notebook calls these so all results come from one implementation."""
from .graph import adjacency_list, adjacency_matrix
from .paths import floyd_warshall, multi_source_dijkstra, path_to


def sectorize(net):
    """t3: assign every node to its closest source (multi-source dijkstra over pipe lengths).
    pipes whose endpoints land in different sectors are the ones to close."""
    adj = adjacency_list(net)
    dist, origin, parent, stats = multi_source_dijkstra(adj, net.sources)
    kept = [p for p in net.pipes if origin[p[0]] == origin[p[1]]]
    closed = [p for p in net.pipes if origin[p[0]] != origin[p[1]]]
    members = {s: sorted(u for u in net.coords if origin[u] == s) for s in net.sources}
    return {'dist': dist, 'origin': origin, 'parent': parent, 'kept': kept,
            'closed': closed, 'members': members, 'stats': stats}


def farthest(net, sec):
    """t4: per sector, the node with the largest shortest-path distance from its source."""
    out = {}
    for s, nodes in sec['members'].items():
        t = max(nodes, key=lambda u: (sec['dist'][u], -u))
        out[s] = {'node': t, 'dist': sec['dist'][t], 'route': path_to(sec['parent'], t)}
    return out


def metric_closure(net):
    """t6 preprocessing: all-pairs network distances over pipe lengths (floyd-warshall)."""
    M, ids, index = adjacency_matrix(net)
    D, nxt = floyd_warshall(M)
    return D, nxt, ids, index
