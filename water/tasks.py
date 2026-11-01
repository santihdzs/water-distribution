"""the task chain shared by the notebooks: lengths (t2) -> sectors (t3) -> farthest nodes (t4)
-> max flow (t5). every notebook calls these so all results come from one implementation."""
import copy

from .flow import dinic, min_cut, pipe_flows
from .graph import adjacency_list, adjacency_matrix
from .paths import floyd_warshall, multi_source_dijkstra, path_to
from .spatial import KDTree


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


def sector_flows(net, sec, far, algorithm=dinic):
    """t5: max flow from each source to its sector's farthest node, inside the sector."""
    out = {}
    for s, f in far.items():
        pipes = [p for p in sec['kept'] if sec['origin'][p[0]] == s]
        value, r, stats = algorithm(pipes, s, f['node'])
        side, cut = min_cut(pipes, r, s)
        out[s] = {'sink': f['node'], 'value': value, 'cut': cut, 'side': side,
                  'flows': pipe_flows(pipes, r), 'pipes': pipes, 'stats': stats}
    return out


def metric_closure(net):
    """t2 and t6: all-pairs network distances over pipe lengths (floyd-warshall)."""
    M, ids, index = adjacency_matrix(net)
    D, nxt = floyd_warshall(M)
    return D, nxt, ids, index


def expand_network(net):
    """t7: insert each new node in file order, linked to the nearest non-source node of the
    network as it stands (earlier new nodes are candidates). returns the grown network and
    one record per insertion."""
    grown = copy.deepcopy(net)
    candidates = {u: xy for u, xy in grown.coords.items() if u not in grown.sources}
    tree = KDTree(candidates)
    added = []
    for x, y, cap in net.new:
        nid = max(grown.coords) + 1
        target, d, visited = tree.nearest((x, y))
        grown.coords[nid] = (x, y)
        grown.pipes.append((nid, target, cap))
        tree.insert(nid, (x, y))
        candidates[nid] = (x, y)
        added.append({'id': nid, 'xy': (x, y), 'capacity': cap, 'target': target,
                      'dist': d, 'visited': visited})
    return grown, added
