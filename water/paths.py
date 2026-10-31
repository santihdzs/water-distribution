"""shortest paths: dijkstra (single, repeated, multi-source) and floyd-warshall."""
import heapq
import math

import numpy as np


def dijkstra(adj, s):
    """greedy: settle the closest unsettled vertex, relax its pipes. binary heap, lazy deletion."""
    dist, parent = {s: 0.0}, {s: None}
    done, heap = set(), [(0.0, s)]
    pops = relax = 0
    while heap:
        d, u = heapq.heappop(heap)
        pops += 1
        if u in done:
            continue
        done.add(u)
        for v, w in adj[u]:
            relax += 1
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v], parent[v] = nd, u
                heapq.heappush(heap, (nd, v))
    return dist, parent, {'pops': pops, 'relaxations': relax}


def repeated_dijkstra(adj, sources):
    """one dijkstra per source, then each node takes the argmin (ties go to the earlier source)."""
    runs = [dijkstra(adj, s) for s in sources]
    dist, origin, parent = {}, {}, {}
    for u in adj:
        k = min(range(len(sources)), key=lambda k: (runs[k][0].get(u, math.inf), k))
        dist[u], origin[u], parent[u] = runs[k][0].get(u, math.inf), sources[k], runs[k][1].get(u)
    stats = {key: sum(r[2][key] for r in runs) for key in ('pops', 'relaxations')}
    return dist, origin, parent, runs, stats


def multi_source_dijkstra(adj, sources):
    """all sources start in the heap at distance 0; each node keeps the source that reached it first."""
    rank = {s: k for k, s in enumerate(sources)}
    dist = {s: 0.0 for s in sources}
    origin = {s: s for s in sources}
    parent = {s: None for s in sources}
    done, heap = set(), [(0.0, rank[s], s) for s in sources]
    heapq.heapify(heap)
    pops = relax = 0
    while heap:
        d, r, u = heapq.heappop(heap)
        pops += 1
        if u in done:
            continue
        done.add(u)
        for v, w in adj[u]:
            relax += 1
            nd = d + w
            # rank breaks exact distance ties the same way repeated_dijkstra does
            if (nd, r) < (dist.get(v, math.inf), rank[origin[v]] if v in origin else math.inf):
                dist[v], origin[v], parent[v] = nd, origin[u], u
                heapq.heappush(heap, (nd, r, v))
    return dist, origin, parent, {'pops': pops, 'relaxations': relax}


def path_to(parent, t):
    route = []
    while t is not None:
        route.append(t)
        t = parent[t]
    return route[::-1]


def floyd_warshall(M):
    """dp over intermediate vertices: D_k[i,j] = min(D_{k-1}[i,j], D_{k-1}[i,k] + D_{k-1}[k,j]).
    M is an adjacency matrix with 0 for no pipe. each k step is one vectorised row/column update.
    returns distances and a next-hop matrix for route reconstruction."""
    n = len(M)
    D = np.where(M > 0, M, np.inf)
    np.fill_diagonal(D, 0.0)
    nxt = np.where(np.isfinite(D), np.arange(n)[None, :], -1)
    for k in range(n):
        via = D[:, k:k + 1] + D[k:k + 1, :]
        better = via < D
        D = np.where(better, via, D)
        nxt = np.where(better, nxt[:, k:k + 1], nxt)
    return D, nxt


def fw_path(nxt, i, j):
    """index route i -> j from the floyd-warshall next-hop matrix."""
    if nxt[i, j] < 0:
        return []
    route = [i]
    while i != j:
        i = int(nxt[i, j])
        route.append(i)
    return route
