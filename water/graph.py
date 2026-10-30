"""adjacency list and adjacency matrix representations, traversals, components."""
from collections import deque

import numpy as np


def adjacency_list(net, pipes=None):
    """dict id -> list of (neighbor, length). pipes defaults to all pipes of net."""
    adj = {u: [] for u in net.coords}
    for u, v, _ in (net.pipes if pipes is None else pipes):
        w = net.length(u, v)
        adj[u].append((v, w))
        adj[v].append((u, w))
    return adj


def adjacency_matrix(net, pipes=None):
    """(M, ids, index): M[i, j] is the pipe length, 0 where there is no pipe."""
    ids = net.nodes
    index = {u: i for i, u in enumerate(ids)}
    M = np.zeros((len(ids), len(ids)))
    for u, v, _ in (net.pipes if pipes is None else pipes):
        M[index[u], index[v]] = M[index[v], index[u]] = net.length(u, v)
    return M, ids, index


def bfs_list(adj, s):
    parent = {s: None}
    order, queue = [], deque([s])
    while queue:
        u = queue.popleft()
        order.append(u)
        for v, _ in adj[u]:
            if v not in parent:
                parent[v] = u
                queue.append(v)
    return order, parent


def bfs_matrix(M, s):
    """s is a row index. scanning a full row per vertex makes this O(V^2)."""
    n = len(M)
    seen = [False] * n
    seen[s] = True
    order, queue = [], deque([s])
    while queue:
        u = queue.popleft()
        order.append(u)
        row = M[u]
        for v in range(n):
            if row[v] and not seen[v]:
                seen[v] = True
                queue.append(v)
    return order


def dfs_list(adj, s):
    seen, order, stack = set(), [], [s]
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        order.append(u)
        stack.extend(v for v, _ in reversed(adj[u]) if v not in seen)
    return order


def dfs_matrix(M, s):
    n = len(M)
    seen, order, stack = [False] * n, [], [s]
    while stack:
        u = stack.pop()
        if seen[u]:
            continue
        seen[u] = True
        order.append(u)
        row = M[u]
        stack.extend(v for v in range(n - 1, -1, -1) if row[v] and not seen[v])
    return order


def components(adj):
    """connected components as a list of sorted id lists, via repeated bfs."""
    seen, comps = set(), []
    for s in sorted(adj):
        if s in seen:
            continue
        order, _ = bfs_list(adj, s)
        seen.update(order)
        comps.append(sorted(order))
    return comps


def components_matrix(M):
    n, seen, comps = len(M), set(), []
    for s in range(n):
        if s not in seen:
            order = bfs_matrix(M, s)
            seen.update(order)
            comps.append(sorted(order))
    return comps
