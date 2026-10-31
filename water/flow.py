"""maximum flow on undirected pipe networks: edmonds-karp, dinic, minimum cut."""
from collections import deque


def residual_graph(pipes):
    """an undirected pipe of capacity c is two opposite arcs of capacity c.
    pushing x along u->v lowers r[u][v] by x and raises r[v][u] by x."""
    r = {}
    for u, v, c in pipes:
        r.setdefault(u, {})
        r.setdefault(v, {})
        r[u][v] = r[u].get(v, 0.0) + c
        r[v][u] = r[v].get(u, 0.0) + c
    return r


def edmonds_karp(pipes, s, t):
    """ford-fulkerson with bfs augmenting paths: each augmentation uses a shortest
    path (in arcs) of the residual graph."""
    r = residual_graph(pipes)
    value, augmentations = 0.0, 0
    while True:
        parent = {s: None}
        queue = deque([s])
        while queue and t not in parent:
            u = queue.popleft()
            for v, cap in r[u].items():
                if cap > 1e-12 and v not in parent:
                    parent[v] = u
                    queue.append(v)
        if t not in parent:
            break
        path, v = [], t
        while parent[v] is not None:
            path.append((parent[v], v))
            v = parent[v]
        push = min(r[u][v] for u, v in path)
        for u, v in path:
            r[u][v] -= push
            r[v][u] += push
        value += push
        augmentations += 1
    return value, r, {'augmentations': augmentations}


def dinic(pipes, s, t):
    """bfs builds a level graph, dfs pushes a blocking flow along level-increasing arcs.
    current-arc pointers make each phase O(VE)."""
    r = residual_graph(pipes)
    value, phases, augmentations = 0.0, 0, 0
    while True:
        level = {s: 0}
        queue = deque([s])
        while queue:
            u = queue.popleft()
            for v, cap in r[u].items():
                if cap > 1e-12 and v not in level:
                    level[v] = level[u] + 1
                    queue.append(v)
        if t not in level:
            break
        phases += 1
        arcs = {u: list(r[u]) for u in r}
        ptr = {u: 0 for u in r}

        def push(u, limit):
            if u == t:
                return limit
            while ptr[u] < len(arcs[u]):
                v = arcs[u][ptr[u]]
                if r[u][v] > 1e-12 and level.get(v) == level[u] + 1:
                    sent = push(v, min(limit, r[u][v]))
                    if sent > 0:
                        r[u][v] -= sent
                        r[v][u] += sent
                        return sent
                ptr[u] += 1
            return 0.0

        while (sent := push(s, float('inf'))) > 0:
            value += sent
            augmentations += 1
    return value, r, {'phases': phases, 'augmentations': augmentations}


def min_cut(pipes, r, s):
    """source side = vertices reachable from s in the final residual graph.
    the pipes crossing it are saturated and their capacities sum to the max flow."""
    side, queue = {s}, deque([s])
    while queue:
        u = queue.popleft()
        for v, cap in r[u].items():
            if cap > 1e-12 and v not in side:
                side.add(v)
                queue.append(v)
    cut = [(u, v, c) for u, v, c in pipes if (u in side) != (v in side)]
    return side, cut


def pipe_flows(pipes, r):
    """net flow on each pipe, signed in the u -> v direction of the record."""
    return {(u, v): (c - r[u][v]) for u, v, c in pipes}
