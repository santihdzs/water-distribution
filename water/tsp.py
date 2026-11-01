"""sampling route: tsp on the shortest-path metric closure.
D is a symmetric distance matrix (floyd-warshall output); tours are index lists that
start at the office and return to it implicitly."""
import itertools
import time

import numpy as np


def tour_length(D, tour):
    return float(sum(D[tour[i], tour[i + 1]] for i in range(len(tour) - 1)) + D[tour[-1], tour[0]])


def nearest_neighbor(D, start=0):
    """greedy: always walk to the closest unvisited location."""
    n = len(D)
    tour, left = [start], set(range(n)) - {start}
    while left:
        u = tour[-1]
        v = min(left, key=lambda v: (D[u, v], v))
        tour.append(v)
        left.remove(v)
    return tour


def two_opt(D, tour):
    """first-improvement 2-opt: replace pipes (a,b),(c,d) by (a,c),(b,d) while that shortens the tour.
    position 0 (the office) never moves."""
    t = list(tour)
    n = len(t)
    moves, improved = 0, True
    while improved:
        improved = False
        for i in range(n - 2):
            a, b = t[i], t[i + 1]
            for j in range(i + 2, n if i else n - 1):
                c, d = t[j], t[(j + 1) % n]
                if D[a, c] + D[b, d] < D[a, b] + D[c, d] - 1e-9:
                    t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                    b = t[i + 1]
                    moves += 1
                    improved = True
    return t, moves


def _mst(sub):
    """prim on a dense matrix, O(k^2)."""
    k = len(sub)
    if k <= 1:
        return 0.0
    key = sub[0].copy()
    used = np.zeros(k, bool)
    used[0] = True
    total = 0.0
    for _ in range(k - 1):
        key_masked = np.where(used, np.inf, key)
        j = int(np.argmin(key_masked))
        total += key_masked[j]
        used[j] = True
        key = np.minimum(key, sub[j])
    return total


def branch_and_bound(D, start=0, incumbent=None, time_limit=None):
    """exact depth-first branch and bound over partial routes from the office.
    bound(partial) = cost so far + mst(unvisited) + min pipe current->unvisited + min unvisited->office,
    which never exceeds the cost of any completion. returns the best tour, its cost, whether
    optimality was proven (search finished), the root lower bound and search counters."""
    n = len(D)
    best = list(incumbent) if incumbent else nearest_neighbor(D, start)
    best_cost = tour_length(D, best)
    stats = {'expanded': 0, 'pruned': 0}
    t0 = time.perf_counter()
    timed_out = False

    def bound(cur, cost, rem):
        if not rem:
            return cost + D[cur, start]
        idx = np.fromiter(rem, int)
        return (cost + _mst(D[np.ix_(idx, idx)])
                + D[cur, idx].min() + D[idx, start].min())

    root = bound(start, 0.0, set(range(n)) - {start})

    def search(path, cost, rem):
        nonlocal best, best_cost, timed_out
        if timed_out:
            return
        if time_limit and time.perf_counter() - t0 > time_limit:
            timed_out = True
            return
        stats['expanded'] += 1
        cur = path[-1]
        if not rem:
            total = cost + D[cur, start]
            if total < best_cost - 1e-9:
                best, best_cost = list(path), total
            return
        for v in sorted(rem, key=lambda v: D[cur, v]):
            c = cost + D[cur, v]
            rem.remove(v)
            if bound(v, c, rem) < best_cost - 1e-9:
                path.append(v)
                search(path, c, rem)
                path.pop()
            else:
                stats['pruned'] += 1
            rem.add(v)

    search([start], 0.0, set(range(n)) - {start})
    stats['seconds'] = time.perf_counter() - t0
    return best, best_cost, not timed_out, root, stats


def brute_force(D, start=0):
    """enumerate all (n-1)! orders; only for tiny n, used to verify branch and bound."""
    rest = [v for v in range(len(D)) if v != start]
    best = min(itertools.permutations(rest), key=lambda p: tour_length(D, [start, *p]))
    tour = [start, *best]
    return tour, tour_length(D, tour)


def expand(tour, nxt, fw_path):
    """turn a closure tour back into a closed walk over real pipes (index route, office at both ends)."""
    walk = [tour[0]]
    for a, b in zip(tour, tour[1:] + tour[:1]):
        walk.extend(fw_path(nxt, a, b)[1:])
    return walk
