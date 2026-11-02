"""selection hyper-heuristic for the sampling route (task 6).
the low-level heuristics are random perturbative moves on a tour whose position 0 (the office)
never moves; a selection rule picks which one to apply and an acceptance rule decides whether
the result replaces the current tour."""
import random

import numpy as np


def two_opt_move(t, rng):
    i, j = sorted(rng.sample(range(1, len(t)), 2))
    return t[:i] + t[i:j + 1][::-1] + t[j + 1:]


def swap(t, rng):
    i, j = rng.sample(range(1, len(t)), 2)
    t = list(t)
    t[i], t[j] = t[j], t[i]
    return t


def relocate(t, rng):
    i, j = rng.sample(range(1, len(t)), 2)
    t = list(t)
    t.insert(j, t.pop(i))
    return t


def or_opt(t, rng):
    """move a segment of 2 or 3 consecutive locations elsewhere, possibly reversed."""
    k = rng.choice((2, 3))
    if len(t) - 1 <= k:
        return list(t)
    i = rng.randrange(1, len(t) - k + 1)
    seg, rest = t[i:i + k], t[:i] + t[i + k:]
    if rng.random() < 0.5:
        seg = seg[::-1]
    j = rng.randrange(1, len(rest) + 1)
    return rest[:j] + seg + rest[j:]


LLH = {'2-opt': two_opt_move, 'swap': swap, 'relocate': relocate, 'or-opt': or_opt}


def cost(D, t):
    a = np.asarray(t)
    return float(D[a, np.roll(a, -1)].sum())


def search(D, start, heuristics, iters=20000, select='adaptive', history=50, rate=0.05, p_min=0.05, seed=0):
    """late acceptance hill climbing driven by a selection rule over the given heuristics.

    select='random': uniform choice each iteration (with one heuristic this is a plain
    single-heuristic search). select='adaptive': probability matching, each heuristic's
    weight tracks how often it recently improved the current tour; every heuristic keeps
    at least p_min probability so none is abandoned for good.
    acceptance: take the candidate if it is no worse than the current tour or than the
    current tour of `history` iterations ago.
    returns best tour, its cost, the best-cost trace and the selection counts."""
    rng = random.Random(seed)
    names = list(heuristics)
    cur, cur_c = list(start), cost(D, start)
    best, best_c = list(cur), cur_c
    hist = [cur_c] * history
    w = dict.fromkeys(names, 1.0)
    used = dict.fromkeys(names, 0)
    trace = np.empty(iters)
    for it in range(iters):
        if select == 'adaptive' and len(names) > 1:
            total = sum(w.values())
            p = [p_min + (1 - p_min * len(names)) * w[h] / total for h in names]
            h = rng.choices(names, p)[0]
        else:
            h = rng.choice(names)
        used[h] += 1
        cand = heuristics[h](cur, rng)
        c = cost(D, cand)
        w[h] = (1 - rate) * w[h] + rate * (1.0 if c < cur_c - 1e-9 else 0.0) + 1e-6
        slot = it % history
        if c <= cur_c or c <= hist[slot]:
            cur, cur_c = cand, c
            if c < best_c - 1e-9:
                best, best_c = list(cand), c
        hist[slot] = cur_c
        trace[it] = best_c
    return best, best_c, trace, used
