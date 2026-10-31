"""timing and memory helpers."""
import statistics
import time
import sys


def median_time(fn, repeats=21, min_sample=2e-3):
    """median seconds per call. each sample loops fn enough times to last min_sample,
    so timer resolution does not dominate on small inputs."""
    fn()
    loops, t = 1, 0.0
    while True:
        t0 = time.perf_counter()
        for _ in range(loops):
            fn()
        t = time.perf_counter() - t0
        if t >= min_sample:
            break
        loops *= 2
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(loops):
            fn()
        samples.append((time.perf_counter() - t0) / loops)
    return statistics.median(samples)


def deep_size(obj):
    """bytes of obj and everything it references, each object counted once.
    deterministic, unlike tracemalloc, which misses objects served from freelists."""
    seen, stack, total = set(), [obj], 0
    while stack:
        o = stack.pop()
        if id(o) in seen:
            continue
        seen.add(id(o))
        total += sys.getsizeof(o)
        if isinstance(o, dict):
            stack.extend(o.keys())
            stack.extend(o.values())
        elif isinstance(o, (list, tuple, set)):
            stack.extend(o)
    return total
