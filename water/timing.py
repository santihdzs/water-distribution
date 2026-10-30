"""timing and memory helpers."""
import statistics
import time
import tracemalloc


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


def allocated_bytes(build):
    """bytes still allocated by the object build() returns (traced python allocations)."""
    tracemalloc.start()
    tracemalloc.clear_traces()
    before = tracemalloc.get_traced_memory()[0]
    obj = build()
    after = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    del obj
    return after - before
