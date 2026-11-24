"""Benchmark rptrhash against bbhash (the existing Python MPHF) and a dict.

    pip install bbhash numpy   # bbhash ships an sdist only (needs a compiler)
    python benchmarks/bench.py [n ...]

Keys are distinct random 64-bit integers (bbhash only supports u64 keys).
"""

import os
import random
import sys
import tempfile
import time
from array import array

from rptrhash import MPHF

try:
    import bbhash
except ImportError:
    bbhash = None

try:
    import numpy as np
except ImportError:
    np = None

THREADS = os.cpu_count()


def timed(f):
    t = time.perf_counter()
    r = f()
    return r, time.perf_counter() - t


def file_bits(n, save):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m")
        save(p)
        return os.path.getsize(p) * 8 / n


def row(name, build, bits, loop, batch):
    print(f"  {name:<22} {build:>8.2f}s {bits:>9.2f} {loop:>11.0f} {batch:>11.1f}")


def bench(n):
    rng = random.Random(42)
    keys = array("Q", rng.sample(range(2**62), n))
    keys_list = list(keys)
    probe = keys_list[: min(n, 1_000_000)]
    print(f"\nn = {n:,}  ({THREADS} threads)")
    print(f"  {'':<22} {'build':>9} {'bits/key':>9} {'ns/lookup':>11} {'batch ns':>11}")

    for compact in (False, True):
        m, tb = timed(lambda: MPHF(keys, compact=compact))
        _, tl = timed(lambda: [m.index(k) for k in probe])
        _, tm = timed(lambda: m.index_many(keys))
        bits = file_bits(n, m.save)
        row(f"rptrhash {m.variant}", tb, bits, tl / len(probe) * 1e9, tm / n * 1e9)

    if bbhash is not None:
        b, tb = timed(lambda: bbhash.PyMPHF(keys_list, n, THREADS, 1.0))
        _, tl = timed(lambda: [b.lookup(k) for k in probe])
        if np is not None:
            arr = np.frombuffer(keys, dtype=np.uint64)
            _, tm = timed(lambda: b.lookup_many(arr))
            batch = tm / n * 1e9
        else:
            batch = float("nan")
        row("bbhash (gamma=1)", tb, file_bits(n, b.save), tl / len(probe) * 1e9, batch)

    d, tb = timed(lambda: {k: i for i, k in enumerate(keys_list)})
    _, tl = timed(lambda: [d[k] for k in probe])
    dict_bits = sys.getsizeof(d) * 8 / n  # table only; excludes the int objects
    row("dict (table only)", tb, dict_bits, tl / len(probe) * 1e9, float("nan"))

    # String keys: rptrhash only (bbhash needs u64 keys).
    skeys = [f"user:{k:x}" for k in keys_list[: min(n, 1_000_000)]]
    m, tb = timed(lambda: MPHF(skeys))
    _, tl = timed(lambda: [m.index(k) for k in skeys])
    _, tm = timed(lambda: m.index_many(skeys))
    row(
        f"rptrhash str ({len(skeys) // 1000}k)",
        tb,
        m.bits_per_key,
        tl / len(skeys) * 1e9,
        tm / len(skeys) * 1e9,
    )


if __name__ == "__main__":
    for n in [int(x) for x in sys.argv[1:]] or [1_000_000, 10_000_000]:
        bench(n)
