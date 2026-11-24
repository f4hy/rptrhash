# rptrhash

Minimal perfect hash functions for Python, based on
[PtrHash](https://github.com/RagnarGrootKoerkamp/ptrhash) (Groot Koerkamp,
SEA 2025) and implemented in Rust via [PyO3](https://pyo3.rs).

An MPHF maps a fixed set of `n` distinct keys one-to-one onto `range(n)`,
using about **3 bits per key** (2.2 with `compact=True`), however long the
keys are. It doesn't store the keys. You use it to put keys into a plain
array (a static hash table) instead of a dict, which costs hundreds of bits
per entry.

```python
from rptrhash import MPHF

keys = ["apple", "banana", "cherry", 42, b"\x00raw"]
mphf = MPHF(keys)
mphf["cherry"]                 # -> an int in range(5); unique per key

# A static key -> value table in two flat arrays
values = [None] * len(mphf)
for k, v in zip(keys, ["red", "yellow", "dark red", "answer", "bytes"]):
    values[mphf[k]] = v
values[mphf["banana"]]         # 'yellow'

mphf.index_many(keys)          # array('Q', [...]), parallel, no GIL
mphf.save("fruit.mphf")        # or to_bytes() / pickle
```

## Why

For static sets of millions to billions of keys, such as k-mer and n-gram
indexes, URL and ID dictionaries, or read-only lookup tables, a minimal
perfect hash plus flat arrays replaces a dict at 1–2% of the memory.
Before this package, the only Python MPHF was
[`bbhash`](https://pypi.org/project/bbhash/), a Cython wrapper of the 2017
BBHash algorithm. It accepts only 64-bit integer keys and ships only as an
sdist, so you need a C++ compiler to install it. PtrHash (2025) is the current state of the art for query throughput,
and is also among the most compact.

What `rptrhash` adds on top of the Rust crate:

- **Any key type**: `int` (up to 128 bits), `str`, `bytes`, and numpy or
  `array` integer buffers. Each key is pre-hashed with 128-bit XXH3, so
  accidental collisions are negligible even at billions of keys.
- **Optional fingerprints** (`fingerprint_bits=8/16/32`) to detect keys that
  were not in the build set. A plain MPHF maps such keys to an arbitrary
  index.
- **Batch queries** that run in parallel with the GIL released.
- **Persistence**: `save`/`load`, `to_bytes`/`from_bytes` and pickle, in a
  versioned, checksummed format.
- Prebuilt `abi3` wheels (Linux/macOS/Windows, Python 3.11+).

## Install

```sh
pip install rptrhash
```

## Benchmarks

Distinct random 64-bit keys, 8 threads. The `bbhash` comparison uses
`gamma=1` (its most compact setting). Source:
[`benchmarks/bench.py`](./benchmarks/bench.py), raw output in
[`benchmarks/results.txt`](./benchmarks/results.txt).

| n = 10M | build | bits/key (saved file) | `index(k)` from Python | batch (per key) |
|---|---|---|---|---|
| **rptrhash** (default) | 2.64s | **2.99** | 235 ns | **11 ns** |
| **rptrhash** `compact=True` | 1.49s | **2.17** | 226 ns | **11 ns** |
| bbhash | 0.71s | 3.15 | 211 ns | 79 ns |
| `dict` (hash table only) | 3.19s | 268 | 133 ns | – |

| n = 1M | build | bits/key | `index(k)` | batch |
|---|---|---|---|---|
| rptrhash (default) | 0.18s | 2.99 | 147 ns | 5.9 ns |
| rptrhash `compact=True` | 0.21s | 2.17 | 142 ns | 7.0 ns |
| bbhash | 0.08s | 3.07 | 150 ns | 49 ns |
| rptrhash, 1M `str` keys | 0.20s | 2.99 | 128 ns | 24 ns |

Results:

- **Batch queries** (`index_many` on a numpy array) are **7x faster** than
  `bbhash.lookup_many`.
- `compact=True` is **31% smaller** than bbhash.
- Single lookups from Python are about the same speed for every MPHF,
  because the cost of a Python method call dominates.
- bbhash builds faster with many threads. The default PtrHash variant
  builds on a single thread; `compact=True` builds in parallel.
- The `dict` row is a baseline for memory only; it doesn't count the key
  objects.

## API

`MPHF(keys, *, compact=False, fingerprint_bits=0)`

- `keys` is an iterable of distinct `int`, `str`, `bytes` or `bytearray`, or a
  32/64-bit integer buffer (a numpy array or `array.array`), which is hashed
  in parallel. Duplicate keys raise `ValueError`. As with dict keys, `True`
  and `1` count as the same key, while `"a"`, `b"a"` and `97` are three
  different keys.
- `compact=True` gives about 2.2 bits/key and multi-threaded construction.
  Single queries are slightly slower in Rust, which doesn't matter for calls
  from Python.
- `fingerprint_bits` can be 0, 8, 16 or 32. It stores that many bits per key,
  so non-members are rejected except with probability `2**-bits`.

| method | returns |
|---|---|
| `m.index(key)` | an `int` in `range(len(m))`. Unique for build keys; arbitrary for others (fingerprints are not checked). |
| `m[key]` | like `index`, but raises `KeyError` if a fingerprint rules the key out |
| `m.get(key, default=None)` | the index, or `default` if a fingerprint rules the key out |
| `key in m` | probabilistic membership (needs fingerprints, otherwise raises `TypeError`). There are no false negatives. |
| `m.index_many(keys)` | `array('Q')` of indices: parallel, GIL released. Use `np.frombuffer(..., dtype=np.uint64)` for a zero-copy numpy view. |
| `m.get_many(keys, default=-1)` | `array('q')`, with `default` wherever a fingerprint rules a key out |
| `len(m)`, `m.variant`, `m.fingerprint_bits`, `m.bits_per_key`, `m.mphf_bits_per_key` | |
| `m.to_bytes()`, `MPHF.from_bytes(b)`, `m.save(path)`, `MPHF.load(path)`, pickle | serialization |

> [!WARNING]
> As with pickle, only load serialized MPHFs from trusted sources. A
> checksum catches corruption, but the data is not validated against
> malicious input.

The key hashing and the file format are versioned and stable within a
format version. The default variant is deterministic: the same keys always
give the same indices. A `compact=True` build is multi-threaded and may
assign different (equally valid) indices each time you build it, so save
it rather than rebuilding when indices must stay stable.

## How it works

1. Each key is hashed to 128 bits with XXH3. There is a separate seed for
   each key kind (int, str, bytes), and ints are hashed as `i128`, so Python
   ints and integer buffers agree.
2. PtrHash runs over those hashes:
   - keys are split into buckets of about 3 keys on average;
   - each bucket gets an 8-bit *pilot*, chosen so that its keys land in free
     slots of a table about 1% larger than `n`, evicting other buckets when
     needed (as in cuckoo hashing);
   - the slots past `n` are remapped to the free slots below `n`, using
     `Vec<u32>` storage or, with `compact=True`, Elias-Fano encoding.
3. A query is one hash, one pilot lookup, and a multiply. Batch queries
   prefetch 32 keys ahead and run across rayon threads.

## Development

```sh
cargo test --no-default-features
pip install -e '.[dev]' && maturin develop --release
pytest
pip install bbhash numpy && python benchmarks/bench.py
```

## License

MIT. Wraps the MIT-licensed [`ptr_hash`](https://crates.io/crates/ptr_hash)
crate by Ragnar Groot Koerkamp.
