# CLAUDE.md

## Overview

`rptrhash` wraps the [`ptr_hash`](https://crates.io/crates/ptr_hash) crate
(PtrHash minimal perfect hashing) for Python via PyO3/maturin.

- `src/core.rs`: PyO3-free.
  - Keys are pre-hashed to `u128` with XXH3 (`hash_int`/`hash_str`/`hash_bytes`,
    one seed per key kind). PtrHash is built over those with the
    pass-through `PreHashed` hasher.
  - Two variants: `Fast` (`DefaultPtrHash`) and `Compact` (multi-part,
    `CubicEps`, `CachelineEfVec` remap; sucds `EliasFano` is not epserde-serializable).
  - Optional fingerprints (`u8`/`u16`/`u32` per slot).
  - Serialization: own header (magic, version, variant, fp bits, n, checksum)
    around epserde `serialize`/`deserialize_full`.
  - Has Rust tests (`--ignored` `timing` gives a Rust-only breakdown).
- `src/lib.rs`: key extraction (int/str/bytes, or integer buffers hashed in
  parallel without the GIL), `MPHF` pyclass. Batch results are written straight
  into a freshly allocated `array.array` (`new_array`).

## Commands

```sh
cargo test --no-default-features
cargo fmt --check && cargo clippy --no-default-features --all-targets -- -D warnings
pip install -e '.[dev]' && maturin develop --release && pytest
ruff check . && ruff format --check .
pip install bbhash numpy && python benchmarks/bench.py
```

## Invariants / gotchas

- **File format compatibility**: the epserde body embeds a type hash of the
  PtrHash type, including our `PreHashed` struct name and `ptr_hash` internals.
  `ptr_hash` is pinned (`=2.1.5`). Bumping it, renaming `PreHashed`, or changing
  key hashing seeds/encoding changes indices or breaks loading. Bump
  `FORMAT_VERSION` and note it in the README.
- `Mphf::build` sorts the hashes in place to reject duplicates up front
  (upstream would retry seeds, printing "Hashes are not distinct!" to stderr).
  The sort also makes the default variant independent of key order.
- Upstream `index_stream` iterators only implement `fold` (`next()` panics):
  consume them with `for_each` (see `fill`).
- Integer keys always hash as `i128` LE, so Python ints, `int64`/`uint64`/
  `int32` buffers and numpy scalars agree.
- Keep `version` in sync between `Cargo.toml` and `pyproject.toml`;
  `abi3-py311` must match `requires-python`.

## Releasing

`release.yml`: `workflow_dispatch` publishes to TestPyPI, and a `v*` tag
publishes to PyPI, both via Trusted Publishing (OIDC). A trusted publisher
must be configured first on pypi.org/test.pypi.org (owner `f4hy`, repo
`rptrhash`, workflow `release.yml`, environment `pypi`/`testpypi`).
