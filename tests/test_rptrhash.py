import pickle
from array import array

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rptrhash import MPHF

keys_st = st.one_of(
    st.integers(min_value=-(2**127), max_value=2**127 - 1),
    st.text(max_size=12),
    st.binary(max_size=12),
)


def distinct(keys):
    # Python equality decides duplicates (True == 1), as for dict keys;
    # only str/bytes/int are generated, so equal values hash identically.
    return list(dict.fromkeys(keys))


def assert_bijection(m, keys):
    idx = [m.index(k) for k in keys]
    assert sorted(idx) == list(range(len(keys)))
    assert list(m.index_many(keys)) == idx
    assert [m[k] for k in keys] == idx


@settings(max_examples=150, deadline=None)
@given(st.lists(keys_st, max_size=300).map(distinct), st.booleans(), st.sampled_from([0, 8, 16]))
def test_bijection_any_keys(keys, compact, fp_bits):
    m = MPHF(keys, compact=compact, fingerprint_bits=fp_bits)
    assert len(m) == len(keys)
    if keys:
        assert_bijection(m, keys)
    if fp_bits:
        assert all(k in m for k in keys)  # no false negatives
        assert list(m.get_many(keys)) == [m.index(k) for k in keys]


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("n", [1, 2, 5, 1000, 100_000])
def test_bijection_sizes(n, compact):
    keys = [f"key-{i}" for i in range(n)]
    m = MPHF(keys, compact=compact)
    assert_bijection(m, keys)


def test_int_buffers_match_python_ints():
    pos = list(range(0, 2**64 - 1, 2**64 // 5000))
    m = MPHF(array("Q", pos))
    assert list(m.index_many(array("Q", pos))) == [m.index(k) for k in pos]
    assert sorted(m.index_many(pos)) == list(range(len(pos)))

    signed = list(range(-2500, 2500))
    for code in "qlih":
        buf = array(code, signed)
        if buf.itemsize not in (4, 8):
            continue
        m = MPHF(buf)
        assert list(m.index_many(buf)) == [m.index(k) for k in signed]
        assert list(m.index_many(signed)) == [m.index(k) for k in signed]


def test_numpy_arrays():
    np = pytest.importorskip("numpy")
    keys = np.arange(-50_000, 50_000, dtype=np.int64) * 7919
    m = MPHF(keys)
    out = np.frombuffer(m.index_many(keys), dtype=np.uint64)
    assert np.array_equal(np.sort(out), np.arange(len(keys), dtype=np.uint64))
    assert m.index(int(keys[123])) == out[123]
    # non-contiguous views are fine too
    view = keys[::3]
    assert np.array_equal(np.frombuffer(m.index_many(view), dtype=np.uint64), out[::3])
    # numpy scalars work as keys
    assert m[np.int64(keys[5])] == out[5]


def test_key_kinds_are_distinct():
    # "a", b"a" and 97 are different keys, like in a dict.
    m = MPHF(["a", b"a", 97, bytearray(b"b")])
    assert sorted(m.index_many(["a", b"a", 97, b"b"])) == [0, 1, 2, 3]


def test_duplicates_rejected():
    for keys in ([1, 2, 1], ["x", "x"], [True, 1], [b"q", bytearray(b"q")]):
        with pytest.raises(ValueError, match="duplicates"):
            MPHF(keys)


def test_bad_inputs():
    with pytest.raises(TypeError):
        MPHF([1.5])
    with pytest.raises(TypeError):
        MPHF([(1, 2)])
    with pytest.raises(TypeError):
        MPHF("abc")  # a single str is almost surely a mistake
    with pytest.raises(OverflowError):
        MPHF([2**127])
    with pytest.raises(ValueError):
        MPHF([1, 2], fingerprint_bits=4)
    m = MPHF([1, 2, 3])
    with pytest.raises(TypeError):
        m.index(1.0)
    with pytest.raises(TypeError, match="fingerprint"):
        _ = 1 in m


def test_empty():
    m = MPHF([])
    assert len(m) == 0
    assert m.get(1) is None
    with pytest.raises(KeyError):
        m[1]
    with pytest.raises(ValueError):
        m.index(1)
    assert list(m.get_many([1, 2])) == [-1, -1]
    assert len(pickle.loads(pickle.dumps(m))) == 0


def test_fingerprints_reject_non_members():
    n = 50_000
    members = range(n)
    others = range(n, 2 * n)
    for bits, max_fp in ((8, 0.01), (16, 0.0002), (32, 1e-6)):
        m = MPHF(members, fingerprint_bits=bits)
        fp = sum(k in m for k in others) / n
        assert fp < max_fp * 2 + 1e-4, (bits, fp)
        assert m.get(n + 10**9, "missing") in ("missing", *range(n))
        got = m.get_many(others, -7)
        assert sum(x != -7 for x in got) / n == fp
    m = MPHF(members, fingerprint_bits=32)
    with pytest.raises(KeyError):
        m[-12345]


def test_serialization(tmp_path):
    keys = [f"k{i}" for i in range(10_000)]
    for compact in (False, True):
        for bits in (0, 16):
            m = MPHF(keys, compact=compact, fingerprint_bits=bits)
            expected = list(m.index_many(keys))
            for m2 in (
                pickle.loads(pickle.dumps(m)),
                MPHF.from_bytes(m.to_bytes()),
            ):
                assert list(m2.index_many(keys)) == expected
                assert m2.variant == m.variant
                assert m2.fingerprint_bits == bits
            p = tmp_path / f"m-{compact}-{bits}.mphf"
            m.save(p)
            assert list(MPHF.load(str(p)).index_many(keys)) == expected


def test_corrupt_data_rejected():
    data = MPHF(range(1000)).to_bytes()
    for bad in (b"", b"garbage!", data[:-1], data[:20], data[:-1] + bytes([data[-1] ^ 1])):
        with pytest.raises(ValueError):
            MPHF.from_bytes(bad)


def test_space_and_repr():
    m = MPHF(range(200_000))
    c = MPHF(range(200_000), compact=True)
    assert 2.5 < m.bits_per_key < 3.5
    assert c.bits_per_key < m.bits_per_key
    assert len(m.to_bytes()) * 8 / 200_000 < 3.5
    f = MPHF(range(200_000), fingerprint_bits=8)
    assert f.bits_per_key == pytest.approx(f.mphf_bits_per_key + 8)
    assert "n=200000" in repr(m) and "fast" in repr(m)


def test_default_variant_is_deterministic_and_order_independent():
    keys = [f"k{i}" for i in range(50_000)]
    a = MPHF(keys)
    b = MPHF(list(reversed(keys)))
    assert a.to_bytes() == b.to_bytes()
    assert list(a.index_many(keys)) == list(b.index_many(keys))
