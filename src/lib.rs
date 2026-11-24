pub mod core;

use std::path::PathBuf;

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyKeyError, PyOverflowError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyInt, PyString};
use rayon::prelude::*;

use crate::core::{BuildError, Mphf, Variant};

/// Hash one Python key: `int` (incl. `bool` and objects with `__index__`),
/// `str`, `bytes` or `bytearray`.
fn key_hash(obj: &Bound<'_, PyAny>) -> PyResult<u128> {
    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(core::hash_str(s.to_str()?.as_bytes()));
    }
    if let Ok(b) = obj.cast::<PyBytes>() {
        return Ok(core::hash_bytes(b.as_bytes()));
    }
    if let Ok(b) = obj.cast::<PyByteArray>() {
        return Ok(core::hash_bytes(&b.to_vec()));
    }
    if obj.is_instance_of::<PyInt>() || obj.hasattr("__index__")? {
        if let Ok(v) = obj.extract::<i64>() {
            return Ok(core::hash_int(v as i128));
        }
        return match obj.extract::<i128>() {
            Ok(v) => Ok(core::hash_int(v)),
            Err(_) => Err(PyOverflowError::new_err(
                "integer keys must fit in a signed 128-bit integer",
            )),
        };
    }
    Err(PyTypeError::new_err(format!(
        "keys must be int, str, bytes or bytearray, not {}",
        obj.get_type().name()?
    )))
}

/// If `obj` is a buffer of 32/64-bit integers (e.g. a numpy array or
/// `array.array`), hash it in parallel without the GIL.
fn int_buffer_hashes(py: Python<'_>, obj: &Bound<'_, PyAny>) -> Option<Vec<u128>> {
    if obj.is_instance_of::<PyBytes>() || obj.is_instance_of::<PyByteArray>() {
        return None;
    }
    macro_rules! try_buf {
        ($t:ty) => {
            if let Ok(buf) = PyBuffer::<$t>::get(obj) {
                if let Ok(v) = buf.to_vec(py) {
                    return Some(
                        py.detach(|| v.par_iter().map(|&x| core::hash_int(x as i128)).collect()),
                    );
                }
            }
        };
    }
    try_buf!(u64);
    try_buf!(i64);
    try_buf!(u32);
    try_buf!(i32);
    None
}

fn hashes(py: Python<'_>, keys: &Bound<'_, PyAny>) -> PyResult<Vec<u128>> {
    if let Some(v) = int_buffer_hashes(py, keys) {
        return Ok(v);
    }
    if keys.is_instance_of::<PyString>() {
        return Err(PyTypeError::new_err(
            "expected an iterable of keys, not a single str",
        ));
    }
    let mut out = Vec::with_capacity(keys.len().unwrap_or(0));
    for k in keys.try_iter()? {
        out.push(key_hash(&k?)?);
    }
    Ok(out)
}

/// Allocate an `array.array(typecode)` of `n` zeroed 8-byte items and let
/// `fill` write into it without the GIL (one allocation, no copies).
fn new_array<'py, T: pyo3::buffer::Element + Copy + Send + Sync>(
    py: Python<'py>,
    typecode: &str,
    n: usize,
    fill: impl FnOnce(&mut [T]) + Send,
) -> PyResult<Bound<'py, PyAny>> {
    let array = py.import("array")?.getattr("array")?;
    if n == 0 {
        return array.call1((typecode,));
    }
    let zero = array.call1((typecode, (0,)))?;
    let arr = zero.call_method1("__mul__", (n,))?;
    let buf = PyBuffer::<T>::get(&arr)?;
    assert!(!buf.readonly() && buf.is_c_contiguous() && buf.item_count() == n);
    let ptr = buf.buf_ptr() as usize;
    py.detach(|| {
        // SAFETY: `arr` was just created and is not reachable from Python
        // code yet; `buf` keeps it exported (unresizable) until we're done.
        let out = unsafe { std::slice::from_raw_parts_mut(ptr as *mut T, n) };
        fill(out)
    });
    drop(buf);
    Ok(arr)
}

/// A minimal perfect hash function (PtrHash) over a fixed set of keys.
///
/// Maps each of the `n` distinct keys it was built from to a unique index in
/// `range(n)`, using ~3 bits per key (`compact=True`: ~2.2), regardless of
/// key size. Keys not in the build set map to an arbitrary index, unless
/// `fingerprint_bits` is set, in which case they are rejected with
/// probability `1 - 2**-fingerprint_bits`.
#[pyclass(name = "MPHF", module = "rptrhash", frozen)]
struct PyMphf {
    inner: Mphf,
}

#[pymethods]
impl PyMphf {
    #[new]
    #[pyo3(signature = (keys, *, compact = false, fingerprint_bits = 0))]
    fn new(
        py: Python<'_>,
        keys: &Bound<'_, PyAny>,
        compact: bool,
        fingerprint_bits: u32,
    ) -> PyResult<Self> {
        let mut hs = hashes(py, keys)?;
        let variant = if compact {
            Variant::Compact
        } else {
            Variant::Fast
        };
        match py.detach(|| Mphf::build(&mut hs, variant, fingerprint_bits)) {
            Ok(inner) => Ok(PyMphf { inner }),
            Err(BuildError::Duplicate) => Err(PyValueError::new_err("keys contain duplicates")),
            Err(BuildError::BadFingerprintBits) => Err(PyValueError::new_err(
                "fingerprint_bits must be 0, 8, 16 or 32",
            )),
            Err(BuildError::Failed) => Err(PyRuntimeError::new_err(
                "PtrHash construction failed (please report this)",
            )),
        }
    }

    /// The index of `key` in `range(len(self))`. Arbitrary (but in range)
    /// for keys that were not in the build set; fingerprints are not checked.
    fn index(&self, key: &Bound<'_, PyAny>) -> PyResult<usize> {
        if self.inner.is_empty() {
            return Err(PyValueError::new_err("index() on an empty MPHF"));
        }
        Ok(self.inner.index(key_hash(key)?))
    }

    /// The index of `key`; with fingerprints, raises `KeyError` for keys
    /// detected as not in the build set.
    fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.inner
            .get(key_hash(key)?)
            .ok_or_else(|| PyKeyError::new_err(key.clone().unbind()))
    }

    /// The index of `key`, or `default` if fingerprints rule it out.
    #[pyo3(signature = (key, default = None))]
    fn get(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        default: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        Ok(match self.inner.get(key_hash(key)?) {
            Some(i) => i.into_pyobject(py)?.into_any().unbind(),
            None => default.unwrap_or_else(|| py.None()),
        })
    }

    /// Probabilistic membership test (false-positive rate
    /// `2**-fingerprint_bits`, no false negatives). Requires fingerprints.
    fn __contains__(&self, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        if self.inner.fingerprint_bits() == 0 {
            return Err(PyTypeError::new_err(
                "membership tests need fingerprints: build with fingerprint_bits=8/16/32",
            ));
        }
        Ok(self.inner.get(key_hash(key)?).is_some())
    }

    /// Indices of many keys, as an `array.array('Q')`. Accepts any iterable
    /// of keys, or a 32/64-bit integer buffer (numpy array, `array.array`)
    /// which is hashed and queried in parallel without the GIL.
    fn index_many<'py>(
        &self,
        py: Python<'py>,
        keys: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if self.inner.is_empty() {
            return Err(PyValueError::new_err("index_many() on an empty MPHF"));
        }
        let hs = hashes(py, keys)?;
        new_array::<u64>(py, "Q", hs.len(), |out| {
            self.inner.index_many_into(&hs, out)
        })
    }

    /// Like `get` for many keys: an `array.array('q')` with `default` where
    /// fingerprints rule a key out.
    #[pyo3(signature = (keys, default = -1))]
    fn get_many<'py>(
        &self,
        py: Python<'py>,
        keys: &Bound<'py, PyAny>,
        default: i64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let hs = hashes(py, keys)?;
        new_array::<i64>(py, "q", hs.len(), |out| {
            self.inner.get_many_into(&hs, default, out)
        })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// `"fast"` (default) or `"compact"`.
    #[getter]
    fn variant(&self) -> &'static str {
        self.inner.variant().name()
    }

    #[getter]
    fn fingerprint_bits(&self) -> u32 {
        self.inner.fingerprint_bits()
    }

    /// Total bits per key, including fingerprints.
    #[getter]
    fn bits_per_key(&self) -> f64 {
        self.inner.bits_per_key()
    }

    /// Bits per key of the hash function alone (excluding fingerprints).
    #[getter]
    fn mphf_bits_per_key(&self) -> f64 {
        self.inner.mphf_bits_per_key()
    }

    /// Serialize (versioned, checksummed binary format).
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let b = py.detach(|| self.inner.to_bytes());
        PyBytes::new(py, &b)
    }

    /// Load from `to_bytes()` output. Like pickle, only load trusted data.
    #[staticmethod]
    fn from_bytes(py: Python<'_>, data: &[u8]) -> PyResult<Self> {
        load(py, data)
    }

    /// Write `to_bytes()` to a file.
    fn save(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        py.detach(|| std::fs::write(&path, self.inner.to_bytes()))?;
        Ok(())
    }

    /// Read a file written by `save()`. Like pickle, only load trusted data.
    #[staticmethod]
    fn load(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let data = py.detach(|| std::fs::read(&path))?;
        load(py, &data)
    }

    fn __reduce__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyBytes>,))> {
        let f = py.import("rptrhash._rptrhash")?.getattr("_from_bytes")?;
        Ok((f, (self.to_bytes(py),)))
    }

    fn __repr__(&self) -> String {
        format!(
            "MPHF(n={}, variant='{}', fingerprint_bits={}, bits_per_key={:.2})",
            self.inner.len(),
            self.inner.variant().name(),
            self.inner.fingerprint_bits(),
            self.inner.bits_per_key()
        )
    }
}

fn load(py: Python<'_>, data: &[u8]) -> PyResult<PyMphf> {
    py.detach(|| Mphf::from_bytes(data))
        .map(|inner| PyMphf { inner })
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn _from_bytes(py: Python<'_>, data: &[u8]) -> PyResult<PyMphf> {
    load(py, data)
}

#[pymodule]
fn _rptrhash(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMphf>()?;
    m.add_function(wrap_pyfunction!(_from_bytes, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
