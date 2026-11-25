//! PyO3-free core: key pre-hashing, the two PtrHash variants, optional
//! fingerprints, and the on-disk format.
//!
//! Every key is first reduced to a 128-bit XXH3 hash (`hash_*`). PtrHash is
//! then built over those hashes with a pass-through hasher, so one PtrHash
//! type serves ints, strings and bytes alike, and accidental collisions are
//! negligible (~n²/2¹²⁹) even for billions of keys.

use cacheline_ef::{CachelineEf, CachelineEfVec};
use epserde::prelude::*;
use ptr_hash::bucket_fn::CubicEps;
use ptr_hash::hash::KeyHasher;
use ptr_hash::{DefaultPtrHash, PtrHash, PtrHashParams};
use rayon::prelude::*;
use xxhash_rust::xxh3::xxh3_128_with_seed;
use xxhash_rust::xxh3::xxh3_64;

// Distinct seeds per key kind, so `"a"`, `b"a"` and `97` never collide
// systematically (Python treats them as distinct dict keys, too).
const SEED_INT: u64 = 0x0001_7172_6874_6168;
const SEED_STR: u64 = 0x0002_7172_6874_6168;
const SEED_BYTES: u64 = 0x0003_7172_6874_6168;

/// Hash of an integer key. All integer inputs (Python `int`, `int64` and
/// `uint64` buffers) go through `i128`, so they agree with each other.
#[inline]
pub fn hash_int(v: i128) -> u128 {
    xxh3_128_with_seed(&v.to_le_bytes(), SEED_INT)
}

#[inline]
pub fn hash_str(s: &[u8]) -> u128 {
    xxh3_128_with_seed(s, SEED_STR)
}

#[inline]
pub fn hash_bytes(b: &[u8]) -> u128 {
    xxh3_128_with_seed(b, SEED_BYTES)
}

/// Keys are already uniformly random 128-bit hashes: only mix in PtrHash's
/// global seed (as upstream's `Xxh3_128` does). The name is part of the
/// epserde type hash, i.e. of the file format: do not rename.
#[derive(Clone, Epserde)]
#[epserde(deep_copy)]
pub struct PreHashed;

impl KeyHasher<u128> for PreHashed {
    type H = u128;
    #[inline(always)]
    fn hash(x: &u128, seed: u64) -> u128 {
        *x ^ (seed as u128 | (seed as u128) << 64)
    }
}

/// Single part, linear bucket function, `Vec<u32>` remap: ~3.0 bits/key,
/// fastest queries.
type FastMphf = DefaultPtrHash<PreHashed, u128>;
/// Multiple parts (parallel construction), cubic bucket function, Elias-Fano
/// remap: ~2.2 bits/key, somewhat slower queries.
type CompactMphf =
    PtrHash<u128, CubicEps, CachelineEfVec<Vec<CachelineEf>>, PreHashed, Vec<u8>, false, true>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    Fast,
    Compact,
}

impl Variant {
    pub fn name(self) -> &'static str {
        match self {
            Variant::Fast => "fast",
            Variant::Compact => "compact",
        }
    }
}

#[derive(Clone)]
enum Inner {
    Fast(FastMphf),
    Compact(CompactMphf),
}

/// Fingerprints indexed by MPHF slot, for (probabilistic) rejection of keys
/// outside the build set.
#[derive(Clone, PartialEq, Debug)]
enum Fingerprints {
    None,
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
}

impl Fingerprints {
    fn bits(&self) -> u32 {
        match self {
            Fingerprints::None => 0,
            Fingerprints::U8(_) => 8,
            Fingerprints::U16(_) => 16,
            Fingerprints::U32(_) => 32,
        }
    }

    fn zeros(bits: u32, n: usize) -> Self {
        match bits {
            8 => Fingerprints::U8(vec![0; n]),
            16 => Fingerprints::U16(vec![0; n]),
            32 => Fingerprints::U32(vec![0; n]),
            _ => Fingerprints::None,
        }
    }

    #[inline]
    fn get(&self, i: usize) -> u32 {
        match self {
            Fingerprints::None => 0,
            Fingerprints::U8(v) => v[i] as u32,
            Fingerprints::U16(v) => v[i] as u32,
            Fingerprints::U32(v) => v[i],
        }
    }

    fn set(&mut self, i: usize, fp: u32) {
        match self {
            Fingerprints::None => {}
            Fingerprints::U8(v) => v[i] = fp as u8,
            Fingerprints::U16(v) => v[i] = fp as u16,
            Fingerprints::U32(v) => v[i] = fp,
        }
    }

    fn write(&self, out: &mut Vec<u8>) {
        match self {
            Fingerprints::None => {}
            Fingerprints::U8(v) => out.extend_from_slice(v),
            Fingerprints::U16(v) => v.iter().for_each(|x| out.extend(x.to_le_bytes())),
            Fingerprints::U32(v) => v.iter().for_each(|x| out.extend(x.to_le_bytes())),
        }
    }

    fn read(bits: u32, n: usize, data: &[u8]) -> Option<Self> {
        let w = bits as usize / 8;
        if data.len() != n.checked_mul(w)? {
            return None;
        }
        Some(match bits {
            0 => Fingerprints::None,
            8 => Fingerprints::U8(data.to_vec()),
            16 => Fingerprints::U16(
                data.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| u16::from_le_bytes(*c))
                    .collect(),
            ),
            32 => Fingerprints::U32(
                data.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| u32::from_le_bytes(*c))
                    .collect(),
            ),
            _ => return None,
        })
    }
}

#[inline]
fn mix64(mut z: u64) -> u64 {
    // splitmix64 finalizer
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Fingerprint of a key hash: an independent mix of all 128 bits, so a
/// non-member landing on a member's slot matches with probability 2^-bits.
#[inline]
fn fingerprint(h: u128, bits: u32) -> u32 {
    if bits == 0 {
        return 0;
    }
    let m = mix64(h as u64 ^ mix64((h >> 64) as u64 ^ 0x6a09_e667_f3bc_c908));
    (m >> (64 - bits)) as u32
}

#[derive(Debug, PartialEq, Eq)]
pub enum BuildError {
    /// Two keys hash identically: almost surely the same key twice.
    Duplicate,
    /// PtrHash could not find a seed (should not happen for distinct keys).
    Failed,
    /// Unsupported `fingerprint_bits`.
    BadFingerprintBits,
}

#[derive(Debug, PartialEq, Eq)]
pub enum LoadError {
    BadMagic,
    UnsupportedVersion(u8),
    Corrupt,
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::BadMagic => write!(f, "not rptrhash data (bad magic)"),
            LoadError::UnsupportedVersion(v) => {
                write!(f, "unsupported rptrhash format version {v}")
            }
            LoadError::Corrupt => write!(f, "corrupt or truncated rptrhash data"),
        }
    }
}

/// Above this many keys, batch queries are split across rayon threads.
const PAR_THRESHOLD: usize = 1 << 15;
const PAR_CHUNK: usize = 1 << 13;

#[derive(Clone)]
pub struct Mphf {
    inner: Inner,
    n: usize,
    fps: Fingerprints,
}

impl Mphf {
    /// Build a minimal perfect hash over distinct key hashes. Sorts
    /// `hashes` in place (PtrHash does not care about key order) to detect
    /// duplicates up front: upstream would otherwise retry many seeds,
    /// printing to stderr each time.
    pub fn build(hashes: &mut [u128], variant: Variant, fp_bits: u32) -> Result<Self, BuildError> {
        if !matches!(fp_bits, 0 | 8 | 16 | 32) {
            return Err(BuildError::BadFingerprintBits);
        }
        hashes.par_sort_unstable();
        if hashes.windows(2).any(|w| w[0] == w[1]) {
            return Err(BuildError::Duplicate);
        }
        // Upstream's pilot search draws its start pilots from
        // `fastrand::Rng::new()`, i.e. from the thread-local fastrand state of
        // whichever (rayon) thread runs a part. Seeding every thread makes
        // single-part (`Fast`) builds reproducible. Multi-part builds remain
        // scheduling-dependent.
        fastrand::seed(BUILD_RNG_SEED);
        rayon::broadcast(|_| fastrand::seed(BUILD_RNG_SEED));
        let hashes: &[u128] = hashes;
        let inner = match variant {
            Variant::Fast => FastMphf::try_new(hashes, PtrHashParams::default()).map(Inner::Fast),
            Variant::Compact => {
                // Upstream: "If construction fails, try again with decreased lambda."
                [3.9, 3.7, 3.5].into_iter().find_map(|lambda| {
                    let params = PtrHashParams {
                        lambda,
                        ..PtrHashParams::default_compact()
                    };
                    CompactMphf::try_new(hashes, params).map(Inner::Compact)
                })
            }
        };
        let Some(inner) = inner else {
            return Err(BuildError::Failed);
        };
        let mut m = Mphf {
            inner,
            n: hashes.len(),
            fps: Fingerprints::zeros(fp_bits, hashes.len()),
        };
        if fp_bits > 0 {
            let idx = m.index_many(hashes);
            let mut fps = std::mem::replace(&mut m.fps, Fingerprints::None);
            for (&h, &i) in hashes.iter().zip(&idx) {
                fps.set(i as usize, fingerprint(h, fp_bits));
            }
            m.fps = fps;
        }
        Ok(m)
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    pub fn variant(&self) -> Variant {
        match self.inner {
            Inner::Fast(_) => Variant::Fast,
            Inner::Compact(_) => Variant::Compact,
        }
    }

    pub fn fingerprint_bits(&self) -> u32 {
        self.fps.bits()
    }

    /// Index in `[0, n)` for a key hash. Arbitrary (but in range) for keys
    /// outside the build set. Requires `n > 0`.
    #[inline]
    pub fn index(&self, h: u128) -> usize {
        match &self.inner {
            Inner::Fast(m) => m.index(&h),
            Inner::Compact(m) => m.index(&h),
        }
    }

    /// Like `index`, but `None` when the fingerprint rules the key out (or
    /// the MPHF is empty). Without fingerprints, every key maps somewhere.
    #[inline]
    pub fn get(&self, h: u128) -> Option<usize> {
        if self.n == 0 {
            return None;
        }
        let i = self.index(h);
        let bits = self.fps.bits();
        (bits == 0 || self.fps.get(i) == fingerprint(h, bits)).then_some(i)
    }

    /// Batch `index`, parallel and prefetched for large inputs.
    pub fn index_many(&self, hs: &[u128]) -> Vec<u64> {
        let mut out = vec![0u64; hs.len()];
        self.index_many_into(hs, &mut out);
        out
    }

    /// `index_many` writing into `out` (`out.len() == hs.len()`).
    pub fn index_many_into(&self, hs: &[u128], out: &mut [u64]) {
        assert_eq!(hs.len(), out.len());
        if self.n == 0 {
            out.fill(0);
            return;
        }
        let run = |hs: &[u128], out: &mut [u64]| match &self.inner {
            Inner::Fast(m) => fill(out, m.index_stream::<32, _>(hs)),
            Inner::Compact(m) => fill(out, m.index_stream::<32, _>(hs)),
        };
        if hs.len() >= PAR_THRESHOLD {
            hs.par_chunks(PAR_CHUNK)
                .zip(out.par_chunks_mut(PAR_CHUNK))
                .for_each(|(h, o)| run(h, o));
        } else {
            run(hs, out);
        }
    }

    /// Batch `get`: `missing` where a fingerprint rules the key out.
    pub fn get_many(&self, hs: &[u128], missing: i64) -> Vec<i64> {
        let mut out = vec![0i64; hs.len()];
        self.get_many_into(hs, missing, &mut out);
        out
    }

    /// `get_many` writing into `out` (`out.len() == hs.len()`).
    pub fn get_many_into(&self, hs: &[u128], missing: i64, out: &mut [i64]) {
        assert_eq!(hs.len(), out.len());
        if self.n == 0 {
            out.fill(missing);
            return;
        }
        let bits = self.fps.bits();
        let run = |hs: &[u128], out: &mut [i64]| {
            // Reuse `out` as u64 scratch for the indices.
            let idx: &mut [u64] = bytemuck_i64_as_u64(out);
            match &self.inner {
                Inner::Fast(m) => fill(idx, m.index_stream::<32, _>(hs)),
                Inner::Compact(m) => fill(idx, m.index_stream::<32, _>(hs)),
            }
            for (o, &h) in out.iter_mut().zip(hs) {
                let i = *o as usize;
                if bits != 0 && self.fps.get(i) != fingerprint(h, bits) {
                    *o = missing;
                }
            }
        };
        if hs.len() >= PAR_THRESHOLD {
            hs.par_chunks(PAR_CHUNK)
                .zip(out.par_chunks_mut(PAR_CHUNK))
                .for_each(|(h, o)| run(h, o));
        } else {
            run(hs, out);
        }
    }

    /// Space of the MPHF itself (pilots + remap), excluding fingerprints.
    pub fn mphf_bits_per_key(&self) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        let (p, r) = match &self.inner {
            Inner::Fast(m) => m.bits_per_element(),
            Inner::Compact(m) => m.bits_per_element(),
        };
        p + r
    }

    pub fn bits_per_key(&self) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        self.mphf_bits_per_key() + self.fps.bits() as f64
    }

    // ---- serialization ----------------------------------------------------
    //
    // magic "RPTRHASH" | version u8 | variant u8 | fp_bits u8 | pad u8 |
    // n u64 LE | body_len u64 LE | checksum u64 LE (xxh3_64 of body) |
    // body = epserde(PtrHash) ++ fingerprints (LE)

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut body = Vec::new();
        // Writing into a Vec cannot fail.
        unsafe {
            match &self.inner {
                Inner::Fast(m) => m.serialize(&mut body),
                Inner::Compact(m) => m.serialize(&mut body),
            }
        }
        .expect("serializing into a Vec cannot fail");
        let mphf_len = body.len() as u64;
        self.fps.write(&mut body);

        let mut out = Vec::with_capacity(HEADER_LEN + body.len());
        out.extend_from_slice(MAGIC);
        out.push(FORMAT_VERSION);
        out.push(match self.variant() {
            Variant::Fast => 0,
            Variant::Compact => 1,
        });
        out.push(self.fps.bits() as u8);
        out.push(0);
        out.extend((self.n as u64).to_le_bytes());
        out.extend(mphf_len.to_le_bytes());
        out.extend(xxh3_64(&body).to_le_bytes());
        out.extend_from_slice(&body);
        out
    }

    /// Load data written by `to_bytes`. A checksum guards against
    /// corruption, but (like pickle) only load data from trusted sources.
    pub fn from_bytes(data: &[u8]) -> Result<Self, LoadError> {
        if data.len() < MAGIC.len() || &data[..MAGIC.len()] != MAGIC {
            return Err(LoadError::BadMagic);
        }
        if data.len() < HEADER_LEN {
            return Err(LoadError::Corrupt);
        }
        let version = data[8];
        if version != FORMAT_VERSION {
            return Err(LoadError::UnsupportedVersion(version));
        }
        let variant = data[9];
        let fp_bits = data[10] as u32;
        let u64_at = |o: usize| u64::from_le_bytes(data[o..o + 8].try_into().unwrap());
        let n = usize::try_from(u64_at(12)).map_err(|_| LoadError::Corrupt)?;
        let mphf_len = usize::try_from(u64_at(20)).map_err(|_| LoadError::Corrupt)?;
        let checksum = u64_at(28);
        let body = &data[HEADER_LEN..];
        if xxh3_64(body) != checksum || mphf_len > body.len() {
            return Err(LoadError::Corrupt);
        }
        let (mphf_bytes, fp_bytes) = body.split_at(mphf_len);
        let fps = Fingerprints::read(fp_bits, n, fp_bytes).ok_or(LoadError::Corrupt)?;
        let mut cur = std::io::Cursor::new(mphf_bytes);
        let inner = match variant {
            0 => {
                let m = unsafe { FastMphf::deserialize_full(&mut cur) }
                    .map_err(|_| LoadError::Corrupt)?;
                (m.n() == n).then_some(Inner::Fast(m))
            }
            1 => {
                let m = unsafe { CompactMphf::deserialize_full(&mut cur) }
                    .map_err(|_| LoadError::Corrupt)?;
                (m.n() == n).then_some(Inner::Compact(m))
            }
            _ => None,
        }
        .ok_or(LoadError::Corrupt)?;
        Ok(Mphf { inner, n, fps })
    }
}

const BUILD_RNG_SEED: u64 = 0x7074_7268_6173_6801;

const MAGIC: &[u8; 8] = b"RPTRHASH";
const FORMAT_VERSION: u8 = 1;
const HEADER_LEN: usize = 8 + 4 + 8 + 8 + 8;

fn bytemuck_i64_as_u64(v: &mut [i64]) -> &mut [u64] {
    // SAFETY: i64 and u64 have identical size, alignment and validity.
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u64, v.len()) }
}

/// Upstream's stream iterator only implements `fold` (not `next`).
fn fill(out: &mut [u64], it: impl Iterator<Item = usize>) {
    let mut o = out.iter_mut();
    it.for_each(|i| *o.next().unwrap() = i as u64);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(n: usize, salt: i128) -> Vec<u128> {
        (0..n as i128).map(|i| hash_int(i * 7 + salt)).collect()
    }

    fn check_bijection(m: &Mphf, hs: &[u128]) {
        let mut seen = vec![false; hs.len()];
        for &h in hs {
            let i = m.index(h);
            assert!(i < hs.len());
            assert!(!seen[i], "collision at {i}");
            seen[i] = true;
        }
        let batch = m.index_many(hs);
        for (&h, &i) in hs.iter().zip(&batch) {
            assert_eq!(m.index(h) as u64, i);
        }
    }

    #[test]
    fn bijection_all_variants_and_sizes() {
        for variant in [Variant::Fast, Variant::Compact] {
            for n in [1, 2, 3, 10, 100, 1000, 50_000] {
                let hs = keys(n, 0);
                let m = Mphf::build(&mut hs.clone(), variant, 0).unwrap();
                assert_eq!(m.len(), n);
                check_bijection(&m, &hs);
            }
        }
    }

    /// Rust-only timing breakdown: `cargo test --release -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn timing() {
        let n = 10_000_000;
        let t = std::time::Instant::now();
        let hs: Vec<u128> = (0..n as i128)
            .into_par_iter()
            .map(|i| hash_int(i * 0x9e37_79b9))
            .collect();
        eprintln!("hash   {:?}", t.elapsed());
        for variant in [Variant::Fast, Variant::Compact] {
            let t = std::time::Instant::now();
            let m = Mphf::build(&mut hs.clone(), variant, 0).unwrap();
            eprintln!("{variant:?} build {:?}", t.elapsed());
            let t = std::time::Instant::now();
            let out = m.index_many(&hs);
            eprintln!("{variant:?} index_many {:?} {}", t.elapsed(), out[0]);
            let t = std::time::Instant::now();
            let s: usize = hs.iter().map(|&h| m.index(h)).sum();
            eprintln!("{variant:?} serial index {:?} {s}", t.elapsed());
        }
    }

    #[test]
    fn empty() {
        let m = Mphf::build(&mut [], Variant::Fast, 8).unwrap();
        assert!(m.is_empty());
        assert_eq!(m.get(hash_int(1)), None);
        assert_eq!(m.index_many(&[hash_int(1)]), vec![0]);
        let m2 = Mphf::from_bytes(&m.to_bytes()).unwrap();
        assert!(m2.is_empty());
    }

    #[test]
    fn duplicates_rejected() {
        let mut hs = keys(1000, 0);
        hs.push(hs[17]);
        assert_eq!(
            Mphf::build(&mut hs.clone(), Variant::Fast, 0).err(),
            Some(BuildError::Duplicate)
        );
        assert_eq!(
            Mphf::build(&mut hs, Variant::Compact, 0).err(),
            Some(BuildError::Duplicate)
        );
    }

    #[test]
    fn fingerprints_accept_members_reject_most_others() {
        let n = 20_000;
        let hs = keys(n, 0);
        let others = keys(n, 3); // i*7+3 never equals j*7
        for bits in [8u32, 16, 32] {
            let m = Mphf::build(&mut hs.clone(), Variant::Fast, bits).unwrap();
            assert!(hs.iter().all(|&h| m.get(h).is_some()));
            let fp = others.iter().filter(|&&h| m.get(h).is_some()).count();
            let expected = n as f64 / 2f64.powi(bits as i32);
            assert!(
                (fp as f64) < 3.0 * expected + 10.0,
                "{bits} bits: {fp} false positives, expected ~{expected}"
            );
            let got = m.get_many(&others, -1);
            assert_eq!(got.iter().filter(|&&x| x >= 0).count(), fp);
        }
    }

    #[test]
    fn roundtrip_bytes() {
        let hs = keys(5000, 0);
        for variant in [Variant::Fast, Variant::Compact] {
            for bits in [0, 16] {
                let m = Mphf::build(&mut hs.clone(), variant, bits).unwrap();
                let b = m.to_bytes();
                let m2 = Mphf::from_bytes(&b).unwrap();
                assert_eq!(m2.variant(), variant);
                assert_eq!(m2.fingerprint_bits(), bits);
                assert_eq!(m.index_many(&hs), m2.index_many(&hs));
                assert_eq!(m2.to_bytes(), b);
            }
        }
    }

    #[test]
    fn corrupt_bytes_rejected() {
        let b = Mphf::build(&mut keys(100, 0), Variant::Fast, 8)
            .unwrap()
            .to_bytes();
        assert_eq!(Mphf::from_bytes(b"nope").err(), Some(LoadError::BadMagic));
        assert_eq!(
            Mphf::from_bytes(&b[..b.len() - 1]).err(),
            Some(LoadError::Corrupt)
        );
        let mut flipped = b.clone();
        let last = flipped.len() - 1;
        flipped[last] ^= 1;
        assert_eq!(Mphf::from_bytes(&flipped).err(), Some(LoadError::Corrupt));
        let mut v = b.clone();
        v[8] = 99;
        assert_eq!(
            Mphf::from_bytes(&v).err(),
            Some(LoadError::UnsupportedVersion(99))
        );
    }

    #[test]
    fn space_is_small() {
        let hs = keys(200_000, 0);
        let fast = Mphf::build(&mut hs.clone(), Variant::Fast, 0).unwrap();
        let compact = Mphf::build(&mut hs.clone(), Variant::Compact, 0).unwrap();
        assert!(fast.bits_per_key() < 3.6, "{}", fast.bits_per_key());
        assert!(compact.bits_per_key() < 2.8, "{}", compact.bits_per_key());
        assert!((fast.to_bytes().len() as f64) * 8.0 / 200_000.0 < 4.0);
    }
}
