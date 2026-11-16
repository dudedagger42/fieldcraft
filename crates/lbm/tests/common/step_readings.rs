//! Readings of the backward-facing step beyond the carried wall shear: where the flow reattaches
//! (two ways, both on the whole averaging window) and how much of the shear stress in the shear
//! layer is resolved. Test support; the physics is read from the flow by the harness.
//!
//! Only what is reported is averaged, and each average is a few rows or columns, never a whole
//! field: the floor's first [`ROWS`] rows over every x column, and at each station one column of
//! heights. A run's memory therefore stays the flow's own.
//!
//! * **Mean-shear reading.** The floor's time-mean span-mean u in its first rows; the harness reads
//!   the rows 2 to 4 crossing from them. The primary reattachment is the *furthest downstream*
//!   negative-to-non-negative change beyond one step height ([`furthest_crossing`]): the crossings
//!   near 0.2 and 1.3 H belong to the corner bubble.
//! * **Forward-flow reading.** The station where the first open row's velocity is positive half of
//!   the time ([`Floor::forward_flow`]).
//! * **Resolved against modelled shear stress.** At a station, over the samples and the span: the
//!   resolved stress −⟨u′w′⟩ and the modelled (ν + ⟨ν_t⟩) dU/dz, from running sums of u, w, uw and
//!   ν_t per height ([`Station`]); the fraction is read at the height of the largest total shear
//!   above the wall layer.

use super::cases::crossings;

/// Rows above the floor kept per column: the wall reader's four, and two more.
pub const ROWS: usize = 6;
/// Most columns, heights or stations one accumulator takes (a hostile size is refused).
const MAX_LEN: usize = 1 << 20;

/// The furthest-downstream negative → non-negative change of `values` along `x` beyond `beyond`
/// (linear between stations), or why there is none.
pub fn furthest_crossing(x: &[f64], values: &[f64], beyond: f64) -> Result<f64, String> {
    let at = crossings(x, values).map_err(|e| format!("{e:?}"))?;
    at.into_iter().rfind(|&c| c > beyond).ok_or_else(|| format!("no negative → non-negative change beyond {beyond} H"))
}

fn finite_all(values: &[f64], what: &str) -> Result<(), String> {
    match values.iter().find(|v| !v.is_finite()) {
        Some(v) => Err(format!("{what} holds {v}")),
        None => Ok(()),
    }
}

/// The floor's time means: span-mean u in the first [`ROWS`] rows of every column, and the share
/// of samples in which the first open row's velocity is positive.
#[derive(Debug, Clone, PartialEq)]
pub struct Floor {
    /// The columns' x, in H, increasing.
    pub x: Vec<f64>,
    count: u64,
    sum: Vec<Vec<f64>>,
    positive: Vec<f64>,
}

impl Floor {
    pub fn new(x: Vec<f64>) -> Result<Floor, String> {
        if x.len() < 2 || x.len() > MAX_LEN {
            return Err(format!("{} columns: two or more, at most {MAX_LEN}", x.len()));
        }
        finite_all(&x, "the columns' x")?;
        let n = x.len();
        Ok(Floor { x, count: 0, sum: vec![vec![0.0; n]; ROWS], positive: vec![0.0; n] })
    }

    /// One sample: `rows[r][i]` the span-mean u of row r at column i, `positive[i]` the share of
    /// the span's first-open-row cells at column i with u > 0.
    pub fn add(&mut self, rows: &[Vec<f64>], positive: &[f64]) -> Result<(), String> {
        let n = self.x.len();
        if rows.len() != ROWS || rows.iter().any(|r| r.len() != n) || positive.len() != n {
            return Err(format!("a floor sample is {ROWS} rows of {n} columns and {n} shares"));
        }
        rows.iter().try_for_each(|r| finite_all(r, "a floor row"))?;
        finite_all(positive, "the forward-flow shares")?;
        for (s, r) in self.sum.iter_mut().zip(rows) {
            s.iter_mut().zip(r).for_each(|(a, b)| *a += b);
        }
        self.positive.iter_mut().zip(positive).for_each(|(a, b)| *a += b);
        self.count += 1;
        Ok(())
    }

    pub fn samples(&self) -> u64 {
        self.count
    }

    /// The time-mean span-mean u of every row (empty rows of zeros before the first sample).
    pub fn mean_rows(&self) -> Vec<Vec<f64>> {
        let c = self.count.max(1) as f64;
        self.sum.iter().map(|r| r.iter().map(|v| v / c).collect()).collect()
    }

    /// The share of samples with a positive first-open-row velocity, per column.
    pub fn positive_share(&self) -> Vec<f64> {
        let c = self.count.max(1) as f64;
        self.positive.iter().map(|v| v / c).collect()
    }

    /// The forward-flow reading: the furthest-downstream station beyond `beyond` where the share
    /// of positive near-floor velocity rises through one half.
    pub fn forward_flow(&self, beyond: f64) -> Result<f64, String> {
        if self.count == 0 {
            return Err("no sample yet".into());
        }
        let shifted: Vec<f64> = self.positive_share().iter().map(|p| p - 0.5).collect();
        furthest_crossing(&self.x, &shifted, beyond)
    }
}

/// One station's running sums over samples and span cells, per height.
#[derive(Debug, Clone, PartialEq)]
pub struct Station {
    /// The station's x, in H.
    pub x: f64,
    /// The heights above the floor, in H, increasing.
    pub z: Vec<f64>,
    count: u64,
    su: Vec<f64>,
    sw: Vec<f64>,
    suw: Vec<f64>,
    snu: Vec<f64>,
}

/// The stresses at one height, lattice units: resolved −⟨u′w′⟩ and modelled (ν + ⟨ν_t⟩) dU/dz.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stress {
    pub z: f64,
    pub resolved: f64,
    pub modelled: f64,
}

impl Station {
    pub fn new(x: f64, z: Vec<f64>) -> Result<Station, String> {
        if z.len() < 3 || z.len() > MAX_LEN || !x.is_finite() {
            return Err(format!("a station needs a finite x and 3 to {MAX_LEN} heights, got {}", z.len()));
        }
        finite_all(&z, "the heights")?;
        if z.windows(2).any(|w| w[1] <= w[0]) {
            return Err("the heights are not increasing".into());
        }
        let n = z.len();
        Ok(Station { x, z, count: 0, su: vec![0.0; n], sw: vec![0.0; n], suw: vec![0.0; n], snu: vec![0.0; n] })
    }

    /// One point of the span at one sample: u, w and ν_t at every height.
    pub fn add(&mut self, u: &[f64], w: &[f64], nu_t: &[f64]) -> Result<(), String> {
        let n = self.z.len();
        if u.len() != n || w.len() != n || nu_t.len() != n {
            return Err(format!("a station sample is {n} heights of u, w and ν_t"));
        }
        finite_all(u, "u")?;
        finite_all(w, "w")?;
        finite_all(nu_t, "ν_t")?;
        for k in 0..n {
            self.su[k] += u[k];
            self.sw[k] += w[k];
            self.suw[k] += u[k] * w[k];
            self.snu[k] += nu_t[k];
        }
        self.count += 1;
        Ok(())
    }

    pub fn points(&self) -> u64 {
        self.count
    }

    /// The stress profile with molecular viscosity `nu`: resolved and modelled per height (the
    /// first and last heights, with no central difference, are left out).
    pub fn profile(&self, nu: f64) -> Result<Vec<Stress>, String> {
        if self.count == 0 || !nu.is_finite() || nu <= 0.0 {
            return Err(format!("no point yet, or ν = {nu}"));
        }
        let c = self.count as f64;
        let mean = |s: &[f64]| s.iter().map(|v| v / c).collect::<Vec<f64>>();
        let (u, w, uw, nut) = (mean(&self.su), mean(&self.sw), mean(&self.suw), mean(&self.snu));
        Ok((1..self.z.len() - 1)
            .map(|k| {
                let dudz = (u[k + 1] - u[k - 1]) / (self.z[k + 1] - self.z[k - 1]);
                Stress { z: self.z[k], resolved: -(uw[k] - u[k] * w[k]), modelled: (nu + nut[k]) * dudz }
            })
            .collect())
    }

    /// The resolved share of the total shear stress at the height, above `z_min` (the wall layer
    /// left out), where the total is largest in size: `(z, share)`.
    pub fn resolved_share(&self, nu: f64, z_min: f64) -> Result<(f64, f64), String> {
        let profile = self.profile(nu)?;
        let peak = profile
            .iter()
            .filter(|s| s.z >= z_min)
            .max_by(|a, b| (a.resolved + a.modelled).abs().total_cmp(&(b.resolved + b.modelled).abs()))
            .ok_or_else(|| format!("no height at or above {z_min} H"))?;
        let total = peak.resolved + peak.modelled;
        if total == 0.0 || !total.is_finite() {
            return Err(format!("the total shear stress is {total} at its peak height {:.3} H", peak.z));
        }
        Ok((peak.z, peak.resolved / total))
    }
}

/// Everything a run accumulates: the floor and its stations.
#[derive(Debug, Clone, PartialEq)]
pub struct Readings {
    pub floor: Floor,
    pub stations: Vec<Station>,
}

const MAGIC: u64 = 0x5354_4550_5244_0001; // "STEPRD", layout 1

fn put(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_f(out: &mut Vec<u8>, values: &[f64]) {
    put(out, values.len() as u64);
    values.iter().for_each(|v| put(out, v.to_bits()));
}

struct Cursor<'a>(&'a [u8]);

impl Cursor<'_> {
    fn u64(&mut self) -> Result<u64, String> {
        let (head, rest) = self.0.split_at_checked(8).ok_or("the readings file ends early")?;
        self.0 = rest;
        Ok(u64::from_le_bytes(head.try_into().map_err(|_| "the readings file is malformed")?))
    }

    fn f64s(&mut self) -> Result<Vec<f64>, String> {
        let n = usize::try_from(self.u64()?).map_err(|_| "a length does not fit")?;
        if n > MAX_LEN || n.checked_mul(8).is_none_or(|b| b > self.0.len()) {
            return Err(format!("a stored length of {n} does not fit the file"));
        }
        (0..n).map(|_| self.u64().map(f64::from_bits)).collect()
    }
}

impl Readings {
    /// The readings as bytes, with the step they were taken at.
    pub fn to_bytes(&self, steps: u64) -> Vec<u8> {
        let mut out = Vec::new();
        put(&mut out, MAGIC);
        put(&mut out, steps);
        put_f(&mut out, &self.floor.x);
        put(&mut out, self.floor.count);
        self.floor.sum.iter().for_each(|r| put_f(&mut out, r));
        put_f(&mut out, &self.floor.positive);
        put(&mut out, self.stations.len() as u64);
        for s in &self.stations {
            put(&mut out, s.x.to_bits());
            put_f(&mut out, &s.z);
            put(&mut out, s.count);
            [&s.su, &s.sw, &s.suw, &s.snu].into_iter().for_each(|v| put_f(&mut out, v));
        }
        out
    }

    /// The readings and their step from [`Readings::to_bytes`] output; an error for anything else.
    pub fn from_bytes(bytes: &[u8]) -> Result<(u64, Readings), String> {
        let mut c = Cursor(bytes);
        if c.u64()? != MAGIC {
            return Err("not a readings file of this layout".into());
        }
        let steps = c.u64()?;
        let mut floor = Floor::new(c.f64s()?)?;
        floor.count = c.u64()?;
        let n = floor.x.len();
        for r in floor.sum.iter_mut() {
            *r = c.f64s()?;
            if r.len() != n {
                return Err("a stored floor row has the wrong length".into());
            }
        }
        floor.positive = c.f64s()?;
        if floor.positive.len() != n {
            return Err("the stored forward-flow shares have the wrong length".into());
        }
        let stations = usize::try_from(c.u64()?).map_err(|_| "a count does not fit")?;
        if stations > 64 {
            return Err(format!("{stations} stations stored: at most 64"));
        }
        let mut list = Vec::new();
        for _ in 0..stations {
            let x = f64::from_bits(c.u64()?);
            let mut s = Station::new(x, c.f64s()?)?;
            s.count = c.u64()?;
            for v in [&mut s.su, &mut s.sw, &mut s.suw, &mut s.snu] {
                *v = c.f64s()?;
                if v.len() != s.z.len() {
                    return Err("a stored station array has the wrong length".into());
                }
            }
            list.push(s);
        }
        Ok((steps, Readings { floor, stations: list }))
    }

    /// Writes the readings next to a restart file, atomically.
    pub fn save(&self, path: &std::path::Path, steps: u64) -> Result<(), String> {
        let tmp = path.with_extension("readings.tmp");
        std::fs::write(&tmp, self.to_bytes(steps)).map_err(|e| format!("{}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// The readings stored at `path`, with their step.
    pub fn load(path: &std::path::Path) -> Result<(u64, Readings), String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Readings::from_bytes(&bytes)
    }
}
