//! Scoring for the turbulence benchmark cases (`tests/oracle/sa-flat-plate-2dzp`,
//! `tests/oracle/ddes-backstep-driver-seegmiller`): the reference correlations, the quantities
//! read from a run (momentum thickness, reattachment), and a gate table that prints each result
//! against its reference and tolerance.
//!
//! References:
//! - Kármán–Schoenherr's skin friction, as the NASA Turbulence Modeling Resource states it for its
//!   2DZP case (<https://tmbwg.github.io/turbmodels/flatplate_val.html>):
//!   c_f = 1 / (17.08 L² + 25.11 L + 6.012), L = log₁₀ Re_θ.
//! - Blasius's laminar layer: c_f = 0.664 / √Re_x and θ = 0.664 x / √Re_x, so c_f Re_θ = 0.664²
//!   (H. Schlichting, *Boundary-Layer Theory*, 7th ed., McGraw-Hill 1979, ch. VII).

/// Kármán–Schoenherr's turbulent flat-plate skin friction at a momentum-thickness Reynolds
/// number.
pub fn karman_schoenherr(re_theta: f64) -> f64 {
    let l = re_theta.log10();
    1.0 / (17.08 * l * l + 25.11 * l + 6.012)
}

/// Blasius's laminar flat plate: c_f Re_θ.
pub const BLASIUS_CF_RE_THETA: f64 = 0.664 * 0.664;

/// The momentum thickness θ = ∫ (u/U_e)(1 − u/U_e) dz of a wall-normal profile sampled at cell
/// centres `spacing` apart, the first half a cell from the wall (the midpoint rule, exact for a
/// piecewise-constant profile).
pub fn momentum_thickness(u: &[f64], spacing: f64, edge: f64) -> f64 {
    u.iter().map(|&v| (v / edge) * (1.0 - v / edge)).sum::<f64>() * spacing
}

/// The displacement thickness δ* = ∫ (1 − u/U_e) dz, sampled as [`momentum_thickness`].
pub fn displacement_thickness(u: &[f64], spacing: f64, edge: f64) -> f64 {
    u.iter().map(|&v| 1.0 - v / edge).sum::<f64>() * spacing
}

/// `y` at `x` by linear interpolation between the samples `(xs, ys)`, `xs` increasing; `None`
/// outside them.
pub fn interpolate(xs: &[f64], ys: &[f64], x: f64) -> Option<f64> {
    let k = xs.windows(2).position(|w| w[0] <= x && x <= w[1])?;
    let (x0, x1, y0, y1) = (xs[k], xs[k + 1], ys[k], ys[k + 1]);
    Some(if x1 > x0 { y0 + (y1 - y0) * (x - x0) / (x1 - x0) } else { y0 })
}

/// The reattachment point of a separated floor: from the wall shear's minimum downstream, the
/// first change of sign from negative to non-negative, by linear interpolation between the
/// samples (`xs` increasing). `None` when the shear never goes negative or never comes back.
pub fn reattachment(xs: &[f64], tau: &[f64]) -> Option<f64> {
    let n = xs.len().min(tau.len());
    let lowest = (0..n).min_by(|&a, &b| tau[a].total_cmp(&tau[b]))?;
    if tau[lowest] >= 0.0 {
        return None;
    }
    (lowest..n.saturating_sub(1)).find(|&k| tau[k] < 0.0 && tau[k + 1] >= 0.0).map(|k| xs[k] - tau[k] * (xs[k + 1] - xs[k]) / (tau[k + 1] - tau[k]))
}

/// Whether a sampled history has settled: every channel's latest value differs from its value
/// `window` samples earlier by less than `relative` of its magnitude. `history` holds one row of
/// channels per sample. Too short a history is not settled.
pub fn settled(history: &[Vec<f64>], window: usize, relative: f64) -> bool {
    let (Some(last), Some(then)) = (history.last(), history.len().checked_sub(window + 1).and_then(|k| history.get(k))) else {
        return false;
    };
    window > 0 && last.len() == then.len() && last.iter().zip(then).all(|(a, b)| a.is_finite() && (a - b).abs() <= relative * a.abs())
}

/// The force groups of every face of obstacle `body`. A wall strip's shear is the sum: a wall
/// link takes the face nearest its midpoint, lowest face first among equally near ones, so the
/// diagonal links at a strip's ends go to its end faces rather than its top.
pub fn body_groups(flow: &fieldcraft_lbm::Flow, body: &str) -> Vec<usize> {
    flow.groups().iter().enumerate().filter(|(_, g)| g.body.as_deref() == Some(body)).map(|(k, _)| k).collect()
}

/// One quantity of a case against its reference.
#[derive(Debug, Clone)]
pub struct Gate {
    pub case: String,
    pub quantity: String,
    pub value: f64,
    pub reference: f64,
    /// The allowed relative error.
    pub relative: f64,
}

impl Gate {
    pub fn new(case: &str, quantity: &str, value: f64, reference: f64, relative: f64) -> Gate {
        Gate { case: case.into(), quantity: quantity.into(), value, reference, relative }
    }

    /// The relative error (value − reference) / |reference|.
    pub fn error(&self) -> f64 {
        (self.value - self.reference) / self.reference.abs()
    }

    pub fn passes(&self) -> bool {
        self.value.is_finite() && self.error().abs() <= self.relative
    }
}

/// The gates of one or more cases, printed as a table.
#[derive(Debug, Clone, Default)]
pub struct Score {
    pub gates: Vec<Gate>,
}

impl Score {
    pub fn push(&mut self, gate: Gate) {
        self.gates.push(gate);
    }

    pub fn passes(&self) -> bool {
        !self.gates.is_empty() && self.gates.iter().all(Gate::passes)
    }

    /// The table: one row per gate, then the verdict.
    pub fn table(&self) -> String {
        let mut s = String::from("| case | quantity | value | reference | error | tolerance | |\n|---|---|---|---|---|---|---|\n");
        for g in &self.gates {
            s += &format!(
                "| {} | {} | {:.5e} | {:.5e} | {:+.2} % | ±{:.1} % | {} |\n",
                g.case,
                g.quantity,
                g.value,
                g.reference,
                100.0 * g.error(),
                100.0 * g.relative,
                if g.passes() { "pass" } else { "FAIL" }
            );
        }
        let passing = self.gates.iter().filter(|g| g.passes()).count();
        s += &format!("\n{passing} of {} gates pass\n", self.gates.len());
        s
    }
}

/// A case's `expected.json` reference values: `(id, value, relative tolerance)` of each quantity.
pub fn expected(case: &str) -> Vec<(String, f64, f64)> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/oracle").join(case).join("expected.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    v["quantities"]
        .as_array()
        .unwrap()
        .iter()
        .map(|q| (q["id"].as_str().unwrap().to_string(), q["value"].as_f64().unwrap(), q["tolerance"]["relative"].as_f64().unwrap()))
        .collect()
}
