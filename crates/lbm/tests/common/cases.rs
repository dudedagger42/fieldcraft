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

/// Why a floor shear profile gives no reattachment point.
#[derive(Debug, Clone, PartialEq)]
pub enum NoReattachment {
    /// The samples cannot be read: the lengths differ, there are fewer than two, `xs` is not
    /// strictly increasing, or a value is not finite.
    Input(String),
    /// The shear never changes from negative to non-negative. `lowest` is its smallest value:
    /// ≥ 0 means attached everywhere, < 0 means it never reattached inside the samples.
    None { lowest: f64 },
    /// It does so more than once, at these x: a secondary bubble, or a mean that has not
    /// converged. Which one is the reattachment cannot be decided, so none is returned.
    Several(Vec<f64>),
}

/// Every place where the floor shear `tau`, sampled at `xs` (strictly increasing), changes from
/// negative to non-negative, in order. Between samples k and k + 1 with τ_k < 0 ≤ τ_{k+1}, the
/// crossing is the zero of the straight line through them,
/// x = x_k − τ_k (x_{k+1} − x_k) / (τ_{k+1} − τ_k),
/// which is exact when τ is linear on that interval. For a smooth τ the error is at most
/// Δx² max|τ''| / (8 min|τ'|) over the interval (the linear interpolation error, Δx²/8 · max|τ''|,
/// divided by the slope). A change from positive to negative (separation, such as the edge of
/// the corner eddy under the step) is not a crossing here.
pub fn crossings(xs: &[f64], tau: &[f64]) -> Result<Vec<f64>, NoReattachment> {
    if xs.len() != tau.len() || xs.len() < 2 {
        return Err(NoReattachment::Input(format!("{} stations and {} shear values: two or more of each, as many", xs.len(), tau.len())));
    }
    if let Some(v) = xs.iter().chain(tau).find(|v| !v.is_finite()) {
        return Err(NoReattachment::Input(format!("a station or shear value is {v}")));
    }
    if let Some(w) = xs.windows(2).find(|w| w[1] <= w[0]) {
        return Err(NoReattachment::Input(format!("the stations are not increasing: {} then {}", w[0], w[1])));
    }
    Ok(xs
        .windows(2)
        .zip(tau.windows(2))
        .filter(|(_, t)| t[0] < 0.0 && t[1] >= 0.0)
        .map(|(x, t)| x[0] - t[0] * (x[1] - x[0]) / (t[1] - t[0]))
        .collect())
}

/// The reattachment point of a separated floor: the one place where the time-mean floor shear
/// changes from negative to non-negative ([`crossings`], linear interpolation). Refused, with
/// what was found, when there is no such crossing or more than one.
pub fn reattachment(xs: &[f64], tau: &[f64]) -> Result<f64, NoReattachment> {
    let at = crossings(xs, tau)?;
    match at.as_slice() {
        [x] => Ok(*x),
        [] => Err(NoReattachment::None { lowest: tau.iter().copied().fold(f64::INFINITY, f64::min) }),
        _ => Err(NoReattachment::Several(at)),
    }
}

/// When a gated case has settled: after at least `min_flow_throughs` flow-throughs, the time
/// mean of every gated quantity over the last averaging window differs from its mean over the
/// window before by less than `drift` of its magnitude. Each case sets its averaging window to
/// whole periods of its slowest oscillation (on the plate, the domain's acoustic quarter wave), so
/// a stationary oscillation does not read as drift.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Settling {
    pub min_flow_throughs: f64,
    pub drift: f64,
}

/// The board's settling rule for the gated benchmark cases: 2DZP, the Driver & Seegmiller step
/// and the SA channel use this, never their own numbers.
pub const GATED: Settling = Settling { min_flow_throughs: 2.0, drift: 1e-3 };

impl Settling {
    /// Whether `history` (one row of quantities per sample) has settled after `flow_throughs`,
    /// with averaging windows of `window` samples.
    pub fn settled(&self, history: &[Vec<f64>], window: usize, flow_throughs: f64) -> bool {
        flow_throughs >= self.min_flow_throughs && means_settled(history, window, self.drift)
    }
}

/// The mean of each column of `rows` (`None` for no rows or rows of different lengths).
pub fn column_means(rows: &[Vec<f64>]) -> Option<Vec<f64>> {
    let first = rows.first()?;
    if rows.iter().any(|r| r.len() != first.len()) {
        return None;
    }
    Some((0..first.len()).map(|c| rows.iter().map(|r| r[c]).sum::<f64>() / rows.len() as f64).collect())
}

/// Whether the means of the last `window` rows of `history` differ from the means of the `window`
/// rows before them by less than `relative` of their magnitude in every column. Too short a
/// history, or a non-finite mean, is not settled.
pub fn means_settled(history: &[Vec<f64>], window: usize, relative: f64) -> bool {
    let Some(start) = history.len().checked_sub(2 * window) else {
        return false;
    };
    let (Some(then), Some(now)) = (column_means(&history[start..start + window]), column_means(&history[start + window..])) else {
        return false;
    };
    window > 0 && now.len() == then.len() && now.iter().zip(&then).all(|(a, b)| a.is_finite() && b.is_finite() && (a - b).abs() <= relative * a.abs())
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
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display())).unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    v["quantities"]
        .as_array()
        .unwrap()
        .iter()
        .map(|q| (q["id"].as_str().unwrap().to_string(), q["value"].as_f64().unwrap(), q["tolerance"]["relative"].as_f64().unwrap()))
        .collect()
}
