//! Driver & Seegmiller's backward-facing step at zero upper-wall angle
//! (`tests/oracle/ddes-backstep-driver-seegmiller`, the acceptance case of slice 15b).
//!
//! The model, lengths in units of the step height H (Re_H = U H / ν with U the inlet speed):
//! - x from −x_up to x_down, z from −Δ to the roof, y periodic: one cell for a 2D run, a span S
//!   for the 3D hybrid run. The roof follows from the expansion ratio ER (downstream over upstream
//!   channel height): ER H / (ER − 1), 9 H for the case's 1.125;
//! - a uniform velocity inlet at x_min above the step and a pressure outlet with convective
//!   outflow at x_max; the roof is a no-slip box face;
//! - the step, x < 0 and z < H, is solid, its top split into strips H/2 long for x > −8 H (local
//!   wall force near the reference station x = −4 H) and one block upstream of them;
//! - the floor downstream, x > 0, is strips of solid cells H/8 long covering the bottom cell row,
//!   so their halfway wall lies at z = 0 and each strip reports its own wall force.
//!
//! [`Step::check`] refuses a configuration that is not the case's shape (ER > 1, an inflow
//! reaching past the reference station x = −4 H, an outlet at least 30 H downstream, the roof
//! and the lengths on cell faces), and [`Step::geometry`] measures the built grid: H, ER and the
//! lengths from its solid and fluid cells.
//!
//! The incoming boundary layers grow from the inlet along the step's top and the roof; x_up is
//! chosen so that Re_θ ≈ 5000 and δ ≈ 1.5 H at x = −4 H, which the run reports. U_ref is the speed
//! at mid-channel above the step at x = −4 H; C_f and C_p are formed with it.
//!
//! **The floor shear and x_r.** A strip's shear in one step is the x momentum its wall links took
//! from the fluid (momentum exchange, Mei et al., PRE 65:041203, 2002): [`Flow::forces`] of every
//! force group of the strip's body, summed, over the strip's area. That is the carried stress:
//! #829's closed-box budget (`tests/support/wall_balance.rs::closed_step`, run by the channel
//! studies in `turbulence.rs`) shows this wall impulse equals what the populations carry, to
//! round-off, on halfway walls and on Han's on-node wall-function bounce. The step is open
//! (inlet and outlet fluxes), so that budget cannot be formed here and the per-strip momentum
//! exchange is read directly. Each sample is the mean over `sample` steps; every strip's samples
//! go into its own `fieldcraft_lbm::statistics::Blocks` (#828), and the time-mean shear is the
//! mean of its complete blocks. x_r is the one change of sign from negative to non-negative of
//! the time-mean floor shear, by linear interpolation between strip centres
//! (`common::cases::reattachment`); no such change, or several, is refused and reported.
//!
//! **Statistics.** After `startup` flow-throughs, each completed block gives a mean shear profile
//! and its reattachment; the run stops when those reattachments pass the shared acceptance of the
//! LES studies (`Blocks::acceptance`, ≥ 20 blocks, 95 % interval and drift under 1 %, #796), or
//! at the step cap, reported inconclusive. Blocks whose profile is refused are counted.
//!
//! The case runs for hours: [`driver_seegmiller`] takes a configured [`Step`]
//! ([`Step::driver_seegmiller`] gives the case's own), and each model gets its own ignored
//! `#[test]` naming it (the first: [`sa_rans_on_han_s_wall_reattaches_where_measured`]). A 100-step laminar run of the case's
//! geometry stays in the trunk gate; the longer pipeline check runs in the heavy job.

mod common;

use common::cases::*;
use common::*;
use fieldcraft_lbm::backend::Backend;
use fieldcraft_lbm::statistics::Blocks;
use fieldcraft_lbm::{Flow, Grid, Limits, Obstacle};
use fieldcraft_model::{AbsorbingLayer, BoxFaces, Collision, FaceCondition, HrrSigma, Outflow, Procedure, Turbulence, WallModel};

/// What steps a run on a device: from the built CPU flow (already holding any carried-in field), a
/// device backend. A test binary that links one (the GPU crate's `step_study`) sets it; then every
/// [`Step::run`] steps the device, reads the wall shear from it each step and mirrors its fields (and ν̃)
/// into the CPU flow at each sample, so the statistics, the restart file and the reports are the same code.
#[allow(dead_code)]
pub(crate) type DeviceFactory = Box<dyn Fn(&Flow) -> Result<Box<dyn Backend>, String> + Send + Sync>;
#[allow(dead_code)]
pub(crate) static DEVICE: std::sync::OnceLock<DeviceFactory> = std::sync::OnceLock::new();

/// The device's density, velocity and ν̃ set in the CPU flow (equilibrium populations at that state).
fn mirror(flow: &mut Flow, device: &mut dyn Backend, for_save: bool) -> Result<(), String> {
    let f = device.fields().map_err(|e| format!("device fields: {e}"))?;
    flow.set_equilibrium(|c| match (f.density.get(c), f.velocity.get(c)) {
        (Some(r), Some(u)) if r.is_finite() && *r > 0.0 && u.iter().all(|v| v.is_finite()) => (*r, *u),
        _ => (1.0, [0.0; 3]),
    });
    let bad_rho = f.density.iter().filter(|v| !v.is_finite()).count();
    let bad_u = f.velocity.iter().filter(|u| !u.iter().all(|v| v.is_finite())).count();
    if let Some(nu) = device.nu_tilde().map_err(|e| format!("device nu-tilde: {e}"))? {
        let bad_nu = nu.iter().filter(|v| !v.is_finite()).count();
        if bad_rho + bad_u + bad_nu > 0 {
            println!("mirror: non-finite device cells: density {bad_rho}, velocity {bad_u}, nu-tilde {bad_nu} of {}", nu.len());
        }
        let clean: Vec<f64> = nu.iter().map(|v| if v.is_finite() { *v } else { 0.0 }).collect();
        flow.set_sa_nu_tilde(&clean).map_err(|e| format!("setting nu-tilde: {e}"))?;
    }
    // A flow that never stepped holds the derived fields (eddy viscosity, cell velocity, ...) unset: one CPU step fills
    // them so the restart file is whole (the step moves the state by one step, the device's run is unchanged).
    if for_save {
        flow.step().map_err(|e| format!("settling step of the mirrored flow: {e}"))?;
        let bad =
            (0..flow.grid().len()).filter(|c| flow.moments(*c).is_some_and(|(r, u)| !r.is_finite() || !u.iter().all(|v| v.is_finite()))).count();
        let bad_sa = flow.spalart_allmaras().map(|s| s.nu_tilde().iter().filter(|v| !v.is_finite()).count()).unwrap_or(0);
        println!(
            "mirror for save: the file holds the device's density, velocity and nu-tilde (equilibrium part only) plus one CPU settling step, so it is one step ahead of the device; non-finite cells: moments {bad}, nu-tilde {bad_sa}"
        );
    }
    Ok(())
}

/// The GPU's memory in use now (`nvidia-smi`), printed with a label; the number in MiB or None.
fn device_memory(label: &str) -> Option<f64> {
    let out =
        std::process::Command::new("nvidia-smi").args(["--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"]).output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    println!("device memory {label} (used, total MiB): {text}");
    text.split(',').next().and_then(|v| v.trim().parse().ok())
}

/// The step model's parameters (lengths in units of H, inlet speed 1).
#[derive(Debug, Clone)]
struct Step {
    re_h: f64,
    cells_per_h: usize,
    lattice_velocity: f64,
    upstream: f64,
    downstream: f64,
    /// Downstream over upstream channel height: the roof is at ER / (ER − 1) H.
    expansion_ratio: f64,
    /// The span in H; 0 for one cell.
    span: f64,
    /// The short domain of the phase-3 runs (an outlet from 20 H instead of 30 H).
    short: bool,
    collision: Collision,
    /// The hybrid collision's fixed σ (HRR only).
    sigma: Option<f64>,
    turbulence: Option<Turbulence>,
    /// The absorbing layer at the inlet and the outlet.
    absorbing: Option<AbsorbingLayer>,
    sample: u64,
    block: usize,
    startup: f64,
    max_steps: u64,
    /// A restart file: loaded when it exists, rewritten (atomically) every [`CHECKPOINT_STEPS`]
    /// steps, so an hours-long run survives a crash or a restart of the machine.
    checkpoint: Option<std::path::PathBuf>,
}

/// A wall strip: its body, its x range and whether it is on the floor (downstream) or the step's
/// top (upstream).
#[derive(Debug, Clone)]
struct Strip {
    body: String,
    x: (f64, f64),
    floor: bool,
}

/// The built grid's shape, measured from its cells (lengths in H).
#[derive(Debug, Clone, PartialEq)]
struct Geometry {
    step_height: f64,
    expansion_ratio: f64,
    upstream: f64,
    downstream: f64,
    roof: f64,
}

/// The time-mean carried shear of every strip: one [`Blocks`] (#828) per strip, fed one sample
/// (the strip's mean momentum exchange per area over `sample` steps) at a time.
struct Shear {
    strips: Vec<Blocks>,
}

impl Shear {
    fn new(strips: usize, block: usize) -> Shear {
        Shear { strips: (0..strips).map(|_| Blocks::new(block, 0).unwrap()).collect() }
    }

    /// Adds one sample of every strip; true when it completed a block.
    fn push(&mut self, sample: &[f64]) -> bool {
        assert_eq!(sample.len(), self.strips.len());
        let before = self.blocks();
        for (b, &t) in self.strips.iter_mut().zip(sample) {
            b.push(t).unwrap_or_else(|e| panic!("strip shear sample {t}: {e}"));
        }
        self.blocks() > before
    }

    fn blocks(&self) -> usize {
        self.strips.first().map_or(0, |b| b.means().len())
    }

    /// Each strip's mean over the last complete block.
    fn last_block(&self) -> Vec<f64> {
        self.strips.iter().map(|b| b.means().last().copied().unwrap_or(f64::NAN)).collect()
    }

    /// Each strip's time mean: the mean of its complete blocks.
    fn mean(&self) -> Vec<f64> {
        self.strips.iter().map(Blocks::mean).collect()
    }
}

/// What a run measured.
#[derive(Debug, Clone)]
struct Run {
    /// The mean shear profile's stations: (x, C_f) along the step's top, then the floor.
    cf: Vec<(f64, f64)>,
    /// The floor's C_p at the strip centres, from the first fluid cell row.
    cp: Vec<(f64, f64)>,
    /// x_r / H from the time-mean floor shear, or why there is none.
    x_r: Result<f64, NoReattachment>,
    /// A diagnostic beside it, not gated: where u = 0 meets the floor, extrapolated from the first
    /// rows above it in the final field ([`WallCrossing`]), or why there is none.
    crossing: Result<WallCrossing, String>,
    /// The reattachment of each completed block's mean floor shear, for the stop rule.
    reattachment: Blocks,
    /// Blocks whose mean floor shear was refused (left out of `reattachment`), and the last refusal.
    missing: (usize, Option<NoReattachment>),
    /// Blocks of a scale-resolving run with no crossing beyond [`PRIMARY_BEYOND`] (read at the
    /// start or at the last station, [`Found`]); counted, never refused.
    censored: usize,
    /// The readings of the whole averaging window, none of them from blocks.
    readings: Readings,
    /// The shear layer's resolved and modelled stress at [`STRESS_STATIONS`].
    stresses: Vec<Result<Stress, String>>,
    /// Re_θ, δ₉₉ (in H) and U_ref (in units of the inlet speed) at x = −4 H, at the end.
    inflow: (f64, f64, f64),
    steps: u64,
    /// Whether a stop rule settled the run: `rule` is `Some`.
    settled: bool,
    /// The rule that settled it, if one did.
    rule: Option<Settled>,
}

/// The rule that settled a run.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Settled {
    /// A steady RANS run ([`STEADY_BLOCKS`]): at `block`, the last blocks' x_r spread (max − min
    /// over |mean|) and floor shear spread (the largest strip's max − min over the floor's scale).
    Steady { block: usize, x_r: f64, shear: f64 },
    /// The `Blocks` acceptance on the reattachments, for scale-resolving runs.
    Blocks,
    /// The f32 floor ([`f32_floor`]) of a run on the device, over the last [`F32_BLOCKS`] blocks.
    F32Floor(F32Floor),
}

/// A steady run stops once the x_r of its last `STEADY_BLOCKS` completed blocks agree within
/// `STEADY_X_R` (max − min over |mean|) and every floor strip's block-mean shear agrees within
/// `STEADY_SHEAR` of the floor's scale (max − min over the largest |block-mean shear| of any floor
/// strip in those blocks) over the same blocks. The one scale keeps a strip whose mean shear is
/// near zero, the one straddling x_r, from failing on round-off alone. A converged RANS run's block means differ only by round-off,
/// which the `Blocks` acceptance (made for turbulent statistics) reads as correlated noise and never
/// accepts; the shear condition catches a slow drift elsewhere on the floor that x_r would miss.
const STEADY_BLOCKS: usize = 5;
const STEADY_X_R: f64 = 1e-5;
const STEADY_SHEAR: f64 = 1e-4;

/// The steady rule for a run on the device (f32 noise; ruled 2026-11-16, revised the same day). The floor shear is the f64
/// rule exactly: over the last [`STEADY_BLOCKS`] blocks every strip's block mean agrees within [`STEADY_SHEAR`] of the
/// largest strip. x_r passes by either branch: the f64 test (the last [`STEADY_BLOCKS`] blocks within [`STEADY_X_R`]
/// relative) or the noise-floor test over the last [`F32_BLOCKS`] blocks (the least-squares drift, slope × blocks, under
/// twice the scatter about the fitted line, and the scatter under [`F32_X_R_SCATTER`] of x_r). No trend term on the shear.
const F32_BLOCKS: usize = 10;
const F32_X_R_SCATTER: f64 = 5e-5;

/// What the rule measured, and how it passed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct F32Floor {
    /// x_r over the last [`STEADY_BLOCKS`] blocks: max − min over |mean| (the f64 test's number).
    x_r_spread: f64,
    /// x_r over the last [`F32_BLOCKS`] blocks: the fitted drift in H (slope × blocks), the scatter about the line in H
    /// and relative to x_r (NaN with fewer blocks).
    x_r_drift: f64,
    x_r_scatter: f64,
    x_r_scatter_rel: f64,
    /// The worst strip's change over the last [`STEADY_BLOCKS`] blocks, relative to the largest strip (the f64 test's number).
    shear_spread: f64,
    /// x_r passed by the f64 test, by the noise-floor test, and the shear passed.
    x_r_f64: bool,
    x_r_noise: bool,
    shear_ok: bool,
}

impl F32Floor {
    /// The run is settled: the shear passed and x_r passed by either branch.
    pub(crate) fn ok(&self) -> bool {
        self.shear_ok && (self.x_r_f64 || self.x_r_noise)
    }

    /// "settled" (x_r by the f64 test), "settled (f32 floor)" (by the noise-floor test only) or "capped".
    pub(crate) fn verdict(&self) -> &'static str {
        match (self.shear_ok, self.x_r_f64, self.x_r_noise) {
            (true, true, _) => "settled",
            (true, false, true) => "settled (f32 floor)",
            _ => "capped",
        }
    }

    /// The numbers, every time.
    pub(crate) fn describe(&self) -> String {
        format!(
            "x_r: last {STEADY_BLOCKS} blocks spread {:.1e} (f64 test < {STEADY_X_R:.0e}: {}); last {F32_BLOCKS} blocks drift {:+.2e} H, scatter {:.2e} H = {:.1e} of x_r (noise-floor test: drift < 2 x scatter and scatter < {F32_X_R_SCATTER:.0e}: {}); worst strip's change over the last {STEADY_BLOCKS} blocks {:.1e} of the largest (< {STEADY_SHEAR:.0e}: {})",
            self.x_r_spread, self.x_r_f64, self.x_r_drift, self.x_r_scatter, self.x_r_scatter_rel, self.x_r_noise, self.shear_spread, self.shear_ok
        )
    }
}

/// Slope × n and the standard deviation about the least-squares line of `values` (n − 2 degrees of freedom).
fn line_fit(values: &[f64]) -> Option<(f64, f64)> {
    let n = values.len();
    if n < 3 || values.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let nf = n as f64;
    let (mx, my) = ((nf - 1.0) / 2.0, values.iter().sum::<f64>() / nf);
    let sxx: f64 = (0..n).map(|i| (i as f64 - mx).powi(2)).sum();
    let sxy: f64 = values.iter().enumerate().map(|(i, v)| (i as f64 - mx) * (v - my)).sum();
    let slope = sxy / sxx;
    let ssr: f64 = values.iter().enumerate().map(|(i, v)| (v - my - slope * (i as f64 - mx)).powi(2)).sum();
    Some((slope * nf, (ssr / (nf - 2.0)).sqrt()))
}

/// The rule over `recent` (oldest first); none for fewer than [`STEADY_BLOCKS`] blocks.
fn f32_floor(recent: &[(f64, Vec<f64>)]) -> Option<F32Floor> {
    let (x_r_spread, shear_spread) = steady_spreads(recent)?;
    let noise = recent.len().checked_sub(F32_BLOCKS).and_then(|from| {
        let x_r: Vec<f64> = recent.get(from..)?.iter().map(|(x, _)| *x).collect();
        let (drift, scatter) = line_fit(&x_r)?;
        Some((drift, scatter, scatter / (x_r.iter().sum::<f64>() / x_r.len() as f64).abs()))
    });
    let (x_r_drift, x_r_scatter, x_r_scatter_rel) = noise.unwrap_or((f64::NAN, f64::NAN, f64::NAN));
    Some(F32Floor {
        x_r_spread,
        x_r_drift,
        x_r_scatter,
        x_r_scatter_rel,
        shear_spread,
        x_r_f64: x_r_spread < STEADY_X_R,
        x_r_noise: noise.is_some_and(|(d, sc, rel)| d.abs() < 2.0 * sc && rel < F32_X_R_SCATTER),
        shear_ok: shear_spread < STEADY_SHEAR,
    })
}

/// max − min of `values`.
fn range(values: &[f64]) -> f64 {
    let (lo, hi) = values.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
    hi - lo
}

/// `range` over `scale`; 0 when the values are all equal (even all zero, at a zero scale).
fn spread(values: &[f64], scale: f64) -> f64 {
    let r = range(values);
    if r == 0.0 { 0.0 } else { r / scale }
}

/// The steady rule over the last completed blocks, each (x_r, the floor strips' block-mean shear),
/// oldest first: the x_r and largest shear spread when the last `STEADY_BLOCKS` blocks pass, else
/// `None` (fewer blocks, a value that is not finite, or a spread over its tolerance).
fn steady(recent: &[(f64, Vec<f64>)]) -> Option<(f64, f64)> {
    steady_spreads(recent).filter(|(x_r, shear)| *x_r < STEADY_X_R && *shear < STEADY_SHEAR)
}

/// The two spreads the steady rule tests over the last `STEADY_BLOCKS` blocks (x_r relative, floor shear against the
/// floor's scale), whether or not they pass; none for fewer blocks or a value that is not finite.
fn steady_spreads(recent: &[(f64, Vec<f64>)]) -> Option<(f64, f64)> {
    let window = recent.get(recent.len().checked_sub(STEADY_BLOCKS)?..)?;
    let strips = window.first()?.1.len();
    if window.iter().any(|(x, t)| !x.is_finite() || t.len() != strips || t.iter().any(|v| !v.is_finite())) {
        return None;
    }
    let x_r: Vec<f64> = window.iter().map(|(x, _)| *x).collect();
    let x_r = spread(&x_r, (x_r.iter().sum::<f64>() / x_r.len() as f64).abs());
    let scale = window.iter().flat_map(|(_, t)| t).fold(0.0, |a: f64, v| a.max(v.abs()));
    let shear = (0..strips).map(|s| spread(&window.iter().filter_map(|(_, t)| t.get(s).copied()).collect::<Vec<f64>>(), scale)).fold(0.0, f64::max);
    Some((x_r, shear))
}

/// Lengths within this of each other are equal (grid coordinates are sums of cell sizes).
const LENGTH_TOLERANCE: f64 = 1e-9;

/// What sets a built flow's field before the run: it is consumed by the call, so what it carries
/// (a [`Field`]) is freed before the first step allocates the streaming targets.
type Init = Box<dyn FnOnce(&mut Flow) -> Result<(), String>>;

impl Step {
    fn spacing(&self) -> f64 {
        1.0 / self.cells_per_h as f64
    }

    fn span_cells(&self) -> usize {
        ((self.span * self.cells_per_h as f64).round() as usize).max(1)
    }

    /// The roof's height above the floor downstream, in H.
    fn roof(&self) -> f64 {
        self.expansion_ratio / (self.expansion_ratio - 1.0)
    }

    /// Whether `length` H is a whole number of cells.
    fn on_faces(&self, length: f64) -> bool {
        let n = length * self.cells_per_h as f64;
        (n - n.round()).abs() <= LENGTH_TOLERANCE * n.abs().max(1.0)
    }

    /// Refuses a configuration that is not the case's shape, with every reason.
    fn check(&self) -> Result<(), String> {
        let mut why = Vec::new();
        let positive = |v: f64| v.is_finite() && v > 0.0;
        if !positive(self.re_h) || !positive(self.lattice_velocity) || self.lattice_velocity > 0.3 {
            why.push(format!("Re_H {} and lattice velocity {} must be positive (≤ 0.3)", self.re_h, self.lattice_velocity));
        }
        if self.cells_per_h < 2 {
            why.push(format!("{} cells per H: at least 2", self.cells_per_h));
        }
        if !(self.expansion_ratio.is_finite() && self.expansion_ratio > 1.0) {
            why.push(format!("expansion ratio {}: must be above 1", self.expansion_ratio));
        } else if !self.on_faces(self.roof()) {
            why.push(format!("the roof at {} H is not on a cell face at {} cells per H", self.roof(), self.cells_per_h));
        }
        // The reference station x = −4 H needs inflow past it: at least two cells beyond.
        if !(self.upstream.is_finite() && self.upstream >= 4.0 + 2.0 * self.spacing() && self.on_faces(self.upstream)) {
            why.push(format!("an inflow of {} H must reach past x = −4 H by two cells and be whole cells", self.upstream));
        }
        let shortest = if self.short { 20.0 } else { 30.0 };
        if !(self.downstream.is_finite() && self.downstream >= shortest && self.on_faces(self.downstream)) {
            why.push(format!("an outlet {} H downstream must be at least {shortest} H and whole cells", self.downstream));
        }
        if !(self.span.is_finite() && self.span >= 0.0) {
            why.push(format!("span {}: must be ≥ 0", self.span));
        }
        if self.sample == 0 || self.block == 0 || !(self.startup.is_finite() && self.startup >= 0.0) || self.max_steps == 0 {
            why.push(format!(
                "sample {}, block {}, startup {} and max steps {} must be positive",
                self.sample, self.block, self.startup, self.max_steps
            ));
        }
        if why.is_empty() { Ok(()) } else { Err(why.join("; ")) }
    }

    /// The built grid's shape, measured from which cells are fluid: the step height from the first
    /// fluid row upstream and downstream, ER from the fluid heights, the lengths from the grid's
    /// extent. Refused when the solid cells are not a step (every column the same along the span,
    /// solid below its first fluid row and fluid above it, one first row on each side), the floor
    /// strips do not cover 0 … the outlet, or a strip has no force group.
    fn geometry(&self, flow: &Flow, strips: &[Strip]) -> Result<Geometry, String> {
        let g = flow.grid();
        let h = g.spacing;
        let [nx, ny, nz] = g.cells;
        // Han's on-node wall turns the first open cells beside a wall into boundary nodes (the
        // wall itself, half a cell inside the cells the builder left open): the shape is measured
        // from the open cells, nodes included, so it is the configured step either way.
        let mut node = vec![false; g.len()];
        for n in flow.wall_nodes().map_or(&[][..], |w| w.nodes()) {
            if let Some(c) = node.get_mut(n.cell as usize) {
                *c = true;
            }
        }
        let open = |c: usize| flow.is_fluid(c) || node.get(c).copied().unwrap_or(false);
        let mut first = Vec::with_capacity(nx);
        for i in 0..nx {
            let column = |j: usize| (0..nz).map(|k| open(g.index([i, j, k]))).collect::<Vec<bool>>();
            let c = column(0);
            if (1..ny).any(|j| column(j) != c) {
                return Err(format!("column {i} differs along the span"));
            }
            let k = c.iter().position(|&f| f).ok_or(format!("column {i} has no fluid"))?;
            if !c[k..].iter().all(|&f| f) {
                return Err(format!("column {i} has solid cells above its first fluid row {k}"));
            }
            first.push(k);
        }
        let x = |i: usize| g.origin[0] + (i as f64 + 0.5) * h;
        let (up, down): (Vec<usize>, Vec<usize>) = (0..nx).partition(|&i| x(i) < 0.0);
        let row = |cols: &[usize]| -> Result<usize, String> {
            let k = first[*cols.first().ok_or("no columns on one side of the step")?];
            cols.iter().all(|&i| first[i] == k).then_some(k).ok_or_else(|| format!("the first fluid row varies over columns {}…", cols[0]))
        };
        let (k_up, k_down) = (row(&up)?, row(&down)?);
        if k_up <= k_down {
            return Err(format!("no step: first fluid rows {k_up} upstream and {k_down} downstream"));
        }
        if let Some(s) = strips.iter().find(|s| body_groups(flow, &s.body).is_empty()) {
            return Err(format!("strip {} has no force group", s.body));
        }
        let floor: Vec<&Strip> = strips.iter().filter(|s| s.floor).collect();
        let joined = floor.windows(2).all(|w| (w[0].x.1 - w[1].x.0).abs() <= LENGTH_TOLERANCE);
        let (start, end) = (floor.first().map_or(f64::NAN, |s| s.x.0), floor.last().map_or(f64::NAN, |s| s.x.1));
        let outlet = g.origin[0] + nx as f64 * h;
        if !(joined && start.abs() <= LENGTH_TOLERANCE && (end - outlet).abs() <= LENGTH_TOLERANCE) {
            return Err(format!("the floor strips do not cover 0 … {outlet}: {start} … {end}"));
        }
        let floor_z = g.origin[2] + k_down as f64 * h;
        Ok(Geometry {
            step_height: (k_up - k_down) as f64 * h,
            expansion_ratio: (nz - k_down) as f64 / (nz - k_up) as f64,
            upstream: -g.origin[0],
            downstream: outlet,
            roof: g.origin[2] + nz as f64 * h - floor_z,
        })
    }

    fn strips(&self) -> Vec<Strip> {
        let h = self.spacing();
        let cells = |l: f64| (l / h).round() as i64;
        let mut s = Vec::new();
        // The step's top: strips H/2 long over −8 H … 0 (whole cells).
        let top = cells(0.5).max(1);
        let mut i = -cells(8.0f64.min(self.upstream));
        while i < 0 {
            let j = (i + top).min(0);
            s.push(Strip { body: format!("Top{}", s.len()), x: (i as f64 * h, j as f64 * h), floor: false });
            i = j;
        }
        let floor = cells(0.125).max(1);
        let (mut i, end) = (0, cells(self.downstream));
        while i < end {
            let j = (i + floor).min(end);
            s.push(Strip { body: format!("Floor{}", s.len()), x: (i as f64 * h, j as f64 * h), floor: true });
            i = j;
        }
        s
    }

    /// The flow, its strips and its measured geometry; panics (test code) when [`Step::check`]
    /// refuses the configuration or the built grid is not the configured step.
    fn build(&self) -> (Flow, Vec<Strip>, Geometry) {
        self.check().unwrap_or_else(|e| panic!("the step model: {e}"));
        let h = self.spacing();
        let width = self.span_cells() as f64 * h;
        let mut d = domain([-self.upstream, 0.0, -h], [self.downstream, width, self.roof()], h);
        d.faces = BoxFaces {
            x_min: FaceCondition::Velocity { velocity: [1.0, 0.0, 0.0], profile: Default::default() },
            x_max: FaceCondition::Pressure { pressure: 0.0, outflow: Outflow::Convective },
            y_min: FaceCondition::Periodic {},
            y_max: FaceCondition::Periodic {},
            z_min: FaceCondition::Wall {},
            z_max: FaceCondition::Wall {},
        };
        d.initial_velocity = [1.0, 0.0, 0.0];
        d.absorbing = self.absorbing;
        // `FIELDCRAFT_STEP_INLET_TABLE`: a profile file (`z/H u/U_ref nu_tilde/nu`, as `write_profile` writes it) as
        // the inlet's tabulated profile: the incoming boundary layer of a full domain on the short one.
        if let Some(path) = std::env::var_os("FIELDCRAFT_STEP_INLET_TABLE").filter(|_| self.short) {
            let rows = read_inlet_table(std::path::Path::new(&path)).unwrap_or_else(|e| panic!("the inlet table: {e}"));
            d.faces.x_min = FaceCondition::Velocity { velocity: [1.0, 0.0, 0.0], profile: fieldcraft_model::Profile::Table };
            d.inflow_table = Some(fieldcraft_model::InflowTable { axis: fieldcraft_model::fluid_zones::Axis::Z, rows });
        }
        let strips = self.strips();
        let (y0, y1) = (-h, width + h);
        let mut obstacles: Vec<Obstacle> = strips
            .iter()
            .map(|s| {
                let top = if s.floor { 0.0 } else { 1.0 };
                obstacle(&s.body, cuboid([s.x.0, y0, -2.0 * h], [s.x.1, y1, top]), &CUBOID_FACES)
            })
            .collect();
        let first = strips.first().map_or(0.0, |s| s.x.0);
        if first > -self.upstream {
            obstacles.push(obstacle("Step", cuboid([-self.upstream - h, y0, -2.0 * h], [first, y1, 1.0]), &CUBOID_FACES));
        }
        d.obstacles = obstacles.iter().map(|o| o.body.clone()).collect();
        let dt = self.lattice_velocity * h;
        let p = Procedure::Fluid {
            domain: "Flow".into(),
            end_time: Some(self.max_steps as f64 * dt),
            steps: None,
            time_step: Some(dt),
            frames: 2,
            magic: None,
            sigma: self.sigma.map(HrrSigma::Value),
            collision: Some(self.collision),
            turbulence: self.turbulence,
            steady: None,
        };
        let m = model(1.0, 1.0 / self.re_h, d, p);
        let mut limits = Limits { max_cell_updates: u64::MAX, max_steps: u64::MAX, ..Limits::default() };
        // `FIELDCRAFT_STEP_MAX_CELLS`: the cell cap (the solver's default when unset), for a spanwise run.
        if let Some(cap) = std::env::var("FIELDCRAFT_STEP_MAX_CELLS").ok().and_then(|v| v.parse().ok()) {
            limits.max_cells = cap;
        }
        let setup = fieldcraft_lbm::Setup::new(&m, None, &obstacles, &limits).unwrap();
        let flow = Flow::new(&setup, &obstacles).unwrap();
        let built = self.geometry(&flow, &strips).unwrap_or_else(|e| panic!("the step model: {e}"));
        let want = Geometry {
            step_height: 1.0,
            expansion_ratio: self.expansion_ratio,
            upstream: self.upstream,
            downstream: self.downstream,
            roof: self.roof(),
        };
        let close = |a: f64, b: f64| (a - b).abs() <= LENGTH_TOLERANCE * b.abs().max(1.0);
        assert!(
            close(built.step_height, want.step_height)
                && close(built.expansion_ratio, want.expansion_ratio)
                && close(built.upstream, want.upstream)
                && close(built.downstream, want.downstream)
                && close(built.roof, want.roof),
            "built {built:?}, configured {want:?}"
        );
        (flow, strips, built)
    }

    /// [`Step::try_run`], panicking (test code) on an error.
    fn run(&self) -> Run {
        self.try_run().unwrap_or_else(|e| panic!("the step run: {e}"))
    }

    /// The run; an error, with the step and the numbers, when the flow stops stepping or a wall
    /// shear sample is not finite.
    fn try_run(&self) -> Result<Run, String> {
        self.try_run_from(None)
    }

    /// [`Step::try_run`] starting from the field `init` sets (a settled field from another
    /// grid, [`transfer`]) instead of the uniform one; a restart file still takes precedence.
    fn try_run_from(&self, init: Option<Init>) -> Result<Run, String> {
        let (mut flow, strips, _) = self.build();
        if let Some(init) = init {
            init(&mut flow)?;
        }
        // What the flow holds, and the process's high-water mark, once built and set: the peak a
        // source field carried in would show here, before the first step allocates the targets.
        println!("memory after setup: flow arrays {:.2} GB, peak {}", flow.heap_bytes() as f64 / 1e9, peak_memory());
        let grid = *flow.grid();
        let h = self.spacing();
        let [nx, ny, nz] = grid.cells;
        // Every force group of each strip: diagonal links at a strip's ends are tied between its
        // top and its end faces and go to the end faces, so the strip's shear is its whole x force.
        let groups: Vec<Vec<usize>> = strips.iter().map(|s| body_groups(&flow, &s.body)).collect();
        let column = |x: f64| ((x + self.upstream) / h).floor().clamp(0.0, nx as f64 - 1.0) as usize;
        let row = |z: f64| ((z + h) / h).floor().clamp(0.0, nz as f64 - 1.0) as usize;
        // Span-averaged u along z and ρ at a column.
        let span_mean = |flow: &Flow, i: usize, k: usize| {
            let (mut rho, mut u) = (0.0, 0.0);
            for j in 0..ny {
                let (r, v) = flow.moments(grid.index([i, j, k])).unwrap_or((f64::NAN, [f64::NAN; 3]));
                rho += r;
                u += v[0];
            }
            (rho / ny as f64, u / ny as f64)
        };
        // Mid-channel above the step (z = 5 H under the case's 9 H roof).
        let middle = 0.5 * (1.0 + self.roof());
        let reference = |flow: &Flow| span_mean(flow, column(-4.0), row(middle));
        let flow_through = ((self.upstream + self.downstream) / h / self.lattice_velocity) as u64;
        let startup = (self.startup * flow_through as f64) as u64;
        let areas: Vec<f64> = strips.iter().map(|s| ((s.x.1 - s.x.0) / h).round() * ny as f64).collect();
        let mut shear = Shear::new(strips.len(), self.block);
        let mut blocks = Blocks::new(1, 0).unwrap();
        let mut missing = (0usize, None);
        let mut steps = 0;
        let mut interval = vec![0.0; strips.len()];
        let mut last_face_mean = 0.0;
        let floor_x: Vec<f64> = strips.iter().filter(|s| s.floor).map(|s| 0.5 * (s.x.0 + s.x.1)).collect();
        let floor_of = |v: &[f64]| strips.iter().zip(v).filter(|(s, _)| s.floor).map(|(_, &t)| t).collect::<Vec<f64>>();
        let mut rule = None;
        let censored = std::cell::Cell::new(0usize);
        // The floor's near-wall velocity over the whole window, for the readings that need no blocks.
        let mut window = Window::new(&flow, self);
        // A steady (RANS, no DDES) run stops by the steady rule; a scale-resolving one by `Blocks`.
        let steady_run = matches!(self.turbulence, Some(Turbulence::SpalartAllmaras { ddes: None, .. }));
        // The last completed blocks' (x_r, floor shear), for the steady rule; a refused block
        // empties it, so the window is consecutive blocks that each reattach.
        let mut recent: Vec<(f64, Vec<f64>)> = Vec::new();
        let clock = std::time::Instant::now();
        // Every sample so far, kept for the restart file: a resumed run replays them into fresh
        // statistics, so it ends where an unbroken run would.
        let mut samples: Vec<Vec<f64>> = Vec::new();
        // The running mean x_r at each completed block, with the step and the wall time, for the
        // timing report's settle point ([`settle_point`]).
        let mut history: Vec<(u64, f64, f64)> = Vec::new();
        // One sample into the statistics; the rule that settles the run, if one does. Each
        // completed block prints a line: its reattachment, the running mean and, when given, the
        // inlet's first four columns split into the face mean of ρ − 1 (the pressure level, whose
        // change since the last sample is what says whether it is still moving) and the largest
        // deviation from that mean (the grid-scale mode the absorbing layer damps).
        let take = |sample: &[f64],
                    steps: u64,
                    inlet: Option<[f64; 3]>,
                    shear: &mut Shear,
                    blocks: &mut Blocks,
                    missing: &mut (usize, Option<NoReattachment>),
                    recent: &mut Vec<(f64, Vec<f64>)>| {
            if !shear.push(sample) {
                return None;
            }
            let floor = floor_of(&shear.last_block());
            // A steady run reads the one crossing and refuses a profile with none or several; a
            // scale-resolving one reads the furthest downstream crossing beyond one step height
            // and never refuses a block, a block with no such crossing counting as censored.
            let last = if steady_run {
                reattachment(&floor_x, &floor)
            } else {
                furthest_crossing(&floor_x, &floor, PRIMARY_BEYOND).map(|(x, found)| {
                    if found != Found::Crossing {
                        censored.set(censored.get() + 1);
                    }
                    x
                })
            };
            match &last {
                Ok(x_r) => {
                    blocks.push(*x_r).unwrap();
                    recent.push((*x_r, floor));
                    if recent.len() > F32_BLOCKS.max(STEADY_BLOCKS) {
                        recent.remove(0);
                    }
                }
                Err(why) => {
                    *missing = (missing.0 + 1, Some(why.clone()));
                    recent.clear();
                }
            }
            println!(
                "step {steps}: block {} x_r {last:?}, mean {:.3} H over {} blocks ({} refused, {} censored), inlet ρ − 1: face mean {}, change {}, deviation {}, {:.0} s",
                shear.blocks(),
                blocks.mean(),
                blocks.means().len(),
                missing.0,
                censored.get(),
                inlet.map_or("(replayed)".into(), |v| format!("{:+.4e}", v[0])),
                inlet.map_or("(replayed)".into(), |v| format!("{:+.1e}", v[1])),
                inlet.map_or("(replayed)".into(), |v| format!("{:.2e}", v[2])),
                clock.elapsed().as_secs_f64()
            );
            if steady_run {
                if DEVICE.get().is_some() {
                    f32_floor(recent).filter(|f| f.ok()).map(Settled::F32Floor)
                } else {
                    steady(recent).map(|(x_r, spread)| Settled::Steady { block: shear.blocks(), x_r, shear: spread })
                }
            } else {
                blocks.acceptance().accepted.then_some(Settled::Blocks)
            }
        };
        if let Some(path) = self.checkpoint.as_ref().filter(|p| p.exists()) {
            let (at, kept, saved) = load_all(path, &mut flow);
            window.restore(&saved);
            steps = at;
            println!("resumed {} at step {steps} ({} samples)", path.display(), kept.len());
            let first = startup + self.sample;
            for (n, sample) in kept.iter().enumerate() {
                let now = take(sample, first + n as u64 * self.sample, None, &mut shear, &mut blocks, &mut missing, &mut recent);
                rule = rule.or(now);
            }
            samples = kept;
        }
        let mut device: Option<Box<dyn Backend>> = match DEVICE.get() {
            Some(make) => Some(make(&flow)?),
            None => None,
        };
        let mut peak_device = 0.0f64;
        if let Some(d) = device.as_ref() {
            println!("stepping on the device: {}", d.name());
            peak_device = peak_device.max(device_memory("at the start").unwrap_or(0.0));
        }
        // Where the wall time goes, in seconds: advance (the call), the per-step force readback (which waits for the
        // device, so holds the step's kernel time), the mirror (device fields to the host flow), the host's samples
        // (inlet read, blocks, window sums) and the restart saves.
        let mut prof = [0.0f64; 5];
        let lap = |t: &mut std::time::Instant, slot: &mut f64| {
            let now = std::time::Instant::now();
            *slot += now.duration_since(*t).as_secs_f64();
            *t = now;
        };
        let mut tick = std::time::Instant::now();
        while steps < self.max_steps && rule.is_none() {
            steps += 1;
            match device.as_mut() {
                Some(d) => d.advance(1).map_err(|e| format!("step {steps}: {e}"))?,
                None => flow.step().map_err(|e| format!("step {steps}: {e}"))?,
            }
            lap(&mut tick, &mut prof[0]);
            if steps <= startup {
                if let Some(path) = self.checkpoint.as_ref().filter(|_| steps.is_multiple_of(CHECKPOINT_STEPS)) {
                    if let Some(d) = device.as_mut() {
                        mirror(&mut flow, d.as_mut(), true)?;
                    }
                    save(path, &mut flow, steps, &samples, Some(&window));
                }
                continue;
            }
            let forces: Vec<[f64; 3]> = match device.as_mut() {
                Some(d) => d.forces().map_err(|e| format!("step {steps}: {e}"))?,
                None => flow.forces().to_vec(),
            };
            for (s, g) in groups.iter().enumerate() {
                interval[s] += g.iter().map(|&g| forces.get(g).map_or(f64::NAN, |f| f[0])).sum::<f64>() / areas[s];
            }
            lap(&mut tick, &mut prof[1]);
            if steps % self.sample != 0 {
                continue;
            }
            // One sample: each strip's mean shear over the interval.
            let sample: Vec<f64> = interval.iter().map(|t| t / self.sample as f64).collect();
            if let Some((strip, v)) = sample.iter().enumerate().find(|(_, v)| !v.is_finite()) {
                return Err(format!(
                    "the wall shear is not finite after {steps} steps: strip {strip} of {} reads {v}, last inlet face mean of ρ − 1 {last_face_mean:+.3e}",
                    sample.len()
                ));
            }
            interval.iter_mut().for_each(|v| *v = 0.0);
            tick = std::time::Instant::now();
            if let Some(d) = device.as_mut() {
                mirror(&mut flow, d.as_mut(), false)?;
                if steps.is_multiple_of(CHECKPOINT_STEPS) {
                    peak_device = peak_device.max(device_memory(&format!("at step {steps}")).unwrap_or(0.0));
                }
            }
            lap(&mut tick, &mut prof[2]);
            let rhos: Vec<f64> = (0..4.min(nx))
                .flat_map(|i| (0..nz).flat_map(move |k| (0..ny).map(move |j| [i, j, k])))
                .filter_map(|c| flow.moments(grid.index(c)).map(|m| m.0 - 1.0))
                .collect();
            let face_mean = rhos.iter().sum::<f64>() / rhos.len().max(1) as f64;
            let deviation = rhos.iter().map(|r| (r - face_mean).abs()).fold(0.0, f64::max);
            let inlet = [face_mean, face_mean - last_face_mean, deviation];
            last_face_mean = face_mean;
            rule = take(&sample, steps, Some(inlet), &mut shear, &mut blocks, &mut missing, &mut recent);
            if blocks.means().len() > history.len() {
                history.push((steps, clock.elapsed().as_secs_f64(), blocks.mean()));
            }
            samples.push(sample);
            window.add(&flow);
            lap(&mut tick, &mut prof[3]);
            // Sample boundaries carry no partial interval, so the file holds the whole state.
            if let Some(path) = self.checkpoint.as_ref().filter(|_| steps.is_multiple_of(CHECKPOINT_STEPS)) {
                save(path, &mut flow, steps, &samples, Some(&window));
            }
            lap(&mut tick, &mut prof[4]);
        }
        {
            let total: f64 = prof.iter().sum();
            let n = steps.max(1) as f64;
            println!(
                "profile over {steps} steps ({total:.1} s in the loop, {:.1} ms/step): advance call {:.1} s ({:.1} ms/step), per-step forces readback incl. the wait for the device {:.1} s ({:.1} ms/step), mirror {:.1} s, host sample work (inlet read, blocks) and window sums {:.1} s, restart saves {:.1} s",
                1e3 * total / n,
                prof[0],
                1e3 * prof[0] / n,
                prof[1],
                1e3 * prof[1] / n,
                prof[2],
                prof[3],
                prof[4]
            );
        }
        // The end of the run is saved too, so a run that stops between two checkpoints leaves its field.
        if let Some(d) = device.as_mut() {
            mirror(&mut flow, d.as_mut(), true)?;
            peak_device = peak_device.max(device_memory("at the end").unwrap_or(0.0));
            println!("device memory peak over the run's readings: {peak_device:.0} MiB");
        }
        if let Some(path) = self.checkpoint.as_ref() {
            save(path, &mut flow, steps, &samples, Some(&window));
        }
        println!(
            "{} fluid cells, {steps} steps: {:.1} M cell updates/s",
            flow.fluid_cells(),
            flow.fluid_cells() as f64 * steps as f64 / clock.elapsed().as_secs_f64() / 1e6
        );
        println!(
            "timing: grid {:?} = {} cells ({} fluid), {} cells per H, span {} H, wall time {:.0} s for {steps} steps, flow arrays {:.2} GB, peak memory {}",
            grid.cells,
            grid.cells.iter().product::<usize>(),
            flow.fluid_cells(),
            self.cells_per_h,
            self.span,
            clock.elapsed().as_secs_f64(),
            flow.heap_bytes() as f64 / 1e9,
            peak_memory()
        );
        match settle_point(&history) {
            Some((at, secs, x_r, before)) => println!(
                "settle point: running x_r {x_r:.3} H changes by under 0.5 % over the last fifth (a fifth earlier {before:.3} H) at step {at}, wall time {secs:.0} s"
            ),
            None => println!("settle point: the running x_r never changes by under 0.5 % over a fifth of the run"),
        }
        if let Some(x_r) = wall_reading(&flow, self) {
            println!("x_r from the flow field, rows 2-4: {x_r:.3} H");
        }
        // `FIELDCRAFT_STEP_PROFILE=<file>`: the incoming layer at x = −4 H, for a shorter domain.
        if let Some(path) = std::env::var_os("FIELDCRAFT_STEP_PROFILE") {
            write_profile(std::path::Path::new(&path), &flow, self)?;
            println!("inlet profile at x = −4 H written to {}", std::path::Path::new(&path).display());
        }
        // The mean profiles, formed with U_ref and ρ at the reference station.
        let (rho_ref, u_ref) = reference(&flow);
        let q = 0.5 * rho_ref * u_ref * u_ref;
        let mean = shear.mean();
        let x_r = if steady_run {
            reattachment(&floor_x, &floor_of(&mean))
        } else {
            furthest_crossing(&floor_x, &floor_of(&mean), PRIMARY_BEYOND).map(|(x, _)| x)
        };
        let readings = Readings::of(&floor_x, &floor_of(&mean), &window.floor);
        for line in window.floor.describe() {
            println!("{line}");
        }
        if rule.is_none() {
            if let Some(f) = f32_floor(&recent) {
                println!("{} (the run reached its step cap): {}", f.verdict(), f.describe());
            }
            match steady_spreads(&recent) {
                Some((x_r, shear)) => println!(
                    "capped: x_r spread {x_r:.1e} over the last {STEADY_BLOCKS} blocks (rule < {STEADY_X_R:.0e}: {}), floor shear spread {shear:.1e} (rule < {STEADY_SHEAR:.0e}: {})",
                    if x_r < STEADY_X_R { "met" } else { "not met" },
                    if shear < STEADY_SHEAR { "met" } else { "not met" }
                ),
                None => println!(
                    "capped: the steady rule's spreads are not available (fewer than {STEADY_BLOCKS} whole blocks, or a block with a non-finite value)"
                ),
            }
        }
        let partial = if rule.is_some() { "" } else { "partial (the run stopped unsettled, at its step cap): " };
        println!("{partial}{readings}");
        let stresses = window.layer.stresses();
        for s in &stresses {
            match s {
                Ok(s) => println!("{partial}resolved against modelled shear stress, {s}"),
                Err(why) => println!("{partial}resolved against modelled shear stress: none ({why})"),
            }
        }
        let cf: Vec<(f64, f64)> = strips.iter().zip(&mean).map(|(s, t)| (0.5 * (s.x.0 + s.x.1), t / q)).collect();
        let p_ref = rho_ref / 3.0;
        let cp: Vec<(f64, f64)> = floor_x.iter().map(|&x| (x, (span_mean(&flow, column(x), 1).0 / 3.0 - p_ref) / q)).collect();
        // The incoming layer on the step's top at x = −4 H, up to mid-channel.
        let i = column(-4.0);
        let u: Vec<f64> = (row(1.0) + 1..=row(middle)).map(|k| span_mean(&flow, i, k).1).collect();
        let theta = momentum_thickness(&u, h, u_ref);
        let delta = u.iter().position(|&v| v >= 0.99 * u_ref).map_or(f64::NAN, |k| (k as f64 + 0.5) * h);
        // U_ref in units of the inlet speed (the lattice speeds are scaled by U_lat).
        let u_ref_inlet = u_ref / self.lattice_velocity;
        let crossing = wall_crossing(&flow, self, 0);
        Ok(Run {
            cf,
            cp,
            x_r,
            crossing,
            reattachment: blocks,
            missing,
            censored: censored.get(),
            readings,
            stresses,
            inflow: (u_ref_inlet * theta * self.re_h, delta, u_ref_inlet),
            steps,
            settled: rule.is_some(),
            rule,
        })
    }
}

/// Steps between two restart files of a run with [`Step::checkpoint`]: a multiple of the case's
/// sample interval, so a file is written only where no partial sample is pending.
const CHECKPOINT_STEPS: u64 = 2000;

/// A restart file for [`Step::run`]: the flow's own state (`Backend::checkpoint`), then the step
/// count and every sample (each strip's mean shear), little-endian.
struct Restart {
    read: Option<std::io::BufReader<std::fs::File>>,
    write: Option<std::io::BufWriter<std::fs::File>>,
}

impl fieldcraft_lbm::checkpoint::Archive for Restart {
    fn loading(&self) -> bool {
        self.read.is_some()
    }
    fn bytes(&mut self, value: &mut [u8]) -> fieldcraft_lbm::Result<()> {
        use std::io::{Read, Write};
        let r = match (&mut self.read, &mut self.write) {
            (Some(r), _) => r.read_exact(value),
            (_, Some(w)) => w.write_all(value),
            _ => Ok(()),
        };
        r.map_err(|e| fieldcraft_lbm::Error::Invalid(format!("step restart file: {e}")))
    }
}

impl Restart {
    /// Read `v` back when loading, write it when saving.
    fn u64(&mut self, v: u64) -> u64 {
        let mut b = v.to_le_bytes();
        fieldcraft_lbm::checkpoint::Archive::bytes(self, &mut b).unwrap();
        u64::from_le_bytes(b)
    }

    /// The next value of a file being read, or none at its end.
    fn try_u64(&mut self) -> Option<u64> {
        let mut b = [0u8; 8];
        fieldcraft_lbm::checkpoint::Archive::bytes(self, &mut b).ok()?;
        Some(u64::from_le_bytes(b))
    }
}

/// Write the restart file, the window's sums (near-wall rows, shear layer) after the samples so a resumed run's window
/// is whole: to `path` with `.tmp` added, then renamed over `path`.
fn save(path: &std::path::Path, flow: &mut Flow, steps: u64, samples: &[Vec<f64>], window: Option<&Window>) {
    use std::io::Write;
    let rows = window.map(Window::state).unwrap_or_default();
    let tmp = path.with_extension("tmp");
    let mut a = Restart { read: None, write: Some(std::io::BufWriter::new(std::fs::File::create(&tmp).unwrap())) };
    flow.checkpoint(&mut a).unwrap();
    a.u64(steps);
    a.u64(samples.len() as u64);
    for row in samples {
        a.u64(row.len() as u64);
        for v in row {
            a.u64(v.to_bits());
        }
    }
    a.u64(rows.len() as u64);
    for v in &rows {
        a.u64(v.to_bits());
    }
    if let Some(w) = a.write.as_mut() {
        w.flush().unwrap();
    }
    drop(a);
    std::fs::rename(&tmp, path).unwrap();
}

/// Read a restart file into `flow`: the step count and the samples.
fn load(path: &std::path::Path, flow: &mut Flow) -> (u64, Vec<Vec<f64>>) {
    let (steps, samples, _) = load_all(path, flow);
    (steps, samples)
}

/// [`load`] with the near-wall rows' sums (none in a file written before they were kept).
fn load_all(path: &std::path::Path, flow: &mut Flow) -> (u64, Vec<Vec<f64>>, Vec<f64>) {
    let mut a = Restart { read: Some(std::io::BufReader::new(std::fs::File::open(path).unwrap())), write: None };
    flow.checkpoint(&mut a).unwrap();
    let steps = a.u64(0);
    let samples = (0..a.u64(0)).map(|_| (0..a.u64(0)).map(|_| f64::from_bits(a.u64(0))).collect()).collect();
    let rows = a.try_u64().map_or_else(Vec::new, |n| (0..n).map(|_| f64::from_bits(a.u64(0))).collect());
    (steps, samples, rows)
}

/// Reattachments are read at or beyond this x (in H): the crossings nearer the step (about 0.2 and
/// 1.3 H) belong to the corner bubble, the primary reattachment is the furthest downstream one.
const PRIMARY_BEYOND: f64 = 1.0;
/// The two readings of the whole averaging window differ by more than this (H): both are reported
/// and the difference is said.
const READINGS_AGREE: f64 = 0.2;
/// Cell rows above the floor's wall the near-wall velocity is accumulated in (rows 1 to 4).
const FLOOR_ROWS: usize = 4;

/// The near-wall velocity of the floor behind the step over the whole averaging window: for each of
/// the first [`FLOOR_ROWS`] rows above the wall and each column, the running sum of the span-mean
/// u and of the fraction of the span with u > 0, one sample (the run's `sample` interval) at a
/// time. Only these `FLOOR_ROWS × nx` triples are kept, not whole fields, so the memory is nothing
/// next to the flow's.
struct FloorRows {
    /// Cell row of row 1 (Han's wall: the node row is row 1).
    first: usize,
    /// The columns' centres, in H.
    x: Vec<f64>,
    /// Row n's height above the wall, in H.
    y: [f64; FLOOR_ROWS],
    /// `[Σ span-mean u, Σ forward fraction, samples]` of row n (0-based), column i, at `n * nx + i`.
    sums: Vec<[f64; 3]>,
}

impl FloorRows {
    fn new(flow: &Flow, step: &Step) -> FloorRows {
        let h = step.spacing();
        let nx = flow.grid().cells[0];
        let wall = if flow.on_node_bounce() { 0.5 * h } else { 0.0 };
        let first = if flow.on_node_bounce() { 2 } else { 1 };
        FloorRows {
            first,
            x: (0..nx).map(|i| -step.upstream + (i as f64 + 0.5) * h).collect(),
            y: std::array::from_fn(|n| ((first + n) as f64 - 0.5) * h - wall),
            sums: vec![[0.0; 3]; FLOOR_ROWS * nx],
        }
    }

    /// One sample of the flow's current field.
    fn add(&mut self, flow: &Flow) {
        let g = *flow.grid();
        let [nx, ny, nz] = g.cells;
        for n in 0..FLOOR_ROWS.min(nz.saturating_sub(self.first)) {
            for i in 0..nx {
                let u: Vec<f64> = (0..ny).filter_map(|j| flow.moments(g.index([i, j, self.first + n])).map(|m| m.1[0])).collect();
                if let Some(s) = self.sums.get_mut(n * nx + i).filter(|_| !u.is_empty()) {
                    s[0] += u.iter().sum::<f64>() / u.len() as f64;
                    s[1] += u.iter().filter(|&&v| v > 0.0).count() as f64 / u.len() as f64;
                    s[2] += 1.0;
                }
            }
        }
    }

    /// The sums, for the restart file.
    fn state(&self) -> Vec<f64> {
        self.sums.iter().flatten().copied().collect()
    }

    /// Takes the sums of a restart file back; one of another shape (or none) is ignored, so the
    /// window then starts at the restart.
    fn restore(&mut self, state: &[f64]) {
        if state.len() == 3 * self.sums.len() {
            for (s, v) in self.sums.iter_mut().flatten().zip(state) {
                *s = *v;
            }
        }
    }

    /// The columns behind the step (x > 0) that took samples and their centres (the column against the step's face holds
    /// the corner's solid cells and has none).
    fn behind(&self) -> (Vec<usize>, Vec<f64>) {
        (0..self.x.len()).filter(|&i| self.x[i] > 0.0 && self.sums.get(i).is_some_and(|s| s[2] > 0.0)).map(|i| (i, self.x[i])).unzip()
    }

    /// Row `n`'s time-mean span-mean u, or its forward fraction, at the columns `at` (NaN where no
    /// sample was taken).
    fn column(&self, n: usize, at: &[usize], what: usize) -> Vec<f64> {
        let nx = self.x.len();
        at.iter().map(|&i| self.sums.get(n * nx + i).map_or(f64::NAN, |s| if s[2] > 0.0 { s[what] / s[2] } else { f64::NAN })).collect()
    }

    /// What the rows hold, one line per row: the columns behind the step with no sample, with a non-finite sum, and
    /// the first such station.
    pub(crate) fn describe(&self) -> Vec<String> {
        let nx = self.x.len();
        (0..FLOOR_ROWS)
            .map(|n| {
                let cols: Vec<(usize, &[f64; 3])> =
                    (0..nx).filter(|&i| self.x[i] > 0.0).filter_map(|i| self.sums.get(n * nx + i).map(|s| (i, s))).collect();
                let empty = cols.iter().filter(|(_, s)| s[2] <= 0.0).count();
                let bad = cols.iter().filter(|(_, s)| !(s[0].is_finite() && s[1].is_finite())).count();
                let first = cols.iter().find(|(_, s)| s[2] <= 0.0 || !(s[0].is_finite() && s[1].is_finite())).map(|(i, _)| self.x[*i]);
                format!(
                    "floor row {}: {} columns behind the step, {empty} without a sample, {bad} with a non-finite sum, first at x = {first:?} H",
                    n + 1,
                    cols.len()
                )
            })
            .collect()
    }

    /// Samples taken (the most of any column of row 1).
    fn samples(&self) -> u64 {
        self.sums.iter().map(|s| s[2]).fold(0.0, f64::max) as u64
    }
}

/// What the whole averaging window says about the reattachment, from no blocks: the time-mean floor
/// shear's furthest downstream crossing, the same crossing of the time-mean u in rows 2 to 4 taken
/// to the wall, and the station where the near-wall flow is forward half the time. Blocks give
/// only the error bar.
#[derive(Debug, Clone)]
struct Readings {
    /// The mean floor shear (the gated reading).
    shear: Result<(f64, Found), NoReattachment>,
    /// The u = 0 crossings of the time-mean u in rows 2, 3 and 4, least squares to y = 0.
    rows_2_4: Result<f64, String>,
    /// The station where rows 1 to 4 are forward half of the time (row 1, the height of the mean shear's wall node, is the
    /// one compared with it).
    forward: Vec<Result<(f64, Found), NoReattachment>>,
    /// The forward-half-the-time crossings of rows 2, 3 and 4, least squares to y = 0 (the rows reader's extrapolation).
    forward_2_4: Result<f64, String>,
    /// The samples the near-wall readings average.
    samples: u64,
}

impl Readings {
    fn of(floor_x: &[f64], mean_shear: &[f64], rows: &FloorRows) -> Readings {
        let (at, xs) = rows.behind();
        let forward: Vec<Result<(f64, Found), NoReattachment>> = (0..FLOOR_ROWS)
            .map(|n| {
                let half: Vec<f64> = rows.column(n, &at, 1).iter().map(|f| f - 0.5).collect();
                furthest_crossing(&xs, &half, PRIMARY_BEYOND)
            })
            .collect();
        let forward_points: Result<Vec<(f64, f64)>, String> = (1..FLOOR_ROWS)
            .map(|n| match forward.get(n) {
                Some(Ok((x, Found::Crossing))) => Ok((rows.y[n], *x)),
                Some(Ok((_, found))) => Err(format!("row {}: {found:?}", n + 1)),
                Some(Err(why)) => Err(format!("row {}: {why:?}", n + 1)),
                None => Err(format!("row {}: missing", n + 1)),
            })
            .collect();
        let crossings: Result<Vec<(f64, f64)>, String> = (1..FLOOR_ROWS)
            .map(|n| match furthest_crossing(&xs, &rows.column(n, &at, 0), PRIMARY_BEYOND) {
                Ok((x, Found::Crossing)) => Ok((rows.y[n], x)),
                Ok((_, found)) => Err(format!("row {}: {found:?}", n + 1)),
                Err(why) => Err(format!("row {}: {why:?}", n + 1)),
            })
            .collect();
        Readings {
            shear: furthest_crossing(floor_x, mean_shear, PRIMARY_BEYOND),
            rows_2_4: crossings.map(|c| intercept(&c, 1)),
            forward,
            forward_2_4: forward_points.map(|c| intercept(&c, 1)),
            samples: rows.samples(),
        }
    }

    /// The gap between the mean-shear and the forward-flow (row 1: the mean shear sits at that height) readings when it exceeds
    /// [`READINGS_AGREE`].
    fn disagreement(&self) -> Option<f64> {
        let (Ok((a, _)), Some(Ok((b, _)))) = (&self.shear, self.forward.first()) else { return None };
        ((a - b).abs() > READINGS_AGREE).then_some((a - b).abs())
    }
}

impl std::fmt::Display for Readings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let show = |r: &Result<(f64, Found), NoReattachment>| match r {
            Ok((x, Found::Crossing)) => format!("{x:.3} H"),
            Ok((x, found)) => format!("{x:.3} H ({found:?})"),
            Err(why) => format!("none ({why:?})"),
        };
        write!(f, "x_r readings of the whole window ({} samples of the near-wall rows): mean floor shear {}", self.samples, show(&self.shear))?;
        match &self.rows_2_4 {
            Ok(x) => write!(f, "; rows 2-4 of the time-mean u at the wall {x:.3} H")?,
            Err(why) => write!(f, "; rows 2-4 of the time-mean u none ({why})")?,
        }
        let rows: Vec<String> = self.forward.iter().enumerate().map(|(n, r)| format!("row {} {}", n + 1, show(r))).collect();
        write!(f, "; forward half the time: {}", rows.join(", "))?;
        match &self.forward_2_4 {
            Ok(x) => write!(f, "; rows 2-4 of the forward fraction at the wall {x:.3} H")?,
            Err(why) => write!(f, "; rows 2-4 of the forward fraction none ({why})")?,
        }
        if let Some(gap) = self.disagreement() {
            write!(f, "; THE MEAN-SHEAR AND FORWARD-FLOW READINGS DIFFER BY {gap:.3} H (over {READINGS_AGREE} H), both are reported")?;
        }
        Ok(())
    }
}

/// The stations behind the step (x, in H) where the resolved and the modelled shear stress are
/// compared in the shear layer.
const STRESS_STATIONS: [f64; 3] = [1.0, 3.0, 5.0];

/// The shear layer's stress at [`STRESS_STATIONS`] over the whole averaging window: for each
/// station and cell row, the running sums of the span-mean u, w, u w and the eddy viscosity ν_t,
/// one sample at a time. The resolved stress is −(⟨u w⟩ − ⟨u⟩⟨w⟩), the modelled one ⟨ν_t⟩ d⟨u⟩/dz
/// (the molecular part is negligible at these Reynolds numbers and is left out of both).
struct ShearLayer {
    /// The column of each station.
    columns: [usize; STRESS_STATIONS.len()],
    /// Cell rows, the cell size in H, and the lattice speed the stress is scaled by.
    nz: usize,
    spacing: f64,
    lattice_velocity: f64,
    /// The fluid's molecular viscosity (lattice units), the argument of the model's ν_t.
    nu: f64,
    /// `[Σ u, Σ w, Σ u w, Σ ν_t, samples]` of station s, row k, at `s * nz + k`.
    sums: Vec<[f64; 5]>,
}

/// The stress of one station: at the height of the largest total stress, and over the layer.
#[derive(Debug, Clone, PartialEq)]
struct Stress {
    x: f64,
    /// The height (H) where the total stress |τ_res + τ_mod| is largest.
    z: f64,
    /// The resolved and the modelled stress there, in units of the inlet speed squared.
    resolved: f64,
    modelled: f64,
    /// Resolved over total at that height, and ∫|τ_res| over ∫(|τ_res| + |τ_mod|) over the rows.
    peak_fraction: f64,
    layer_fraction: f64,
}

impl ShearLayer {
    fn new(flow: &Flow, step: &Step) -> ShearLayer {
        let g = flow.grid();
        let h = step.spacing();
        let column = |x: f64| ((x + step.upstream) / h).floor().clamp(0.0, g.cells[0] as f64 - 1.0) as usize;
        let (omega, _) = flow.omegas();
        ShearLayer {
            columns: STRESS_STATIONS.map(column),
            nz: g.cells[2],
            spacing: h,
            lattice_velocity: step.lattice_velocity,
            nu: (1.0 / omega - 0.5) / 3.0,
            sums: vec![[0.0; 5]; STRESS_STATIONS.len() * g.cells[2]],
        }
    }

    /// One sample of the flow's current field.
    fn add(&mut self, flow: &Flow) {
        let g = *flow.grid();
        let ny = g.cells[1];
        let sa = flow.spalart_allmaras();
        for (s, &i) in self.columns.iter().enumerate() {
            for k in 0..self.nz {
                let cells: Vec<(f64, f64, f64)> = (0..ny)
                    .filter_map(|j| {
                        let c = g.index([i, j, k]);
                        flow.moments(c).map(|m| (m.1[0], m.1[2], sa.map_or(0.0, |f| f.nu_t(c, self.nu))))
                    })
                    .collect();
                if let Some(a) = self.sums.get_mut(s * self.nz + k).filter(|_| !cells.is_empty()) {
                    let n = cells.len() as f64;
                    a[0] += cells.iter().map(|c| c.0).sum::<f64>() / n;
                    a[1] += cells.iter().map(|c| c.1).sum::<f64>() / n;
                    a[2] += cells.iter().map(|c| c.0 * c.1).sum::<f64>() / n;
                    a[3] += cells.iter().map(|c| c.2).sum::<f64>() / n;
                    a[4] += 1.0;
                }
            }
        }
    }

    fn state(&self) -> Vec<f64> {
        self.sums.iter().flatten().copied().collect()
    }

    fn restore(&mut self, state: &[f64]) {
        if state.len() == 5 * self.sums.len() {
            for (s, v) in self.sums.iter_mut().flatten().zip(state) {
                *s = *v;
            }
        }
    }

    /// Each station's stress, or why there is none (no samples, or no fluid row with both
    /// neighbours to difference).
    fn stresses(&self) -> Vec<Result<Stress, String>> {
        (0..self.columns.len())
            .map(|s| {
                let rows = &self.sums[s * self.nz..(s + 1) * self.nz];
                let mean = |k: usize, what: usize| rows.get(k).filter(|r| r[4] > 0.0).map(|r| r[what] / r[4]);
                let scale = self.lattice_velocity * self.lattice_velocity;
                // (height in H, resolved, modelled) of every row with fluid in it and in both neighbours.
                let layer: Vec<(f64, f64, f64)> = (1..self.nz.saturating_sub(1))
                    .filter_map(|k| {
                        let (up, down) = (mean(k + 1, 0)?, mean(k - 1, 0)?);
                        let resolved = -(mean(k, 2)? - mean(k, 0)? * mean(k, 1)?);
                        let modelled = mean(k, 3)? * (up - down) / 2.0;
                        Some(((k as f64 - 0.5) * self.spacing, resolved / scale, modelled / scale))
                    })
                    .collect();
                let peak = layer
                    .iter()
                    .copied()
                    .max_by(|a, b| (a.1 + a.2).abs().total_cmp(&(b.1 + b.2).abs()))
                    .ok_or_else(|| format!("x = {} H: no sample", STRESS_STATIONS[s]))?;
                let (res, tot): (f64, f64) = layer.iter().fold((0.0, 0.0), |(r, t), l| (r + l.1.abs(), t + l.1.abs() + l.2.abs()));
                Ok(Stress {
                    x: STRESS_STATIONS[s],
                    z: peak.0,
                    resolved: peak.1,
                    modelled: peak.2,
                    peak_fraction: peak.1 / (peak.1 + peak.2),
                    layer_fraction: if tot > 0.0 { res / tot } else { f64::NAN },
                })
            })
            .collect()
    }
}

impl std::fmt::Display for Stress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "x = {} H: at the stress peak z = {:.2} H, resolved {:.3e} and modelled {:.3e} U²: resolved fraction {:.2} (whole layer {:.2})",
            self.x, self.z, self.resolved, self.modelled, self.peak_fraction, self.layer_fraction
        )
    }
}

/// Everything a run averages over its window apart from the floor shear's blocks.
struct Window {
    floor: FloorRows,
    layer: ShearLayer,
}

impl Window {
    fn new(flow: &Flow, step: &Step) -> Window {
        Window { floor: FloorRows::new(flow, step), layer: ShearLayer::new(flow, step) }
    }

    fn add(&mut self, flow: &Flow) {
        self.floor.add(flow);
        self.layer.add(flow);
    }

    /// The floor rows' sums, then the shear layer's, for the restart file.
    fn state(&self) -> Vec<f64> {
        let mut v = self.floor.state();
        v.extend(self.layer.state());
        v
    }

    fn restore(&mut self, state: &[f64]) {
        let split = 3 * self.floor.sums.len();
        if state.len() >= split {
            self.floor.restore(&state[..split]);
            self.layer.restore(&state[split..]);
        }
    }
}

/// Where u = 0 meets the floor behind the step, from the final field's first four rows above the
/// wall: a diagnostic reported beside the gated x_r, never gated.
///
/// The gated x_r is where the floor's carried shear changes sign. On Han's on-node wall that shear
/// comes from the node's interior neighbour one cell up, so its sign change is where u = 0 crosses
/// the row at y = Δ, upstream of where u = 0 meets the wall: near reattachment the u = 0 line
/// leans downstream towards the wall (about 8 H along per H down on the 16/H SA run). This reads
/// each row's last negative → non-negative change of u (linear between cell centres) and
/// extrapolates those crossings to y = 0 four ways; their spread is the sensitivity band. The
/// field is the last step's, which is the mean for a settled RANS run, not for a scale-resolving
/// one.
#[derive(Debug, Clone)]
struct WallCrossing {
    /// (y, x) of each row's crossing, in H, y above the wall (Han's wall: the node row's centre).
    rows: Vec<(f64, f64)>,
    /// x at y = 0: linear through rows 1–2, 1–3 and 1–4 (least squares), quadratic through 1–4.
    fits: [f64; 4],
}

impl WallCrossing {
    /// The band the fits span.
    fn range(&self) -> (f64, f64) {
        self.fits.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| (a.min(x), b.max(x)))
    }
}

impl std::fmt::Display for WallCrossing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let rows: Vec<String> = self.rows.iter().map(|(y, x)| format!("y {y:.4}: {x:.3}")).collect();
        let [l2, l3, l4, q4] = self.fits;
        let (a, b) = self.range();
        write!(
            f,
            "u = 0 at the floor (diagnostic, not gated): rows {}; at y = 0: linear 1–2 {l2:.3}, 1–3 {l3:.3}, 1–4 {l4:.3}, quadratic 1–4 {q4:.3}; band {a:.3} … {b:.3} H",
            rows.join(", ")
        )
    }
}

/// The value at 0 of the least-squares polynomial of `degree` (1 or 2) through `(y, x)`.
fn intercept(points: &[(f64, f64)], degree: usize) -> f64 {
    let n = degree + 1;
    // Normal equations A c = r, row j: A_jk = Σ y^(j+k) and r_j = Σ x y^j (the last column),
    // solved by Gauss–Jordan elimination with partial pivoting.
    let mut a: Vec<Vec<f64>> = (0..n)
        .map(|j| {
            let mut row: Vec<f64> = (0..n).map(|k| points.iter().map(|&(y, _)| y.powi((j + k) as i32)).sum()).collect();
            row.push(points.iter().map(|&(y, x)| x * y.powi(j as i32)).sum());
            row
        })
        .collect();
    for c in 0..n {
        let pivot = (c..n).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs())).unwrap_or(c);
        a.swap(c, pivot);
        let p = a[c].clone();
        if p[c] == 0.0 {
            continue;
        }
        for (r, row) in a.iter_mut().enumerate() {
            if r != c {
                let m = row[c] / p[c];
                row.iter_mut().zip(&p).skip(c).for_each(|(v, q)| *v -= m * q);
            }
        }
    }
    a[0][n] / a[0][0]
}

/// The last negative → non-negative change of u along cell row `k` behind the step (linear between
/// cell centres): `(y, x)` with y the row's height above the wall (Han's wall: the node row's centre).
fn crossing_at(flow: &Flow, step: &Step, k: usize) -> Result<(f64, f64), String> {
    let g = flow.grid();
    let h = step.spacing();
    let [nx, ny, _] = g.cells;
    let wall = if flow.on_node_bounce() { 0.5 * h } else { 0.0 };
    let x = |i: usize| -step.upstream + (i as f64 + 0.5) * h;
    let u = |i: usize, k: usize| (0..ny).filter_map(|j| flow.moments(g.index([i, j, k])).map(|m| m.1[0])).sum::<f64>() / ny as f64;
    let y = (k as f64 - 0.5) * h - wall;
    let crossing = (0..nx.saturating_sub(1))
        .rev()
        .filter(|&i| x(i) > 0.0)
        .find_map(|i| {
            let (a, b) = (u(i, k), u(i + 1, k));
            (a < 0.0 && b >= 0.0).then(|| x(i) + h * a / (a - b))
        })
        .ok_or_else(|| format!("no negative → non-negative change of u in the row at y = {y:.4} H"))?;
    Ok((y, crossing))
}

/// The floor's u = 0 crossing behind the step in the final field ([`WallCrossing`]).
fn wall_crossing(flow: &Flow, step: &Step, shift: isize) -> Result<WallCrossing, String> {
    // Cell row k's centre is at z = (k − ½) Δ (the box starts at z = −Δ). The floor's halfway wall
    // is at z = 0, so its first rows are 1…4 at y = Δ/2 …; Han's wall is the node row 1 at z = Δ/2,
    // so its first rows are 2…5 at y = Δ … 4Δ.
    let first = if flow.on_node_bounce() { 2 } else { 1 };
    // `shift` moves the four rows the crossing is read from, up (+) or down (−) the wall in cells.
    let first = usize::try_from(first as isize + shift).map_err(|_| format!("a shift of {shift} rows puts the first row below the box"))?;
    let rows = (first..first + 4).map(|k| crossing_at(flow, step, k)).collect::<Result<Vec<_>, _>>()?;
    let fits = [intercept(&rows[..2], 1), intercept(&rows[..3], 1), intercept(&rows, 1), intercept(&rows, 2)];
    Ok(WallCrossing { rows, fits })
}

fn report(step: &Step, r: &Run) -> Score {
    let experiment = experiment();
    match r.rule {
        Some(Settled::Steady { block, x_r, shear }) => println!(
            "{} steps, settled by the steady rule at block {block}: the last {STEADY_BLOCKS} blocks' x_r spread {x_r:.1e} (< {STEADY_X_R:.0e}), largest floor shear spread {shear:.1e} (< {STEADY_SHEAR:.0e})",
            r.steps
        ),
        Some(Settled::Blocks) => println!("{} steps, settled by the Blocks acceptance", r.steps),
        Some(Settled::F32Floor(f)) => println!("{} steps, {}: {}", r.steps, f.verdict(), f.describe()),
        None => println!("{} steps, not settled", r.steps),
    }
    println!(
        "inflow at x = −4 H: Re_θ {:.0} (5000), δ₉₉ {:.2} H (≈ 1.5), U_ref {:.4}, Re_H {:.0}",
        r.inflow.0,
        r.inflow.1,
        r.inflow.2,
        r.inflow.2 * step.re_h
    );
    println!("x_r from the time-mean floor shear: {:?}", r.x_r);
    match &r.crossing {
        Ok(c) => println!("{c}"),
        Err(why) => println!("u = 0 at the floor: none ({why})"),
    }
    println!(
        "reattachment blocks: {} ({} refused, last {:?}), mean {:.3} H, 95 % half-width {:.3}, lag-1 {:.2}",
        r.reattachment.means().len(),
        r.missing.0,
        r.missing.1,
        r.reattachment.mean(),
        r.reattachment.half_width(),
        r.reattachment.lag1()
    );
    let (xs, cfs): (Vec<f64>, Vec<f64>) = r.cf.iter().copied().unzip();
    println!("| x/H | C_f measured | C_f run |");
    for (x, c, _) in &experiment.0 {
        println!("| {x:7.3} | {c:.3e} | {} |", interpolate(&xs, &cfs, *x).map_or("—".into(), |v| format!("{v:.3e}")));
    }
    let (px, pcs): (Vec<f64>, Vec<f64>) = r.cp.iter().copied().unzip();
    println!("| x/H | C_p measured (re x/H −4) | C_p run |");
    for (x, c) in &experiment.1 {
        println!("| {x:5.1} | {c:+.4} | {} |", interpolate(&px, &pcs, *x).map_or("—".into(), |v| format!("{v:+.4}")));
    }
    let refs = expected("ddes-backstep-driver-seegmiller");
    let mut score = Score::default();
    for (id, value, tolerance) in refs {
        let run = match id.as_str() {
            "reattachment" => r.x_r.clone().unwrap_or(f64::NAN),
            "cf_inflow" => interpolate(&xs, &cfs, -3.956).unwrap_or(f64::NAN),
            _ => f64::NAN,
        };
        score.push(Gate::new("Driver & Seegmiller", &id, run, value, tolerance));
    }
    println!("{}", score.table());
    score
}

/// The measured floor C_f (x/H, C_f, uncertainty) and C_p (x/H, C_p referenced to x/H = −4) of
/// the case's experiment.json.
fn experiment() -> (Vec<(f64, f64, f64)>, Vec<(f64, f64)>) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/oracle/ddes-backstep-driver-seegmiller/experiment.json");
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let rows = |key: &str| {
        v[key]["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect::<Vec<f64>>())
            .collect::<Vec<_>>()
    };
    (rows("floor_cf").iter().map(|r| (r[0], r[1], r[2])).collect(), rows("floor_cp").iter().map(|r| (r[0], r[2])).collect())
}

/// The case's geometry (ER 1.125, 70 H of inflow, 30 H downstream, one cell of span) at 4 cells
/// per H, laminar at Re_H 20 (τ = 0.53) on the existing uniform velocity inlet with TRT.
fn laminar(max_steps: u64) -> Step {
    Step {
        re_h: 20.0,
        cells_per_h: 4,
        lattice_velocity: 0.05,
        collision: Collision::Trt,
        sigma: None,
        turbulence: None,
        absorbing: None,
        sample: 10,
        block: 2,
        startup: 0.0,
        max_steps,
        checkpoint: None,
        ..Step::case(0.0)
    }
}

/// The builder makes the case's step (H, ER and the lengths measured from the built cells), and
/// 100 laminar steps on the existing inlet run without error and give a finite time-mean shear on
/// every strip.
#[test]
fn the_case_geometry_builds_and_runs_100_laminar_steps() {
    let step = laminar(100);
    let (_, strips, built) = step.build();
    assert_eq!(built, Geometry { step_height: 1.0, expansion_ratio: 1.125, upstream: 70.0, downstream: 30.0, roof: 9.0 });
    assert_eq!(strips.iter().filter(|s| s.floor).count(), 120, "floor strips H/8 long, rounded up to one cell (H/4) at 4 cells per H, over 30 H");
    let r = step.run();
    assert_eq!((r.steps, r.settled), (100, false));
    assert_eq!(r.cf.len(), strips.len());
    assert!(r.cf.iter().all(|(_, c)| c.is_finite()) && r.cp.iter().all(|(_, c)| c.is_finite()), "{:?}", r.cf);
    // Five blocks of two samples, each read for a reattachment or refused with a reason.
    assert_eq!(r.reattachment.means().len() + r.missing.0, 5);
    println!("after 100 steps: x_r {:?}, inflow {:?}", r.x_r, r.inflow);
}

/// Configurations that are not the case's shape are refused, with the reason.
#[test]
fn the_builder_refuses_a_step_that_is_not_the_case() {
    let ok = laminar(100);
    assert_eq!(ok.check(), Ok(()));
    let bad = [
        Step { expansion_ratio: 1.0, ..ok.clone() },
        Step { expansion_ratio: 0.5, ..ok.clone() },
        Step { expansion_ratio: f64::NAN, ..ok.clone() },
        // The roof at 1.3 / 0.3 H falls inside a cell.
        Step { expansion_ratio: 1.3, ..ok.clone() },
        Step { downstream: 29.0, ..ok.clone() },
        Step { downstream: 30.1, ..ok.clone() },
        Step { upstream: 4.0, ..ok.clone() },
        Step { upstream: f64::INFINITY, ..ok.clone() },
        Step { cells_per_h: 1, ..ok.clone() },
        Step { span: -1.0, ..ok.clone() },
        Step { re_h: 0.0, ..ok.clone() },
        Step { lattice_velocity: 0.5, ..ok.clone() },
        Step { sample: 0, ..ok.clone() },
    ];
    for step in bad {
        let why = step.check().expect_err("refused");
        assert!(!why.is_empty(), "{step:?}");
    }
    // ER 1.5 puts the roof at 3 H, a cell face: accepted.
    assert_eq!(Step { expansion_ratio: 1.5, ..ok }.check(), Ok(()));
}

/// The longer pipeline check: 2000 laminar steps read finite floor shear, C_p and an inflow
/// profile, and the flow separates behind the step (no reference: the pipeline only).
#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn the_backstep_model_reads_its_floor_and_inflow() {
    let step = Step { re_h: 50.0, cells_per_h: 8, upstream: 6.0, expansion_ratio: 1.5, sample: 20, block: 5, startup: 0.1, ..laminar(2000) };
    let r = step.run();
    let score = report(&step, &r);
    assert_eq!(score.gates.len(), 2);
    assert_eq!(r.cf.len(), step.strips().len());
    assert!(r.cf.iter().all(|(_, c)| c.is_finite()) && r.cp.iter().all(|(_, c)| c.is_finite()));
    assert!(r.inflow.2 > 0.5 && r.inflow.0 > 0.0, "{:?}", r.inflow);
    // Laminar flow over a step separates behind it: the floor shear is negative just downstream.
    assert!(r.cf.iter().any(|&(x, c)| x > 0.0 && x < 2.0 && c < 0.0), "{:?}", r.cf);
}

impl Step {
    /// Driver & Seegmiller's geometry and conditions (Re_H 36 000, expansion ratio 1.125 so the
    /// roof is at 9 H, 70 H of inflow development, 30 H downstream) at 16 cells per H, without a
    /// turbulence model; `span` in H, 0 for one cell (2D). U = 0.1 in lattice units keeps
    /// τ − ½ = 3 U N / Re_H = 1.3 × 10⁻⁴ above the wall model's floor of 10⁻⁴, and puts the first
    /// cell upstream at y⁺ ≈ 43.
    ///
    /// The hybrid collision at fixed σ 0.99 and the default absorbing layer (10 cells) at the
    /// inlet and the outlet: at this τ a velocity inlet's first columns grow a density mode one
    /// cell wide, which the layer damps (`inlet_free_stream.rs`); the step's uniform inlet held
    /// 20 000 steps with it and drifted to |ρ − 1| 0.25 without it.
    fn case(span: f64) -> Step {
        Step {
            re_h: 36_000.0,
            cells_per_h: 16,
            lattice_velocity: 0.1,
            upstream: 70.0,
            downstream: 30.0,
            expansion_ratio: 1.125,
            span,
            short: false,
            collision: Collision::Hrr,
            sigma: Some(0.99),
            turbulence: None,
            absorbing: Some(AbsorbingLayer { thickness: None, strength: None, filter_time: None }),
            sample: 500,
            block: 10,
            startup: 2.0,
            max_steps: 2_000_000,
            checkpoint: None,
        }
    }

    /// The case with a turbulence model: what each model's run uses.
    fn driver_seegmiller(turbulence: Turbulence, span: f64) -> Step {
        Step { turbulence: Some(turbulence), ..Step::case(span) }
    }
}

/// Driver & Seegmiller's step with a configured run: the 15b gate (reattachment within 10 % of
/// 6.26 H) and the inflow check, printed with the measured C_f and C_p. Each model gets its own
/// `#[test]` calling this, so the test that ran names the model that ran, e.g. once slice 15b
/// exists:
///
/// ```text
/// #[test]
/// #[ignore = "hours: tests/oracle/ddes-backstep-driver-seegmiller/README.md"]
/// fn ddes_reattaches_where_measured() {
///     let (run, score) = driver_seegmiller(&Step::driver_seegmiller(Turbulence::SpalartAllmaras { .. }, 4.0));
///     assert!(run.settled && score.passes());
/// }
/// ```
fn driver_seegmiller(step: &Step) -> (Run, Score) {
    println!("Driver & Seegmiller: {step:?}");
    let r = step.run();
    let score = report(step, &r);
    (r, score)
}

/// The SA run's step: the case at `FIELDCRAFT_STEP_CELLS_PER_H` cells per H (16 by default). The
/// inlet is a uniform velocity that develops a boundary layer, so `FIELDCRAFT_STEP_EXTRA_UPSTREAM`
/// (H, a whole number of cells; 0 by default) lengthens the inflow to raise the Re_θ at x = −4 H:
/// ΔL/H ≈ 1270 / (0.00155 Re_H), about 22 H at the case's Re_H 37 349 (Re_θ 5000 from about 3900).
fn sa_step(sa: Turbulence) -> Step {
    let short = std::env::var("FIELDCRAFT_STEP_SHORT").is_ok_and(|v| v == "1");
    // `FIELDCRAFT_STEP_SPAN`: the periodic span in H (0, the default, is one cell: the 2D run).
    let span: f64 = std::env::var("FIELDCRAFT_STEP_SPAN").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
    sa_step_in(sa, short, span)
}

/// [`sa_step`] with the domain and the span given. `short`: the phase-3 domain, from 4 H and two
/// cells upstream of the step (the reference station x = −4 H is the second cell of the inflow) to
/// `FIELDCRAFT_STEP_DOWNSTREAM` H downstream (20 by default).
fn sa_step_in(sa: Turbulence, short: bool, span: f64) -> Step {
    let cells_per_h = std::env::var("FIELDCRAFT_STEP_CELLS_PER_H").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
    let extra: f64 = std::env::var("FIELDCRAFT_STEP_EXTRA_UPSTREAM").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let case = Step::driver_seegmiller(sa, span);
    if short {
        let downstream = std::env::var("FIELDCRAFT_STEP_DOWNSTREAM").ok().and_then(|v| v.parse().ok()).unwrap_or(20.0);
        return Step { cells_per_h, upstream: 4.0 + 2.0 / cells_per_h as f64, downstream, short: true, ..case };
    }
    Step { cells_per_h, upstream: case.upstream + extra, ..case }
}

/// The SA step as the environment describes it ([`sa_step`], with the wall law of
/// `FIELDCRAFT_WALL_LAW`), built, and its configuration: what the GPU probe
/// (`crates/lbm-gpu/tests/step_probe.rs`, which includes this file) runs.
#[allow(dead_code)]
pub(crate) fn built_sa_step() -> (Flow, String) {
    let step = sa_step(sa_rans());
    let (flow, _, _) = step.build();
    (flow, format!("{step:?}"))
}

/// A small short-domain SA-DDES step in T3's zone (`cells_per_h` per H, `span` H across), built: what the device's
/// zone-mask parity test steps on the CPU and on the device side by side.
#[allow(dead_code)]
pub(crate) fn built_zone_step(cells_per_h: usize, span: f64) -> (Flow, String) {
    let Turbulence::SpalartAllmaras { prandtl, wall_model, inflow_ratio, .. } = sa_rans() else { panic!("sa_rans is Spalart–Allmaras") };
    let h = 1.0 / cells_per_h as f64;
    let zone = fieldcraft_model::DdesZone { min: [-1.0, -1.0, -h], max: [12.0, span + 1.0, 3.0] };
    let ddes =
        Turbulence::SpalartAllmaras { prandtl, wall_model, inflow_ratio, ddes: Some(fieldcraft_model::Ddes { c_des: 0.65, zone: Some(zone) }) };
    let step = Step { cells_per_h, upstream: 4.0 + 2.0 / cells_per_h as f64, ..sa_step_in(ddes, true, span) };
    let (flow, _, _) = step.build();
    (flow, format!("{step:?}"))
}

/// The SA run's model: on Han's on-node wall with the model's own law, or with Spalding's with
/// `FIELDCRAFT_WALL_LAW=spalding` ([`study_wall_law`]; the law is printed).
fn sa_rans() -> Turbulence {
    let (wall, name) = study_wall_law();
    println!("step study: wall law {name}");
    Turbulence::SpalartAllmaras { prandtl: 0.9, wall_model: Some(wall), inflow_ratio: 3.0, ddes: None }
}

/// The floor's u = 0 crossing ([`WallCrossing`]) in an SA run's restart file
/// (`FIELDCRAFT_STEP_RESTART`, a copy: a running job rewrites its own), with the floor shear's
/// sign change after 500 more steps beside it. Seconds:
/// `FIELDCRAFT_STEP_RESTART=<copy> cargo test --release -p fieldcraft-lbm --test ddes_backstep the_floor_crossing_of_a_restart_file -- --ignored --nocapture`.
#[test]
#[ignore = "reads a restart file: FIELDCRAFT_STEP_RESTART"]
fn the_floor_crossing_of_a_restart_file() {
    let file = std::path::PathBuf::from(std::env::var("FIELDCRAFT_STEP_RESTART").unwrap());
    let step = sa_step(sa_rans());
    let (mut flow, strips, _) = step.build();
    let (at, samples) = load(&file, &mut flow);
    println!("{}: step {at}, {} samples", file.display(), samples.len());
    println!("{}", wall_crossing(&flow, &step, 0).unwrap());
    // The floor shear's sign change over the next 500 steps, as the run reads it.
    let h = step.spacing();
    let floor: Vec<&Strip> = strips.iter().filter(|s| s.floor).collect();
    let groups: Vec<Vec<usize>> = floor.iter().map(|s| body_groups(&flow, &s.body)).collect();
    let mut shear = vec![0.0; floor.len()];
    for _ in 0..500 {
        flow.step().unwrap();
        let forces = flow.forces();
        for (t, g) in shear.iter_mut().zip(&groups) {
            *t += g.iter().map(|&g| forces.get(g).map_or(f64::NAN, |f| f[0])).sum::<f64>();
        }
    }
    let x: Vec<f64> = floor.iter().map(|s| 0.5 * (s.x.0 + s.x.1)).collect();
    let per_area: Vec<f64> = shear.iter().zip(&floor).map(|(t, s)| t / ((s.x.1 - s.x.0) / h).round()).collect();
    println!("the floor shear's sign change over the next 500 steps: {:?}", reattachment(&x, &per_area));
}

/// T3's whole-window readings from a T3 restart file (`span` as the run's): the stored samples replayed into the strips' block
/// means (blocks of 30 H/U), the near-wall rows' sums read back, and [`Readings`] formed as the run forms them, with the
/// current reader. Seconds, no stepping.
pub(crate) fn t3_readings_of_restart(file: &std::path::Path, span: f64) -> Result<String, String> {
    let rans = sa_rans();
    let Turbulence::SpalartAllmaras { prandtl, wall_model, inflow_ratio, .. } = rans else { return Err("sa_rans is Spalart–Allmaras".into()) };
    let base = sa_step_in(rans, true, span);
    let h = base.spacing();
    let zone = fieldcraft_model::DdesZone { min: [-1.0, -1.0, -h], max: [12.0, span + 1.0, 3.0] };
    let ddes =
        Turbulence::SpalartAllmaras { prandtl, wall_model, inflow_ratio, ddes: Some(fieldcraft_model::Ddes { c_des: 0.65, zone: Some(zone) }) };
    let mut step = sa_step_in(ddes, true, span);
    let h_over_u = (step.cells_per_h as f64 / step.lattice_velocity).round() as u64;
    step.sample = 100;
    step.block = (30 * h_over_u).div_ceil(100) as usize;
    let (mut flow, strips, _) = step.build();
    let (steps, samples, saved) = load_all(file, &mut flow);
    let mut window = Window::new(&flow, &step);
    window.restore(&saved);
    let mut shear = Shear::new(strips.len(), step.block);
    samples.iter().for_each(|s| {
        shear.push(s);
    });
    let floor_x: Vec<f64> = strips.iter().filter(|s| s.floor).map(|s| 0.5 * (s.x.0 + s.x.1)).collect();
    let floor: Vec<f64> = strips.iter().zip(shear.mean()).filter(|(s, _)| s.floor).map(|(_, t)| t).collect();
    let readings = Readings::of(&floor_x, &floor, &window.floor);
    Ok(format!("{} at step {steps}, {} shear samples in {} blocks: {readings}", file.display(), samples.len(), shear.blocks()))
}

/// The first completed block at which the running mean x_r differs by under 0.5 % from its value
/// at the block a fifth of the steps earlier: `(step, wall seconds, x_r, x_r a fifth earlier)`.
/// `history` holds `(step, wall seconds, running mean x_r)` per block.
fn settle_point(history: &[(u64, f64, f64)]) -> Option<(u64, f64, f64, f64)> {
    history.iter().find_map(|&(at, secs, x_r)| {
        let earlier = history.iter().rev().find(|h| h.0 as f64 <= 0.8 * at as f64)?;
        let change = ((x_r - earlier.2) / x_r).abs();
        (x_r.is_finite() && earlier.2.is_finite() && change < 0.005).then_some((at, secs, x_r, earlier.2))
    })
}

/// The process's peak resident memory, from `/proc/self/status` (Linux); "n/a" elsewhere.
fn peak_memory() -> String {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| s.lines().find_map(|l| l.strip_prefix("VmHWM:").map(|v| v.trim().to_string())))
        .unwrap_or_else(|| "n/a".to_string())
}

/// A settled flow's field as `transfer` needs it: its grid and, per cell, ρ, u and ν̃ (ρ NaN in a
/// solid cell), 40 bytes a cell. A flow holds hundreds of bytes a cell (two sets of populations,
/// the model's arrays), so a run that carries a field into a new flow takes it with [`snapshot`]
/// and drops the source before building the target: the two flows never coexist, which they did
/// when `transfer` read the source flow itself, and the peak memory was twice the flow's.
struct Field {
    grid: Grid,
    cells: Vec<[f64; 5]>,
}

/// The field of `from` ([`Field`]). Refused for a flow without a Spalart–Allmaras ν̃.
fn snapshot(from: &Flow) -> Result<Field, String> {
    let nu = from.spalart_allmaras().map(|f| f.nu_tilde()).ok_or("the source flow has no Spalart–Allmaras field")?;
    let cells = (0..from.grid().len())
        .map(|c| {
            let (rho, u) = from.moments(c).unwrap_or((f64::NAN, [0.0; 3]));
            [rho, u[0], u[1], u[2], nu.get(c).copied().unwrap_or(0.0)]
        })
        .collect();
    Ok(Field { grid: *from.grid(), cells })
}

/// A settled field set in `to`: every fluid cell of `to` takes the density, velocity and ν̃ of the
/// cell of `from` at the same x and z (the span: the source cell at the same index along y modulo
/// its own, so a 2D field is extruded). The two grids share H, the step's position and the floor;
/// their x and z offsets are whole cells. Equilibrium populations at that state, so the
/// non-equilibrium part is rebuilt in a few steps. Cells of `to` outside `from` (a longer outlet)
/// take the nearest column.
fn transfer(from: &Field, to: &mut Flow) -> Result<(), String> {
    let (gf, gt) = (from.grid, *to.grid());
    if (gf.spacing - gt.spacing).abs() > LENGTH_TOLERANCE * gf.spacing {
        return Err(format!("the grids' spacings differ: {} and {}", gf.spacing, gt.spacing));
    }
    let cells = |a: f64, b: f64| ((a - b) / gf.spacing).round() as i64;
    let (dx, dz) = (cells(gt.origin[0], gf.origin[0]), cells(gt.origin[2], gf.origin[2]));
    let source = |c: usize| -> Option<usize> {
        let [i, j, k] = gt.coords(c);
        let (i, k) = (i as i64 + dx, k as i64 + dz);
        let clamp = |v: i64, n: usize| v.clamp(0, n as i64 - 1) as usize;
        Some(gf.index([clamp(i, gf.cells[0]), j % gf.cells[1], clamp(k, gf.cells[2])]))
    };
    let at = |c: usize| source(c).and_then(|s| from.cells.get(s));
    let state = |c: usize| at(c).filter(|v| v[0].is_finite()).map_or((1.0, [0.0; 3]), |v| (v[0], [v[1], v[2], v[3]]));
    to.set_equilibrium(state);
    let values: Vec<f64> = (0..gt.len()).map(|c| at(c).map_or(0.0, |v| v[4])).collect();
    to.set_sa_nu_tilde(&values).map_err(|e| format!("setting ν̃: {e}"))
}

/// The incoming boundary layer at x = −4 H as a table, one line per row from the first fluid row
/// above the step's top to mid-channel: `z/H  u/U_ref  ν̃/ν` (span means; ν̃ in units of ν). It is the
/// inlet profile of a shorter domain, which the solver cannot impose yet (no tabulated inflow
/// profile on a velocity face): written so the profile is a file, not a number in a log.
fn write_profile(path: &std::path::Path, flow: &Flow, step: &Step) -> Result<(), String> {
    let g = *flow.grid();
    let [_, ny, nz] = g.cells;
    let h = step.spacing();
    let i = ((step.upstream - 4.0) / h).floor().clamp(0.0, g.cells[0] as f64 - 1.0) as usize;
    let nu = flow.spalart_allmaras().map(|f| f.nu_tilde().to_vec()).ok_or("the flow has no Spalart–Allmaras field")?;
    let nu_lat = step.lattice_velocity * h * step.re_h.recip() / h / h;
    let mut text = String::from(
        "# z/H  u/U_ref  nu_tilde/nu
",
    );
    let k0 = ((1.0 + h) / h).floor() as usize;
    let middle = ((0.5 * (1.0 + step.roof()) + 1.0) / h).floor().clamp(0.0, nz as f64 - 1.0) as usize;
    // The span-mean of `f` over the fluid cells of row k (None when the row has none: an on-node wall's nodes are not fluid).
    let mean = |k: usize, f: &dyn Fn(usize) -> Option<f64>| {
        let v: Vec<f64> = (0..ny).filter_map(|j| f(g.index([i, j, k]))).collect();
        (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64)
    };
    let u_of = |c: usize| flow.moments(c).map(|m| m.1[0]);
    let u_ref = mean(middle, &u_of).ok_or("the middle row has no fluid cell")?;
    // The profile starts at the first fluid row; rows without fluid (the wall nodes) are left out.
    for k in k0..=middle {
        let (Some(u), Some(t)) = (mean(k, &u_of), mean(k, &|c| nu.get(c).copied())) else { continue };
        let z = g.origin[2] + (k as f64 + 0.5) * g.spacing;
        text.push_str(&format!(
            "{z:.6} {:.8} {:.6}
",
            u / u_ref,
            t / nu_lat
        ));
    }
    std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))
}

/// The rows `[z, u/U_ref, nu_tilde/nu]` of a profile file ([`write_profile`]): comment lines skipped.
fn read_inlet_table(path: &std::path::Path) -> Result<Vec<[f64; 3]>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut rows = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#')) {
        let v: Vec<f64> = line.split_whitespace().filter_map(|w| w.parse().ok()).collect();
        let [z, u, t] = v[..] else { return Err(format!("{}: `{line}` is not three numbers", path.display())) };
        rows.push([z, u, t]);
    }
    if rows.is_empty() {
        return Err(format!("{}: no rows", path.display()));
    }
    Ok(rows)
}

/// The gate reading from the flow field: x_r is the wall (y = 0) value of the least-squares line
/// through the u = 0 crossings of rows 2, 3 and 4 (row 1, the wall-modelled row, reads long on
/// every grid and is reported beside it). No variant is picked by its nearness to the experiment.
/// Returns that x_r (H), if the rows cross.
fn wall_reading(flow: &Flow, step: &Step) -> Option<f64> {
    let first_row = if flow.on_node_bounce() { 2 } else { 1 };
    let rows: Vec<Result<(f64, f64), String>> = (0..6).map(|n| crossing_at(flow, step, first_row + n)).collect();
    for (n, r) in rows.iter().enumerate() {
        match r {
            Ok((y, x)) => println!("u = 0 crossing, row {} (y = {y:.4} H): x {x:.3} H", n + 1),
            Err(why) => println!("u = 0 crossing, row {}: none ({why})", n + 1),
        }
    }
    if let [Ok(r1), Ok(r2), Ok(r3), Ok(r4), ..] = rows.as_slice() {
        let wall = intercept(&[*r2, *r3, *r4], 1);
        println!(
            "x_r at the wall (y = 0, rows 2-4 least squares): {wall:.3} H; row 1 beside it: {:.3} H; rows 2-3 only: {:.3} H (experiment 6.26 ± 0.10); wall law {}",
            r1.1,
            intercept(&[*r2, *r3], 1),
            study_wall_law().1
        );
        return Some(wall);
    }
    None
}

/// What a finished SA run's restart file says without stepping: the gated x_r from the stored
/// samples, the floor crossing read from rows shifted one cell each way, and the incoming layer at
/// x = −4 H (Re_θ and δ₉₉ against the experiment's 5000 and about 1.5 H, and the inflow C_f the
/// scorer gates against). The wall law is the run's: the model's own by default, or
/// `FIELDCRAFT_WALL_LAW=spalding`, `FIELDCRAFT_STEP_CELLS_PER_H` the run's grid. Seconds, no solver step:
/// `FIELDCRAFT_STEP_RESTART=<copy> FIELDCRAFT_STEP_CELLS_PER_H=24 cargo test --release -p fieldcraft-lbm --test ddes_backstep a_restart_files_readings -- --ignored --nocapture`.
#[test]
#[ignore = "reads a restart file: FIELDCRAFT_STEP_RESTART"]
fn a_restart_files_readings() {
    let file = std::path::PathBuf::from(std::env::var("FIELDCRAFT_STEP_RESTART").unwrap());
    let sa_law = study_wall_law().0 == WallModel::SpalartAllmarasBounce;
    let step = sa_step(sa_rans());
    let (mut flow, strips, _) = step.build();
    let (at, samples) = load(&file, &mut flow);
    let h = step.spacing();
    println!(
        "wall law {}; {}: step {at}, {} samples, {} cells per H",
        if sa_law { "spalart_allmaras" } else { "spalding" },
        file.display(),
        samples.len(),
        step.cells_per_h
    );
    // The gated reading: the stored samples replayed into the strips' block means, as the run does.
    let mut shear = Shear::new(strips.len(), step.block);
    samples.iter().for_each(|s| {
        shear.push(s);
    });
    let mean = shear.mean();
    let floor_x: Vec<f64> = strips.iter().filter(|s| s.floor).map(|s| 0.5 * (s.x.0 + s.x.1)).collect();
    let floor: Vec<f64> = strips.iter().zip(&mean).filter(|(s, _)| s.floor).map(|(_, &t)| t).collect();
    println!("gated x_r from the stored samples: {:?} (experiment 6.26 ± 0.10)", reattachment(&floor_x, &floor));
    // The floor crossing, the wall reader moved one cell down and up the wall.
    for shift in [-1isize, 0, 1] {
        match wall_crossing(&flow, &step, shift) {
            Ok(c) => println!("wall reader shifted {shift:+} cells: first row x {:.3}; {c}", c.rows.first().map_or(f64::NAN, |r| r.1)),
            Err(why) => println!("wall reader shifted {shift:+} cells: none ({why})"),
        }
    }
    wall_reading(&flow, &step);
    // The incoming layer at x = −4 H, up to mid-channel, as the run reads it, and with the wall
    // interval (the first node above the wall to the wall node) from the wall law.
    let g = *flow.grid();
    let [nx, ny, nz] = g.cells;
    let column = |x: f64| ((x + step.upstream) / h).floor().clamp(0.0, nx as f64 - 1.0) as usize;
    let row = |z: f64| ((z + h) / h).floor().clamp(0.0, nz as f64 - 1.0) as usize;
    let span_mean = |i: usize, k: usize| {
        let (mut rho, mut u) = (0.0, 0.0);
        for j in 0..ny {
            let (r, v) = flow.moments(g.index([i, j, k])).unwrap_or((f64::NAN, [f64::NAN; 3]));
            rho += r;
            u += v[0];
        }
        (rho / ny as f64, u / ny as f64)
    };
    let middle = 0.5 * (1.0 + step.roof());
    let (rho_ref, u_ref) = span_mean(column(-4.0), row(middle));
    let i = column(-4.0);
    let u: Vec<f64> = (row(1.0) + 1..=row(middle)).map(|k| span_mean(i, k).1).collect();
    let lat = step.lattice_velocity;
    let (edge, nu) = (u_ref / lat, 1.0 / step.re_h);
    let up: Vec<f64> = u.iter().map(|v| v / lat).collect();
    let theta_run = momentum_thickness(&u, h, u_ref);
    // Trapezoid over the nodes from the first one above the wall to mid-channel, plus the interval below.
    let f = |v: f64| (v / edge) * (1.0 - v / edge);
    let beyond = h * (up.iter().skip(1).map(|&v| f(v)).sum::<f64>() + 0.5 * f(up[0]) - 0.5 * f(up[up.len() - 1]));
    let interval = up.first().and_then(|&u1| if sa_law { WallInterval::new_sa_law(u1, h, nu) } else { WallInterval::new(u1, h, nu) });
    let [wall_law, wall_trap] = interval.map_or([f64::NAN; 2], |w| w.momentum_thickness(edge));
    let delta = u.iter().position(|&v| v >= 0.99 * u_ref).map_or(f64::NAN, |k| (k as f64 + 0.5) * h);
    let re = |theta: f64| edge * theta * step.re_h;
    println!(
        "inflow at x = −4 H (U_ref {edge:.4}, Re_H {:.0}): Re_θ as the run logs it {:.0}; trapezoid from u = 0 at the wall node {:.0}; with the {} wall interval {:.0}; experiment 5000. δ₉₉ {:.2} H (experiment ≈ 1.5)",
        edge * step.re_h,
        re(theta_run),
        re(beyond + wall_trap),
        if sa_law { "SA-law" } else { "Spalding" },
        re(beyond + wall_law),
        delta
    );
    let q = 0.5 * rho_ref * u_ref * u_ref;
    let cf: Vec<(f64, f64)> = strips.iter().zip(&mean).map(|(s, t)| (0.5 * (s.x.0 + s.x.1), t / q)).collect();
    let (xs, cfs): (Vec<f64>, Vec<f64>) = cf.iter().copied().unzip();
    let reference = expected("ddes-backstep-driver-seegmiller").into_iter().find(|(id, ..)| id == "cf_inflow");
    println!(
        "inflow C_f at x/H −3.956: run {:.3e}; the scorer's reference {:?} (tolerance, relative)",
        interpolate(&xs, &cfs, -3.956).unwrap_or(f64::NAN),
        reference.map(|(_, v, t)| (v, t))
    );
}

/// Spalart–Allmaras RANS on Han's wall in 2D (one cell of span), the case's grid and inlet: the
/// first model run toward x_r / H. Hours on 5 threads; `FIELDCRAFT_STEP_CHECKPOINT=<file>` keeps a
/// restart file there every 2000 steps, and a run started with it resumes from it;
/// `FIELDCRAFT_STEP_CELLS_PER_H` refines it (16 by default):
/// `FIELDCRAFT_THREADS=5 cargo test --release -p fieldcraft-lbm --test ddes_backstep sa_rans_on_han_s_wall -- --ignored --nocapture`.
#[test]
#[ignore = "hours: tests/oracle/ddes-backstep-driver-seegmiller/README.md"]
pub(crate) fn sa_rans_on_han_s_wall_reattaches_where_measured() {
    let sa = sa_rans();
    let steps = std::env::var("FIELDCRAFT_STEP_MAX_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(400_000);
    let checkpoint = std::env::var_os("FIELDCRAFT_STEP_CHECKPOINT").map(std::path::PathBuf::from);
    let (run, score) = driver_seegmiller(&Step { max_steps: steps, checkpoint, ..sa_step(sa) });
    assert!(run.settled && score.passes(), "x_r {:?}, settled {}", run.x_r, run.settled);
}

/// A run with a restart file resumes bit for bit: a small laminar step run 2000 steps, which
/// writes its restart file, then resumed to 2400, gives the same shear profile and block
/// reattachments as an unbroken run of 2400 steps.
#[test]
fn a_step_run_resumes_from_its_restart_file_bit_for_bit() {
    let dir = std::env::temp_dir().join(format!("fieldcraft-step-restart-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("step.restart");
    let base = Step { re_h: 50.0, cells_per_h: 8, upstream: 6.0, expansion_ratio: 1.5, sample: 20, block: 5, ..laminar(0) };
    let whole = Step { max_steps: 2400, ..base.clone() }.run();
    let first = Step { max_steps: 2000, checkpoint: Some(path.clone()), ..base.clone() }.run();
    assert!(path.exists(), "a restart file at step 2000");
    assert_eq!(first.steps, 2000);
    let resumed = Step { max_steps: 2400, checkpoint: Some(path.clone()), ..base }.run();
    std::fs::remove_dir_all(&dir).ok();
    assert_eq!(resumed.steps, whole.steps);
    let bits = |r: &Run| r.cf.iter().map(|(x, c)| (x.to_bits(), c.to_bits())).collect::<Vec<_>>();
    assert_eq!(bits(&resumed), bits(&whole), "the resumed run's shear differs");
    assert_eq!(resumed.reattachment.means(), whole.reattachment.means());
}

/// The wall-crossing fits: points on a line give its intercept by every fit, points on a parabola
/// give its intercept by the quadratic fit.
#[test]
fn the_wall_crossing_fits_recover_lines_and_parabolas() {
    let line: Vec<(f64, f64)> = [1.0, 2.0, 3.0, 4.0].iter().map(|&y| (y / 16.0, 6.2 - 8.0 * y / 16.0)).collect();
    for degree in [1, 2] {
        assert!((intercept(&line, degree) - 6.2).abs() < 1e-12);
    }
    assert!((intercept(&line[..2], 1) - 6.2).abs() < 1e-12);
    let parabola: Vec<(f64, f64)> = [1.0, 2.0, 3.0, 4.0].iter().map(|&y| (y / 16.0, 6.2 - 8.0 * y / 16.0 + 30.0 * (y / 16.0) * (y / 16.0))).collect();
    assert!((intercept(&parabola, 2) - 6.2).abs() < 1e-10);
}

/// The steady rule: five identical blocks pass (a strip at zero shear too), four do not; an x_r
/// spread, or a strong floor strip's drift over the floor's scale, fails; a strip near zero
/// shear with round-off jitter passes; a value that is not finite never settles.
#[test]
fn the_furthest_downstream_crossing_beyond_one_step_height_is_the_reattachment() {
    let xs: Vec<f64> = (0..20).map(|i| 0.5 * i as f64 + 0.25).collect();
    // The corner bubble's change at about 0.9 H and the primary one at about 6.3 H: negative,
    // positive (the corner eddy), negative, positive.
    let tau: Vec<f64> = xs
        .iter()
        .map(|&x| {
            if x < 0.5 {
                -1.0
            } else if x < 1.3 {
                0.5
            } else if x < 6.3 {
                -1.0
            } else {
                0.5
            }
        })
        .collect();
    assert!(matches!(reattachment(&xs, &tau), Err(NoReattachment::Several(_))), "the old reader refuses two changes");
    let (x, found) = furthest_crossing(&xs, &tau, 1.0).unwrap();
    assert_eq!(found, Found::Crossing);
    assert!((6.0..6.6).contains(&x), "{x}");
    // Never refused: attached everywhere beyond the start reads the start, never reattached reads the last station.
    let attached = vec![0.5; xs.len()];
    assert_eq!(furthest_crossing(&xs, &attached, 1.0).unwrap(), (1.0, Found::Attached));
    let open = vec![-0.5; xs.len()];
    assert_eq!(furthest_crossing(&xs, &open, 1.0).unwrap(), (xs[xs.len() - 1], Found::Open));
    // A crossing nearer than the start is a corner bubble's, so with none beyond it the profile reads as attached.
    let near: Vec<f64> = xs.iter().map(|&x| if x < 0.6 { -1.0 } else { 1.0 }).collect();
    assert_eq!(furthest_crossing(&xs, &near, 1.0).unwrap().1, Found::Attached);
    assert!(furthest_crossing(&xs[..1], &tau[..1], 1.0).is_err(), "one sample cannot be read");
}

/// A [`FloorRows`] holding a given forward fraction and mean u in every row, as if sampled `n` times.
fn synthetic_rows(x: &[f64], mean_u: impl Fn(usize, f64) -> f64, forward: impl Fn(usize, f64) -> f64) -> FloorRows {
    let nx = x.len();
    let mut sums = vec![[0.0; 3]; FLOOR_ROWS * nx];
    for n in 0..FLOOR_ROWS {
        for (i, &xi) in x.iter().enumerate() {
            sums[n * nx + i] = [4.0 * mean_u(n, xi), 4.0 * forward(n, xi), 4.0];
        }
    }
    FloorRows { first: 1, x: x.to_vec(), y: [0.02, 0.06, 0.1, 0.14], sums }
}

#[test]
fn the_window_readings_come_from_the_time_means_and_say_when_they_differ() {
    let x: Vec<f64> = (0..120).map(|i| -2.0 + 0.1 * (i as f64 + 0.5)).collect();
    // Row n's u = 0 line leans downstream towards the wall: x = 6.0 + 4 y. Forward half of the time at 6.1 in row 2.
    let u = |n: usize, x: f64| x - (6.0 + 4.0 * [0.02, 0.06, 0.1, 0.14][n]);
    let forward = |_: usize, x: f64| 0.5 + 0.4 * (x - 6.1).clamp(-1.0, 1.0);
    let rows = synthetic_rows(&x, u, forward);
    let floor_x: Vec<f64> = (0..80).map(|i| 0.1 * i as f64 + 0.05).collect();
    let shear: Vec<f64> = floor_x.iter().map(|&x| x - 6.0).collect();
    let r = Readings::of(&floor_x, &shear, &rows);
    let (shear_x, _) = r.shear.clone().unwrap();
    assert!((shear_x - 6.0).abs() < 1e-9, "{shear_x}");
    // The least-squares line through rows 2-4 to y = 0 recovers the wall value 6.0.
    assert!((r.rows_2_4.clone().unwrap() - 6.0).abs() < 1e-6, "{:?}", r.rows_2_4);
    let (forward_x, _) = r.forward[0].clone().unwrap();
    assert!((forward_x - 6.1).abs() < 1e-9, "{forward_x}");
    assert_eq!(r.forward.len(), FLOOR_ROWS, "rows 1 to 4");
    assert!((r.forward_2_4.clone().unwrap() - 6.1).abs() < 1e-6, "{:?}", r.forward_2_4);
    assert_eq!(r.disagreement(), None, "0.1 H apart is inside the 0.2 H band");
    assert_eq!(r.samples, 4);
    // Moved 0.5 H apart they differ, and the report says so.
    let far = Readings::of(&floor_x, &floor_x.iter().map(|&x| x - 5.5).collect::<Vec<f64>>(), &rows);
    assert!(far.disagreement().is_some_and(|g| (g - 0.6).abs() < 1e-6));
    assert!(far.to_string().contains("DIFFER"), "{far}");
    // No samples: the readings say so, never a number.
    let empty = FloorRows { sums: vec![[0.0; 3]; rows.sums.len()], ..synthetic_rows(&x, u, forward) };
    assert!(Readings::of(&floor_x, &shear, &empty).forward[0].is_err());
}

#[test]
fn the_shear_layers_stress_splits_into_resolved_and_modelled() {
    // One station's rows: u rises linearly (du/dz = 2 per cell row), w = 0 on average, u w
    // correlated to give a resolved stress of 0.004 at row 5 and 0.001 elsewhere, ν_t = 0.003.
    let (nz, spacing, lattice_velocity) = (10, 0.1, 1.0);
    let station = |tau_res: &dyn Fn(usize) -> f64| -> Vec<[f64; 5]> {
        (0..nz).map(|k| [4.0 * 2.0 * k as f64, 0.0, 4.0 * -tau_res(k), 4.0 * 0.003, 4.0]).collect()
    };
    let rows = station(&|k| if k == 5 { 0.004 } else { 0.001 });
    let layer = ShearLayer { columns: [0; 3], nz, spacing, lattice_velocity, nu: 0.01, sums: rows.iter().cycle().take(3 * nz).copied().collect() };
    let s = layer.stresses().remove(0).unwrap();
    // Modelled: ⟨ν_t⟩ d⟨u⟩/dz = 0.003 × 2 = 0.006 everywhere; the total peaks at row 5 (0.004 + 0.006).
    assert!((s.z - 4.5 * spacing).abs() < 1e-12, "{}", s.z);
    assert!((s.resolved - 0.004).abs() < 1e-12 && (s.modelled - 0.006).abs() < 1e-12, "{s:?}");
    assert!((s.peak_fraction - 0.4).abs() < 1e-12);
    // Over rows 1 to 8: resolved 7 × 0.001 + 0.004 = 0.011 against 8 × 0.006 modelled.
    assert!((s.layer_fraction - 0.011 / (0.011 + 0.048)).abs() < 1e-12, "{}", s.layer_fraction);
    // No samples: no stress.
    let none = ShearLayer { sums: vec![[0.0; 5]; 3 * nz], ..layer };
    assert!(none.stresses().iter().all(Result::is_err));
}

#[test]
fn the_near_wall_rows_ride_in_the_restart_file_and_an_old_file_reads_without_them() {
    let step = laminar(10);
    let (mut flow, _, _) = step.build();
    let mut rows = Window::new(&flow, &step);
    flow.step().unwrap();
    rows.add(&flow);
    rows.add(&flow);
    assert_eq!(rows.floor.samples(), 2);
    let dir = std::env::temp_dir().join(format!("fieldcraft-rows-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("step.restart");
    save(&file, &mut flow, 1, &[vec![1.0, 2.0]], Some(&rows));
    let (mut again, _, _) = step.build();
    let (steps, samples, state) = load_all(&file, &mut again);
    assert_eq!((steps, samples), (1, vec![vec![1.0, 2.0]]));
    let mut back = Window::new(&again, &step);
    back.restore(&state);
    assert_eq!(back.state(), rows.state());
    // A file written without them (the old layout ends after the samples) reads as an empty window.
    save(&file, &mut flow, 1, &[vec![1.0, 2.0]], None);
    let (_, _, none) = load_all(&file, &mut again);
    let mut empty = Window::new(&again, &step);
    empty.restore(&none);
    assert_eq!(empty.floor.samples(), 0);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn the_steady_rule_needs_five_agreeing_blocks_in_x_r_and_every_floor_strip() {
    let block = |x_r: f64, drift: f64| (x_r, vec![-7.0e-4 * (1.0 + drift), 0.0, 3.0e-4]);
    let still: Vec<(f64, Vec<f64>)> = (0..5).map(|_| block(5.7834, 0.0)).collect();
    assert_eq!(steady(&still), Some((0.0, 0.0)));
    assert_eq!(steady(&still[..4]), None);
    let round_off: Vec<(f64, Vec<f64>)> = (0..5).map(|n| block(5.7834 * (1.0 + 1e-8 * n as f64), 1e-7 * n as f64)).collect();
    assert!(steady(&round_off).is_some_and(|(x, t)| x < 1e-7 && t < 1e-6));
    let moving: Vec<(f64, Vec<f64>)> = (0..5).map(|n| block(5.7834 * (1.0 + 5e-6 * n as f64), 0.0)).collect();
    assert_eq!(steady(&moving), None, "x_r spread 2e-5");
    let drifting: Vec<(f64, Vec<f64>)> = (0..5).map(|n| block(5.7834, 5e-5 * n as f64)).collect();
    assert_eq!(steady(&drifting), None, "the strongest strip's shear spread 2e-4");
    // A strip straddling x_r: mean 1e-12 with round-off jitter of its own size (a relative spread
    // of about 1) passes, since it is measured against the floor's scale.
    let straddling: Vec<(f64, Vec<f64>)> = (0..5).map(|n| (5.7834, vec![-7.0e-4, 1e-12 * (1.0 + 0.5 * (n % 2) as f64), 3.0e-4])).collect();
    assert!(steady(&straddling).is_some_and(|(_, t)| t < 1e-8));
    for bad in [f64::NAN, f64::INFINITY] {
        let mut broken = still.clone();
        broken[2].1[1] = bad;
        assert_eq!(steady(&broken), None, "{bad}");
        broken[2] = (bad, vec![-7.0e-4, 0.0, 3.0e-4]);
        assert_eq!(steady(&broken), None, "x_r {bad}");
    }
    // The window is the last five: an older moving block no longer counts.
    let settled_late: Vec<(f64, Vec<f64>)> = std::iter::once(block(5.0, 0.0)).chain(still).collect();
    assert_eq!(steady(&settled_late), Some((0.0, 0.0)));
    assert_eq!(spread(&[0.0, 0.0], 0.0), 0.0);
}

/// The settle point is the first block whose running mean is within 0.5 % of the one a fifth of
/// the steps earlier, and none while it still drifts.
#[test]
fn the_settle_point_is_the_first_block_within_half_a_percent_of_a_fifth_earlier() {
    let drifting: Vec<(u64, f64, f64)> = (1..=20).map(|n| (n * 100, n as f64, 5.0 + 0.1 * n as f64)).collect();
    assert_eq!(settle_point(&drifting), None);
    let mut settling = drifting.clone();
    settling.iter_mut().skip(10).for_each(|h| h.2 = 6.0);
    // Step 1300 is the first whose fifth-earlier block (step 1000, running mean 6.0) agrees.
    assert_eq!(settle_point(&settling).map(|p| p.0), Some(1300));
    assert_eq!(settle_point(&[]), None);
}

/// The f32 floor over the last blocks of T1's restart file: its stored samples replayed into the blocks, the floor's
/// x_r and strip shears of each, and the rule of [`f32_floor`] on the last [`F32_BLOCKS`]; with those blocks' x_r.
pub(crate) fn f32_floor_of_restart(file: &std::path::Path) -> Result<(F32Floor, Vec<f64>), String> {
    let step = sa_step_in(sa_rans(), false, 0.0);
    let (mut flow, strips, _) = step.build();
    let (_, samples) = load(file, &mut flow);
    let mut shear = Shear::new(strips.len(), step.block);
    let x: Vec<f64> = strips.iter().filter(|s| s.floor).map(|s| 0.5 * (s.x.0 + s.x.1)).collect();
    let mut recent: Vec<(f64, Vec<f64>)> = Vec::new();
    for sample in &samples {
        if shear.push(sample) {
            let floor: Vec<f64> = strips.iter().zip(shear.last_block()).filter(|(s, _)| s.floor).map(|(_, t)| t).collect();
            let x_r = reattachment(&x, &floor).map_err(|e| format!("block {}: {e:?}", shear.blocks()))?;
            recent.push((x_r, floor));
        }
    }
    let last: Vec<f64> = recent.iter().rev().take(F32_BLOCKS).rev().map(|(x, _)| *x).collect();
    f32_floor(&recent).map(|f| (f, last)).ok_or_else(|| format!("{} blocks, fewer than {F32_BLOCKS}", recent.len()))
}

/// The gated x_r of a finished run's restart file: the stored samples replayed into the strips'
/// block means, as the run reads them.
fn restart_x_r(step: &Step, file: &std::path::Path) -> Result<f64, String> {
    let (mut flow, strips, _) = step.build();
    let (_, samples) = load(file, &mut flow);
    let mut shear = Shear::new(strips.len(), step.block);
    samples.iter().for_each(|s| {
        shear.push(s);
    });
    let mean = shear.mean();
    let x: Vec<f64> = strips.iter().filter(|s| s.floor).map(|s| 0.5 * (s.x.0 + s.x.1)).collect();
    let floor: Vec<f64> = strips.iter().zip(&mean).filter(|(s, _)| s.floor).map(|(_, &t)| t).collect();
    reattachment(&x, &floor).map_err(|e| format!("{}: {e:?}", file.display()))
}

/// T1s: the short domain (4 H upstream to 20 H downstream, 2D), started from T1's settled field,
/// run to the steady rule; x_r against T1's (the gate: within 0.5 %). The inlet is the uniform
/// velocity: the solver has no tabulated inflow profile, so until it has one the short domain's
/// boundary layer at x = −4 H is not T1's and this run only checks the plumbing. `FIELDCRAFT_STEP_RESTART` is T1's
/// restart file (a copy), `FIELDCRAFT_STEP_CELLS_PER_H` its grid, `FIELDCRAFT_STEP_CHECKPOINT` T1s's own:
/// `FIELDCRAFT_STEP_RESTART=<copy> FIELDCRAFT_STEP_CELLS_PER_H=24 cargo test --release -p fieldcraft-lbm --test ddes_backstep t1s -- --ignored --nocapture`.
#[test]
#[ignore = "hours: the short step from T1's field, FIELDCRAFT_STEP_RESTART"]
pub(crate) fn t1s_the_short_step_from_t1s_field_reattaches_with_t1() {
    let file = std::path::PathBuf::from(std::env::var("FIELDCRAFT_STEP_RESTART").expect("FIELDCRAFT_STEP_RESTART"));
    let sa = sa_rans();
    let t1 = sa_step_in(sa, false, 0.0);
    let (mut full, _, _) = t1.build();
    load(&file, &mut full);
    let x_t1 = restart_x_r(&t1, &file).unwrap_or(f64::NAN);
    let checkpoint = std::env::var_os("FIELDCRAFT_STEP_CHECKPOINT").map(std::path::PathBuf::from);
    let steps = std::env::var("FIELDCRAFT_STEP_MAX_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(400_000);
    let short = Step { max_steps: steps, checkpoint, startup: 0.5, ..sa_step_in(sa, true, 0.0) };
    let field = snapshot(&full).unwrap_or_else(|e| panic!("T1s: {e}"));
    drop(full);
    let run = short.try_run_from(Some(Box::new(move |flow: &mut Flow| transfer(&field, flow)))).unwrap_or_else(|e| panic!("T1s: {e}"));
    let x_s = run.x_r.as_ref().map_or(f64::NAN, |x| *x);
    println!(
        "T1s x_r {x_s:.4} H against T1 {x_t1:.4} H: {:+.2} % (gate 0.5 %: {}); settled {}",
        100.0 * (x_s - x_t1) / x_t1,
        if ((x_s - x_t1) / x_t1).abs() <= 0.005 { "pass" } else { "fail" },
        run.settled
    );
}

/// T2: T1s's field extruded across a periodic span (`FIELDCRAFT_STEP_SPAN` H, RANS, 20 H/U steps),
/// timed: cells, peak memory, seconds per step and x_r from the flow field (rows 2-4) against T1s's
/// (the gate: within 0.5 %). `FIELDCRAFT_STEP_RESTART` is T1s's restart file (a copy). One block is
/// 1000 steps, so the gated shear x_r is read from the last whole blocks as well.
/// `FIELDCRAFT_STEP_RESTART=<copy> FIELDCRAFT_STEP_CELLS_PER_H=24 FIELDCRAFT_STEP_SPAN=4 cargo test --release -p fieldcraft-lbm --test ddes_backstep t2 -- --ignored --nocapture`.
#[test]
#[ignore = "hours: the extruded short step from T1s's field, FIELDCRAFT_STEP_RESTART"]
pub(crate) fn t2_the_extruded_short_step_keeps_t1s_reattachment() {
    let file = std::path::PathBuf::from(std::env::var("FIELDCRAFT_STEP_RESTART").expect("FIELDCRAFT_STEP_RESTART"));
    let span: f64 = std::env::var("FIELDCRAFT_STEP_SPAN").ok().and_then(|v| v.parse().ok()).unwrap_or(4.0);
    let sa = sa_rans();
    let t1s = sa_step_in(sa, true, 0.0);
    let (mut flat, _, _) = t1s.build();
    load(&file, &mut flat);
    let x_flat = wall_reading(&flat, &t1s);
    let h_over_u = (t1s.cells_per_h as f64 / t1s.lattice_velocity).round() as u64;
    let checkpoint = std::env::var_os("FIELDCRAFT_STEP_CHECKPOINT").map(std::path::PathBuf::from);
    let t2 = Step { max_steps: 20 * h_over_u, sample: 100, block: 10, startup: 0.0, checkpoint, ..sa_step_in(sa, true, span) };
    let field = snapshot(&flat).unwrap_or_else(|e| panic!("T2: {e}"));
    drop(flat);
    let run = t2.try_run_from(Some(Box::new(move |flow: &mut Flow| transfer(&field, flow)))).unwrap_or_else(|e| panic!("T2: {e}"));
    println!("T2 (span {span} H, {} steps = 20 H/U): x_r (gated shear) {:?}; T1s from its field {x_flat:?}", run.steps, run.x_r);
}

/// The short domain's check, and a field carried from the full domain to a spanwise one: the
/// target's cells hold the source's density at the same x and z, the span extruded.
#[test]
fn a_field_carries_from_the_full_step_to_the_short_extruded_one() {
    let sa = Turbulence::SpalartAllmaras { prandtl: 0.9, wall_model: None, inflow_ratio: 3.0, ddes: None };
    let base = Step { turbulence: Some(sa), ..laminar(100) };
    let short = Step { upstream: 4.5, downstream: 20.0, short: true, span: 1.0, ..base.clone() };
    assert_eq!(short.check(), Ok(()));
    assert!(Step { short: false, ..short.clone() }.check().is_err(), "20 H downstream is the short domain's alone");
    let (mut full, _, _) = base.build();
    full.set_equilibrium(|c| (1.0 + 1e-3 * (c % 7) as f64, [0.02, 0.0, 0.0]));
    let (mut to, _, _) = short.build();
    let field = snapshot(&full).unwrap();
    // What a run keeps while the target is built is a small part of a flow: this is what stands in the
    // way of the target's peak being twice the source's (a built flow here holds a few hundred bytes a cell).
    let kept = (field.cells.len() * std::mem::size_of::<[f64; 5]>()) as f64;
    assert!(kept < 0.2 * full.heap_bytes() as f64, "{kept} bytes of {}", full.heap_bytes());
    transfer(&field, &mut to).unwrap();
    let (gf, gt) = (*full.grid(), *to.grid());
    let (dx, dz) = (((gt.origin[0] - gf.origin[0]) / gf.spacing).round() as usize, ((gt.origin[2] - gf.origin[2]) / gf.spacing).round() as usize);
    let mut checked = 0;
    for j in 0..gt.cells[1] {
        for (i, k) in [(3, 5), (20, 9), (40, 12)] {
            let (a, b) = (gt.index([i, j, k]), gf.index([i + dx, 0, k + dz]));
            if let (Some(t), Some(f)) = (to.moments(a), full.moments(b)) {
                assert!((t.0 - f.0).abs() < 1e-12 && (t.1[0] - f.1[0]).abs() < 1e-12, "cell {i},{j},{k}: {t:?} against {f:?}");
                checked += 1;
            }
        }
    }
    assert!(checked >= gt.cells[1], "{checked} cells compared");
}

/// Writes the inlet profile (`FIELDCRAFT_STEP_PROFILE`) from a restart file (`FIELDCRAFT_STEP_RESTART`) without stepping, at the
/// grid of `FIELDCRAFT_STEP_CELLS_PER_H` (the case's other knobs as for the run that wrote the restart).
#[test]
#[ignore = "local: re-extracts the profile from a restart, FIELDCRAFT_STEP_RESTART and FIELDCRAFT_STEP_PROFILE"]
fn a_restart_writes_its_profile() {
    let file = std::path::PathBuf::from(std::env::var("FIELDCRAFT_STEP_RESTART").expect("FIELDCRAFT_STEP_RESTART"));
    let out = std::path::PathBuf::from(std::env::var("FIELDCRAFT_STEP_PROFILE").expect("FIELDCRAFT_STEP_PROFILE"));
    let step = sa_step_in(sa_rans(), false, 0.0);
    let (mut flow, _, _) = step.build();
    load(&file, &mut flow);
    write_profile(&out, &flow, &step).expect("profile written");
    println!("profile written to {}", out.display());
}

/// T3: the DDES study. The span step of T2 (`FIELDCRAFT_STEP_SPAN` H, short domain) with DDES
/// (C_DES 0.65, Spalart et al. 2006) acting in the box from 1 H upstream of the step to 12 H
/// downstream and from the floor to 3 H, RANS everywhere else. It starts from T2's restart file
/// (`FIELDCRAFT_STEP_RESTART`, a copy: RANS, same grid and span, read as [`transfer`] reads T1s's),
/// discards the first 50 H/U (the start-up, which the statistics never see), then averages x_r in
/// blocks of 1000 steps (100 samples of 10) until the blocks' acceptance holds (at least 20 blocks,
/// 95 % half-width and half-window drift under 1 %, `Blocks::acceptance`), at most 300 H/U in all.
/// H/U is `FIELDCRAFT_STEP_CELLS_PER_H` over the lattice speed, in steps. Prints `timing:` (cells,
/// wall time, peak memory), the M cell updates/s, the steps, and x_r with its batch-means error.
/// `FIELDCRAFT_STEP_CHECKPOINT` is T3's own restart file, resumed when it exists; `FIELDCRAFT_STEP_MAX_STEPS` is the step cap
/// (600 H/U by default), and a run that reaches it unsettled prints its readings marked "partial".
/// `FIELDCRAFT_STEP_RESTART=<copy> FIELDCRAFT_STEP_CELLS_PER_H=24 FIELDCRAFT_STEP_SPAN=4 cargo test --release -p fieldcraft-lbm --test ddes_backstep t3 -- --ignored --nocapture`.
#[test]
#[ignore = "hours: the DDES study from T2's restart, FIELDCRAFT_STEP_RESTART"]
pub(crate) fn t3_ddes_in_the_zone_from_t2s_field_reports_x_r_with_its_error() {
    let file = std::path::PathBuf::from(std::env::var("FIELDCRAFT_STEP_RESTART").expect("FIELDCRAFT_STEP_RESTART"));
    let span: f64 = std::env::var("FIELDCRAFT_STEP_SPAN").ok().and_then(|v| v.parse().ok()).unwrap_or(4.0);
    let checkpoint = std::env::var_os("FIELDCRAFT_STEP_CHECKPOINT").map(std::path::PathBuf::from);
    let rans = sa_rans();
    let t2 = Step { startup: 0.0, ..sa_step_in(rans, true, span) };
    let (mut from, _, _) = t2.build();
    load(&file, &mut from);
    let h = t2.spacing();
    let zone = fieldcraft_model::DdesZone { min: [-1.0, -1.0, -h], max: [12.0, span + 1.0, 3.0] };
    let Turbulence::SpalartAllmaras { prandtl, wall_model, inflow_ratio, .. } = rans else { panic!("sa_rans is Spalart–Allmaras") };
    let ddes =
        Turbulence::SpalartAllmaras { prandtl, wall_model, inflow_ratio, ddes: Some(fieldcraft_model::Ddes { c_des: 0.65, zone: Some(zone) }) };
    let base = sa_step_in(ddes, true, span);
    let h_over_u = (base.cells_per_h as f64 / base.lattice_velocity).round() as u64;
    let flow_through = ((base.upstream + base.downstream) / h / base.lattice_velocity) as u64;
    // The cap: 600 H/U, or `FIELDCRAFT_STEP_MAX_STEPS` steps.
    let max_steps = std::env::var("FIELDCRAFT_STEP_MAX_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(600 * h_over_u);
    let t3 = Step {
        max_steps,
        sample: 100,
        // A block is 30 H/U at least, for the error bar only: no block is refused.
        block: (30 * h_over_u).div_ceil(100) as usize,
        startup: (50 * h_over_u) as f64 / flow_through as f64,
        checkpoint,
        ..base
    };
    println!(
        "T3: DDES zone {:?} to {:?} (H), C_DES 0.65; H/U = {h_over_u} steps, discard 50 H/U, cap {max_steps} steps = {:.0} H/U",
        zone.min,
        zone.max,
        max_steps as f64 / h_over_u as f64
    );
    // The source flow goes before the target is built: the two never coexist (see [`Field`]).
    let field = snapshot(&from).unwrap_or_else(|e| panic!("T3: {e}"));
    drop(from);
    let run = t3.try_run_from(Some(Box::new(move |flow: &mut Flow| transfer(&field, flow)))).unwrap_or_else(|e| panic!("T3: {e}"));
    let a = run.reattachment.acceptance();
    println!(
        "T3 x_r {:?} H (mean-shear reading); mean of {} block reattachments {:.4} H, 95 % half-width {:?} (relative {:?}), half-window drift {:?}; {} steps = {:.1} H/U, settled {} ({:?}), {} blocks refused, {} censored
{}",
        run.x_r,
        run.reattachment.means().len(),
        run.reattachment.mean(),
        a.half_width_95,
        a.relative_half_width,
        a.relative_half_window_drift,
        run.steps,
        run.steps as f64 / h_over_u as f64,
        run.settled,
        run.rule,
        run.missing.0,
        run.censored,
        run.readings
    );
    // The plan's check: under half of the stress resolved at 3 H is the delayed break-up of the shear
    // layer, not physics.
    match run.stresses.get(1) {
        Some(Ok(s)) if s.peak_fraction < 0.5 => {
            println!("T3: resolved stress is {:.2} of the total at 3 H, under half: the shear layer breaks up late", s.peak_fraction)
        }
        Some(Ok(s)) => println!("T3: resolved stress is {:.2} of the total at 3 H, at least half", s.peak_fraction),
        other => println!("T3: no stress reading at 3 H: {other:?}"),
    }
}
