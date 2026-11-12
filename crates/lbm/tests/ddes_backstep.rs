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
use fieldcraft_lbm::{Flow, Limits, Obstacle};
use fieldcraft_model::{AbsorbingLayer, BoxFaces, Collision, FaceCondition, HrrSigma, Outflow, Procedure, Turbulence, WallModel};

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
    let window = recent.get(recent.len().checked_sub(STEADY_BLOCKS)?..)?;
    let strips = window.first()?.1.len();
    if window.iter().any(|(x, t)| !x.is_finite() || t.len() != strips || t.iter().any(|v| !v.is_finite())) {
        return None;
    }
    let x_r: Vec<f64> = window.iter().map(|(x, _)| *x).collect();
    let x_r = spread(&x_r, (x_r.iter().sum::<f64>() / x_r.len() as f64).abs());
    let scale = window.iter().flat_map(|(_, t)| t).fold(0.0, |a: f64, v| a.max(v.abs()));
    let shear = (0..strips).map(|s| spread(&window.iter().filter_map(|(_, t)| t.get(s).copied()).collect::<Vec<f64>>(), scale)).fold(0.0, f64::max);
    (x_r < STEADY_X_R && shear < STEADY_SHEAR).then_some((x_r, shear))
}

/// Lengths within this of each other are equal (grid coordinates are sums of cell sizes).
const LENGTH_TOLERANCE: f64 = 1e-9;

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
        if !(self.downstream.is_finite() && self.downstream >= 30.0 && self.on_faces(self.downstream)) {
            why.push(format!("an outlet {} H downstream must be at least 30 H and whole cells", self.downstream));
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
        let limits = Limits { max_cell_updates: u64::MAX, max_steps: u64::MAX, ..Limits::default() };
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

    fn run(&self) -> Run {
        let (mut flow, strips, _) = self.build();
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
        let floor_x: Vec<f64> = strips.iter().filter(|s| s.floor).map(|s| 0.5 * (s.x.0 + s.x.1)).collect();
        let floor_of = |v: &[f64]| strips.iter().zip(v).filter(|(s, _)| s.floor).map(|(_, &t)| t).collect::<Vec<f64>>();
        let mut rule = None;
        // A steady (RANS, no DDES) run stops by the steady rule; a scale-resolving one by `Blocks`.
        let steady_run = matches!(self.turbulence, Some(Turbulence::SpalartAllmaras { ddes: None, .. }));
        // The last completed blocks' (x_r, floor shear), for the steady rule; a refused block
        // empties it, so the window is consecutive blocks that each reattach.
        let mut recent: Vec<(f64, Vec<f64>)> = Vec::new();
        let clock = std::time::Instant::now();
        // Every sample so far, kept for the restart file: a resumed run replays them into fresh
        // statistics, so it ends where an unbroken run would.
        let mut samples: Vec<Vec<f64>> = Vec::new();
        // One sample into the statistics; the rule that settles the run, if one does. Each
        // completed block prints a line: its reattachment, the running mean and, when given, the
        // inlet's largest |ρ − 1| (its first four columns).
        let take = |sample: &[f64],
                    steps: u64,
                    inlet: Option<f64>,
                    shear: &mut Shear,
                    blocks: &mut Blocks,
                    missing: &mut (usize, Option<NoReattachment>),
                    recent: &mut Vec<(f64, Vec<f64>)>| {
            if !shear.push(sample) {
                return None;
            }
            let floor = floor_of(&shear.last_block());
            let last = reattachment(&floor_x, &floor);
            match &last {
                Ok(x_r) => {
                    blocks.push(*x_r).unwrap();
                    recent.push((*x_r, floor));
                    if recent.len() > STEADY_BLOCKS {
                        recent.remove(0);
                    }
                }
                Err(why) => {
                    *missing = (missing.0 + 1, Some(why.clone()));
                    recent.clear();
                }
            }
            println!(
                "step {steps}: block {} x_r {last:?}, mean {:.3} H over {} blocks ({} refused), inlet |ρ − 1| {}, {:.0} s",
                shear.blocks(),
                blocks.mean(),
                blocks.means().len(),
                missing.0,
                inlet.map_or("(replayed)".into(), |v| format!("{v:.2e}")),
                clock.elapsed().as_secs_f64()
            );
            if steady_run {
                steady(recent).map(|(x_r, spread)| Settled::Steady { block: shear.blocks(), x_r, shear: spread })
            } else {
                blocks.acceptance().accepted.then_some(Settled::Blocks)
            }
        };
        if let Some(path) = self.checkpoint.as_ref().filter(|p| p.exists()) {
            let (at, kept) = load(path, &mut flow);
            steps = at;
            println!("resumed {} at step {steps} ({} samples)", path.display(), kept.len());
            let first = startup + self.sample;
            for (n, sample) in kept.iter().enumerate() {
                let now = take(sample, first + n as u64 * self.sample, None, &mut shear, &mut blocks, &mut missing, &mut recent);
                rule = rule.or(now);
            }
            samples = kept;
        }
        while steps < self.max_steps && rule.is_none() {
            flow.step().unwrap();
            steps += 1;
            if steps <= startup {
                if let Some(path) = self.checkpoint.as_ref().filter(|_| steps.is_multiple_of(CHECKPOINT_STEPS)) {
                    save(path, &mut flow, steps, &samples);
                }
                continue;
            }
            let forces = flow.forces();
            for (s, g) in groups.iter().enumerate() {
                interval[s] += g.iter().map(|&g| forces.get(g).map_or(f64::NAN, |f| f[0])).sum::<f64>() / areas[s];
            }
            if steps % self.sample != 0 {
                continue;
            }
            // One sample: each strip's mean shear over the interval.
            let sample: Vec<f64> = interval.iter().map(|t| t / self.sample as f64).collect();
            assert!(sample.iter().all(|s| s.is_finite()), "the wall shear is not finite after {steps} steps");
            interval.iter_mut().for_each(|v| *v = 0.0);
            let inlet = (0..4.min(nx))
                .flat_map(|i| (0..nz).flat_map(move |k| (0..ny).map(move |j| [i, j, k])))
                .filter_map(|c| flow.moments(grid.index(c)).map(|m| (m.0 - 1.0).abs()))
                .fold(0.0, f64::max);
            rule = take(&sample, steps, Some(inlet), &mut shear, &mut blocks, &mut missing, &mut recent);
            samples.push(sample);
            // Sample boundaries carry no partial interval, so the file holds the whole state.
            if let Some(path) = self.checkpoint.as_ref().filter(|_| steps.is_multiple_of(CHECKPOINT_STEPS)) {
                save(path, &mut flow, steps, &samples);
            }
        }
        println!(
            "{} fluid cells, {steps} steps: {:.1} M cell updates/s",
            flow.fluid_cells(),
            flow.fluid_cells() as f64 * steps as f64 / clock.elapsed().as_secs_f64() / 1e6
        );
        // The mean profiles, formed with U_ref and ρ at the reference station.
        let (rho_ref, u_ref) = reference(&flow);
        let q = 0.5 * rho_ref * u_ref * u_ref;
        let mean = shear.mean();
        let x_r = reattachment(&floor_x, &floor_of(&mean));
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
        Run {
            cf,
            cp,
            x_r,
            crossing,
            reattachment: blocks,
            missing,
            inflow: (u_ref_inlet * theta * self.re_h, delta, u_ref_inlet),
            steps,
            settled: rule.is_some(),
            rule,
        }
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
}

/// Write the restart file: to `path` with `.tmp` added, then renamed over `path`.
fn save(path: &std::path::Path, flow: &mut Flow, steps: u64, samples: &[Vec<f64>]) {
    use std::io::Write;
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
    if let Some(w) = a.write.as_mut() {
        w.flush().unwrap();
    }
    drop(a);
    std::fs::rename(&tmp, path).unwrap();
}

/// Read a restart file into `flow`: the step count and the samples.
fn load(path: &std::path::Path, flow: &mut Flow) -> (u64, Vec<Vec<f64>>) {
    let mut a = Restart { read: Some(std::io::BufReader::new(std::fs::File::open(path).unwrap())), write: None };
    flow.checkpoint(&mut a).unwrap();
    let steps = a.u64(0);
    let samples = (0..a.u64(0)).map(|_| (0..a.u64(0)).map(|_| f64::from_bits(a.u64(0))).collect()).collect();
    (steps, samples)
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
    let cells_per_h = std::env::var("FIELDCRAFT_STEP_CELLS_PER_H").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
    let extra: f64 = std::env::var("FIELDCRAFT_STEP_EXTRA_UPSTREAM").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let case = Step::driver_seegmiller(sa, 0.0);
    Step { cells_per_h, upstream: case.upstream + extra, ..case }
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
    // The gate reading: x_r is the wall (y = 0) value, the linear extrapolation of the u = 0
    // crossings of rows 1 and 2 to y = 0; the uncertainty band is that against the row-1 value and
    // the extrapolation from rows 2 and 3. No variant is picked by its nearness to the experiment.
    let first_row = if flow.on_node_bounce() { 2 } else { 1 };
    let rows: Vec<Result<(f64, f64), String>> = (0..6).map(|n| crossing_at(&flow, &step, first_row + n)).collect();
    for (n, r) in rows.iter().enumerate() {
        match r {
            Ok((y, x)) => println!("u = 0 crossing, row {} (y = {y:.4} H): x {x:.3} H", n + 1),
            Err(why) => println!("u = 0 crossing, row {}: none ({why})", n + 1),
        }
    }
    if let [Ok(r1), Ok(r2), Ok(r3), ..] = rows.as_slice() {
        let (wall, row1, from_23) = (intercept(&[*r1, *r2], 1), r1.1, intercept(&[*r2, *r3], 1));
        let (lo, hi) = (wall.min(row1).min(from_23), wall.max(row1).max(from_23));
        println!(
            "x_r at the wall (y = 0, rows 1–2 extrapolated): {wall:.3} H; band: row-1 value {row1:.3}, rows 2–3 extrapolated {from_23:.3}; {lo:.3} … {hi:.3} H (experiment 6.26 ± 0.10); wall law {}",
            study_wall_law().1
        );
    }
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
fn sa_rans_on_han_s_wall_reattaches_where_measured() {
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
