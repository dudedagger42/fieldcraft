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
    /// The reattachment of each completed block's mean floor shear, for the stop rule.
    reattachment: Blocks,
    /// Blocks whose mean floor shear was refused (left out of `reattachment`), and the last refusal.
    missing: (usize, Option<NoReattachment>),
    /// Re_θ, δ₉₉ (in H) and U_ref (in units of the inlet speed) at x = −4 H, at the end.
    inflow: (f64, f64, f64),
    steps: u64,
    settled: bool,
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
        let mut settled = false;
        let clock = std::time::Instant::now();
        while steps < self.max_steps {
            flow.step().unwrap();
            steps += 1;
            if steps <= startup {
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
            if shear.push(&sample) {
                let last = reattachment(&floor_x, &floor_of(&shear.last_block()));
                match &last {
                    Ok(x_r) => blocks.push(*x_r).unwrap(),
                    Err(why) => missing = (missing.0 + 1, Some(why.clone())),
                }
                // A long run says where it is once per block: the block's reattachment, the
                // running mean, and the inlet's largest |ρ − 1| (its first four columns).
                let inlet = (0..4.min(nx))
                    .flat_map(|i| (0..nz).flat_map(move |k| (0..ny).map(move |j| [i, j, k])))
                    .filter_map(|c| flow.moments(grid.index(c)).map(|m| (m.0 - 1.0).abs()))
                    .fold(0.0, f64::max);
                println!(
                    "step {steps}: block {} x_r {last:?}, mean {:.3} H over {} blocks ({} refused), inlet |ρ − 1| {inlet:.2e}, {:.0} s",
                    shear.blocks(),
                    blocks.mean(),
                    blocks.means().len(),
                    missing.0,
                    clock.elapsed().as_secs_f64()
                );
                if blocks.acceptance().accepted {
                    settled = true;
                    break;
                }
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
        Run { cf, cp, x_r, reattachment: blocks, missing, inflow: (u_ref_inlet * theta * self.re_h, delta, u_ref_inlet), steps, settled }
    }
}

fn report(step: &Step, r: &Run) -> Score {
    let experiment = experiment();
    println!("{} steps, settled: {}", r.steps, r.settled);
    println!(
        "inflow at x = −4 H: Re_θ {:.0} (5000), δ₉₉ {:.2} H (≈ 1.5), U_ref {:.4}, Re_H {:.0}",
        r.inflow.0,
        r.inflow.1,
        r.inflow.2,
        r.inflow.2 * step.re_h
    );
    println!("x_r from the time-mean floor shear: {:?}", r.x_r);
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

/// Spalart–Allmaras RANS on Han's wall in 2D (one cell of span), the case's grid and inlet: the
/// first model run toward x_r / H. Hours on 5 threads:
/// `FIELDCRAFT_THREADS=5 cargo test --release -p fieldcraft-lbm --test ddes_backstep sa_rans_on_han_s_wall -- --ignored --nocapture`.
#[test]
#[ignore = "hours: tests/oracle/ddes-backstep-driver-seegmiller/README.md"]
fn sa_rans_on_han_s_wall_reattaches_where_measured() {
    let sa = Turbulence::SpalartAllmaras { prandtl: 0.9, wall_model: Some(WallModel::SpaldingBounce), inflow_ratio: 3.0, ddes: None };
    let steps = std::env::var("FIELDCRAFT_STEP_MAX_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(400_000);
    let (run, score) = driver_seegmiller(&Step { max_steps: steps, ..Step::driver_seegmiller(sa, 0.0) });
    assert!(run.settled && score.passes(), "x_r {:?}, settled {}", run.x_r, run.settled);
}
