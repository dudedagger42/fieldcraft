//! Driver & Seegmiller's backward-facing step at zero upper-wall angle
//! (`tests/oracle/ddes-backstep-driver-seegmiller`, the acceptance case of slice 15b).
//!
//! The model, lengths in units of the step height H (Re_H = U H / ν with U the inlet speed):
//! - x from −x_up to x_down, z from −Δ to the roof (9 H), y periodic: one cell for a 2D run, a
//!   span S for the 3D hybrid run;
//! - a uniform velocity inlet at x_min above the step and a pressure outlet with convective
//!   outflow at x_max; the roof is a no-slip box face;
//! - the step, x < 0 and z < H, is solid, its top split into strips H/2 long for x > −8 H (local
//!   wall force near the reference station x = −4 H) and one block upstream of them;
//! - the floor downstream, x > 0, is strips of solid cells H/8 long covering the bottom cell row,
//!   so their halfway wall lies at z = 0 and each strip reports its own wall force.
//!
//! The incoming boundary layers grow from the inlet along the step's top and the roof; x_up is
//! chosen so that Re_θ ≈ 5000 and δ ≈ 1.5 H at x = −4 H, which the run reports. U_ref is the speed
//! at mid-channel above the step (z = 5 H under the 9 H roof) at x = −4 H; C_f and C_p are formed with it, and the
//! time-mean floor shear's zero crossing after its minimum is the reattachment x_r
//! (common::cases::reattachment).
//!
//! **Statistics.** After `startup` flow-throughs, each block of `block` samples gives a mean
//! shear profile and its reattachment; the run stops when the reattachments' block means pass the
//! shared acceptance of the LES studies (`fieldcraft_lbm::statistics::Blocks::acceptance`, #796),
//! or at the step cap, reported inconclusive. A steady run passes once its reattachment stops
//! moving and 20 blocks are in.
//!
//! The case runs for hours: [`driver_seegmiller`] takes a configured [`Step`]
//! ([`Step::driver_seegmiller`] gives the case's own), and each model gets its own ignored
//! `#[test]` naming it (none yet: slice 15b adds the first). The pipeline check runs in the heavy
//! job.

mod common;

use common::cases::*;
use fieldcraft_lbm::statistics::Blocks;
use common::*;
use fieldcraft_lbm::{Flow, Limits, Obstacle};
use fieldcraft_model::{BoxFaces, Collision, FaceCondition, Outflow, Procedure, Turbulence};

/// The step model's parameters (lengths in units of H, inlet speed 1).
#[derive(Debug, Clone)]
struct Step {
    re_h: f64,
    cells_per_h: usize,
    lattice_velocity: f64,
    upstream: f64,
    downstream: f64,
    roof: f64,
    /// The span in H; 0 for one cell.
    span: f64,
    collision: Collision,
    turbulence: Option<Turbulence>,
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

/// What a run measured.
#[derive(Debug, Clone)]
struct Run {
    /// The mean shear profile's stations: (x, C_f) along the step's top, then the floor.
    cf: Vec<(f64, f64)>,
    /// The floor's C_p at the strip centres, from the first fluid cell row.
    cp: Vec<(f64, f64)>,
    reattachment: Blocks,
    /// Blocks whose mean floor shear had no reattachment (left out of `reattachment`).
    missing: usize,
    /// Re_θ, δ₉₉ (in H) and U_ref (in units of the inlet speed) at x = −4 H, at the end.
    inflow: (f64, f64, f64),
    steps: u64,
    settled: bool,
}

impl Step {
    fn spacing(&self) -> f64 {
        1.0 / self.cells_per_h as f64
    }

    fn span_cells(&self) -> usize {
        ((self.span * self.cells_per_h as f64).round() as usize).max(1)
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

    fn build(&self) -> (Flow, Vec<Strip>) {
        let h = self.spacing();
        let width = self.span_cells() as f64 * h;
        let mut d = domain([-self.upstream, 0.0, -h], [self.downstream, width, self.roof], h);
        d.faces = BoxFaces {
            x_min: FaceCondition::Velocity { velocity: [1.0, 0.0, 0.0], profile: Default::default() },
            x_max: FaceCondition::Pressure { pressure: 0.0, outflow: Outflow::Convective },
            y_min: FaceCondition::Periodic {},
            y_max: FaceCondition::Periodic {},
            z_min: FaceCondition::Wall {},
            z_max: FaceCondition::Wall {},
        };
        d.initial_velocity = [1.0, 0.0, 0.0];
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
            sigma: None,
            collision: Some(self.collision),
            turbulence: self.turbulence,
            steady: None,
        };
        let m = model(1.0, 1.0 / self.re_h, d, p);
        let limits = Limits { max_cell_updates: u64::MAX, max_steps: u64::MAX, ..Limits::default() };
        let setup = fieldcraft_lbm::Setup::new(&m, None, &obstacles, &limits).unwrap();
        (Flow::new(&setup, &obstacles).unwrap(), strips)
    }

    fn run(&self) -> Run {
        let (mut flow, strips) = self.build();
        let grid = flow.grid().clone();
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
        let middle = 0.5 * (1.0 + self.roof);
        let reference = |flow: &Flow| span_mean(flow, column(-4.0), row(middle));
        let flow_through = ((self.upstream + self.downstream) / h / self.lattice_velocity) as u64;
        let startup = (self.startup * flow_through as f64) as u64;
        let areas: Vec<f64> = strips.iter().map(|s| ((s.x.1 - s.x.0) / h).round() * ny as f64).collect();
        let (mut sums, mut block_sums, mut in_block) = (vec![0.0; strips.len()], vec![0.0; strips.len()], 0usize);
        let mut total_samples = 0usize;
        let mut blocks = Blocks::new(1, 0).unwrap();
        // Blocks whose mean shear has no reattachment (still attached, or not yet back).
        let mut missing = 0usize;
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
            // One sample: the mean shear over the interval.
            for s in 0..strips.len() {
                let t = interval[s] / self.sample as f64;
                sums[s] += t;
                block_sums[s] += t;
                interval[s] = 0.0;
            }
            assert!(sums.iter().all(|s| s.is_finite()), "the wall shear is not finite after {steps} steps");
            total_samples += 1;
            in_block += 1;
            if in_block == self.block {
                match reattachment(&floor_x, &floor_of(&block_sums)) {
                    Some(x_r) => blocks.push(x_r).unwrap(),
                    None => missing += 1,
                }
                block_sums.iter_mut().for_each(|v| *v = 0.0);
                in_block = 0;
                if blocks.acceptance().accepted {
                    settled = true;
                    break;
                }
            }
        }
        println!("{} fluid cells, {steps} steps: {:.1} M cell updates/s", flow.fluid_cells(), flow.fluid_cells() as f64 * steps as f64 / clock.elapsed().as_secs_f64() / 1e6);
        // The mean profiles, formed with U_ref and ρ at the reference station.
        let (rho_ref, u_ref) = reference(&flow);
        let q = 0.5 * rho_ref * u_ref * u_ref;
        let n = total_samples.max(1) as f64;
        let cf: Vec<(f64, f64)> = strips.iter().zip(&sums).map(|(s, t)| (0.5 * (s.x.0 + s.x.1), t / n / q)).collect();
        let p_ref = rho_ref / 3.0;
        let cp: Vec<(f64, f64)> = floor_x.iter().map(|&x| (x, (span_mean(&flow, column(x), 1).0 / 3.0 - p_ref) / q)).collect();
        // The incoming layer on the step's top at x = −4 H, up to mid-channel.
        let i = column(-4.0);
        let u: Vec<f64> = (row(1.0) + 1..=row(middle)).map(|k| span_mean(&flow, i, k).1).collect();
        let theta = momentum_thickness(&u, h, u_ref);
        let delta = u.iter().position(|&v| v >= 0.99 * u_ref).map_or(f64::NAN, |k| (k as f64 + 0.5) * h);
        // U_ref in units of the inlet speed (the lattice speeds are scaled by U_lat).
        let u_ref_inlet = u_ref / self.lattice_velocity;
        Run { cf, cp, reattachment: blocks, missing, inflow: (u_ref_inlet * theta * self.re_h, delta, u_ref_inlet), steps, settled }
    }
}

fn report(step: &Step, r: &Run) -> Score {
    let experiment = experiment();
    println!("{} steps, settled: {}", r.steps, r.settled);
    println!("inflow at x = −4 H: Re_θ {:.0} (5000), δ₉₉ {:.2} H (≈ 1.5), U_ref {:.4}, Re_H {:.0}", r.inflow.0, r.inflow.1, r.inflow.2, r.inflow.2 * step.re_h);
    println!(
        "reattachment blocks: {} ({} without one), mean {:.3} H, 95 % half-width {:.3}, lag-1 {:.2}",
        r.reattachment.means().len(),
        r.missing,
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
            "reattachment" => r.reattachment.mean(),
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
    let rows = |key: &str| v[key]["rows"].as_array().unwrap().iter().map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect::<Vec<f64>>()).collect::<Vec<_>>();
    (rows("floor_cf").iter().map(|r| (r[0], r[1], r[2])).collect(), rows("floor_cp").iter().map(|r| (r[0], r[2])).collect())
}

/// The model builds, every strip has its force group, and a short laminar run reads finite floor
/// shear and an inflow profile (no reference: the pipeline only).
#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn the_backstep_model_reads_its_floor_and_inflow() {
    let step = Step {
        re_h: 50.0,
        cells_per_h: 8,
        lattice_velocity: 0.05,
        upstream: 6.0,
        downstream: 10.0,
        roof: 3.0,
        span: 0.0,
        collision: Collision::Trt,
        turbulence: None,
        sample: 20,
        block: 5,
        startup: 0.1,
        max_steps: 2000,
    };
    let r = step.run();
    let score = report(&step, &r);
    assert_eq!(score.gates.len(), 2);
    assert_eq!(r.cf.len(), step.strips().len());
    assert!(r.cf.iter().all(|(_, c)| c.is_finite()) && r.cp.iter().all(|(_, c)| c.is_finite()));
    assert!(r.inflow.2 > 0.5 && r.inflow.0 > 0.0, "{:?}", r.inflow);
    // Laminar flow over a step separates behind it: the floor shear is negative just downstream.
    assert!(r.cf.iter().any(|&(x, c)| x > 0.0 && x < 2.0 && c < 0.0), "{:?}", &r.cf);
}

impl Step {
    /// Driver & Seegmiller's conditions (Re_H 36 000, roof at 9 H, 70 H of inflow development,
    /// 30 H downstream) at 16 cells per H, 2D (one cell of span); `span` in H for the hybrid run.
    /// U = 0.1 in lattice units keeps τ − ½ = 3 U N / Re_H = 1.3 × 10⁻⁴ above the wall model's
    /// floor of 10⁻⁴, and puts the first cell upstream at y⁺ ≈ 43.
    #[allow(dead_code)]
    fn driver_seegmiller(turbulence: Turbulence, span: f64) -> Step {
        Step {
            re_h: 36_000.0,
            cells_per_h: 16,
            lattice_velocity: 0.1,
            upstream: 70.0,
            downstream: 30.0,
            roof: 9.0,
            span,
            collision: Collision::Mrt,
            turbulence: Some(turbulence),
            sample: 500,
            block: 10,
            startup: 2.0,
            max_steps: 2_000_000,
        }
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
#[allow(dead_code)]
fn driver_seegmiller(step: &Step) -> (Run, Score) {
    println!("Driver & Seegmiller: {step:?}");
    let r = step.run();
    let score = report(step, &r);
    (r, score)
}
