//! The turbulent flat plate at zero pressure gradient, NASA TMR's 2DZP case
//! (`tests/oracle/sa-flat-plate-2dzp`, the acceptance case of slice 15a), and the laminar plate
//! that checks its measuring pipeline against Blasius.
//!
//! The model, lengths in units of the reference length (Re per unit length = U / ν):
//! - x from −x_up to the end of the plate, z from −Δ to H, one cell periodic in y (2D);
//! - a uniform velocity inlet at x_min, a slip floor (the box face at z = −Δ) ahead of the plate,
//!   a pressure outlet with convective outflow at x_max, and the top face either slip or a
//!   convective pressure face (a far field the displaced flow can leave through);
//! - the plate, 0 ≤ x ≤ L, as strips of solid cells covering the bottom cell row, one obstacle
//!   each, so the momentum exchange gives each strip its own wall force: the halfway wall lies at
//!   z = 0, a one-cell step above the slip floor at the leading edge.
//!
//! C_f of a strip is its x wall force per wetted area over ½ ρ U_e², U_e the fastest speed in the
//! column above the strip's centre; Re_θ = U_e θ / ν, θ integrated up to it. Each is a time mean
//! over two periods of the domain's slowest acoustic mode, which the velocity inlet and pressure
//! outlet trap and the plate's C_f follows. The gate interpolates C_f at each reference Re_θ between the strips that
//! bracket it (common::cases).
//!
//! On Han's on-node wall (WL1) the strips' top row is the wall's node row, so the wall sits at
//! z = +Δ/2 and the leading edge 1.5 Δ above the slip floor. θ and δ* take the interval from the wall
//! to the first fluid node from the wall law (`WallInterval`) and report the trapezoid beside it.
//! Re_θ for the gate is the law's. A strip's force includes its nodes' share, and its request is
//! the law's (`wanted`).
//!
//! The turbulent case runs for hours: [`tmr_2dzp`] takes the turbulence model and grid, and each
//! model gets its own ignored `#[test]` naming it (none yet: slice 15a's Spalart–Allmaras adds the
//! first). The laminar check is a study run by hand (docs/verification.md, "Long studies"); the
//! heavy job runs its first flow-through against frozen values.

mod common;

use common::cases::*;
use common::*;
use fieldcraft_lbm::{Flow, Limits, Obstacle, Setup};
use fieldcraft_model::{AbsorbingLayer, BoxFaces, Collision, FaceCondition, Outflow, Procedure, Turbulence};

/// The plate model's parameters (lengths in units of the reference length, U = 1).
#[derive(Debug, Clone)]
struct Plate {
    /// Re per unit length, U / ν.
    re: f64,
    /// The plate runs from x = 0 to x = `length`; the domain from −`upstream`.
    length: f64,
    upstream: f64,
    /// The domain's height above the plate.
    height: f64,
    /// Grid spacing Δ.
    spacing: f64,
    /// U in lattice units.
    lattice_velocity: f64,
    /// Strip length (a whole number of cells).
    strip: f64,
    /// Slip (`false`) or a convective pressure face (`true`) on top.
    pressure_top: bool,
    collision: Collision,
    turbulence: Option<Turbulence>,
    max_steps: u64,
    sample: u64,
    /// The x range whose strips decide settling: the gated range (the first and last strips' end
    /// faces carry the leading edge's stagnation pressure and the outflow's, not shear).
    settle: (f64, f64),
    /// When the time-mean C_f of those strips has settled (averaged over two acoustic periods,
    /// [`Plate::acoustic_period`]).
    settling: Settling,
    /// An absorbing layer at the inlet and the outlet (none in the gated cases).
    absorbing: Option<AbsorbingLayer>,
}

/// One strip's time-mean reading.
#[derive(Debug, Clone, Copy)]
struct Station {
    x: f64,
    cf: f64,
    /// Re_θ from θ's law entry (the gate's), and from its trapezoid entry.
    re_theta: f64,
    re_theta_trapezoid: f64,
    /// The momentum and displacement thickness (lattice units), [law, trapezoid]: on Han's on-node
    /// wall the interval from the wall to the first node is integrated from the wall law
    /// (`WallInterval`) or by the trapezoid from u = 0; on a halfway wall both entries are the
    /// midpoint sum.
    theta: [f64; 2],
    delta_star: [f64; 2],
    /// The wall model's requested C_f, when it reports one.
    requested: Option<f64>,
    edge: f64,
}

impl Plate {
    fn cells(&self, length: f64) -> usize {
        (length / self.spacing).round() as usize
    }

    fn time_step(&self) -> f64 {
        self.lattice_velocity * self.spacing
    }

    /// ν in lattice units.
    fn lattice_nu(&self) -> f64 {
        (1.0 / self.re) * self.time_step() / (self.spacing * self.spacing)
    }

    /// The period of the domain's slowest acoustic mode, in steps: a quarter wave between the
    /// velocity inlet (a closed end for sound) and the pressure outlet (an open end), 4 L_x / c_s
    /// with c_s = 1/√3, slowed by the mean flow by 1/(1 − Ma²). The laminar plate rings in it
    /// without visible decay (C_f ±3 %): measured 1785 steps on a 250-cell domain and 2500 on a
    /// 350-cell one, against 1786 and 2500 from this formula.
    fn acoustic_period(&self) -> f64 {
        let cells = self.cells(self.upstream + self.length) as f64;
        let mach2 = 3.0 * self.lattice_velocity * self.lattice_velocity;
        4.0 * cells * 3f64.sqrt() / (1.0 - mach2)
    }

    /// The strips' x ranges.
    fn strips(&self) -> Vec<(f64, f64)> {
        let per = self.cells(self.strip).max(1);
        let total = self.cells(self.length);
        (0..total.div_ceil(per))
            .map(|s| (s * per, ((s + 1) * per).min(total)))
            .map(|(a, b)| (a as f64 * self.spacing, b as f64 * self.spacing))
            .collect()
    }

    fn build(&self) -> (Setup, Flow, Vec<(f64, f64)>) {
        let h = self.spacing;
        let mut d = domain([-self.upstream, 0.0, -h], [self.length, h, self.height], h);
        d.faces = BoxFaces {
            x_min: FaceCondition::Velocity { velocity: [1.0, 0.0, 0.0], profile: Default::default() },
            x_max: FaceCondition::Pressure { pressure: 0.0, outflow: Outflow::Convective },
            y_min: FaceCondition::Periodic {},
            y_max: FaceCondition::Periodic {},
            z_min: FaceCondition::Slip {},
            z_max: if self.pressure_top { FaceCondition::Pressure { pressure: 0.0, outflow: Outflow::Convective } } else { FaceCondition::Slip {} },
        };
        d.initial_velocity = [1.0, 0.0, 0.0];
        d.absorbing = self.absorbing;
        let strips = self.strips();
        let obstacles: Vec<Obstacle> = strips
            .iter()
            .enumerate()
            .map(|(s, &(a, b))| obstacle(&format!("Strip{s}"), cuboid([a, -h, -2.0 * h], [b, 2.0 * h, 0.0]), &CUBOID_FACES))
            .collect();
        d.obstacles = obstacles.iter().map(|o| o.body.clone()).collect();
        let dt = self.time_step();
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
        let m = model(1.0, 1.0 / self.re, d, p);
        let limits = Limits { max_cell_updates: u64::MAX, max_steps: u64::MAX, ..Limits::default() };
        let setup = Setup::new(&m, None, &obstacles, &limits).unwrap();
        let flow = Flow::new(&setup, &obstacles).unwrap();
        (setup, flow, strips)
    }

    /// Run until the time-mean C_f of the strips in `settle` has settled (`settling`, with
    /// averaging windows of two acoustic periods rounded to whole samples) or the step cap; each
    /// strip's means over the last window, whether they settled, and the steps taken.
    fn run(&self) -> (Vec<Station>, bool, u64) {
        let (_, mut flow, strips) = self.build();
        let grid = *flow.grid();
        // Every force group of each strip (its shear is its whole x force; see body_groups).
        let groups: Vec<Vec<usize>> = (0..strips.len()).map(|s| body_groups(&flow, &format!("Strip{s}"))).collect();
        let nu = self.lattice_nu();
        let [nx, _, nz] = grid.cells;
        let column = |x: f64| ((x + self.upstream) / self.spacing).floor().clamp(0.0, nx as f64 - 1.0) as usize;
        let on_node = flow.on_node_bounce();
        let gated: Vec<usize> =
            (0..strips.len()).filter(|&s| (self.settle.0..=self.settle.1).contains(&(0.5 * (strips[s].0 + strips[s].1)))).collect();
        let window = ((2.0 * self.acoustic_period() / self.sample as f64).round() as usize).max(1);
        // Each strip's reading per sample interval, and the gated strips' C_f.
        let mut samples: Vec<Vec<Station>> = Vec::new();
        let mut history: Vec<Vec<f64>> = Vec::new();
        let (mut carried, mut requested) = (vec![0.0; strips.len()], vec![0.0; strips.len()]);
        let mut any_requested = false;
        let mut steps = 0;
        let flow_through = (self.upstream + self.length) / self.spacing / self.lattice_velocity;
        let clock = std::time::Instant::now();
        let throughput = |steps: u64, flow: &Flow| {
            println!(
                "{} fluid cells, {steps} steps: {:.1} M cell updates/s",
                flow.fluid_cells(),
                flow.fluid_cells() as f64 * steps as f64 / clock.elapsed().as_secs_f64() / 1e6
            )
        };
        // Each strip's mean over the last window of samples.
        let mean = |samples: &[Vec<Station>]| -> Vec<Station> {
            let rows = &samples[samples.len().saturating_sub(window)..];
            let n = rows.len() as f64;
            let avg = |f: &dyn Fn(&Station) -> f64, s: usize| rows.iter().map(|r| f(&r[s])).sum::<f64>() / n;
            let Some(first) = rows.first() else { return Vec::new() };
            (0..first.len())
                .map(|s| Station {
                    x: first[s].x,
                    cf: avg(&|t| t.cf, s),
                    re_theta: avg(&|t| t.re_theta, s),
                    re_theta_trapezoid: avg(&|t| t.re_theta_trapezoid, s),
                    theta: [avg(&|t| t.theta[0], s), avg(&|t| t.theta[1], s)],
                    delta_star: [avg(&|t| t.delta_star[0], s), avg(&|t| t.delta_star[1], s)],
                    requested: first[s].requested.map(|_| avg(&|t| t.requested.unwrap_or(f64::NAN), s)),
                    edge: avg(&|t| t.edge, s),
                })
                .collect()
        };
        while steps < self.max_steps {
            flow.step().unwrap();
            steps += 1;
            let forces = flow.forces();
            for (s, g) in groups.iter().enumerate() {
                carried[s] += g.iter().map(|&g| forces.get(g).map_or(f64::NAN, |f| f[0])).sum::<f64>();
            }
            if let Some(w) = flow.walls() {
                for (_, _, g, stress) in w.stresses() {
                    if let Some(s) = groups.iter().position(|x| x.contains(&g)) {
                        requested[s] += stress[0];
                        any_requested = true;
                    }
                }
            }
            // Han's nodes: the stress the law asks for (`wanted`; the applied `stress` except at a
            // strip's end nodes, along the axes they can't carry).
            if let Some(nodes) = flow.wall_nodes() {
                for n in nodes.nodes() {
                    if let Some(s) = groups.iter().position(|x| x.contains(&(n.group as usize))) {
                        requested[s] += n.wanted[0];
                        any_requested = true;
                    }
                }
            }
            if steps % self.sample != 0 {
                continue;
            }
            let stations: Vec<Station> = strips
                .iter()
                .enumerate()
                .map(|(s, &(a, b))| {
                    let cells = ((b - a) / self.spacing).round();
                    let i = column(0.5 * (a + b));
                    // The edge is the profile's fastest cell below 80 % of the height (the cells
                    // next to a pressure top are not free stream, and over a slip top the outer flow
                    // is not uniform); θ integrates up to it, where every term is non-negative. The
                    // profile starts at the first fluid cell: row 1 over a halfway wall (at z = 0,
                    // half a cell below it), row 2 over Han's node row (row 1, the wall itself, at
                    // z = +Δ/2, a whole cell below it).
                    let first = if on_node { 2 } else { 1 };
                    let profile: Vec<f64> =
                        (first..(nz * 4 / 5).max(first + 1)).map(|k| flow.moments(grid.index([i, 0, k])).map_or(f64::NAN, |(_, u)| u[0])).collect();
                    let top = (0..profile.len()).max_by(|&a, &b| profile[a].total_cmp(&profile[b])).unwrap_or(0);
                    let u = &profile[..=top.min(profile.len().saturating_sub(1))];
                    let edge = u.last().copied().unwrap_or(f64::NAN);
                    let q = 0.5 * edge * edge;
                    let n = self.sample as f64 * cells;
                    let (theta, delta_star) = thicknesses(u, edge, on_node, nu);
                    Station {
                        x: 0.5 * (a + b),
                        cf: carried[s] / n / q,
                        re_theta: edge * theta[0] / nu,
                        re_theta_trapezoid: edge * theta[1] / nu,
                        theta,
                        delta_star,
                        requested: any_requested.then(|| requested[s] / n / q),
                        edge,
                    }
                })
                .collect();
            carried.iter_mut().chain(requested.iter_mut()).for_each(|v| *v = 0.0);
            assert!(stations.iter().all(|s| s.cf.is_finite()), "C_f is not finite after {steps} steps: {stations:?}");
            history.push(gated.iter().map(|&s| stations[s].cf).collect());
            samples.push(stations);
            if self.settling.settled(&history, window, steps as f64 / flow_through) {
                throughput(steps, &flow);
                return (mean(&samples), true, steps);
            }
        }
        throughput(steps, &flow);
        if let Some(start) = history.len().checked_sub(2 * window)
            && let (Some(then), Some(now)) = (column_means(&history[start..start + window]), column_means(&history[start + window..]))
        {
            let drift = now.iter().zip(&then).map(|(a, b)| ((a - b) / a).abs()).fold(0.0f64, f64::max);
            println!("not settled: the largest relative change of the {window}-sample mean C_f is {drift:.2e}");
        }
        (mean(&samples), false, steps)
    }
}

/// The momentum and displacement thickness of a profile `u` sampled one cell apart up to its edge
/// speed `edge` (lattice units), each as [law, trapezoid].
///
/// Over a halfway wall the first sample is half a cell from the wall and both entries are the
/// midpoint sum Σ f(u_j). Over Han's on-node wall the first sample is a whole cell above the wall,
/// where u = 0: from the first sample up it is the trapezoid, Σ f(u_j) − ½ f(u_1) (f(U_e) = 0 at
/// the top), and the interval below it is added from the wall law or by its own trapezoid
/// (`WallInterval`, board WL3 P1). For δ* the integrand is 1 at the wall, so its trapezoid share is
/// not ½ f(u_1).
fn thicknesses(u: &[f64], edge: f64, on_node: bool, nu: f64) -> ([f64; 2], [f64; 2]) {
    let theta = momentum_thickness(u, 1.0, edge);
    let delta_star = displacement_thickness(u, 1.0, edge);
    if !on_node {
        return ([theta; 2], [delta_star; 2]);
    }
    let Some((&u1, w)) = u.first().and_then(|u1| Some((u1, WallInterval::new(*u1, 1.0, nu)?))) else {
        return ([f64::NAN; 2], [f64::NAN; 2]);
    };
    let r = u1 / edge;
    let (theta_above, delta_above) = (theta - 0.5 * r * (1.0 - r), delta_star - 0.5 * (1.0 - r));
    let [t_law, t_trap] = w.momentum_thickness(edge);
    let [d_law, d_trap] = w.displacement_thickness(edge);
    ([theta_above + t_law, theta_above + t_trap], [delta_above + d_law, delta_above + d_trap])
}

fn print_stations(stations: &[Station], reference: impl Fn(f64) -> f64) {
    println!("|   x   | Re_θ law, trapezoid | δ* law, trapezoid |  U_e  | C_f carried | C_f requested | reference | error |");
    for s in stations {
        let r = reference(s.re_theta);
        println!(
            "| {:.3} | {:6.0}, {:6.0} | {:.3}, {:.3} | {:.4} | {:.5e} | {} | {:.5e} | {:+.2} % |",
            s.x,
            s.re_theta,
            s.re_theta_trapezoid,
            s.delta_star[0],
            s.delta_star[1],
            s.edge,
            s.cf,
            s.requested.map_or("—".into(), |c| format!("{c:.5e}")),
            r,
            100.0 * (s.cf - r) / r
        );
    }
}

/// C_f at each reference Re_θ, interpolated between the strips that bracket it, among the
/// strips from `from` on (the leading edge's first strips are left out).
fn gate(case: &str, stations: &[Station], from: f64, refs: &[(f64, f64, f64)], score: &mut Score) {
    let kept: Vec<&Station> = stations.iter().filter(|s| s.x >= from).collect();
    let re: Vec<f64> = kept.iter().map(|s| s.re_theta).collect();
    let cf: Vec<f64> = kept.iter().map(|s| s.cf).collect();
    for &(re_theta, reference, tolerance) in refs {
        let value = interpolate(&re, &cf, re_theta).unwrap_or(f64::NAN);
        score.push(Gate::new(case, &format!("C_f at Re_θ {re_theta:.0}"), value, reference, tolerance));
    }
}

/// Settling for the laminar plate, a check of the measuring pipeline and not a gated benchmark
/// case: 4 flow-throughs, then a time-mean drift under 1 %, a fifth of its 5 % gate. The plate
/// rings in its acoustic modes (±3 % in C_f, [`Plate::acoustic_period`]) and the two-period
/// means of the gated C_f still move by 2–5 × 10⁻³ between windows as the modes beat, so
/// [`GATED`]'s 0.1 % is out of reach here (measured over 30 000 steps). Its mean C_f is steady: the two halves of steps
/// 10 000–30 000 agree to 0.3 %.
const LAMINAR_PIPELINE: Settling = Settling { min_flow_throughs: 4.0, drift: 1e-2 };

/// The laminar plate at Re 2000 per unit length (Δ = 1/200, TRT, slip top two plate lengths up),
/// run for at most `max_steps` steps.
fn laminar_plate(max_steps: u64) -> Plate {
    Plate {
        re: 2000.0,
        length: 1.0,
        upstream: 0.25,
        height: 2.0,
        spacing: 1.0 / 200.0,
        lattice_velocity: 0.1,
        strip: 0.05,
        pressure_top: false,
        collision: Collision::Trt,
        turbulence: None,
        max_steps,
        sample: 100,
        settle: (0.3, 0.9),
        settling: LAMINAR_PIPELINE,
        absorbing: None,
    }
}

/// The environment variable that runs the plate studies (docs/verification.md, "Long studies").
const STUDY: &str = "FIELDCRAFT_PLATE_STUDY";

/// The measuring pipeline on a laminar plate at Re 2000 per unit length: C_f Re_θ within 5 % of
/// Blasius's 0.441 from x = 0.3 to 0.9 (Re_x 600–1800), clear of the leading edge and the outlet.
/// τ − ½ = 3 U_lat / (Re Δ) = 0.03: at Re 10⁴ (τ − ½ = 0.006) the leading edge went unstable with
/// either a slip or a pressure top. The top is a slip wall two plate lengths up; the layer's
/// displacement accelerates the edge flow (U_e ≈ 1.07 U); C_f and Re_θ are both formed with the
/// measured edge speed. About two minutes on four threads (10⁵ cells), so it runs only when asked:
/// `FIELDCRAFT_PLATE_STUDY=1 cargo test --release -p fieldcraft-lbm --test sa_flat_plate -- --ignored --nocapture study`.
/// The bounded regression below checks its first flow-through on every heavy run; the recorded
/// results are in docs/verification.md ("Long studies").
#[test]
#[ignore = "minutes: set FIELDCRAFT_PLATE_STUDY=1 (docs/verification.md, \"Long studies\")"]
fn study_a_laminar_plate_against_blasius() {
    if std::env::var_os(STUDY).is_none_or(|v| v != "1") {
        println!("the laminar plate study runs only with {STUDY}=1: skipped");
        return;
    }
    let (stations, converged, steps) = laminar_plate(40_000).run();
    println!("laminar plate, Re 2000 per unit length, Δ = 1/200: {steps} steps, settled: {converged}");
    print_stations(&stations, |re_theta| BLASIUS_CF_RE_THETA / re_theta);
    let mut score = Score::default();
    for s in stations.iter().filter(|s| (0.3..=0.9).contains(&s.x)) {
        score.push(Gate::new("laminar plate", &format!("C_f Re_θ at x {:.3}", s.x), s.cf * s.re_theta, BLASIUS_CF_RE_THETA, 0.05));
    }
    println!("{}", score.table());
    assert!(converged, "the laminar plate did not settle in {steps} steps");
    assert!(score.passes());
}

/// The absorbing layer leaves the laminar plate's friction alone: the study above with and without
/// a 20-cell layer at the inlet and the outlet (x < −0.15 and x > 0.9, clear of the gated
/// stations), each run until it settles. At every gated station (0.3 ≤ x ≤ 0.9) C_f Re_θ agrees
/// within 0.5 %. The layer's filtered reference makes it vanish on a steady flow; what it removes
/// is the quarter-wave ringing, so the run with it should also settle sooner (printed, not gated).
/// About five minutes on four threads: `FIELDCRAFT_PLATE_STUDY=1 cargo test --release -p
/// fieldcraft-lbm --test sa_flat_plate -- --ignored --nocapture study_the_absorbing`.
#[test]
#[ignore = "minutes: set FIELDCRAFT_PLATE_STUDY=1 (docs/verification.md, \"Long studies\")"]
fn study_the_absorbing_layer_keeps_the_laminar_friction() {
    if std::env::var_os(STUDY).is_none_or(|v| v != "1") {
        println!("the laminar plate study runs only with {STUDY}=1: skipped");
        return;
    }
    let off = laminar_plate(40_000);
    let on = Plate { absorbing: Some(AbsorbingLayer { thickness: 20.0 * off.spacing, strength: None, filter_time: None }), ..off.clone() };
    let (without, settled_without, steps_without) = off.run();
    let (with, settled_with, steps_with) = on.run();
    println!("settled without the layer: {settled_without} after {steps_without} steps; with it: {settled_with} after {steps_with} steps");
    let mut worst = 0.0f64;
    for (a, b) in without.iter().zip(&with).filter(|(a, _)| (0.3..=0.9).contains(&a.x)) {
        let (fa, fb) = (a.cf * a.re_theta, b.cf * b.re_theta);
        let change = (fb / fa - 1.0).abs();
        worst = worst.max(change);
        println!("x {:.3}: C_f Re_θ {fa:.5} without, {fb:.5} with ({:+.3} %)", a.x, 100.0 * (fb / fa - 1.0));
    }
    assert!(settled_without && settled_with, "a run did not settle");
    assert!(worst <= 5e-3, "C_f Re_θ moved by {:.3} % with the layer", 100.0 * worst);
}

/// The study above, bounded: its first flow-through (2500 steps), with C_f and Re_θ at three
/// gated stations, each the mean of the 25 samples (the averaging window, two acoustic periods,
/// is longer than the run). Frozen at its first measurement; the run is laminar and
/// deterministic, so the band of 10⁻⁹ (relative) is round-off room only, and any change to
/// the collision, the walls, the inlet or outlet, or the measuring pipeline fails it. About half
/// a minute on four threads.
#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn a_short_laminar_plate_keeps_the_study_s_friction() {
    // x, C_f, Re_θ: measured when frozen (2026-11-07, Windows x86_64; the same bits on two and four
    // threads). Still the start-up: C_f is near 0.043 at every station, against 0.016–0.024 settled.
    const FROZEN: [(f64, f64, f64); 3] = [
        (0.325, 4.309038167324105e-2, 1.4895299537647583e1),
        (0.625, 4.335373848472823e-2, 1.68498305361289e1),
        (0.875, 4.413276296504364e-2, 1.7141221700117505e1),
    ];
    let (stations, _, steps) = laminar_plate(2500).run();
    assert_eq!(steps, 2500);
    let found: Vec<(f64, f64, f64, &Station)> =
        FROZEN.iter().map(|&(x, cf, re_theta)| (x, cf, re_theta, stations.iter().find(|s| (s.x - x).abs() < 1e-9).unwrap())).collect();
    for (x, _, _, s) in &found {
        println!("x {x}: C_f {:e}, Re_θ {:e}", s.cf, s.re_theta);
    }
    for (x, cf, re_theta, s) in found {
        assert!(
            (s.cf / cf - 1.0).abs() < 1e-9 && (s.re_theta / re_theta - 1.0).abs() < 1e-9,
            "x {x}: C_f {} (frozen {cf}), Re_θ {} (frozen {re_theta})",
            s.cf,
            s.re_theta
        );
    }
}

/// The 2DZP grid and run controls (lengths in units of the reference length).
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // until slice 15a's model test calls tmr_2dzp
struct PlateGrid {
    /// 1/Δ.
    cells_per_unit: f64,
    lattice_velocity: f64,
    /// The domain's height above the plate, H.
    height: f64,
    /// The slip floor's length ahead of the plate.
    upstream: f64,
    strip: f64,
    pressure_top: bool,
    collision: Collision,
    max_steps: u64,
    sample: u64,
}

impl PlateGrid {
    /// The wall-function grid: Δ = 1/2000, so the first cell centre lies at y⁺ ≈ 40–55 over the
    /// gated range (Re_θ 4000–13000), with U = 0.1 in lattice units.
    #[allow(dead_code)]
    fn wall_function() -> PlateGrid {
        PlateGrid {
            cells_per_unit: 2000.0,
            lattice_velocity: 0.1,
            height: 0.25,
            upstream: 0.33,
            strip: 0.05,
            pressure_top: false,
            collision: Collision::Mrt,
            max_steps: 400_000,
            sample: 2000,
        }
    }
}

/// The result of a 2DZP run: each strip's reading, the 15a gate at the expected.json Re_θ, and
/// whether the run settled.
#[derive(Debug, Clone)]
#[allow(dead_code)] // until slice 15a's model test calls tmr_2dzp
struct PlateResult {
    stations: Vec<Station>,
    score: Score,
    settled: bool,
    steps: u64,
}

/// NASA TMR 2DZP (Re 5 × 10⁶ per unit length, plate 0 ≤ x ≤ 2) with a turbulence model on a
/// grid: C_f of each strip and the 15a gate, C_f within 5 % of Kármán–Schoenherr at the Re_θ of
/// the case's expected.json (strips from x = 0.1 on). Each model gets its own `#[test]` calling
/// this, so the test that ran names the model that ran, e.g. once slice 15a's model exists:
///
/// ```text
/// #[test]
/// #[ignore = "hours: tests/oracle/sa-flat-plate-2dzp/README.md"]
/// fn spalart_allmaras_on_the_musker_wall_meets_karman_schoenherr() {
///     let r = tmr_2dzp(Turbulence::SpalartAllmaras { .. }, PlateGrid::wall_function());
///     assert!(r.settled && r.score.passes());
/// }
/// ```
#[allow(dead_code)]
fn tmr_2dzp(turbulence: Turbulence, grid: PlateGrid) -> PlateResult {
    let plate = Plate {
        re: 5e6,
        length: 2.0,
        upstream: grid.upstream,
        height: grid.height,
        spacing: 1.0 / grid.cells_per_unit,
        lattice_velocity: grid.lattice_velocity,
        strip: grid.strip,
        pressure_top: grid.pressure_top,
        collision: grid.collision,
        turbulence: Some(turbulence),
        max_steps: grid.max_steps,
        sample: grid.sample,
        settle: (0.1, 1.95),
        settling: GATED,
        absorbing: None,
    };
    println!("2DZP: {plate:?}");
    let (stations, settled, steps) = plate.run();
    println!("{steps} steps, settled: {settled}");
    print_stations(&stations, karman_schoenherr);
    let refs: Vec<(f64, f64, f64)> =
        expected("sa-flat-plate-2dzp").into_iter().map(|(id, v, t)| (id.trim_start_matches("cf_re").parse().unwrap(), v, t)).collect();
    let mut score = Score::default();
    gate("2DZP", &stations, 0.1, &refs, &mut score);
    println!("{}", score.table());
    PlateResult { stations, score, settled, steps }
}

/// Sizing probe (board task C1): the throughput of the wall-function 2DZP grid (Δ = 1/2000,
/// 2.3 million cells) on MRT with Smagorinsky's model and Musker's wall, the nearest
/// configuration to slice 15a's until SA exists, over 200 steps.
#[test]
#[ignore = "sizing probe: about a minute in release"]
fn sizing_the_wall_function_plate() {
    let grid = PlateGrid { max_steps: 200, sample: 100, ..PlateGrid::wall_function() };
    let plate = Plate {
        re: 5e6,
        length: 2.0,
        upstream: grid.upstream,
        height: grid.height,
        spacing: 1.0 / grid.cells_per_unit,
        lattice_velocity: grid.lattice_velocity,
        strip: grid.strip,
        pressure_top: grid.pressure_top,
        collision: grid.collision,
        turbulence: Some(Turbulence::Smagorinsky {
            constant: 0.1,
            prandtl: 0.9,
            van_driest: None,
            wall_model: Some(fieldcraft_model::WallModel::Musker),
        }),
        max_steps: grid.max_steps,
        sample: grid.sample,
        settle: (0.1, 1.95),
        settling: GATED,
        absorbing: None,
    };
    let (stations, _, steps) = plate.run();
    assert_eq!(steps, 200);
    assert!(stations.iter().all(|s| s.cf.is_finite()));
}

/// A small SA plate on Han's on-node wall (WL1), a few hundred steps: the profile starts above the
/// node row, θ and δ* come as [law, trapezoid] with the first interval from the wall law, and the
/// strips report the law's request. A check of the reading, not of the flow (under a second).
#[test]
fn a_plate_on_han_s_wall_reads_its_thicknesses_from_the_wall_law() {
    let plate = Plate {
        re: 1e5,
        length: 0.5,
        upstream: 0.1,
        height: 0.2,
        spacing: 1.0 / 100.0,
        lattice_velocity: 0.1,
        strip: 0.05,
        pressure_top: false,
        collision: Collision::Mrt,
        turbulence: Some(Turbulence::SpalartAllmaras {
            prandtl: 0.9,
            wall_model: Some(fieldcraft_model::WallModel::SpaldingBounce),
            inflow_ratio: 3.0,
            ddes: None,
        }),
        max_steps: 300,
        sample: 100,
        settle: (0.1, 0.45),
        settling: GATED,
        absorbing: None,
    };
    let (stations, _, steps) = plate.run();
    assert_eq!(steps, 300);
    print_stations(&stations, karman_schoenherr);
    for s in stations.iter().filter(|s| (0.1..=0.45).contains(&s.x)) {
        let finite = [s.cf, s.re_theta, s.re_theta_trapezoid].iter().chain(&s.theta).chain(&s.delta_star).all(|v| v.is_finite());
        assert!(finite && s.theta[0] > 0.0 && s.delta_star[0] > 0.0, "{s:?}");
        assert!(s.theta[0] != s.theta[1] && s.delta_star[0] != s.delta_star[1], "the law's first interval differs from the trapezoid's: {s:?}");
        assert!(s.requested.is_some_and(|r| r > 0.0 && r.is_finite()), "Han's nodes report the law's request: {s:?}");
    }
}
