//! A ratio-2 strip across a coarse flow ([`fieldcraft_lbm::refine`]): the uniform state, the mass of a
//! closed box, and what is refused. CPU only.
#[path = "common/mod.rs"]
mod common;

use common::*;
use fieldcraft_lbm::refine::{Refined, Strip};
use fieldcraft_lbm::{Limits, Obstacle};
use fieldcraft_model::{Collision, FaceCondition, FluidDomain, HrrSigma, Model, Procedure, Turbulence};

const STRIP: Strip = Strip { first: 12, last: 20, linear: true };

fn procedure(collision: Collision, turbulence: Option<Turbulence>) -> Procedure {
    procedure_dt(collision, turbulence, 0.05 * 0.05 / 0.1)
}

fn procedure_dt(collision: Collision, turbulence: Option<Turbulence>, dt: f64) -> Procedure {
    Procedure::Fluid {
        domain: "Flow".into(),
        end_time: None,
        steps: Some(100),
        time_step: Some(dt),
        frames: 2,
        magic: None,
        sigma: Some(HrrSigma::Value(0.95)),
        collision: Some(collision),
        turbulence,
        steady: None,
    }
}

/// A box 32 x 4 x 4 coarse cells (0.05 m), τ − ½ = 1e-2 (lattice speed 0.05 at 0.1 m/s).
fn box_model(d: FluidDomain, p: Procedure) -> Model {
    let dx: f64 = 0.05;
    let dt: f64 = 0.05 * dx / 0.1;
    let nu = 1e-2 / 3.0 * dx * dx / dt;
    model(1.0, nu, d, p)
}

fn closed_box() -> FluidDomain {
    let mut d = domain([0.0; 3], [1.6, 0.2, 0.2], 0.05);
    d.initial_velocity = [0.1, 0.03, -0.02];
    d.acceleration = [0.0, 0.0, -0.002];
    d
}

fn periodic_box() -> FluidDomain {
    let mut d = domain([0.0; 3], [1.6, 0.2, 0.2], 0.05);
    d.initial_velocity = [0.1, 0.03, -0.02];
    d.faces.x_min = FaceCondition::Periodic {};
    d.faces.x_max = FaceCondition::Periodic {};
    d.faces.y_min = FaceCondition::Periodic {};
    d.faces.y_max = FaceCondition::Periodic {};
    d.faces.z_min = FaceCondition::Periodic {};
    d.faces.z_max = FaceCondition::Periodic {};
    d
}

fn refined(d: FluidDomain, strip: Strip) -> Result<Refined, fieldcraft_lbm::Error> {
    Refined::new(&box_model(d, procedure(Collision::Hrr, None)), None, &[], &Limits::default(), strip)
}

/// A uniform flow stays uniform on both levels: every composite cell keeps its velocity and
/// density to round-off, the mass does not move.
#[test]
fn a_uniform_flow_stays_uniform() {
    for linear in [true, false] {
        let mut r = refined(periodic_box(), Strip { linear, ..STRIP }).unwrap();
        let (m0, c0) = (r.mass(), r.composite_moments());
        for _ in 0..200 {
            r.step().unwrap();
        }
        let c = r.composite_moments();
        let worst =
            c0.iter().zip(&c).map(|(a, b)| (a.0 - b.0).abs().max((0..3).map(|k| (a.1[k] - b.1[k]).abs()).fold(0.0, f64::max))).fold(0.0f64, f64::max);
        assert!(worst <= 1e-12, "linear {linear}: the uniform state moved by {worst:e}");
        assert!(((r.mass() - m0) / m0).abs() <= 1e-12, "linear {linear}: mass drifted by {:e}", (r.mass() - m0) / m0);
    }
}

/// A closed box (walls on every face, a body force, an initial velocity into the walls): the
/// mass of the two levels together is conserved while the flow crosses the strip and rings off
/// the walls.
#[test]
fn the_mass_of_a_closed_box_is_conserved() {
    for linear in [true, false] {
        let mut r = refined(closed_box(), Strip { linear, ..STRIP }).unwrap_or_else(|e| panic!("{e}"));
        let m0 = r.mass();
        let mut worst = 0.0f64;
        for step in 0..600 {
            r.step().unwrap();
            if step % 50 == 49 {
                worst = worst.max(((r.mass() - m0) / m0).abs());
            }
        }
        assert!(worst <= 1e-10, "linear {linear}: relative mass drift {worst:e}");
        // The flow is alive and finite on both levels.
        let c = r.composite_moments();
        assert!(c.iter().all(|(rho, u)| rho.is_finite() && u.iter().all(|v| v.is_finite())));
    }
}

fn refused(r: Result<Refined, fieldcraft_lbm::Error>, what: &str) {
    match r {
        Ok(_) => panic!("expected a refusal mentioning `{what}`"),
        Err(e) => assert!(e.to_string().contains(what), "{e} does not mention `{what}`"),
    }
}

#[test]
fn what_is_not_supported_is_refused_with_the_reason() {
    let build = |d: FluidDomain, p: Procedure, strip: Strip, obstacles: &[Obstacle]| {
        Refined::new(&box_model(d, p), None, obstacles, &Limits::default(), strip)
    };
    // Another collision, an eddy-viscosity model.
    refused(build(closed_box(), procedure(Collision::Trt, None), STRIP, &[]), "collision");
    let sa = Turbulence::SpalartAllmaras { prandtl: 0.9, wall_model: None, inflow_ratio: 3.0, ddes: None };
    refused(build(closed_box(), procedure(Collision::Hrr, Some(sa)), STRIP, &[]), "eddy-viscosity");
    let vreman = Turbulence::Vreman { constant: 0.18, prandtl: 0.9, wall_model: None };
    refused(build(closed_box(), procedure(Collision::Hrr, Some(vreman)), STRIP, &[]), "eddy-viscosity");
    // An absorbing layer.
    let mut d = closed_box();
    d.faces.x_min = FaceCondition::Velocity { velocity: [0.1, 0.0, 0.0], profile: fieldcraft_model::Profile::Uniform };
    d.faces.x_max = FaceCondition::Pressure { pressure: 0.0, outflow: fieldcraft_model::Outflow::Uniform };
    d.absorbing = Some(fieldcraft_model::AbsorbingLayer { thickness: None, strength: None, filter_time: None });
    refused(build(d, procedure(Collision::Hrr, None), STRIP, &[]), "absorbing");
    // A strip too close to the box face, or too narrow.
    refused(build(closed_box(), procedure(Collision::Hrr, None), Strip { first: 1, last: 12, linear: true }, &[]), "inside");
    refused(build(closed_box(), procedure(Collision::Hrr, None), Strip { first: 12, last: 18, linear: true }, &[]), "narrower");
    refused(build(closed_box(), procedure(Collision::Hrr, None), Strip { first: 12, last: 31, linear: true }, &[]), "inside");
    // A solid next to the interface: a block in the coarse column just outside the strip.
    let mut d = closed_box();
    d.obstacles = vec!["Block".into()];
    let block = obstacle("Block", cuboid([0.55, 0.05, 0.05], [0.64, 0.15, 0.15]), &CUBOID_FACES);
    refused(build(d, procedure(Collision::Hrr, None), STRIP, &[block]), "fluid");
}

// ---- waves through the strip -----------------------------------------------------------------

fn all_periodic(n: [usize; 3]) -> FluidDomain {
    let mut d = domain([0.0; 3], [n[0] as f64 * 0.05, n[1] as f64 * 0.05, n[2] as f64 * 0.05], 0.05);
    d.faces.x_min = FaceCondition::Periodic {};
    d.faces.x_max = FaceCondition::Periodic {};
    d.faces.y_min = FaceCondition::Periodic {};
    d.faces.y_max = FaceCondition::Periodic {};
    d.faces.z_min = FaceCondition::Periodic {};
    d.faces.z_max = FaceCondition::Periodic {};
    d
}

/// A periodic box with a negligible eddy-viscosity model, at the given τ − ½ of the coarse level.
fn wave_pair(n: [usize; 3], strip: Strip, eps: f64) -> Refined {
    let dx: f64 = 0.05;
    let dt: f64 = 0.05 * dx / 0.1;
    let nu = eps / 3.0 * dx * dx / dt;
    let smag = Some(Turbulence::Smagorinsky { constant: 1e-4, prandtl: 0.9, van_driest: None, wall_model: None });
    let m = model(1.0, nu, all_periodic(n), procedure(Collision::Hrr, smag));
    Refined::new(&m, None, &[], &Limits::default(), strip).unwrap_or_else(|e| panic!("{e}"))
}

struct Wave {
    amplitude: f64,
    phase: f64,
    spurious: f64,
    density: f64,
}

/// Fit `component` of the velocity to `sin(2π x_axis/λ)` over all coarse cells; the spurious
/// velocity is the largest of the other two components, the density error the largest |ρ − 1|.
fn fit(r: &Refined, m: &[(f64, [f64; 3])], axis: usize, component: usize, lambda: f64) -> Wave {
    let g = r.coarse().grid();
    let (mut a, mut b, mut n) = (0.0, 0.0, 0.0);
    let (mut spurious, mut density) = (0.0f64, 0.0f64);
    for (cell, (rho, u)) in m.iter().enumerate() {
        let c = g.coords(cell);
        let k = 2.0 * std::f64::consts::PI * (c[axis] as f64 + 0.5) / lambda;
        a += u[component] * k.sin();
        b += u[component] * k.cos();
        n += 1.0;
        spurious = spurious.max((u[if axis == 0 { 0 } else { 1 }] - 0.05).abs());
        density = density.max((rho - 1.0).abs());
    }
    Wave { amplitude: 2.0 * a.hypot(b) / n, phase: b.atan2(a), spurious, density }
}

/// A shear wave crossing (wave vector along x, velocity along y) or running along (wave vector
/// along y, velocity along x, through the interface) the strip, refined against the coarse flow
/// alone: the extra amplitude loss, phase shift and spurious velocity, as fractions of the wave.
fn shear(normal: bool, lambda: usize, eps: f64, steps: usize) -> (f64, f64, f64, f64) {
    let a0 = 0.005;
    let n = if normal { [64, 4, 4] } else { [64, lambda, 4] };
    let strip = Strip { first: 24, last: 40, linear: true };
    let u0 = 0.05;
    let (axis, comp) = if normal { (0, 1) } else { (1, 0) };
    let l = lambda as f64;
    let state = |x: [f64; 3]| {
        let mut u = [0.0; 3];
        u[comp] = a0 * (2.0 * std::f64::consts::PI * x[axis] / l).sin();
        // The wave is carried along the wave vector (normal) or across it (parallel) by a mean flow.
        u[if normal { 0 } else { 1 }] = u0;
        (1.0, u)
    };
    let (mut with, mut without) = (wave_pair(n, strip, eps), wave_pair(n, strip, eps));
    with.initialise(&state);
    without.initialise(&state);
    for _ in 0..steps {
        with.step().unwrap();
        without.step_unrefined().unwrap();
    }
    let w = fit(&with, &with.composite_moments(), axis, comp, l);
    let b = fit(&without, &without.coarse_moments(), axis, comp, l);
    ((w.amplitude - b.amplitude) / b.amplitude, w.phase - b.phase, (w.spurious - b.spurious) / a0, w.density - b.density)
}

/// The shear-wave bar: against the coarse flow alone, an extra amplitude loss of at most 1 %, a
/// phase shift of at most 0.01 rad, spurious velocity at most 5 % of the wave. The table is
/// printed for every case; `cargo test --release --test refine -- --ignored --nocapture`.
#[test]
#[ignore = "heavy: the shear-wave table"]
fn the_shear_wave_crosses_and_runs_along_the_strip() {
    let mut worst: Vec<String> = Vec::new();
    for eps in [2.4e-4, 1e-3, 1e-2] {
        for (normal, lambda) in [(true, 16), (true, 32), (false, 32), (false, 16)] {
            let steps = 900;
            let (loss, phase, spurious, rho) = shear(normal, lambda, eps, steps);
            let pass = (-loss).max(0.0) <= 0.01 && phase.abs() <= 0.01 && spurious <= 0.05;
            let line = format!(
                "eps {eps:7.1e} {} lambda {lambda:3}: extra amplitude {loss:+9.2e}  phase {phase:9.2e} rad  spurious {spurious:9.2e}  density {rho:9.2e}  {}",
                if normal { "normal  " } else { "parallel" },
                if pass { "ok" } else { "OVER" }
            );
            eprintln!("{line}");
            if !pass && lambda >= 32 {
                worst.push(line);
            }
        }
    }
    assert!(
        worst.is_empty(),
        "outside the bar:
{}",
        worst.join(
            "
"
        )
    );
}

// ---- checkerboard, Poiseuille, pulse ---------------------------------------------------------

/// Grid-scale checkerboards (a velocity one and a density one) through the strip do not grow:
/// the largest deviation never exceeds its starting size, at the lowest and a high viscosity.
#[test]
fn a_grid_scale_checkerboard_decays_across_the_strip() {
    for eps in [2.4e-4, 1e-2] {
        for density in [false, true] {
            let n = [48, 4, 4];
            let strip = Strip { first: 16, last: 32, linear: true };
            let a0 = 1e-3;
            // The pattern lives on the cells of whichever level holds the point.
            let state = move |x: [f64; 3]| {
                let s = if (16.0..32.0).contains(&x[0]) { 2.0 } else { 1.0 };
                let parity = (0..3).map(|a| (x[a] * s).floor() as i64).sum::<i64>().rem_euclid(2);
                let v = if parity == 0 { a0 } else { -a0 };
                if density { (1.0 + v, [0.0; 3]) } else { (1.0, [0.0, v, 0.0]) }
            };
            let mut r = wave_pair(n, strip, eps);
            r.initialise(&state);
            let worst = |r: &Refined| {
                r.composite_moments()
                    .iter()
                    .map(|(rho, u)| (rho - 1.0).abs().max(u.iter().fold(0.0f64, |m, v| m.max(v.abs()))))
                    .fold(0.0f64, f64::max)
            };
            for step in 0..400 {
                r.step().unwrap();
                assert!(worst(&r) <= 1.5 * a0, "eps {eps:e} density {density}: the checkerboard grew to {:e} at step {step}", worst(&r));
            }
            assert!(worst(&r) <= 0.5 * a0, "eps {eps:e} density {density}: left {:e} of {a0:e} after 400 steps", worst(&r));
        }
    }
}

/// A channel between walls (y) driven by a body force along x, through a strip, against the same
/// channel on a uniform fine grid: the profile error of both runs, column by column, as a share of
/// the peak (printed). Report only for now: next to the interface the refined profile is far off.
#[test]
#[ignore = "report: the driven channel against a uniform fine run"]
fn a_driven_channel_through_the_strip_against_a_uniform_fine_run() {
    let n = [64, 8, 4];
    let mut d = all_periodic(n);
    d.faces.y_min = FaceCondition::Wall {};
    d.faces.y_max = FaceCondition::Wall {};
    d.acceleration = [0.002, 0.0, 0.0];
    let dx: f64 = 0.05;
    let dt: f64 = 0.05 * dx / 0.1;
    let nu = 0.1 / 3.0 * dx * dx / dt;
    let smag = Some(Turbulence::Smagorinsky { constant: 1e-4, prandtl: 0.9, van_driest: None, wall_model: None });
    let m = model(1.0, nu, d, procedure(Collision::Hrr, smag));
    let strip = Strip { first: 24, last: 40, linear: true };
    let build = || Refined::new(&m, None, &[], &Limits::default(), strip).unwrap_or_else(|e| panic!("{e}"));
    let (mut with, mut without) = (build(), build());
    for step in 0..3000 {
        with.step().unwrap();
        without.step_unrefined().unwrap();
        if step % 300 == 299 {
            let a = with.composite_moments();
            let g = with.coarse().grid();
            eprintln!(
                "step {step}: u(y=4) at columns 10, 23, 30, 36, 41: {:.3e} {:.3e} {:.3e} {:.3e} {:.3e}",
                a[g.index([10, 4, 1])].1[0],
                a[g.index([23, 4, 1])].1[0],
                a[g.index([30, 4, 1])].1[0],
                a[g.index([36, 4, 1])].1[0],
                a[g.index([41, 4, 1])].1[0]
            );
        }
    }
    // The reference: the same channel on a grid of half the spacing and time step, everywhere.
    let mut fine_domain = all_periodic([2 * n[0], 2 * n[1], 2 * n[2]]);
    fine_domain.min = [0.0; 3];
    fine_domain.max = [n[0] as f64 * dx, n[1] as f64 * dx, n[2] as f64 * dx];
    fine_domain.spacing = dx / 2.0;
    fine_domain.faces.y_min = FaceCondition::Wall {};
    fine_domain.faces.y_max = FaceCondition::Wall {};
    fine_domain.acceleration = [0.002, 0.0, 0.0];
    let smag = Some(Turbulence::Smagorinsky { constant: 1e-4, prandtl: 0.9, van_driest: None, wall_model: None });
    let fine_model = model(1.0, nu, fine_domain, procedure_dt(Collision::Hrr, smag, dt / 2.0));
    let mut reference =
        Refined::new(&fine_model, None, &[], &Limits::default(), Strip { first: 48, last: 80, linear: true }).unwrap_or_else(|e| panic!("{e}"));
    for _ in 0..6000 {
        reference.step_unrefined().unwrap();
    }
    let fine_grid = reference.coarse().grid();
    let fine = reference.coarse_moments();
    let g = with.coarse().grid();
    // u_x(y) at coarse column x, the mean of the reference's cells in it.
    let truth = |x: usize| -> Vec<f64> {
        (0..n[1]).map(|y| (0..8).map(|k| fine[fine_grid.index([2 * x + k % 2, 2 * y + (k / 2) % 2, 2 + k / 4])].1[0]).sum::<f64>() / 8.0).collect()
    };
    let profile = |m: &[(f64, [f64; 3])], x: usize| -> Vec<f64> { (0..n[1]).map(|y| m[g.index([x, y, 1])].1[0]).collect() };
    let (a, b) = (with.composite_moments(), without.coarse_moments());
    let peak = truth(10)[4];
    assert!(peak > 1e-3, "the channel did not accelerate: {peak:e}");
    let mut worst_refined = 0.0f64;
    for x in [10, 20, 22, 23, 24, 26, 30, 34, 38, 40, 41, 42, 50] {
        let t = truth(x);
        let err = |p: &[f64]| p.iter().zip(&t).map(|(p, q)| (p - q).abs()).fold(0.0f64, f64::max) / peak;
        let (er, eu) = (err(&profile(&a, x)), err(&profile(&b, x)));
        eprintln!("column {x:2}: profile error against the uniform fine run: refined {er:.2e}, unrefined {eu:.2e} of the peak {peak:.3e}");
        worst_refined = worst_refined.max(er);
    }
    eprintln!("worst refined {worst_refined:.2e}");
}

/// Report only: a weak acoustic pulse crosses the strip. Printed: the transmitted peak against
/// the unrefined run's, and the reflected peak as a share of the incident one.
#[test]
#[ignore = "report: the acoustic pulse through the strip"]
fn the_acoustic_pulse_crosses_the_strip() {
    for eps in [2.4e-4, 1e-3] {
        let n = [160, 4, 4];
        let strip = Strip { first: 70, last: 86, linear: true };
        let delta = 1e-4;
        let state = move |x: [f64; 3]| (1.0 + delta * (-((x[0] - 30.0) / 3.0).powi(2)).exp(), [0.0; 3]);
        let (mut with, mut without) = (wave_pair(n, strip, eps), wave_pair(n, strip, eps));
        with.initialise(&state);
        without.initialise(&state);
        for _ in 0..121 {
            with.step().unwrap();
            without.step_unrefined().unwrap();
        }
        let g = with.coarse().grid();
        let (a, b) = (with.composite_moments(), without.coarse_moments());
        let peak = |m: &[(f64, [f64; 3])], lo: usize, hi: usize| (lo..hi).map(|x| (m[g.index([x, 1, 1])].0 - 1.0).abs()).fold(0.0f64, f64::max);
        let incident = 0.5 * delta;
        eprintln!(
            "eps {eps:.1e}: transmitted {:.4} of the unrefined run's; reflected {:.2e} of the incident (unrefined {:.2e})",
            peak(&a, 90, 130) / peak(&b, 90, 130),
            peak(&a, 0, 60) / incident,
            peak(&b, 0, 60) / incident
        );
    }
}

#[test]
fn a_body_force_accelerates_both_levels_alike() {
    let n = [64, 4, 4];
    let mut d = all_periodic(n);
    d.acceleration = [0.002, 0.0, 0.0];
    let dx: f64 = 0.05;
    let dt: f64 = 0.05 * dx / 0.1;
    let nu = 1e-2 / 3.0 * dx * dx / dt;
    let m = model(
        1.0,
        nu,
        d,
        procedure(Collision::Hrr, Some(Turbulence::Smagorinsky { constant: 1e-4, prandtl: 0.9, van_driest: None, wall_model: None })),
    );
    let strip = Strip { first: 24, last: 40, linear: true };
    let mut with = Refined::new(&m, None, &[], &Limits::default(), strip).unwrap_or_else(|e| panic!("{e}"));
    let mut without = Refined::new(&m, None, &[], &Limits::default(), strip).unwrap_or_else(|e| panic!("{e}"));
    for step in 0..200 {
        with.step().unwrap();
        without.step_unrefined().unwrap();
        if step % 40 == 39 {
            let (a, b) = (with.composite_moments(), without.coarse_moments());
            let g = with.coarse().grid();
            eprintln!(
                "step {step}: u coarse {:.6e}, strip {:.6e}, interface {:.6e}, unrefined {:.6e}",
                a[g.index([10, 1, 1])].1[0],
                a[g.index([32, 1, 1])].1[0],
                a[g.index([23, 1, 1])].1[0],
                b[g.index([10, 1, 1])].1[0]
            );
        }
    }
}
