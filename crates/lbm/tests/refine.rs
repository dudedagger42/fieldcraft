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
    Procedure::Fluid {
        domain: "Flow".into(),
        end_time: None,
        steps: Some(100),
        time_step: Some(0.05 * 0.05 / 0.1),
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
