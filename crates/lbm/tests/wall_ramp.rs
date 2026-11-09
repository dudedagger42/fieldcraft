//! The moving walls' ramp (`FluidDomain::moving_wall_ramp`): without one, or with one shorter than
//! half a step, the run is the unramped run bit for bit; with one, a wall moving along its normal
//! injects exactly its links' moving-wall terms times the ramp's factors; hostile ramps are refused,
//! and a ramp beside a wall model on a moving wall is refused as not supported yet.
mod common;

use common::*;
use fieldcraft_lbm::{Limits, Setup, lattice, solve_fluid};
use fieldcraft_model::{Collision, FaceCondition, GeometryRef, Procedure, SurfaceBoundary, SurfaceCondition, Turbulence};

/// The lid-driven cavity, 4 × 4 × 4 cells, 40 steps, its lid ramped over `ramp`.
fn cavity(ramp: Option<f64>, collision: Collision) -> fieldcraft_model::Model {
    let mut d = domain([0.0; 3], [4.0; 3], 1.0);
    d.faces.y_max = FaceCondition::MovingWall { velocity: [0.05, 0.0, 0.0] };
    d.moving_wall_ramp = ramp;
    model(1.0, 0.1, d, collide(40.0, 1.0, None, collision))
}

#[test]
fn the_factor_is_a_half_cosine_that_ends_at_exactly_one() {
    assert_eq!(lattice::wall_ramp(1, 0), 1.0, "no ramp");
    assert_eq!(lattice::wall_ramp(0, 10), 0.0);
    assert!((lattice::wall_ramp(5, 10) - 0.5).abs() < 1e-15);
    assert_eq!(lattice::wall_ramp(10, 10), 1.0, "the last step of the ramp is at full speed");
    assert_eq!(lattice::wall_ramp(11, 10), 1.0);
    assert!((1..10).all(|n| lattice::wall_ramp(n, 10) < lattice::wall_ramp(n + 1, 10)), "rises");
    assert_eq!(lattice::ramp_steps(None, 1.0), 0);
    assert_eq!(lattice::ramp_steps(Some(0.0), 1.0), 0);
    assert_eq!(lattice::ramp_steps(Some(0.2), 1.0), 1, "any ramp above 0 is at least a step");
    assert_eq!(lattice::ramp_steps(Some(29.6), 1.0), 30);
    assert_eq!(lattice::ramp_steps(Some(1e300), 1e-300), u64::MAX, "beyond any run: never reaches 1");
    assert!(lattice::wall_ramp(1_000_000, u64::MAX) < 1e-20);
}

/// None, 0 and a ramp shorter than half a step (one step at full speed) are today's run, bit
/// for bit, on every collision.
#[test]
fn no_ramp_is_the_unramped_run_bit_for_bit() {
    for collision in [Collision::Trt, Collision::Mrt, Collision::Hrr] {
        let none = solve_fluid(&cavity(None, collision), None, &[], &Limits::default()).unwrap();
        for ramp in [0.0, 0.4] {
            let ramped = solve_fluid(&cavity(Some(ramp), collision), None, &[], &Limits::default()).unwrap();
            assert_eq!(ramped, none, "{collision:?}, ramp {ramp}");
        }
        // A real ramp changes the run.
        let ramped = solve_fluid(&cavity(Some(20.0), collision), None, &[], &Limits::default()).unwrap();
        assert_ne!(ramped, none, "{collision:?}");
    }
}

/// A closed box with a block whose x_max face moves out along its normal: a mass source. Each step
/// the box gains its links' moving-wall terms Σ du times that step's factor, so after n steps the
/// mass is up by Σ du × Σ_{k ≤ n} r(k), to round-off, during the ramp and after it.
#[test]
fn the_ramp_scales_the_injected_mass_step_by_step() {
    let mut d = domain([0.0; 3], [8.0; 3], 1.0);
    d.obstacles = vec!["Block".into()];
    d.surfaces.push(SurfaceBoundary {
        face: GeometryRef::face("Block", "x_max").unwrap(),
        condition: SurfaceCondition::MovingWall { velocity: [0.01, 0.0, 0.0] },
    });
    d.moving_wall_ramp = Some(20.0);
    let block = obstacle("Block", cuboid([2.0; 3], [4.0; 3]), &CUBOID_FACES);
    let m = model(1.0, 0.1, d, collide(30.0, 1.0, None, Collision::Trt));
    let (setup, mut flow): (Setup, _) = prepare(&m, std::slice::from_ref(&block));
    assert_eq!(flow.ramp_steps(), 20);
    let du: f64 = flow
        .links()
        .iter()
        .filter_map(|l| match l.rule {
            fieldcraft_lbm::flow::Rule::Bounce { du, .. } => Some(du),
            _ => None,
        })
        .sum();
    assert!(du > 1e-3, "the face injects: Σ du {du}");
    let m0 = flow.mass();
    let mut factors = 0.0;
    for n in 1..=setup.steps {
        flow.step().unwrap();
        factors += lattice::wall_ramp(n, 20);
        let gained = flow.mass() - m0;
        assert!((gained - du * factors).abs() < 1e-12 * m0, "step {n}: gained {gained}, expected {}", du * factors);
    }
    assert!(factors > 10.0 && factors < 30.0, "{factors}");
}

#[test]
fn hostile_ramps_are_refused_and_a_long_one_runs() {
    for bad in [f64::NAN, -1.0, f64::INFINITY, f64::NEG_INFINITY] {
        let mut d = domain([0.0; 3], [4.0; 3], 1.0);
        d.moving_wall_ramp = Some(bad);
        let mut m = fieldcraft_model::Model::default();
        let err = m.add_fluid_domain("Flow", d).unwrap_err().to_string();
        assert!(err.contains("moving wall ramp"), "{bad}: {err}");
    }
    // Longer than the run: the lid never reaches full speed, and that is fine.
    let r = solve_fluid(&cavity(Some(1e9), Collision::Trt), None, &[], &Limits::default()).unwrap();
    assert_eq!(r.stats.steps, 40);
}

/// The wall model and Van Driest's damping take a moving wall's own velocity, which the ramp
/// does not scale: refused while a wall moves, accepted with every wall at rest.
#[test]
fn a_ramp_beside_a_wall_model_on_a_moving_wall_is_not_supported_yet() {
    let turbulence = Some(Turbulence::Smagorinsky { constant: 0.17, prandtl: 0.9, van_driest: Some(26.0), wall_model: None });
    let with = |moving: bool| {
        let mut d = domain([0.0; 3], [4.0; 3], 1.0);
        if moving {
            d.faces.y_max = FaceCondition::MovingWall { velocity: [0.05, 0.0, 0.0] };
        }
        d.moving_wall_ramp = Some(10.0);
        let mut p = collide(20.0, 1.0, None, Collision::Trt);
        if let Procedure::Fluid { turbulence: t, .. } = &mut p {
            *t = turbulence;
        }
        model(1.0, 0.1, d, p)
    };
    let err = solve_fluid(&with(true), None, &[], &Limits::default()).unwrap_err().to_string();
    assert!(err.contains("not supported yet") && err.contains("moving_wall_ramp"), "{err}");
    solve_fluid(&with(false), None, &[], &Limits::default()).unwrap();
}
