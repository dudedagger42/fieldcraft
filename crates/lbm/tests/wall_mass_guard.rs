//! The walls' mass correction is refused with the regularised and hybrid regularised collisions
//! below τ − ½ = 5e-2 (where it destabilises them) and still enabled with MRT and TRT there. A flow starts in
//! the automatic mode: ON where it is supported (the same rule), OFF below the limit with a note that says why.
//! A laminar plane channel of ten cells whose walls cut the links at q = 0.75, lattice units.

mod common;

use common::*;
use fieldcraft_lbm::Limits;
use fieldcraft_lbm::flow::{Flow, WALL_MASS_MIN_RELAXATION_EXCESS, WallMassMode};
use fieldcraft_model::{Collision, FaceCondition, WallTreatment};

fn channel(collision: Collision, tau_minus_half: f64) -> Flow {
    let (m, obstacles) = channel_model(collision, tau_minus_half);
    prepare(&m, &obstacles).1
}

fn channel_model(collision: Collision, tau_minus_half: f64) -> (fieldcraft_model::Model, Vec<fieldcraft_lbm::Obstacle>) {
    let (n, slab) = (10usize, 2usize);
    let nu = tau_minus_half / 3.0;
    let s = 0.75 - 0.5;
    let width = n as f64 + 2.0 * s;
    let umax = 0.05;
    let t = slab as f64;
    let height = n as f64;
    let mut d = domain([0.0, -t, 0.0], [1.0, height + t, 1.0], 1.0);
    for f in [&mut d.faces.x_min, &mut d.faces.x_max, &mut d.faces.z_min, &mut d.faces.z_max, &mut d.faces.y_min, &mut d.faces.y_max] {
        *f = FaceCondition::Periodic {};
    }
    let obstacles = vec![
        obstacle("Floor", cuboid([-1.0, -t, -1.0], [2.0, -s, 2.0]), &CUBOID_FACES),
        obstacle("Roof", cuboid([-1.0, height + s, -1.0], [2.0, height + t, 2.0]), &CUBOID_FACES),
    ];
    d.obstacles = obstacles.iter().map(|o| o.body.clone()).collect();
    d.acceleration = [8.0 * nu * umax / (width * width), 0.0, 0.0];
    d.initial_velocity = [umax, 0.0, 0.0];
    let p = with_walls(collide(100.0, 1.0, None, collision), WallTreatment::Interpolated);
    (model(1.0, nu, d, p), obstacles)
}

#[test]
fn the_regularised_collisions_refuse_the_correction_below_the_limit() {
    for collision in [Collision::Regularized, Collision::Hrr] {
        for excess in [5e-3, 1e-2, 4e-2] {
            let mut flow = channel(collision, excess);
            let why = flow.set_wall_mass_correction(true).expect_err("refused below the limit").to_string();
            assert!(why.contains("mass correction") && why.contains(collision.name()), "{why}");
            assert!(flow.wall_mass().is_none(), "a refused correction must not be left on");
        }
        for excess in [WALL_MASS_MIN_RELAXATION_EXCESS * 1.2, 1e-1] {
            let mut flow = channel(collision, excess);
            assert!(flow.set_wall_mass_correction(true).is_ok(), "{collision:?} at {excess}");
            assert!(flow.wall_mass().is_some());
        }
    }
}

#[test]
fn mrt_and_trt_still_enable_the_correction_at_small_relaxation_times() {
    for collision in [Collision::Mrt, Collision::Trt] {
        for excess in [5e-3, 1e-2, 4e-2] {
            let mut flow = channel(collision, excess);
            flow.set_wall_mass_correction(true).unwrap_or_else(|e| panic!("{collision:?} at {excess}: {e}"));
            assert!(flow.wall_mass().is_some());
            for _ in 0..50 {
                flow.step().unwrap_or_else(|e| panic!("{collision:?} at {excess}: {e}"));
            }
        }
    }
}

/// The automatic mode (every flow's start) picks ON and OFF by the guard's rule, on each side of the limit.
#[test]
fn auto_picks_the_correction_on_each_side_of_the_limit() {
    for collision in [Collision::Regularized, Collision::Hrr] {
        for excess in [5e-3, 1e-2, 4e-2, 4.9e-2] {
            let flow = channel(collision, excess);
            assert!(flow.wall_mass().is_none(), "{collision:?} at {excess}: auto must leave it off");
            let c = flow.wall_mass_choice().expect("a flow with interpolated walls says what it chose");
            assert!(!c.on && c.mode == WallMassMode::Auto, "{c:?}");
            // the guard's own text, the word automatically, and where the drift is read
            assert!(c.reason.contains("switched OFF automatically"), "{}", c.reason);
            assert!(c.reason.contains(&format!("not supported with the {} collision", collision.name())), "{}", c.reason);
            assert!(c.reason.contains("below 5e-2"), "{}", c.reason);
            assert!(c.reason.contains("Flow::mass"), "{}", c.reason);
        }
        for excess in [WALL_MASS_MIN_RELAXATION_EXCESS * 1.2, 1e-1] {
            let flow = channel(collision, excess);
            assert!(flow.wall_mass().is_some(), "{collision:?} at {excess}: auto must switch it on");
            let c = flow.wall_mass_choice().expect("chosen");
            assert!(c.on && c.mode == WallMassMode::Auto && c.reason.contains("ON (auto mode"), "{c:?}");
        }
    }
    for collision in [Collision::Mrt, Collision::Trt] {
        for excess in [5e-3, 1e-2, 4e-2, 1e-1] {
            let flow = channel(collision, excess);
            assert!(flow.wall_mass().is_some(), "{collision:?} at {excess}: auto must switch it on");
            assert!(flow.wall_mass_choice().is_some_and(|c| c.on && c.reason.contains(collision.name())), "{collision:?}");
        }
    }
}

/// The explicit choices are what they were: On is the guard's refusal below the limit, Off is off, and Auto can be asked again.
#[test]
fn explicit_on_and_off_are_unchanged() {
    let mut flow = channel(Collision::Hrr, 1e-2);
    let why = flow.set_wall_mass_mode(WallMassMode::On).expect_err("explicit on is refused below the limit").to_string();
    assert!(why.contains("not supported with the hrr collision"), "{why}");
    assert!(flow.wall_mass().is_none());
    let mut flow = channel(Collision::Mrt, 1e-2);
    assert!(flow.wall_mass().is_some());
    flow.set_wall_mass_correction(false).expect("explicit off");
    assert!(flow.wall_mass().is_none());
    let c = flow.wall_mass_choice().expect("chosen");
    assert!(!c.on && c.mode == WallMassMode::Off && c.reason.contains("OFF (chosen explicitly)"), "{c:?}");
    flow.set_wall_mass_correction(true).expect("explicit on");
    assert!(flow.wall_mass().is_some() && flow.wall_mass_choice().is_some_and(|c| c.on && c.mode == WallMassMode::On));
    flow.set_wall_mass_mode(WallMassMode::Auto).expect("auto again");
    assert!(flow.wall_mass_choice().is_some_and(|c| c.mode == WallMassMode::Auto));
}

/// The note is in the run's warnings on every run, in both states, and a checkpointed run turns an automatic ON into OFF with a note.
#[test]
fn the_choice_is_in_every_runs_warnings_and_a_checkpointed_run_switches_it_off() {
    let (m, obstacles) = channel_model(Collision::Hrr, 1e-2);
    let r = fieldcraft_lbm::solve::solve_fluid(&m, None, &obstacles, &Limits::default()).expect("runs with the correction off");
    assert!(r.warnings.iter().any(|w| w.contains("switched OFF automatically") && w.contains("Flow::mass")), "{:?}", r.warnings);
    let (m, obstacles) = channel_model(Collision::Mrt, 1e-2);
    let r = fieldcraft_lbm::solve::solve_fluid(&m, None, &obstacles, &Limits::default()).expect("runs with the correction on");
    assert!(r.warnings.iter().any(|w| w.contains("ON (auto mode")), "{:?}", r.warnings);
    let mut flow = channel(Collision::Mrt, 1e-2);
    flow.wall_mass_auto_for_checkpoint();
    assert!(flow.wall_mass().is_none());
    assert!(flow.wall_mass_choice().is_some_and(|c| !c.on && c.reason.contains("checkpointed")));
    // an explicit ON stays on
    let mut flow = channel(Collision::Mrt, 1e-2);
    flow.set_wall_mass_correction(true).expect("on");
    flow.wall_mass_auto_for_checkpoint();
    assert!(flow.wall_mass().is_some());
}
