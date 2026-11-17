//! The walls' mass correction is refused with the regularised and hybrid regularised collisions
//! below τ − ½ = 5e-2 (where it destabilises them) and still enabled with MRT and TRT there.
//! A laminar plane channel of ten cells whose walls cut the links at q = 0.75, lattice units.

mod common;

use common::*;
use fieldcraft_lbm::flow::{Flow, WALL_MASS_MIN_RELAXATION_EXCESS};
use fieldcraft_model::{Collision, FaceCondition, WallTreatment};

fn channel(collision: Collision, tau_minus_half: f64) -> Flow {
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
    prepare(&model(1.0, nu, d, p), &obstacles).1
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
