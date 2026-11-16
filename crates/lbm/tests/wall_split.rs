//! The wall links' momentum exchange split by the wall's normal (`Flow::track_wall_split`): the
//! friction and the pressure part, summed over the tracked walls.
//!
//! - A laminar channel driven by a body force: at the steady state the walls carry all of the force
//!   the fluid gets, `ρ a` per fluid cell, and all of it is friction (the wall normals are across
//!   the flow), so the friction share along the flow is 1 and the normal part along it 0.
//! - Fluid at rest in a box under gravity: the floor carries the weight as a normal force, no friction:
//!   tangential part 0 (the friction share along x is undefined, both parts being zero along it).

mod common;

use common::*;
use fieldcraft_lbm::interpolated::WallOrder;
use fieldcraft_model::FaceCondition;

fn channel(q: f64, steps: u64) -> (fieldcraft_lbm::Flow, f64, usize) {
    // Lattice units: spacing 1, time step 1.
    let (n, slab) = (10usize, 2usize);
    let nu = 0.1;
    let s = q - 0.5;
    let width = n as f64 + 2.0 * s;
    let t = slab as f64;
    let height = n as f64;
    let mut d = domain([0.0, -t, 0.0], [4.0, height + t, 4.0], 1.0);
    for f in [&mut d.faces.x_min, &mut d.faces.x_max, &mut d.faces.z_min, &mut d.faces.z_max, &mut d.faces.y_min, &mut d.faces.y_max] {
        *f = FaceCondition::Periodic {};
    }
    let obstacles = vec![
        obstacle("Floor", cuboid([-1.0, -t, -1.0], [5.0, -s, 5.0]), &CUBOID_FACES),
        obstacle("Roof", cuboid([-1.0, height + s, -1.0], [5.0, height + t, 5.0]), &CUBOID_FACES),
    ];
    d.obstacles = obstacles.iter().map(|o| o.body.clone()).collect();
    let umax = 0.02;
    let a = 8.0 * nu * umax / (width * width);
    d.acceleration = [a, 0.0, 0.0];
    let p = collide(steps as f64, 1.0, None, fieldcraft_model::Collision::Trt);
    let (setup, mut flow) = prepare(&model(1.0, nu, d, p), &obstacles);
    flow.interpolate_walls_with(&setup, &obstacles, WallOrder::Linear).unwrap();
    let tracked = flow.track_wall_split(&obstacles, &["Floor", "Roof"]).unwrap();
    assert!(tracked > 0);
    for _ in 0..steps {
        flow.step().unwrap();
    }
    let fluid = (0..flow.grid().len()).filter(|&c| flow.is_fluid(c)).count();
    (flow, a, fluid)
}

#[test]
fn a_driven_channel_carries_its_body_force_as_friction() {
    for q in [0.5, 0.75, 0.25] {
        let (flow, a, fluid) = channel(q, 6000);
        let split = flow.wall_split().unwrap();
        // The walls take the force the body force puts in: rho a per fluid cell (density about 1).
        let expected = a * fluid as f64;
        // The force on the walls is along the flow.
        let friction = split.tangential[0];
        println!("q {q}: friction {friction:.6e} (body force {expected:.6e}), normal along x {:.2e}", split.normal[0]);
        assert!((friction / expected - 1.0).abs() < 2e-3, "q {q}: {friction} against {expected}");
        assert!(split.normal[0].abs() < 1e-3 * expected, "q {q}: normal {}", split.normal[0]);
        let share = split.friction_share([1.0, 0.0, 0.0]).unwrap();
        assert!(share > 0.99, "q {q}: share {share}");
    }
}

#[test]
fn fluid_at_rest_under_gravity_presses_on_the_floor_and_does_not_rub() {
    let (n, t) = (8usize, 2.0);
    let mut d = domain([0.0, -t, 0.0], [4.0, n as f64 + t, 4.0], 1.0);
    for f in [&mut d.faces.x_min, &mut d.faces.x_max, &mut d.faces.z_min, &mut d.faces.z_max, &mut d.faces.y_min, &mut d.faces.y_max] {
        *f = FaceCondition::Periodic {};
    }
    let obstacles = vec![
        obstacle("Floor", cuboid([-1.0, -t, -1.0], [5.0, 0.0, 5.0]), &CUBOID_FACES),
        obstacle("Roof", cuboid([-1.0, n as f64, -1.0], [5.0, n as f64 + t, 5.0]), &CUBOID_FACES),
    ];
    d.obstacles = obstacles.iter().map(|o| o.body.clone()).collect();
    let g = 1e-5;
    d.acceleration = [0.0, -g, 0.0];
    let p = collide(4000.0, 1.0, None, fieldcraft_model::Collision::Trt);
    let (setup, mut flow) = prepare(&model(1.0, 0.1, d, p), &obstacles);
    flow.interpolate_walls_with(&setup, &obstacles, WallOrder::Linear).unwrap();
    flow.track_wall_split(&obstacles, &["Floor", "Roof"]).unwrap();
    for _ in 0..4000 {
        flow.step().unwrap();
    }
    let fluid = (0..flow.grid().len()).filter(|&c| flow.is_fluid(c)).count();
    let split = flow.wall_split().unwrap();
    // The walls hold the fluid up: the normal forces on them sum to the weight (the force on the floor
    // points down), and nothing rubs.
    let weight = g * fluid as f64;
    println!("normal y {:.4e}, weight {weight:.4e}, tangential {:?}", split.normal[1], split.tangential);
    assert!((split.normal[1].abs() / weight - 1.0).abs() < 5e-2, "{} against {weight}", split.normal[1]);
    assert!(split.tangential.iter().all(|v| v.abs() < 1e-3 * weight));
}
