//! The population flux through a lattice plane (`Flow::population_flux`, `Flow::open_face_flux`): the mass that crossed in the
//! last streaming, from the populations.

mod common;

use common::*;
use fieldcraft_lbm::backend::Backend;
use fieldcraft_lbm::{Flow, Limits, Setup};
use fieldcraft_model::{Collision, FaceCondition};

fn periodic(d: &mut fieldcraft_model::FluidDomain) {
    for f in [&mut d.faces.x_min, &mut d.faces.x_max, &mut d.faces.y_min, &mut d.faces.y_max, &mut d.faces.z_min, &mut d.faces.z_max] {
        *f = FaceCondition::Periodic {};
    }
}

#[test]
fn uniform_flow_flux_is_rho_u_a() {
    let (n, u) = (8usize, 0.01f64);
    let dx = 1.0 / n as f64;
    let dt = 0.05 * dx * dx;
    let mut d = domain([0.0; 3], [1.0; 3], dx);
    periodic(&mut d);
    let m = model(1.0, 1.0, d, collide(1e6 * dt, dt, None, Collision::Hrr));
    let setup = Setup::new(&m, None, &[], &Limits::default()).expect("setup");
    let mut flow = Flow::new(&setup, &[]).expect("flow");
    flow.set_equilibrium(|_| (1.0, [u, 0.0, 0.0]));
    Backend::advance(&mut flow, 3).expect("advance");
    let area = (n * n) as f64;
    for plane in 0..n {
        let q = flow.population_flux(0, plane, None).expect("flux");
        assert!((q - u * area).abs() < 1e-12, "x plane {plane}: {q} against {}", u * area);
        assert!(flow.population_flux(1, plane, None).expect("flux").abs() < 1e-12);
        assert!(flow.population_flux(2, plane, None).expect("flux").abs() < 1e-12);
    }
    // a mask takes its cells' share: one row of the plane out of eight
    let g = *flow.grid();
    let row: Vec<bool> = (0..g.len()).map(|c| g.coords(c)[1] == 3).collect();
    let q = flow.population_flux(0, 2, Some(&row)).expect("flux");
    assert!((q - u * n as f64).abs() < 1e-12, "{q}");
    // the closed box between two planes of each axis: what comes in goes out
    let net: f64 = (0..3).map(|a| flow.population_flux(a, 2, None).expect("flux") - flow.population_flux(a, 6, None).expect("flux")).sum();
    assert!(net.abs() < 1e-12, "{net}");
    // hostile arguments are errors, not panics
    assert!(flow.population_flux(3, 0, None).is_err());
    assert!(flow.population_flux(0, n, None).is_err());
    assert!(flow.population_flux(0, 1, Some(&[true])).is_err());
}

#[test]
fn walled_channel_flux_matches_the_cell_sum() {
    // a plane channel of 8 cells between walls (y), periodic along x and z, driven by a body force to its steady state
    let w = 8usize;
    let dx = 1.0 / w as f64;
    let nu_lat = 0.1;
    let dt = nu_lat * dx * dx;
    let ny = w + 4;
    let (lx, ly, lz) = (8.0 * dx, ny as f64 * dx, 4.0 * dx);
    let mut d = domain([0.0; 3], [lx, ly, lz], dx);
    periodic(&mut d);
    let ho = w as f64 * dx;
    let wall = obstacle("Wall0", cuboid([-3.0 * lx, ho, -dx], [3.0 * lx, ly, lz + dx]), &CUBOID_FACES);
    d.obstacles = vec![wall.body.clone()];
    let g_lat = 12.0 * nu_lat * 0.01 / (w * w) as f64;
    d.acceleration = [g_lat * dx / (dt * dt), 0.0, 0.0];
    let m = model(1.0, 1.0, d, collide(1e6 * dt, dt, None, Collision::Hrr));
    let setup = Setup::new(&m, None, std::slice::from_ref(&wall), &Limits::default()).expect("setup");
    let mut flow = Flow::new(&setup, std::slice::from_ref(&wall)).expect("flow");
    Backend::advance(&mut flow, 4000).expect("advance");
    let g = *flow.grid();
    let n = g.len();
    // the flux through an x plane is the sum of the plane's cell momenta Σ c f when the flow is uniform along x
    let mut cell_sum = 0.0;
    for c in (0..n).filter(|&c| g.coords(c)[0] == 3) {
        if let Some((rho, u)) = flow.moments(c) {
            cell_sum += rho * u[0];
        }
    }
    let q = flow.population_flux(0, 3, None).expect("flux");
    assert!((cell_sum - q).abs() < 1e-12, "{q} against {cell_sum}");
    // the driven mean speed (a few percent: the staircase wall), the same through every plane
    let reference = 0.01 * w as f64 * 4.0;
    assert!((q / reference - 1.0).abs() < 0.05, "{q} against {reference}");
    for plane in 0..8 {
        let qp = flow.population_flux(0, plane, None).expect("flux");
        assert!((qp - q).abs() < 1e-9 * q.abs(), "plane {plane}: {qp} against {q}");
    }
    // nothing crosses the wall: the plane between the last fluid row and the first solid one, over the fluid cells
    let in_channel: Vec<bool> = (0..n).map(|c| g.coords(c)[1] < w).collect();
    assert!(flow.population_flux(1, w, Some(&in_channel)).expect("flux").abs() < 1e-9);
    assert!(flow.open_face_flux(0, false, None).expect("flux") > 0.0);
}
