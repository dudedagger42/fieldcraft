//! A velocity inlet under the MRT collision stays stable. The Zou-He closure leaves the
//! non-hydrodynamic part of the inlet cell's non-equilibrium unconstrained; the regularised
//! collisions project it every step, MRT relaxes it raw, so without the projection of the inlet
//! cell (`zou_he`, `project`) a plate-with-slot channel at tau - 1/2 = 2.4e-3 went unstable in
//! about 80 steps, in the first columns behind the inlet. This channel (uniform inlet at 0.05 in
//! lattice units, pressure outlet, no-slip walls on y) runs 3000 steps at
//! tau - 1/2 = 2.4e-3 and 1e-2 and must stay finite with a bounded speed.

mod common;

use common::*;
use fieldcraft_model::{Collision, FaceCondition, Outflow, Procedure, Profile, Turbulence};

fn run(tau_minus_half: f64, collision: Collision, steps: u64) -> f64 {
    let nu = tau_minus_half / 3.0;
    let u = 0.05;
    let (nx, ny, nz) = (40usize, 12usize, 4usize);
    let mut d = domain([0.0; 3], [nx as f64, ny as f64, nz as f64], 1.0);
    for f in [&mut d.faces.z_min, &mut d.faces.z_max] {
        *f = FaceCondition::Periodic {};
    }
    d.faces.x_min = FaceCondition::Velocity { velocity: [u, 0.02, 0.0], profile: Profile::Uniform };
    d.faces.x_max = FaceCondition::Pressure { pressure: 0.0, outflow: Outflow::Uniform };
    d.initial_velocity = [u, 0.0, 0.0];
    let mut p = collide(steps as f64, 1.0, None, collision);
    // The setup refuses tau - 1/2 below 5e-3 without a model; SA adds nothing in this laminar flow.
    if let Procedure::Fluid { turbulence, .. } = &mut p
        && tau_minus_half < 5e-3
    {
        *turbulence = Some(Turbulence::SpalartAllmaras { prandtl: 0.9, wall_model: None, inflow_ratio: 3.0, ddes: None });
    }
    let (_, mut flow) = prepare(&model(1.0, nu, d, p), &[]);
    let mut peak = 0.0f64;
    for _ in 0..steps {
        flow.step().unwrap();
    }
    for c in 0..flow.grid().len() {
        if flow.is_fluid(c)
            && let Some((_, v)) = flow.moments(c)
        {
            peak = peak.max((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt());
        }
    }
    peak
}

#[test]
fn a_velocity_inlet_under_mrt_stays_stable_at_low_viscosity() {
    for tau in [2.4e-3, 1e-2] {
        let peak = run(tau, Collision::Mrt, 3000);
        assert!(peak.is_finite() && peak < 0.2, "tau - 1/2 {tau}: peak speed {peak}");
    }
}
