//! The inlet ramp (`FluidDomain::inlet_ramp`): without one, or with one shorter than half a step, the
//! run is the unramped run bit for bit; with one, every velocity open cell runs at the ramp's factor
//! times its full velocity until the ramp ends and at exactly the full velocity after it; hostile
//! ramps are refused.
mod common;

use common::*;
use fieldcraft_lbm::flow::Target;
use fieldcraft_lbm::{Flow, Setup, lattice};
use fieldcraft_model::{Collision, FaceCondition, Outflow, Procedure, Profile};

const DX: f64 = 1.0 / 12.0;
const ULAT: f64 = 0.05;

/// A channel with a uniform velocity inlet and a pressure outlet, `steps` steps long at the lattice
/// velocity [`ULAT`] (one unit of SI speed, the time step fixing it), its inlet ramped over `ramp`
/// time steps' worth of time.
fn channel(ramp_steps: Option<f64>, steps: u64) -> (Setup, Flow) {
    let mut d = domain([0.0; 3], [2.0, 0.5, DX], DX);
    d.faces.x_min = FaceCondition::Velocity { velocity: [1.0, 0.0, 0.0], profile: Profile::Uniform };
    d.faces.x_max = FaceCondition::Pressure { pressure: 0.0, outflow: Outflow::Uniform };
    d.faces.y_min = FaceCondition::Slip {};
    d.faces.y_max = FaceCondition::Slip {};
    d.faces.z_min = FaceCondition::Periodic {};
    d.faces.z_max = FaceCondition::Periodic {};
    let dt = ULAT * DX;
    d.inlet_ramp = ramp_steps.map(|n| n * dt);
    let p: Procedure = collide(steps as f64 * dt, dt, None, Collision::Hrr);
    prepare(&model(1.0, 0.02, d, p), &[])
}

fn inlet_speeds(flow: &Flow) -> Vec<f64> {
    flow.open_cells()
        .iter()
        .filter_map(|oc| match oc.target {
            Target::Velocity(v) => Some(v[0]),
            _ => None,
        })
        .collect()
}

#[test]
fn no_ramp_is_the_unramped_run_bit_for_bit() {
    let (setup, mut none) = channel(None, 30);
    run_all(&setup, &mut none);
    for ramp in [0.0, 0.4] {
        let (setup, mut ramped) = channel(Some(ramp), 30);
        assert_eq!(ramped.inlet_ramp_steps(), if ramp == 0.0 { 0 } else { 1 }, "ramp {ramp}");
        run_all(&setup, &mut ramped);
        assert_eq!(ramped.mass().to_bits(), none.mass().to_bits(), "ramp {ramp}");
        assert_eq!(inlet_speeds(&ramped), inlet_speeds(&none), "ramp {ramp}");
    }
}

#[test]
fn the_inlet_runs_at_the_ramps_factor_and_at_full_speed_after_it() {
    let (setup, mut full) = channel(None, 40);
    let base = inlet_speeds(&full);
    assert!(base.iter().all(|v| *v > 0.0), "{base:?}");
    let (_, mut flow) = channel(Some(20.0), 40);
    assert_eq!(flow.inlet_ramp_steps(), 20);
    for n in 1..=40u64 {
        flow.step().unwrap();
        let want = lattice::wall_ramp(n, 20);
        for (v, b) in inlet_speeds(&flow).iter().zip(&base) {
            assert!((v - b * want).abs() <= 1e-15 * b.abs(), "step {n}: {v} vs {}", b * want);
        }
    }
    assert_eq!(inlet_speeds(&flow), base, "full speed after the ramp, exactly");
    run_all(&setup, &mut full);
    assert_ne!(flow.mass().to_bits(), full.mass().to_bits(), "the ramp changed the run");
}

#[test]
fn a_ramped_inlet_starts_with_a_smaller_density_pulse_than_an_impulsive_one() {
    // The largest departure of the density from 1 over the channel in the first steps.
    let pulse = |ramp: Option<f64>| {
        let (setup, mut flow) = channel(ramp, 400);
        let n: usize = flow.grid().cells.iter().product();
        let mut worst = 0.0f64;
        for _ in 0..setup.steps {
            flow.step().unwrap();
            for cell in 0..n {
                if let Some((rho, _)) = flow.moments(cell) {
                    worst = worst.max((rho - 1.0).abs());
                }
            }
        }
        worst
    };
    let (impulsive, gentle) = (pulse(None), pulse(Some(400.0)));
    assert!(impulsive > 1e-3 && gentle < 0.5 * impulsive, "impulsive {impulsive:e} gentle {gentle:e}");
}

#[test]
fn hostile_ramps_are_refused() {
    for bad in [f64::NAN, -1.0, f64::INFINITY, f64::NEG_INFINITY] {
        let mut d = domain([0.0; 3], [2.0, 0.5, DX], DX);
        d.inlet_ramp = Some(bad);
        let mut m = fieldcraft_model::Model::default();
        let err = m.add_fluid_domain("Flow", d).unwrap_err().to_string();
        assert!(err.contains("inlet ramp"), "{bad}: {err}");
    }
}
