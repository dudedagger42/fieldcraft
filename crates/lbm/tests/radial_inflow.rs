//! A velocity inlet with a radial profile: a pipe through a flange, fed from the box's inlet face.
//!
//! The box face carries one condition, so a jet from a pipe is a velocity face with a solid
//! flange (a tube with a large outer radius) over it: the inlet feeds only the fluid cells under
//! the bore. The profile is the 1/n power law of a turbulent pipe, scaled so the given velocity is
//! the bulk velocity of the cells it reaches.

mod common;

use common::*;
use fieldcraft_lbm::Target;
use fieldcraft_model::{FaceCondition, Outflow, Profile, RadialInflow};

const SPACING: f64 = 0.05;
const BORE: f64 = 0.4;
const AXIS: [f64; 2] = [1.0, 1.0];

/// A 1.5 × 2 × 2 box, the flange and pipe wall over the inlet face (x from 0 to 0.5, bore radius
/// 0.4 about the axis through (y, z) = (1, 1), outer radius large enough to cover the face), a
/// radial inlet of bulk velocity 0.1 and a pressure outlet.
fn jet() -> (fieldcraft_model::Model, fieldcraft_lbm::Obstacle) {
    let mut d = domain([0.0; 3], [1.5, 2.0, 2.0], SPACING);
    d.faces.x_min = FaceCondition::Velocity { velocity: [0.1, 0.0, 0.0], profile: Profile::Radial };
    d.faces.x_max = FaceCondition::Pressure { pressure: 0.0, outflow: Outflow::Uniform };
    for f in [&mut d.faces.y_min, &mut d.faces.y_max, &mut d.faces.z_min, &mut d.faces.z_max] {
        *f = FaceCondition::Slip {};
    }
    d.obstacles = vec!["Pipe".into()];
    d.radial_inflow = Some(RadialInflow::power_law([0.0, AXIS[0], AXIS[1]], BORE, 6.6, 256).unwrap());
    let pipe = obstacle("Pipe", tube(-0.5, 0.5, AXIS, BORE, 2.0, 128), &["inner", "outer", "start", "end"]);
    (model(1.0, 0.01, d, collide(1.0, 0.025, None, fieldcraft_model::Collision::Trt)), pipe)
}

/// The inlet cells are the fluid cells under the bore, none behind the flange, and their velocities
/// are the power law scaled to the given bulk velocity.
#[test]
fn a_radial_inlet_feeds_only_the_bore_at_the_bulk_velocity() {
    let (m, pipe) = jet();
    let (setup, flow) = prepare(&m, &[pipe]);
    let grid = *flow.grid();
    let lattice = setup.to_lattice([0.1, 0.0, 0.0])[0];
    let inlet: Vec<(f64, f64)> = flow
        .open_cells()
        .iter()
        .filter(|oc| oc.axis == 0 && oc.inward > 0.0)
        .filter_map(|oc| {
            let c = grid.centre(oc.cell as usize);
            let r = (c[1] - AXIS[0]).hypot(c[2] - AXIS[1]);
            match oc.target {
                Target::Velocity(v) => Some((r, v[0])),
                _ => None,
            }
        })
        .collect();
    let fed: Vec<&(f64, f64)> = inlet.iter().filter(|(_, u)| *u > 0.0).collect();
    assert!(!fed.is_empty(), "no cell under the bore is fed");
    let bore_cells = std::f64::consts::PI * BORE * BORE / (SPACING * SPACING);
    assert!(
        fed.len() as f64 > 0.8 * bore_cells && (fed.len() as f64) < 1.2 * bore_cells,
        "{} cells fed, the bore holds about {bore_cells:.0}",
        fed.len()
    );
    assert!(
        inlet.iter().all(|(r, _)| *r < BORE + SPACING),
        "a cell behind the flange is an inlet cell: {:?}",
        inlet.iter().map(|x| x.0).fold(0.0, f64::max)
    );
    let mean = fed.iter().map(|(_, u)| u).sum::<f64>() / fed.len() as f64;
    assert!((mean / lattice - 1.0).abs() < 1e-12, "the mean over the fed cells is {} of the given velocity", mean / lattice);
    // The power law: the centre runs at (n + 1)(2n + 1) / 2n² of the bulk, the cell nearest the wall slower.
    let peak = fed.iter().map(|(_, u)| *u).fold(0.0, f64::max) / lattice;
    let near_wall = fed.iter().filter(|(r, _)| *r > BORE - 1.5 * SPACING).map(|(_, u)| *u).fold(f64::INFINITY, f64::min) / lattice;
    assert!((peak / 1.2373 - 1.0).abs() < 0.05, "peak {peak}");
    assert!(near_wall < 0.8, "the cell next to the wall runs at {near_wall} of the bulk");
}

/// The jet runs: 100 steps with the flange over the face stay finite, and the bore's centre
/// cell moves with the inlet (a smoke test; the developed profile is for the jet studies).
#[test]
fn a_jet_from_a_radial_inlet_runs() {
    let (m, pipe) = jet();
    let (setup, mut flow) = prepare(&m, &[pipe]);
    let lattice = setup.to_lattice([0.1, 0.0, 0.0])[0];
    for _ in 0..100 {
        flow.step().unwrap();
    }
    let grid = *flow.grid();
    let cell = grid.cell_at([0.3, AXIS[0], AXIS[1]]).unwrap();
    let (rho, u) = flow.moments(cell).unwrap();
    assert!(rho.is_finite() && (rho - 1.0).abs() < 0.05, "{rho}");
    // Started from rest, so the bore is still in its start-up transient (the first pressure wave overshoots).
    assert!(u[0] > 0.5 * lattice && u[0] < 2.0 * lattice, "{} of the bulk velocity", u[0] / lattice);
}

/// Hostile tables: a spike on four cells among thousands of near-zero factors would run one cell
/// far past the lattice speed limit, so the flow refuses it with the profile named; the table
/// itself is refused when it is not finite, not increasing or too long (the model's checks).
#[test]
fn an_uneven_radial_table_is_refused_before_the_run() {
    let (mut m, pipe) = jet();
    let mut d = m.fluid_domain("Flow").unwrap().clone();
    // The four diagonal cells next to the axis sit at r = 0.0354 (axis at a cell corner).
    d.radial_inflow = Some(RadialInflow {
        axis_point: [0.0, AXIS[0], AXIS[1]],
        table: vec![[0.0, 1e-9], [0.0353, 1e-9], [0.03536, 1e9], [0.0354, 1e-9], [0.4, 1e-9]],
    });
    m.replace_fluid_domain("Flow", d).unwrap();
    let setup = fieldcraft_lbm::Setup::new(&m, None, std::slice::from_ref(&pipe), &fieldcraft_lbm::Limits::default()).unwrap();
    let e = fieldcraft_lbm::Flow::new(&setup, &[pipe]).err().map(|e| e.to_string());
    assert!(e.as_deref().is_some_and(|e| e.contains("radial inflow") && e.contains("lattice speed") && e.contains("x_min")), "{e:?}");
}

/// Without the profile nothing changes: a uniform inlet's cells hold the given velocity to the
/// bit, and a parabolic inlet's hold it times 4 s (1 − s) per direction.
#[test]
fn the_uniform_and_parabolic_inlets_are_unchanged() {
    for profile in [Profile::Uniform, Profile::Parabolic] {
        let mut d = domain([0.0; 3], [1.0, 1.0, 0.5], SPACING);
        d.faces.x_min = FaceCondition::Velocity { velocity: [0.1, 0.0, 0.0], profile };
        d.faces.x_max = FaceCondition::Pressure { pressure: 0.0, outflow: Outflow::Uniform };
        let m = model(1.0, 0.01, d, collide(1.0, 0.025, None, fieldcraft_model::Collision::Trt));
        let (setup, flow) = prepare(&m, &[]);
        let u = setup.to_lattice([0.1, 0.0, 0.0])[0];
        let grid = *flow.grid();
        let inlet: Vec<_> = flow.open_cells().iter().filter(|oc| oc.axis == 0 && oc.inward > 0.0).collect();
        assert_eq!(inlet.len(), 20 * 10, "{profile:?}");
        for oc in inlet {
            let Target::Velocity(v) = oc.target else { panic!("not a velocity cell") };
            let [_, j, k] = grid.coords(oc.cell as usize);
            let s = |i: usize, n: usize| (i as f64 + 0.5) / n as f64;
            // The box's z faces are walls, so the parabola runs across y and z both.
            let parabola = |i: usize, n: usize| 4.0 * s(i, n) * (1.0 - s(i, n));
            let scale = if profile == Profile::Parabolic { 1.0 * parabola(j, 20) * parabola(k, 10) } else { 1.0 };
            assert_eq!(v[0].to_bits(), (u * scale).to_bits(), "{profile:?} at ({j}, {k})");
        }
    }
}
