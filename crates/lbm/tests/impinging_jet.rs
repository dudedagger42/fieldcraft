//! A round jet from a pipe onto a flat plate, quarter domain (Cooper et al. 1993, the round
//! impinging jet): a smoke study of the set-up, not a gate. It reports the actual inflow against the
//! bore's area times the bulk velocity (a staircase bore feeds 80–120 % of the true area, so the
//! flow and the Reynolds number can shift by that much), the inlet's density deviation and the
//! fastest cell, as the flow develops.
//!
//! `FIELDCRAFT_JET_STUDY=1 cargo test --release -p fieldcraft-lbm --test impinging_jet -- --ignored
//! --nocapture smoke` (`FIELDCRAFT_JET_N` cells across the pipe diameter, default 32;
//! `FIELDCRAFT_JET_RE`, default 23000; `FIELDCRAFT_JET_STEPS`, default 20000;
//! `FIELDCRAFT_JET_LAYER=1` adds the absorbing layer; the inflow is scaled to the bore's area,
//! `FIELDCRAFT_JET_FED=1` to the mean over the fed cells).

mod common;

use common::*;
use fieldcraft_lbm::{Limits, Obstacle, Setup, Target};
use fieldcraft_model::{
    AbsorbingLayer, Collision, FaceCondition, HrrSigma, Outflow, Procedure, Profile, RadialInflow, RadialScale, Turbulence, WallModel,
};

fn var(name: &str, default: f64) -> f64 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// The set-up: lengths in pipe diameters (D = 1), bulk velocity 1; x along the jet from the pipe's
/// inlet (x = 0), a pipe of 2 D in a thin flange, the plate `h` D past the pipe's exit; the axis on
/// the domain's edge (y = z = 0, symmetry faces), pressure outlets at 4 D radius.
fn jet(n: f64, re: f64, h: f64, layer: bool) -> (fieldcraft_model::Model, Vec<Obstacle>) {
    let (pipe_len, width, wall) = (2.0, 4.0, 0.05);
    let mut d = domain([0.0; 3], [pipe_len + h, width, width], 1.0 / n);
    d.faces.x_min = FaceCondition::Velocity { velocity: [1.0, 0.0, 0.0], profile: Profile::Radial };
    d.faces.x_max = FaceCondition::Wall {};
    d.faces.y_min = FaceCondition::Slip {};
    d.faces.z_min = FaceCondition::Slip {};
    d.faces.y_max = FaceCondition::Pressure { pressure: 0.0, outflow: Outflow::Convective };
    d.faces.z_max = FaceCondition::Pressure { pressure: 0.0, outflow: Outflow::Convective };
    d.radial_inflow = RadialInflow::power_law([0.0, 0.0, 0.0], 0.5, if re > 4.0e4 { 7.0 } else { 6.6 }, 256).ok().map(|r| RadialInflow {
        scale: if std::env::var("FIELDCRAFT_JET_FED").is_ok_and(|v| v == "1") { RadialScale::Fed } else { RadialScale::Area },
        ..r
    });
    d.absorbing = layer.then_some(AbsorbingLayer { thickness: None, strength: None, filter_time: None });
    let flange = obstacle("Flange", tube(0.0, 2.0 / n, [0.0, 0.0], 0.5, 8.0, 128), &["inner", "outer", "start", "end"]);
    let pipe = obstacle("Pipe", tube(0.0, pipe_len, [0.0, 0.0], 0.5, 0.5 + wall, 128), &["inner", "outer", "start", "end"]);
    d.obstacles = vec!["Flange".into(), "Pipe".into()];
    let dt = 0.08 / n; // the bulk velocity is 0.08 lattice units
    let steps = var("FIELDCRAFT_JET_STEPS", 20000.0);
    let p = Procedure::Fluid {
        domain: "Flow".into(),
        end_time: Some(steps * dt),
        steps: None,
        time_step: Some(dt),
        frames: 2,
        magic: None,
        sigma: Some(HrrSigma::Value(0.95)),
        collision: Some(Collision::Hrr),
        turbulence: Some(Turbulence::SpalartAllmaras { prandtl: 0.9, wall_model: Some(WallModel::SpaldingBounce), inflow_ratio: 3.0, ddes: None }),
        steady: None,
    };
    (model(1.0, 1.0 / re, d, p), vec![flange, pipe])
}

#[test]
#[ignore = "minutes: set FIELDCRAFT_JET_STUDY=1 (docs/verification.md, \"Long studies\")"]
fn smoke() {
    if std::env::var("FIELDCRAFT_JET_STUDY").is_err() {
        return;
    }
    let (n, re) = (var("FIELDCRAFT_JET_N", 32.0), var("FIELDCRAFT_JET_RE", 23000.0));
    let layer = std::env::var("FIELDCRAFT_JET_LAYER").is_ok_and(|v| v == "1");
    let (m, obstacles) = jet(n, re, 2.0, layer);
    let limits = Limits { max_cell_updates: u64::MAX, max_steps: u64::MAX, ..Limits::default() };
    let setup = Setup::new(&m, None, &obstacles, &limits).unwrap();
    let mut flow = fieldcraft_lbm::Flow::new(&setup, &obstacles).unwrap();
    let grid = *flow.grid();
    let u = setup.to_lattice([1.0, 0.0, 0.0])[0];
    println!("jet smoke: N {n}, Re {re}, {:?} cells, bulk velocity {u:.4} lattice, steps {}, warnings {:?}", grid.cells, setup.steps, setup.warnings);
    // The actual inflow: the sum of the inlet cells' velocities (the inlet face's cell area is dx²)
    // against the bulk velocity times the quarter bore's area.
    let (mut fed, mut sum) = (0usize, 0.0);
    for oc in flow.open_cells().iter().filter(|oc| oc.axis == 0 && oc.inward > 0.0) {
        if let Target::Velocity(v) = oc.target
            && v[0] > 0.0
        {
            fed += 1;
            sum += v[0];
        }
    }
    let quarter = std::f64::consts::PI * 0.25 / 4.0 * n * n; // cells
    println!("inflow: {fed} cells fed, {:.4} of the true quarter bore area, flux {:.4} of U·πR²/4", fed as f64 / quarter, sum / (u * quarter));
    let report = (setup.steps / 20).max(1);
    for s in 1..=setup.steps {
        flow.step().unwrap();
        if s % report == 0 || s == setup.steps {
            let [nx, ny, nz] = grid.cells;
            let (mut inlet, mut fastest) = (0.0f64, 0.0f64);
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..4.min(nx) {
                        if let Some((r, _)) = flow.moments(grid.index([i, j, k])) {
                            inlet = inlet.max((r - 1.0).abs());
                        }
                    }
                    for i in 0..nx {
                        if let Some((_, v)) = flow.moments(grid.index([i, j, k])) {
                            fastest = fastest.max(v.iter().map(|c| c * c).sum::<f64>().sqrt());
                        }
                    }
                }
            }
            println!("step {s}: inlet columns 0–3 max |ρ − 1| {inlet:.3e}; fastest cell {fastest:.4} lattice");
            assert!(inlet.is_finite() && fastest.is_finite(), "the flow blew up at step {s}");
        }
    }
}
