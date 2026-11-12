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
//! `FIELDCRAFT_JET_FED=1` to the mean over the fed cells; `FIELDCRAFT_JET_RIM_TAPER=1` reads the
//! profile at the distance from the staircase wall).

mod common;

use common::cases::study_wall_law;
use common::*;
use fieldcraft_lbm::{Limits, Obstacle, Setup, Target};
use fieldcraft_model::{AbsorbingLayer, Collision, FaceCondition, HrrSigma, Outflow, Procedure, Profile, RadialInflow, RadialScale, Turbulence};

/// The hybrid collision's σ.
const SIGMA: f64 = 0.95;

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
        wall_taper: std::env::var("FIELDCRAFT_JET_RIM_TAPER").is_ok_and(|v| v == "1"),
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
        sigma: Some(HrrSigma::Value(SIGMA)),
        collision: Some(Collision::Hrr),
        turbulence: Some(Turbulence::SpalartAllmaras { prandtl: 0.9, wall_model: Some(study_wall_law().0), inflow_ratio: 3.0, ddes: None }),
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
    let layer_cells = m
        .fluid_domain("Flow")
        .and_then(|d| d.absorbing.as_ref().map(|a| a.thickness.map_or(fieldcraft_model::fluids::DEFAULT_ABSORBING_CELLS as f64, |t| t * n)));
    println!(
        "header: σ {SIGMA} (hybrid collision); wall law {}; absorbing layer {layer_cells:?} cells (None: no layer); τ − ½ from the warning above; D = {n} cells",
        study_wall_law().1
    );
    // Where a cell sits in the face plane (D = 1, the axis at y = z = 0, the bore's rim at r = 0.5).
    let place = |cell: usize| {
        let [x, y, z] = grid.centre(cell);
        let r = y.hypot(z);
        let label = if r < 0.15 {
            "axis"
        } else if (r - 0.5).abs() < 0.1 {
            "rim"
        } else if y.min(z) < 3.0 / n {
            "symmetry edge"
        } else if r < 0.5 {
            "bore"
        } else {
            "outside the bore"
        };
        format!("x {x:.3} y {y:.3} z {z:.3} r {r:.3} ({label})")
    };
    let report = var("FIELDCRAFT_JET_REPORT", 1000.0) as u64;
    let clock = std::time::Instant::now();
    for s in 1..=setup.steps {
        flow.step().unwrap();
        if s % report == 0 || s == setup.steps {
            let [nx, ny, nz] = grid.cells;
            // The inlet columns 0–3: the face mean of ρ − 1 (physics: the pipe's pressure offset)
            // and the largest deviation from it (the mode that should decay).
            let (mut sum, mut count) = (0.0, 0usize);
            let (mut fastest, mut fastest_cell) = (0.0f64, 0usize);
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..4.min(nx) {
                        if let Some((r, _)) = flow.moments(grid.index([i, j, k])) {
                            sum += r - 1.0;
                            count += 1;
                        }
                    }
                    for i in 0..nx {
                        let cell = grid.index([i, j, k]);
                        if let Some((_, v)) = flow.moments(cell) {
                            let speed = v.iter().map(|c| c * c).sum::<f64>().sqrt();
                            if speed > fastest {
                                (fastest, fastest_cell) = (speed, cell);
                            }
                        }
                    }
                }
            }
            let mean = sum / count.max(1) as f64;
            let (mut deviation, mut deviation_cell) = (0.0f64, 0usize);
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..4.min(nx) {
                        let cell = grid.index([i, j, k]);
                        if let Some((r, _)) = flow.moments(cell) {
                            let d = (r - 1.0 - mean).abs();
                            if d > deviation {
                                (deviation, deviation_cell) = (d, cell);
                            }
                        }
                    }
                }
            }
            println!(
                "step {s}: inlet columns 0–3 mean ρ − 1 {mean:+.3e}, max deviation from the mean {deviation:.3e} at {}; fastest cell {fastest:.4} lattice at {}; {:.0} s",
                place(deviation_cell),
                place(fastest_cell),
                clock.elapsed().as_secs_f64()
            );
            assert!(mean.is_finite() && deviation.is_finite() && fastest.is_finite(), "the flow blew up at step {s}");
            if s == setup.steps {
                // Columns 4…10 (past the flange): in each, the mean of ρ − 1 over its fluid cells
                // and the largest deviation from it, over every cell and away from the rim cells
                // (r within 0.1 D of the bore's wall), each with the worst cell's place.
                for i in 4..11.min(nx) {
                    let cells: Vec<(usize, f64)> = (0..nz)
                        .flat_map(|k| (0..ny).map(move |j| (j, k)))
                        .filter_map(|(j, k)| {
                            let cell = grid.index([i, j, k]);
                            flow.moments(cell).map(|(r, _)| (cell, r - 1.0))
                        })
                        .collect();
                    let section_mean = cells.iter().map(|c| c.1).sum::<f64>() / cells.len().max(1) as f64;
                    let worst = |away_from_rim: bool| {
                        cells
                            .iter()
                            .filter(|(cell, _)| {
                                let [_, y, z] = grid.centre(*cell);
                                !away_from_rim || (y.hypot(z) - 0.5).abs() >= 0.1
                            })
                            .map(|&(cell, r)| (cell, (r - section_mean).abs()))
                            .fold((0usize, 0.0f64), |a, b| if b.1 > a.1 { b } else { a })
                    };
                    let (all, away) = (worst(false), worst(true));
                    println!(
                        "column {i}: mean ρ − 1 {section_mean:+.3e}; max deviation from the mean, all cells {:.3e} at {}; away from the rim {:.3e} at {}",
                        all.1,
                        place(all.0),
                        away.1,
                        place(away.0)
                    );
                }
            }
        }
    }
}
