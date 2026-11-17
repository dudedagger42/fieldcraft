//! A reproduction, not a gate: what the hybrid recursive-regularised collision (`sigma` below 1) does in the cells next to a
//! wall. Both tests are `#[ignore]` and print numbers; nothing is asserted, because the point is to show a problem.
//!
//! (a) The aligned plane channel, 8 cells wide, driven by a body force that gives a mean speed of 0.02 lattice units at
//! tau - 1/2 = 5e-3, with plain half-way bounce-back walls. The exact flow rate is known (12 nu U / H^2 gives the force,
//! the parabola gives the rate), so the flow-rate error is read straight: sigma = 0.95 against sigma = 1.0 (the fully
//! regularised collision) shows how much the blend costs in a flow that is a pure parabola.
//!
//! (b) The same channel with a pocket (2 cells wide, 4 deep) cut into one wall, tau - 1/2 = 5e-4, sigma 0.95: every
//! 500 steps the largest speed in the pocket is printed, and the run reports the step at which the solver stops it
//! (the lattice speed reaching the limit, or a non-finite value). The sigma = 1.0 and MRT controls run the same steps.
//!
//! Run: `cargo test --release -p fieldcraft-lbm --test hybrid_wall_cells -- --ignored --nocapture`
//! A few minutes for (a), under a minute for (b).

mod common;

use common::*;
use fieldcraft_lbm::backend::Backend;
use fieldcraft_lbm::{Flow, Limits, Obstacle, Setup};
use fieldcraft_model::{Collision, FaceCondition, HrrSigma, Procedure, Turbulence};

const PERIODIC: FaceCondition = FaceCondition::Periodic {};
/// The exact mean speed of the channel, in lattice units.
const U_LAT: f64 = 0.02;
/// The channel's width in cells.
const W: usize = 8;

#[derive(Clone, Copy)]
enum Scheme {
    Hybrid(f64),
    Mrt,
}

impl Scheme {
    fn label(self) -> String {
        match self {
            Scheme::Hybrid(s) => format!("hybrid sigma {s}"),
            Scheme::Mrt => "MRT".to_string(),
        }
    }
}

/// The channel (walls on the cell faces at y = 0 and y = W cells, periodic all round), with a pocket of `(width, depth)` cells cut
/// into the upper wall when asked. Returns the setup, the obstacles and the flow-rate reference per layer of cells along x.
fn build(tau_minus_half: f64, scheme: Scheme, pocket: Option<(usize, usize)>) -> Result<(Setup, Vec<Obstacle>, f64, bool), String> {
    let nu_lat = tau_minus_half / 3.0;
    let dx = 1.0 / W as f64;
    let dt = nu_lat * dx * dx;
    let wf = W as f64;
    let depth = pocket.map_or(0, |(_, d)| d);
    let ny = W + 4 + depth;
    let cells_x = if pocket.is_some() { 16.0 } else { 8.0 };
    let (lx, ly, lz) = (cells_x * dx, ny as f64 * dx, 4.0 * dx);
    let ho = wf * dx;
    let obstacles = match pocket {
        None => vec![obstacle("Wall0", cuboid([-3.0 * lx, ho, -dx], [3.0 * lx, ly, lz + dx]), &CUBOID_FACES)],
        Some((pw, pd)) => {
            let (x0, x1) = (4.0 * dx, (4 + pw) as f64 * dx);
            vec![
                obstacle("WallLeft", cuboid([-3.0 * lx, ho, -dx], [x0, ly, lz + dx]), &CUBOID_FACES),
                obstacle("WallRight", cuboid([x1, ho, -dx], [3.0 * lx, ly, lz + dx]), &CUBOID_FACES),
                obstacle("WallCap", cuboid([x0, ho + pd as f64 * dx, -dx], [x1, ly, lz + dx]), &CUBOID_FACES),
            ]
        }
    };
    let mut d = domain([0.0; 3], [lx, ly, lz], dx);
    for f in [&mut d.faces.x_min, &mut d.faces.x_max, &mut d.faces.y_min, &mut d.faces.y_max, &mut d.faces.z_min, &mut d.faces.z_max] {
        *f = PERIODIC;
    }
    d.obstacles = obstacles.iter().map(|o| o.body.clone()).collect();
    // the body force that gives the exact mean speed of a plane channel: 12 nu U / H^2
    let g_lat = 12.0 * nu_lat * U_LAT / (wf * wf);
    d.acceleration = [g_lat * dx / (dt * dt), 0.0, 0.0];
    let make = |turbulence: Option<Turbulence>| {
        let mut p = collide(1e6 * dt, dt, None, if matches!(scheme, Scheme::Mrt) { Collision::Mrt } else { Collision::Hrr });
        if let Procedure::Fluid { sigma, turbulence: t, .. } = &mut p {
            if let Scheme::Hybrid(s) = scheme {
                *sigma = Some(HrrSigma::Value(s));
            }
            *t = turbulence;
        }
        model(1.0, 1.0, d.clone(), p)
    };
    // below tau - 1/2 = 5e-3 the setup wants a turbulence model; Vreman's model is exactly zero in a pure shear flow
    // (its tensor has one nonzero row), so it stays idle in the channel, and is named in the output
    let (m, vreman) = if tau_minus_half < 5e-3 - 1e-12 {
        (make(Some(Turbulence::Vreman { constant: 0.18, prandtl: 0.9, wall_model: None })), true)
    } else {
        (make(None), false)
    };
    let setup = Setup::new(&m, None, &obstacles, &Limits::default()).map_err(|e| e.to_string())?;
    Ok((setup, obstacles, U_LAT * wf * (lx / dx) * 4.0, vreman))
}

/// The flow-rate error and the largest speed (all cells, then the pocket's) of the fluid cells in the channel rows.
fn read(flow: &mut Flow, fluid: &[bool], reference: f64) -> Result<(f64, f64, f64, bool), String> {
    let f = Backend::fields(flow).map_err(|e| e.to_string())?;
    let g = *flow.grid();
    let (mut sum, mut all, mut pocket, mut finite) = (0.0f64, 0.0f64, 0.0f64, true);
    for c in (0..fluid.len()).filter(|&c| fluid[c]) {
        let u = f.velocity[c];
        let m = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
        finite &= m.is_finite() && f.density[c].is_finite();
        all = all.max(m);
        if g.coords(c)[1] >= W {
            pocket = pocket.max(m);
        } else {
            sum += u[0];
        }
    }
    Ok((sum / reference - 1.0, all, pocket, finite))
}

#[test]
#[ignore = "reproduction: prints the channel's flow-rate error for sigma 0.95 and 1.0 (cargo test --release -p fieldcraft-lbm --test hybrid_wall_cells -- --ignored --nocapture)"]
fn the_hybrid_blend_costs_flow_rate_in_a_plane_channel() {
    for scheme in [Scheme::Hybrid(0.95), Scheme::Hybrid(1.0), Scheme::Mrt] {
        let tau = 5e-3;
        let (setup, obstacles, reference, _) = match build(tau, scheme, None) {
            Ok(b) => b,
            Err(e) => {
                println!("CHANNEL {}: setup refused: {e}", scheme.label());
                continue;
            }
        };
        let mut flow = match Flow::new(&setup, &obstacles) {
            Ok(f) => f,
            Err(e) => {
                println!("CHANNEL {}: refused: {e}", scheme.label());
                continue;
            }
        };
        let fluid: Vec<bool> = (0..flow.grid().len()).map(|c| Backend::is_fluid(&flow, c)).collect();
        let mut history = Vec::new();
        let mut done = 0u64;
        // 40,000 steps are ten diffusion times of the channel (1.5 W^2 / (tau - 1/2) = 19,200 is one and a half of them)
        while done < 40_000 {
            if let Err(e) = Backend::advance(&mut flow, 4_000) {
                println!("CHANNEL {}: stopped by the solver after step {done}: {e}", scheme.label());
                break;
            }
            done += 4_000;
            match read(&mut flow, &fluid, reference) {
                Ok((err, ..)) => history.push(format!("{done}:{:+.3}%", 100.0 * err)),
                Err(e) => {
                    println!("CHANNEL {}: could not read the fields: {e}", scheme.label());
                    break;
                }
            }
        }
        println!("CHANNEL {} tau-1/2 {tau:e} width {W} cells, flow-rate error by step: {}", scheme.label(), history.join(" "));
    }
}

#[test]
#[ignore = "reproduction: the pocket at tau - 1/2 = 5e-4 (cargo test --release -p fieldcraft-lbm --test hybrid_wall_cells -- --ignored --nocapture)"]
fn the_hybrid_blend_stops_a_pocket_in_the_wall() {
    let tau = 5e-4;
    for scheme in [Scheme::Hybrid(0.95), Scheme::Hybrid(1.0), Scheme::Mrt] {
        let (setup, obstacles, reference, vreman) = match build(tau, scheme, Some((2, 4))) {
            Ok(b) => b,
            Err(e) => {
                println!("POCKET {}: setup refused: {e}", scheme.label());
                continue;
            }
        };
        let mut flow = match Flow::new(&setup, &obstacles) {
            Ok(f) => f,
            Err(e) => {
                println!("POCKET {}: refused: {e}", scheme.label());
                continue;
            }
        };
        let fluid: Vec<bool> = (0..flow.grid().len()).map(|c| Backend::is_fluid(&flow, c)).collect();
        let mut series = Vec::new();
        let (mut done, mut stop, mut eddy) = (0u64, None::<String>, 0.0f64);
        while done < 8_000 && stop.is_none() {
            if let Err(e) = Backend::advance(&mut flow, 500) {
                stop = Some(format!("stopped by the solver after step {done}: {e}"));
                break;
            }
            done += 500;
            eddy = eddy.max(flow.eddy_viscosity_ratio());
            match read(&mut flow, &fluid, reference) {
                Ok((_, all, pocket, finite)) => {
                    series.push(format!("{done}:{pocket:.3e}"));
                    if !finite || all > 0.3 {
                        stop = Some(format!("speed {all:.3} (or a non-finite number) at step {done}"));
                    }
                }
                Err(e) => stop = Some(format!("could not read the fields: {e}")),
            }
        }
        println!(
            "POCKET {} tau-1/2 {tau:e} (turbulence model {}, largest eddy viscosity over molecular {eddy:.3e}) pocket 2x4 cells, width {W}: {}; pocket max |u| by step: {}",
            scheme.label(),
            if vreman { "Vreman, zero in pure shear" } else { "none" },
            stop.unwrap_or_else(|| "ran to step 8000 without stopping".to_string()),
            series.join(" ")
        );
    }
}
