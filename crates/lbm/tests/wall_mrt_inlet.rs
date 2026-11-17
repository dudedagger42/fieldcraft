//! A study (print-only, no assertions): the wall-layer MRT remedy (`Flow::set_wall_layer_mrt`) next to a velocity inlet.
//!
//! The wall cells that touch the inlet face are MRT cells beside an inlet, the combination that needed the inlet fix when MRT
//! ran everywhere. The case: a channel with a uniform velocity inlet at x min (U = ULAT, 0.014 by default as in the slot series), a pressure outlet at x max,
//! two plates of solid cells (the channel walls, starting at the inlet face so the wall cells touch it), periodic y, the hybrid collision at sigma 0.95 with Spalart-Allmaras (the setup needs a model below
//! tau - 1/2 = 5e-3), tau - 1/2 = 1.8e-4, and the default absorbing layer (10 cells) at the inlet and the outlet as every
//! small-tau case here has. Each run takes 20,000 steps with the remedy off and then on, and prints one line: the steps
//! done (the run stops at the first non-finite or runaway field), the peak speed, the largest |rho - 1|, and the mass flux
//! through the planes at a quarter and three quarters of the length, with their ratio (pass: within 1e-3 of 1).
//!
//! `cargo test --release -p fieldcraft-lbm --test wall_mrt_inlet -- --ignored --nocapture wall_mrt_inlet`
//! Environment: `STEPS` (20000), `TAU_MINUS_HALF` (1.8e-4), `LAYER` (1; 0 runs without the absorbing layer), `LENGTH` (160
//! cells), `HEIGHT` (16 cells), `SIGMA` (0.95), `ULAT` (0.014).

mod common;

use common::*;
use fieldcraft_model::{AbsorbingLayer, Collision, FaceCondition, HrrSigma, Outflow, Procedure, Profile, Turbulence};

fn env(name: &str, default: f64) -> f64 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// The mass flux through the plane at cell column `i`: the sum of rho u_x over its fluid cells, and the largest |rho - 1|.
fn plane(flow: &fieldcraft_lbm::Flow, i: usize) -> f64 {
    let g = flow.grid();
    let mut sum = 0.0;
    for c in 0..g.len() {
        if g.coords(c)[0] == i
            && flow.is_fluid(c)
            && let Some((rho, v)) = flow.moments(c)
        {
            sum += rho * v[0];
        }
    }
    sum
}

fn run(wall_mrt: bool, tau_minus_half: f64, steps: u64) -> String {
    let nu = tau_minus_half / 3.0;
    let u = env("ULAT", 0.014);
    // the channel is the gap between two plates (solid cells) that start at the inlet face, as the slot's plates do: its wall cells
    // touch the inlet face; z is the wall-normal axis, y (4 cells) is periodic
    let (nx, ny, h) = (env("LENGTH", 160.0) as usize, 4usize, env("HEIGHT", 16.0) as usize);
    let nz = h + 16;
    let mut d = domain([0.0; 3], [nx as f64, ny as f64, nz as f64], 1.0);
    d.faces.y_min = FaceCondition::Periodic {};
    d.faces.y_max = FaceCondition::Periodic {};
    d.faces.z_min = FaceCondition::Slip {};
    d.faces.z_max = FaceCondition::Slip {};
    d.obstacles = vec!["PlateLow".into(), "PlateHigh".into()];
    let (xl, xh, yl, yh) = (-2.0, nx as f64 + 2.0, -2.0, ny as f64 + 2.0);
    let obstacles = vec![
        obstacle("PlateLow", cuboid([xl, yl, -2.0], [xh, yh, 8.0]), &CUBOID_FACES),
        obstacle("PlateHigh", cuboid([xl, yl, 8.0 + h as f64], [xh, yh, nz as f64 + 2.0]), &CUBOID_FACES),
    ];
    d.faces.x_min = FaceCondition::Velocity { velocity: [u, 0.0, 0.0], profile: Profile::Parabolic };
    d.faces.x_max = FaceCondition::Pressure { pressure: 0.0, outflow: Outflow::Uniform };
    d.initial_velocity = [u, 0.0, 0.0];
    if env("LAYER", 1.0) > 0.5 {
        d.absorbing = Some(AbsorbingLayer { thickness: None, strength: None, filter_time: None });
    }
    let mut p = collide(steps as f64, 1.0, None, Collision::Hrr);
    if let Procedure::Fluid { turbulence, sigma, .. } = &mut p {
        *turbulence = Some(Turbulence::SpalartAllmaras { prandtl: 0.9, wall_model: None, inflow_ratio: 3.0, ddes: None });
        *sigma = Some(HrrSigma::Value(env("SIGMA", 0.95)));
    }
    let (_, mut flow) = prepare(&model(1.0, nu, d, p), &obstacles);
    flow.set_wall_layer_mrt(wall_mrt);
    let (a, b) = (nx / 4, 3 * nx / 4);
    let mut done = 0u64;
    let mut stop = String::new();
    let mut peak = 0.0f64;
    let mut worst_rho = 0.0f64;
    for s in 0..steps {
        if let Err(e) = flow.step() {
            stop = format!(" STOPPED at step {s}: {e}");
            break;
        }
        done = s + 1;
        if done % 250 == 0 || done == steps {
            peak = 0.0;
            worst_rho = 0.0;
            for c in 0..flow.grid().len() {
                if flow.is_fluid(c)
                    && let Some((rho, v)) = flow.moments(c)
                {
                    peak = peak.max((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt());
                    worst_rho = worst_rho.max((rho - 1.0).abs());
                }
            }
            if !peak.is_finite() || peak > 0.5 {
                stop = format!(" RUNAWAY at step {done}: peak speed {peak}");
                break;
            }
        }
    }
    let (fa, fb) = (plane(&flow, a), plane(&flow, b));
    format!(
        "WALLMRT-INLET wall_mrt {wall_mrt} tau-1/2 {tau_minus_half:e} sigma {} layer {} steps {done} of {steps}{stop}: peak speed {peak:.4}, largest |rho-1| {worst_rho:.3e}, flux at x={a} {fa:.6}, at x={b} {fb:.6}, ratio {:.6} (within 1e-3: {})",
        env("SIGMA", 0.95),
        env("LAYER", 1.0),
        fb / fa,
        ((fb / fa) - 1.0).abs() < 1e-3
    )
}

#[test]
#[ignore = "print-only study: reads STEPS, TAU_MINUS_HALF, LAYER, LENGTH, HEIGHT, SIGMA"]
fn wall_mrt_inlet() {
    let steps = env("STEPS", 20_000.0) as u64;
    let tau = env("TAU_MINUS_HALF", 1.8e-4);
    for wall_mrt in [false, true] {
        println!("{}", run(wall_mrt, tau, steps));
    }
}
