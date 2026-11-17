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
//! cells), `HEIGHT` (16 cells), `SIGMA` (0.95), `ULAT` (0.014), `HEIGHT2` (the gap past column `STRIP`, default HEIGHT), `STRIP` (33), `UNIFORM`, `EVERY` (10, the sampling interval), `SHOW` (1000, the pair gap list), `AVG` (2000, the time-mean flux window), `BUDGET` (0, steps between the mass and flux budget lines), `INIT` (1, the initial velocity as a share of ULAT), `ARMS` (1: the three-arm strip study, 2: WALL_MRT only).

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

/// One arm: `sigma1` (the wall-layer sigma = 1 study knob, 0 off), `wall_mrt`. The channel is the gap between two plates that start at
/// the inlet face: `h` cells high for the first `strip` columns (the strip: solid below, one more plate above), then `h2` cells high
/// (the region it opens into). Returns the report line.
fn run(sigma1: u8, wall_mrt: bool, tau_minus_half: f64, steps: u64) -> String {
    let nu = tau_minus_half / 3.0;
    let u = env("ULAT", 0.014);
    let (nx, ny) = (env("LENGTH", 160.0) as usize, 4usize);
    let (h, h2, strip) = (env("HEIGHT", 16.0) as usize, env("HEIGHT2", env("HEIGHT", 16.0)) as usize, env("STRIP", 33.0));
    let (gap_lo, nz) = (8.0, h.max(h2) + 16);
    let mut d = domain([0.0; 3], [nx as f64, ny as f64, nz as f64], 1.0);
    d.faces.y_min = FaceCondition::Periodic {};
    d.faces.y_max = FaceCondition::Periodic {};
    d.faces.z_min = FaceCondition::Slip {};
    d.faces.z_max = FaceCondition::Slip {};
    d.obstacles = vec!["PlateLow".into(), "PlateHigh".into(), "PlateHigh2".into()];
    let (xl, xh, yl, yh) = (-2.0, nx as f64 + 2.0, -2.0, ny as f64 + 2.0);
    let obstacles = vec![
        obstacle("PlateLow", cuboid([xl, yl, -2.0], [xh, yh, gap_lo]), &CUBOID_FACES),
        obstacle("PlateHigh", cuboid([xl, yl, gap_lo + h as f64], [strip, yh, nz as f64 + 2.0]), &CUBOID_FACES),
        obstacle("PlateHigh2", cuboid([strip, yl, gap_lo + h2 as f64], [xh, yh, nz as f64 + 2.0]), &CUBOID_FACES),
    ];
    let uniform = env("UNIFORM", 0.0) > 0.5;
    d.faces.x_min = FaceCondition::Velocity { velocity: [u, 0.0, 0.0], profile: if uniform { Profile::Uniform } else { Profile::Parabolic } };
    d.faces.x_max = FaceCondition::Pressure { pressure: 0.0, outflow: Outflow::Uniform };
    d.initial_velocity = [u * env("INIT", 1.0), 0.0, 0.0];
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
    flow.set_wall_layer_sigma_one(sigma1);
    let (a, b) = (nx / 4, 3 * nx / 4);
    let (mut done, mut stop) = (0u64, String::new());
    let (mut peak, mut worst_rho, mut worst_cell) = (0.0f64, 0.0f64, [0usize; 3]);
    let mut pair: Vec<String> = Vec::new();
    let mut last = String::new();
    // the plane fluxes (x 10 inside the strip, x a and b) averaged over the last AVG steps (every EVERY steps)
    let avg_steps = env("AVG", 2000.0) as u64;
    let (mut sum, mut n_avg) = ([0.0f64; 3], 0u64);
    // BUDGET (steps, default 0 = off): every BUDGET steps print the time means over the last BUDGET*2 steps of the flux at x 10, 40, 80, 120, the
    // outlet flux (the last column) and the inlet flux (column 0), and the total mass of the fluid cells, from samples every EVERY steps.
    let budget = env("BUDGET", 0.0) as u64;
    let cols = [10usize.min(nx - 1), 40usize.min(nx - 1), 80usize.min(nx - 1), 120usize.min(nx - 1), nx - 1, 0];
    let mut series: Vec<[f64; 6]> = Vec::new();
    let mut budget_lines: Vec<String> = Vec::new();
    let every = env("EVERY", 10.0) as u64;
    let show = env("SHOW", 1000.0) as u64;
    let g = *flow.grid();
    for s in 0..steps {
        if let Err(e) = flow.step() {
            stop = format!(" STOPPED at step {s}: {e}");
            break;
        }
        done = s + 1;
        if budget > 0 && done % every == 0 {
            let mut row = [0.0f64; 6];
            for (r, &i) in row.iter_mut().zip(cols.iter()) {
                *r = plane(&flow, i);
            }
            series.push(row);
            if done % budget == 0 {
                let w = ((2 * budget / every) as usize).min(series.len());
                let tail = &series[series.len() - w..];
                let m: Vec<f64> = (0..6).map(|k| tail.iter().map(|r| r[k]).sum::<f64>() / w as f64).collect();
                let mass: f64 = (0..g.len()).filter(|&c| flow.is_fluid(c)).filter_map(|c| flow.moments(c).map(|(rho, _)| rho)).sum();
                budget_lines.push(format!(
                    "BUDGET step {done}: time-mean over the last {} steps: flux x=10 {:.6} x=40 {:.6} x=80 {:.6} x=120 {:.6}; outlet column {:.6}; inlet column 0 {:.6}; net (inlet - outlet) {:+.3e}; total mass {mass:.4}",
                    w as u64 * every,
                    m[0], m[1], m[2], m[3], m[4], m[5], m[5] - m[4]
                ));
            }
        }
        if done % every == 0 && done + avg_steps > steps {
            for (acc, i) in sum.iter_mut().zip([10usize.min(nx - 1), a, b]) {
                *acc += plane(&flow, i);
            }
            n_avg += 1;
        }
        if done % every == 0 || done == steps {
            peak = 0.0;
            worst_rho = 0.0;
            let mut pair_gap = 0.0f64;
            for c in 0..g.len() {
                if !flow.is_fluid(c) {
                    continue;
                }
                if let Some((rho, v)) = flow.moments(c) {
                    peak = peak.max((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt());
                    if (rho - 1.0).abs() > worst_rho {
                        worst_rho = (rho - 1.0).abs();
                        worst_cell = g.coords(c);
                    }
                    // the 2 dx pair: this cell's density against the mean of its two x neighbours (fluid ones only)
                    let p = g.coords(c);
                    if p[0] >= 1 && p[0] + 1 < nx {
                        let (l, r) = (g.index([p[0] - 1, p[1], p[2]]), g.index([p[0] + 1, p[1], p[2]]));
                        if flow.is_fluid(l)
                            && flow.is_fluid(r)
                            && let (Some((rl, _)), Some((rr, _))) = (flow.moments(l), flow.moments(r))
                        {
                            pair_gap = pair_gap.max((rho - 0.5 * (rl + rr)).abs());
                        }
                    }
                }
            }
            last = format!(
                "last sample step {done}: peak speed {peak:.4}, largest |rho-1| {worst_rho:.3e} at cell {worst_cell:?}, 2dx pair {pair_gap:.2e}"
            );
            if done % show == 0 {
                pair.push(format!("{done}:{pair_gap:.2e}"));
            }
            if !peak.is_finite() || peak > 0.5 {
                stop = format!(" RUNAWAY at step {done}: peak speed {peak}");
                break;
            }
        }
    }
    let (fa, fb) = (plane(&flow, a), plane(&flow, b));
    for l in &budget_lines {
        println!("{l}");
    }
    format!(
        "WALLMRT-STRIP sigma1 {sigma1} wall_mrt {wall_mrt} u {u} {} gap {h}->{h2} strip {strip} tau-1/2 {tau_minus_half:e} steps {done} of {steps}{stop}: {last}; flux at x={a} {fa:.6}, at x={b} {fb:.6}, ratio {:.6}; time-mean flux over the last {n_avg} samples (every {every} steps) at x=10 {:.6}, x={a} {:.6}, x={b} {:.6}, ratio b/a {:.6}, a/inlet {:.6}; 2dx pair gap (step:max |rho - mean of x neighbours|) {}",
        if uniform { "uniform" } else { "parabolic" },
        fb / fa,
        sum[0] / n_avg.max(1) as f64,
        sum[1] / n_avg.max(1) as f64,
        sum[2] / n_avg.max(1) as f64,
        sum[2] / sum[1],
        sum[1] / sum[0],
        pair.join(" ")
    )
}

#[test]
#[ignore = "print-only study: reads STEPS, TAU_MINUS_HALF, LAYER, LENGTH, HEIGHT, HEIGHT2, STRIP, UNIFORM, SIGMA, ULAT, EVERY, ARMS"]
fn wall_mrt_inlet() {
    let steps = env("STEPS", 20_000.0) as u64;
    let tau = env("TAU_MINUS_HALF", 1.8e-4);
    // ARMS: 0 = off and WALL_MRT (default, test 4), 1 = off, WALL_SIGMA1 = 1 and WALL_MRT (the strip study's three arms)
    let arms: &[(u8, bool)] = if env("ARMS", 0.0) > 1.5 {
        &[(0, true)]
    } else if env("ARMS", 0.0) > 0.5 {
        &[(0, false), (1, false), (0, true)]
    } else {
        &[(0, false), (0, true)]
    };
    for &(s1, mrt) in arms {
        println!("{}", run(s1, mrt, tau, steps));
    }
}
