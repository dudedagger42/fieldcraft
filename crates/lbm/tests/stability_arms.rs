//! SCRATCH (not for a pull request): where interpolated walls are stable. A laminar plane
//! channel, no turbulence model, no-slip walls moved so the links cut at `q` (0.75 or 0.25), run
//! from an impulsive start at each relaxation time minus one half, in lattice units (spacing 1,
//! time step 1, ν = (τ − ½) / 3, peak speed 0.05, driven by the force that balances it).
//!
//! The arms of the S-1 gate that run from here (collision, wall rule):
//! 1. regularised, linear;  2. MRT, linear;  3. regularised, linear with every link of q > ½
//! forced to half-way (the diagnostic switch of `flow.rs`);  4. regularised, half-way walls.
//! Also quadratic and HRR (σ 0.95) rows, for the earlier scan. Stable = all steps ran; otherwise
//! the step at which a step returned an error. K is the mean kinetic energy per fluid cell over
//! the fluid cells, u·u/2 in lattice units, at the last sampled step, and `row` is the row (index
//! across the channel) of the largest |u| at the last sampled step (every 25 steps), the best
//! guess where the growth starts. Nothing is asserted; the table is the result
//! (`--ignored --nocapture`, `SCAN_TAU` adds τ − ½ values, `SCAN_STEPS` sets the steps).

mod common;

use common::*;
use fieldcraft_lbm::flow::set_halfway_above_half_diagnostic;
use fieldcraft_model::{Collision, FaceCondition, HrrSigma, Procedure, WallTreatment};

#[derive(Clone, Copy, Debug)]
struct Arm {
    name: &'static str,
    collision: Collision,
    sigma: Option<f64>,
    walls: WallTreatment,
    halfway_above_half: bool,
}

enum Outcome {
    Stable { peak: f64, k: f64, row: usize },
    Blew { step: usize, peak: f64, k: f64, row: usize, message: String },
    Refused(String),
}

fn run(arm: Arm, tau_minus_half: f64, q: f64, steps: usize) -> Outcome {
    let (n, slab) = (10usize, 2usize);
    let nu = tau_minus_half / 3.0;
    let s = q - 0.5;
    let width = n as f64 + 2.0 * s;
    let umax = 0.05;
    let t = slab as f64;
    let height = n as f64;
    let mut d = domain([0.0, -t, 0.0], [1.0, height + t, 1.0], 1.0);
    for f in [&mut d.faces.x_min, &mut d.faces.x_max, &mut d.faces.z_min, &mut d.faces.z_max, &mut d.faces.y_min, &mut d.faces.y_max] {
        *f = FaceCondition::Periodic {};
    }
    let obstacles = vec![
        obstacle("Floor", cuboid([-1.0, -t, -1.0], [2.0, -s, 2.0]), &CUBOID_FACES),
        obstacle("Roof", cuboid([-1.0, height + s, -1.0], [2.0, height + t, 2.0]), &CUBOID_FACES),
    ];
    d.obstacles = obstacles.iter().map(|o| o.body.clone()).collect();
    d.acceleration = [8.0 * nu * umax / (width * width), 0.0, 0.0];
    d.initial_velocity = [umax, 0.0, 0.0];
    let mut p = collide(steps as f64, 1.0, None, arm.collision);
    if let Procedure::Fluid { sigma, .. } = &mut p {
        *sigma = arm.sigma.map(HrrSigma::Value);
    }
    let p = with_walls(p, arm.walls);
    let m = model(1.0, nu, d, p);
    set_halfway_above_half_diagnostic(arm.halfway_above_half);
    let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| prepare(&m, &obstacles).1));
    set_halfway_above_half_diagnostic(false);
    let mut flow = match built {
        Ok(f) => f,
        Err(_) => return Outcome::Refused("setup refused".into()),
    };
    let grid = *flow.grid();
    let fluid: Vec<usize> = (0..grid.len()).filter(|&c| flow.is_fluid(c)).collect();
    let probe = |flow: &fieldcraft_lbm::flow::Flow| -> (f64, f64, usize) {
        let (mut peak, mut k, mut row) = (0.0f64, 0.0, 0usize);
        for &c in &fluid {
            if let Some((_, u)) = flow.moments(c) {
                let e = 0.5 * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
                k += e;
                let m = (2.0 * e).sqrt();
                if m > peak || m.is_nan() {
                    peak = m;
                    row = grid.coords(c)[1];
                }
            }
        }
        (peak, k / fluid.len().max(1) as f64, row)
    };
    let mut last = probe(&flow);
    for step in 1..=steps {
        if let Err(e) = flow.step() {
            return Outcome::Blew { step, peak: last.0, k: last.1, row: last.2, message: e.to_string().split(':').next().unwrap_or("").to_string() };
        }
        if step % 25 == 0 {
            last = probe(&flow);
        }
    }
    let end = probe(&flow);
    Outcome::Stable { peak: end.0, k: end.1, row: end.2 }
}

#[test]
#[ignore = "heavy: scratch stability scan, cargo test --release --test stability_arms -- --ignored --nocapture"]
fn the_stability_arms() {
    let mut taus = vec![1e-1, 7e-2, 5e-2, 4e-2, 3e-2, 2e-2, 1.5e-2, 1.25e-2, 1e-2, 5e-3];
    if let Ok(v) = std::env::var("SCAN_TAU") {
        taus.extend(v.split(',').filter_map(|t| t.trim().parse::<f64>().ok()));
    }
    let steps = std::env::var("SCAN_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(20_000usize);
    let lin = WallTreatment::Interpolated;
    let arms = [
        Arm { name: "S1 arm 1: regularised, linear", collision: Collision::Regularized, sigma: None, walls: lin, halfway_above_half: false },
        Arm { name: "S1 arm 2: MRT, linear", collision: Collision::Mrt, sigma: None, walls: lin, halfway_above_half: false },
        Arm {
            name: "S1 arm 3: regularised, linear, q>1/2 half-way",
            collision: Collision::Regularized,
            sigma: None,
            walls: lin,
            halfway_above_half: true,
        },
        Arm {
            name: "S1 arm 4: regularised, half-way walls",
            collision: Collision::Regularized,
            sigma: None,
            walls: WallTreatment::Halfway,
            halfway_above_half: false,
        },
        Arm { name: "HRR 0.95, linear", collision: Collision::Hrr, sigma: Some(0.95), walls: lin, halfway_above_half: false },
        Arm {
            name: "HRR 0.95, quadratic",
            collision: Collision::Hrr,
            sigma: Some(0.95),
            walls: WallTreatment::InterpolatedQuadratic,
            halfway_above_half: false,
        },
        Arm {
            name: "MRT, quadratic",
            collision: Collision::Mrt,
            sigma: None,
            walls: WallTreatment::InterpolatedQuadratic,
            halfway_above_half: false,
        },
    ];
    for arm in arms {
        for q in [0.75, 0.25] {
            let row: Vec<String> = taus
                .iter()
                .map(|&t| match run(arm, t, q, steps) {
                    Outcome::Stable { peak, k, row } => format!("{t:.1e}: stable (peak {peak:.4}, K {k:.3e}, row {row})"),
                    Outcome::Blew { step, peak, k, row, message } => {
                        format!("{t:.1e}: BLEW at step {step} ({message}; last sample peak {peak:.3e}, K {k:.3e}, row {row})")
                    }
                    Outcome::Refused(e) => format!("{t:.1e}: refused ({e})"),
                })
                .collect();
            println!("{} q {q}: {}", arm.name, row.join("; "));
        }
    }
}
