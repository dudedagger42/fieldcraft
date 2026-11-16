//! SCRATCH (stability study, not a gate): the four arms of the S-1 gate on the aligned laminar
//! channel at q = 0.75 (and 0.25), at the τ − ½ values in `SCAN_TAU` (comma separated; default the
//! scan's list), 20 000 steps (`SCAN_STEPS`), Spalart–Allmaras below τ − ½ 5e-3 (its eddy
//! viscosity is zero in this laminar flow). Arms: 1 regularised, linear; 2 MRT, linear;
//! 3 regularised, linear with every link of q above ½ forced to half-way
//! (`Flow::force_half_way_above(Some(0.5))`); 4 regularised, plain half-way walls. Prints stable
//! (peak |u|, mean_ke = mean u·u/2 over the fluid cells, row of the largest |u|) or the step at
//! which a step failed, with the last sample (every 25 steps). mean_ke is not the slot's K.

mod common;

use common::*;
use fieldcraft_lbm::interpolated::WallOrder;
use fieldcraft_model::{Collision, FaceCondition, Procedure, Turbulence};

#[derive(Clone, Copy)]
struct Arm {
    name: &'static str,
    collision: Collision,
    interpolate: bool,
    forced: bool,
    extrapolated: bool,
}

fn run(arm: Arm, tau_minus_half: f64, q: f64, steps: usize) -> String {
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
    if let Procedure::Fluid { turbulence, .. } = &mut p
        && tau_minus_half < 5e-3
    {
        *turbulence = Some(Turbulence::SpalartAllmaras { prandtl: 0.9, wall_model: None, inflow_ratio: 3.0, ddes: None });
    }
    let m = model(1.0, nu, d, p);
    let Ok((setup, mut flow)) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| prepare(&m, &obstacles))) else {
        return "refused".into();
    };
    if arm.interpolate {
        flow.extrapolate_above_half(arm.extrapolated);
        if arm.forced {
            flow.force_half_way_above(Some(0.5));
        }
        if let Err(e) = flow.interpolate_walls_with(&setup, &obstacles, WallOrder::Linear) {
            return format!("refused ({e})");
        }
    }
    let grid = *flow.grid();
    let fluid: Vec<usize> = (0..grid.len()).filter(|&c| flow.is_fluid(c)).collect();
    let probe = |flow: &fieldcraft_lbm::Flow| -> (f64, f64, usize) {
        let (mut peak, mut ke, mut row) = (0.0f64, 0.0, 0usize);
        for &c in &fluid {
            if let Some((_, u)) = flow.moments(c) {
                let e = 0.5 * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
                ke += e;
                let v = (2.0 * e).sqrt();
                if v > peak || v.is_nan() {
                    peak = v;
                    row = grid.coords(c)[1];
                }
            }
        }
        (peak, ke / fluid.len().max(1) as f64, row)
    };
    let mut last = probe(&flow);
    for step in 1..=steps {
        if let Err(e) = flow.step() {
            let msg = e.to_string();
            return format!(
                "BLEW at step {step} ({}; last sample peak {:.3e}, mean_ke {:.3e}, row {})",
                msg.split(':').next().unwrap_or(""),
                last.0,
                last.1,
                last.2
            );
        }
        if step % 25 == 0 {
            last = probe(&flow);
        }
    }
    let end = probe(&flow);
    format!("stable (peak {:.4}, mean_ke {:.3e}, row {})", end.0, end.1, end.2)
}

#[test]
#[ignore = "scratch stability study: cargo test --release --test s1_arms -- --ignored --nocapture"]
fn the_s1_arms_on_the_aligned_channel() {
    let taus: Vec<f64> = match std::env::var("SCAN_TAU") {
        Ok(v) => v.split(',').filter_map(|t| t.trim().parse::<f64>().ok()).collect(),
        Err(_) => vec![1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 1e-3],
    };
    let steps = std::env::var("SCAN_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(20_000usize);
    let arms = [
        Arm { name: "arm 1: regularised, linear", collision: Collision::Regularized, interpolate: true, forced: false, extrapolated: false },
        Arm { name: "arm 2: MRT, linear", collision: Collision::Mrt, interpolate: true, forced: false, extrapolated: false },
        Arm {
            name: "arm 3: regularised, linear, q>1/2 half-way",
            collision: Collision::Regularized,
            interpolate: true,
            forced: true,
            extrapolated: false,
        },
        Arm { name: "arm 4: regularised, half-way walls", collision: Collision::Regularized, interpolate: false, forced: false, extrapolated: false },
        Arm {
            name: "arm 5: regularised, linear, q>1/2 extrapolated",
            collision: Collision::Regularized,
            interpolate: true,
            forced: false,
            extrapolated: true,
        },
    ];
    let qs: Vec<f64> = match std::env::var("SCAN_Q") {
        Ok(v) => v.split(',').filter_map(|t| t.trim().parse::<f64>().ok()).collect(),
        Err(_) => vec![0.75, 0.25],
    };
    for arm in arms {
        for &q in &qs {
            for &t in &taus {
                println!("{} q {q} tau-1/2 {t:.2e}: {}", arm.name, run(arm, t, q, steps));
            }
        }
    }
}
