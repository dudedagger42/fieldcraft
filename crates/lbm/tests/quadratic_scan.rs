//! Where the quadratic interpolated wall rule is stable: a laminar plane channel with no
//! turbulence model and no-slip walls moved so the links cut at q = 0.75 and 0.25, run 20 000
//! steps from an impulsive start at each relaxation time minus one half, with each collision.
//! The table is printed (`--nocapture`); nothing is asserted, the numbers are the result.

mod common;

use common::*;
use fieldcraft_lbm::interpolated::WallOrder;
use fieldcraft_model::{Collision, FaceCondition, HrrSigma, Procedure, Turbulence};

/// One run: `n` open rows between walls cutting the links at `q`; whether it stayed stable.
fn stable(collision: Collision, sigma: Option<f64>, tau_minus_half: f64, q: f64, order: WallOrder, steps: usize) -> Result<f64, String> {
    // Lattice units: spacing 1, time step 1, so ν = (τ − ½)/3 and the speed is the lattice's.
    let (n, slab) = (10usize, 2usize);
    let nu = tau_minus_half / 3.0;
    // The first fluid row's centre is 0.5 cell above the box face; the wall lies q − ½ cells
    // below it (q = 0.75: out by 0.25).
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
    let mut p = collide(steps as f64, 1.0, None, collision);
    if let Procedure::Fluid { sigma: sg, turbulence, .. } = &mut p {
        *sg = sigma.map(HrrSigma::Value);
        // The setup refuses τ − ½ below 5e-3 without a model: there the SA model is on (its
        // eddy viscosity is zero in this laminar flow, so the walls and collision are what is tested).
        if tau_minus_half < 5e-3 {
            *turbulence = Some(Turbulence::SpalartAllmaras { prandtl: 0.9, wall_model: None, inflow_ratio: 3.0, ddes: None });
        }
    }
    let (setup, mut flow) = prepare(&model(1.0, nu, d, p), &obstacles);
    flow.interpolate_walls_with(&setup, &obstacles, order).map_err(|e| e.to_string())?;
    for _ in 0..steps {
        flow.step().map_err(|e| e.to_string())?;
    }
    let peak = (0..flow.grid().len()).filter(|&c| flow.is_fluid(c)).filter_map(|c| flow.moments(c)).map(|(_, u)| u[0].abs()).fold(0.0, f64::max);
    Ok(peak)
}

/// The scan: quadratic and linear, MRT and HRR σ 0.95, q 0.75 and 0.25, τ − ½ from 1e-1 to 1e-3
/// (and the harness values when `SCAN_TAU` lists them, comma separated). Below 5e-3 the setup
/// refuses a run without a model, so those cells run with Spalart–Allmaras.
#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn the_quadratic_rule_stability_table() {
    let mut taus = vec![1e-1, 1e-2, 1e-3];
    if let Ok(v) = std::env::var("SCAN_TAU") {
        taus.extend(v.split(',').filter_map(|t| t.trim().parse::<f64>().ok()));
    }
    let steps = std::env::var("SCAN_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(20_000usize);
    for (name, collision, sigma) in [("MRT", Collision::Mrt, None), ("HRR σ0.95", Collision::Hrr, Some(0.95))] {
        for order in [WallOrder::Quadratic, WallOrder::Linear] {
            for q in [0.75, 0.25] {
                let row: Vec<String> = taus
                    .iter()
                    .map(|&t| match stable(collision, sigma, t, q, order, steps) {
                        Ok(peak) => format!("τ−½ {t:.1e}: stable (peak {peak:.3})"),
                        Err(e) => format!("τ−½ {t:.1e}: UNSTABLE ({})", e.split(':').next().unwrap_or("")),
                    })
                    .collect();
                println!("{name:10} {order:?} q {q}: {}", row.join("; "));
            }
        }
    }
}
