//! SCRATCH (not for a pull request): the mass the interpolated walls create on a curved body. A
//! stationary sphere (r = 6, 24-cell periodic box, lattice units, body force 1e-5) with linear
//! interpolated walls, run with the walls' mass correction off and on at τ − ½ in `SCAN_TAU` (default
//! 1e-1, 2e-2, 5e-3): total mass drift at a quarter, a half and the end of `SCAN_STEPS` (80 000),
//! the sphere's drag (mean over the last 5 000 steps) and the mean u_x at the end. Nothing is asserted.

mod common;

use common::*;
use fieldcraft_lbm::flow::Flow;
use fieldcraft_model::{Collision, FaceCondition, HrrSigma, Procedure, WallTreatment};

fn flow(collision: Collision, sigma: Option<f64>, tau_minus_half: f64, steps: usize) -> Flow {
    let side = 24.0;
    let mut d = domain([0.0; 3], [side; 3], 1.0);
    for f in [&mut d.faces.x_min, &mut d.faces.x_max, &mut d.faces.y_min, &mut d.faces.y_max, &mut d.faces.z_min, &mut d.faces.z_max] {
        *f = FaceCondition::Periodic {};
    }
    d.acceleration = [1e-5, 0.0, 0.0];
    d.obstacles = vec!["Ball".into()];
    let ball = obstacle("Ball", sphere([side / 2.0; 3], 6.0, 3), &["skin"]);
    let mut p = collide(steps as f64, 1.0, None, collision);
    if let Procedure::Fluid { sigma: s, .. } = &mut p {
        *s = sigma.map(HrrSigma::Value);
    }
    let p = with_walls(p, WallTreatment::Interpolated);
    prepare(&model(1.0, tau_minus_half / 3.0, d, p), std::slice::from_ref(&ball)).1
}

fn mass_and_ux(flow: &Flow, fluid: &[usize]) -> (f64, f64) {
    let (mut m, mut u) = (0.0, 0.0);
    for &c in fluid {
        if let Some((r, v)) = flow.moments(c) {
            m += r;
            u += v[0];
        }
    }
    (m, u / fluid.len().max(1) as f64)
}

#[test]
#[ignore = "heavy: scratch sphere leak scan, cargo test --release --test sphere_leak -- --ignored --nocapture"]
fn the_sphere_leak() {
    let steps = std::env::var("SCAN_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(80_000usize);
    let taus: Vec<f64> = std::env::var("SCAN_TAU").ok().map(|v| v.split(',').filter_map(|t| t.trim().parse().ok()).collect()).unwrap_or_else(|| vec![1e-1, 2e-2, 5e-3]);
    let arms: [(&str, Collision, Option<f64>); 3] = [("HRR 0.95", Collision::Hrr, Some(0.95)), ("HRR 1.0", Collision::Hrr, Some(1.0)), ("MRT", Collision::Mrt, None)];
    let only = std::env::var("SCAN_ONLY").unwrap_or_default();
    for (name, collision, sigma) in arms.into_iter().filter(|a| only.is_empty() || only.split(';').any(|o| a.0 == o)) {
        for &t in &taus {
            for on in [false, true] {
                let mut f = flow(collision, sigma, t, steps);
                if on && let Err(e) = f.set_wall_mass_correction(true) {
                    println!("SPHERE {name} tau-1/2 {t:.1e} correction on: refused ({e})");
                    continue;
                }
                let fluid: Vec<usize> = (0..f.grid().len()).filter(|&c| f.is_fluid(c)).collect();
                let skin = f.groups().iter().position(|g| g.name == "skin");
                let m0 = mass_and_ux(&f, &fluid).0;
                let mut drift = [f64::NAN; 3];
                let (mut drag, mut n) = (0.0, 0usize);
                let mut blew = None;
                for step in 1..=steps {
                    if let Err(e) = f.step() {
                        blew = Some((step, e.to_string()));
                        break;
                    }
                    if step > steps.saturating_sub(5_000)
                        && let Some(g) = skin
                        && let Some(force) = f.forces().get(g)
                    {
                        drag += force[0];
                        n += 1;
                    }
                    let k = [steps / 4, steps / 2, steps].iter().position(|&s| s == step);
                    if let Some(k) = k {
                        let m = mass_and_ux(&f, &fluid).0 - f.wall_mass_pending();
                        drift[k] = (m - m0) / m0;
                    }
                }
                let (_, ux) = mass_and_ux(&f, &fluid);
                match blew {
                    Some((step, why)) => println!("SPHERE {name} tau-1/2 {t:.1e} correction {}: BLEW at step {step} ({})", if on { "on" } else { "off" }, why.split(':').next().unwrap_or("")),
                    None => println!(
                        "SPHERE {name} tau-1/2 {t:.1e} correction {}: drift {:.3e} / {:.3e} / {:.3e} (quarter/half/end, relative, less pending), drag {:.5e}, mean_ux {ux:.5e}",
                        if on { "on" } else { "off" },
                        drift[0],
                        drift[1],
                        drift[2],
                        drag / n.max(1) as f64
                    ),
                }
            }
        }
    }
}
