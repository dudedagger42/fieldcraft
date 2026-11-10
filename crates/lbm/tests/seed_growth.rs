//! How a small perturbation behaves in the slotted channel at very low τ − ½ (CPU only).
//!
//! The case of `crates/lbm-gpu/tests/parity_low_tau.rs` (slotted plate, hybrid collision σ 0.99, Spalart–Allmaras,
//! parabolic inlet, pressure outlet, slip face, body force): two flows from the same state, the second with `SEED`
//! added to the density of one fluid cell. Every 100 steps the table gives the largest lattice speed of each, the
//! largest density and velocity difference between them, and the growth of the density difference over the last
//! 100 steps. A run stops, with the step, when a field is not finite, a lattice speed passes 0.5 or a density
//! leaves 0.5..1.5. Whether the difference saturates (the speeds stay bounded and the difference stops growing)
//! or the run fails shows in the last rows.
//!
//! `STUDY_TAUS` (τ⁺ − ½, comma-separated), `STUDY_ULATS` (the inlet's peak lattice speed), `STUDY_STEPS`.
#[path = "common/mod.rs"]
mod common;

use common::*;
use fieldcraft_lbm::backend::Backend;
use fieldcraft_lbm::{Flow, Limits, Setup};
use fieldcraft_model::{Collision, FaceCondition, HrrSigma, Outflow, Procedure, Profile, Turbulence};

const SEED: f64 = 1e-7;

fn list(k: &str, d: &str) -> Vec<f64> {
    std::env::var(k).unwrap_or_else(|_| d.into()).split(',').filter_map(|s| s.trim().parse().ok()).collect()
}

fn setup_of(ulat: f64, tau_half: f64, steps: u64) -> Option<(Setup, Vec<fieldcraft_lbm::Obstacle>)> {
    let dx = 0.05;
    let dt = ulat * dx / 0.1;
    let nu = tau_half / 3.0 * dx * dx / dt;
    let mut d = domain([0.0; 3], [2.0, 1.0, 1.0], dx);
    d.faces.x_min = FaceCondition::Velocity { velocity: [0.1, 0.02, 0.0], profile: Profile::Parabolic };
    d.faces.x_max = FaceCondition::Pressure { pressure: 0.0, outflow: Outflow::Uniform };
    d.faces.z_max = FaceCondition::Slip {};
    d.acceleration = [0.0, 0.0, -0.01];
    d.obstacles = vec!["PlateLow".into(), "PlateHigh".into()];
    let bodies = vec![
        obstacle("PlateLow", cuboid([0.6, 0.0, 0.0], [0.8, 1.0, 0.45]), &CUBOID_FACES),
        obstacle("PlateHigh", cuboid([0.6, 0.0, 0.55], [0.8, 1.0, 1.0]), &CUBOID_FACES),
    ];
    let sa = Turbulence::SpalartAllmaras { prandtl: 0.9, wall_model: None, inflow_ratio: 3.0, ddes: None };
    let p = Procedure::Fluid {
        domain: "Flow".into(),
        end_time: None,
        steps: Some(steps),
        time_step: Some(dt),
        frames: 2,
        magic: None,
        sigma: Some(HrrSigma::Value(0.99)),
        collision: Some(Collision::Hrr),
        turbulence: Some(sa),
        steady: None,
    };
    let m = model(1.0, nu, d, p);
    Setup::new(&m, None, &bodies, &Limits::default()).ok().map(|s| (s, bodies))
}

/// (largest lattice speed, whether the fluid cells are finite with a density in 0.5..1.5).
fn health(b: &mut Flow, scale: f64) -> (f64, bool) {
    let Ok(f) = Backend::fields(b) else { return (f64::NAN, false) };
    let (mut umax, mut ok) = (0.0f64, true);
    for c in (0..f.density.len()).filter(|&c| Backend::is_fluid(b, c)) {
        let u = f.velocity[c].iter().map(|v| v * v).sum::<f64>().sqrt() / scale;
        ok &= u.is_finite() && f.density[c].is_finite() && f.density[c] > 0.5 && f.density[c] < 1.5;
        umax = umax.max(u);
    }
    (umax, ok && umax < 0.5)
}

#[test]
#[ignore = "study: cargo test -p fieldcraft-lbm --release --test seed_growth -- --ignored --nocapture"]
fn a_seed_in_the_slotted_channel_at_low_tau() {
    let steps = std::env::var("STUDY_STEPS").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(3000);
    for ulat in list("STUDY_ULATS", "0.012") {
        for tau_half in list("STUDY_TAUS", "2.4e-4,5e-4,1e-3,1e-2") {
            let Some((setup, bodies)) = setup_of(ulat, tau_half, steps) else {
                println!("u_lat {ulat} tau-1/2 {tau_half:e}: setup refused");
                continue;
            };
            let scale = setup.velocity_scale();
            let (Ok(mut a), Ok(mut b)) = (Flow::new(&setup, &bodies), Flow::new(&setup, &bodies)) else {
                println!("u_lat {ulat} tau-1/2 {tau_half:e}: flow refused");
                continue;
            };
            let Ok(f0) = Backend::fields(&mut a) else { continue };
            let fluid: Vec<usize> = (0..f0.density.len()).filter(|&c| Backend::is_fluid(&a, c)).collect();
            let Some(&seed) = fluid.get(fluid.len() / 2) else { continue };
            let state = |c: usize, e: f64| (f0.density[c] + if c == seed { e } else { 0.0 }, [0, 1, 2].map(|k| f0.velocity[c][k] / scale));
            a.set_equilibrium(|c| state(c, 0.0));
            b.set_equilibrium(|c| state(c, SEED));
            println!("== u_lat {ulat} tau-1/2 {tau_half:e} (tau- {:.3e}), seed {SEED:e} in cell {seed}, {steps} steps", setup.tau_minus - 0.5);
            let (mut last, mut peak_ratio, mut done, mut verdict) = (0.0f64, 0.0f64, 0u64, "ran all steps");
            while done < steps {
                if Backend::advance(&mut a, 100).is_err() || Backend::advance(&mut b, 100).is_err() {
                    verdict = "advance failed";
                    break;
                }
                done += 100;
                let ((ua, oka), (ub, okb)) = (health(&mut a, scale), health(&mut b, scale));
                if !(oka && okb) {
                    println!("  step {done}: FAILED (umax unseeded {ua:.4}, seeded {ub:.4}, ok {oka}/{okb})");
                    verdict = "FAILED";
                    break;
                }
                let (Ok(fa), Ok(fb)) = (Backend::fields(&mut a), Backend::fields(&mut b)) else { break };
                let (mut dr, mut du) = (0.0f64, 0.0f64);
                for &c in &fluid {
                    dr = dr.max((fa.density[c] - fb.density[c]).abs());
                    du = du.max((0..3).map(|k| ((fa.velocity[c][k] - fb.velocity[c][k]) / scale).powi(2)).sum::<f64>().sqrt());
                }
                let ratio = if last > 0.0 { dr / last } else { f64::NAN };
                if done > 300 && ratio.is_finite() {
                    peak_ratio = peak_ratio.max(ratio);
                }
                println!("  step {done}: umax {ua:.4} / {ub:.4}; max |d_rho| {dr:.3e}; max |d_u| {du:.3e}; growth over 100 steps {ratio:.2}x");
                last = dr;
            }
            println!("  -> {verdict} at step {done}; peak growth per 100 steps after step 300: {peak_ratio:.2}x; last max |d_rho| {last:.3e}");
        }
    }
}
