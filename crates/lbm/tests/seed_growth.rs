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
//! Variants, all off by default: `STUDY_SIGMA` (the hybrid collision's σ, or `dynamic` with `STUDY_MODEL=vreman` or
//! `smag`), `STUDY_MODEL` (`sa`, `sa_ddes`, `vreman`, `smag`), `STUDY_COLLISION` (`hrr`, `regularized`, `mrt`, `trt`),
//! `STUDY_YFACES=slip` (slip instead of walls at y min and max), `STUDY_SLOT` (the slot's width in m, 0.1), `STUDY_INFLOW` (Spalart–Allmaras's ν̃/ν at the inflow) and `STUDY_LAYER=1` (the default absorbing layer).
//!
//! `STUDY_TAUS` (τ⁺ − ½, comma-separated), `STUDY_ULATS` (the inlet's peak lattice speed), `STUDY_STEPS`.
#[path = "common/mod.rs"]
mod common;

use common::*;
use fieldcraft_lbm::backend::Backend;
use fieldcraft_lbm::{Flow, Limits, Setup};
use fieldcraft_model::{AbsorbingLayer, Collision, Ddes, FaceCondition, HrrSigma, Outflow, Procedure, Profile, Turbulence};

const SEED: f64 = 1e-7;

fn list(k: &str, d: &str) -> Vec<f64> {
    std::env::var(k).unwrap_or_else(|_| d.into()).split(',').filter_map(|s| s.trim().parse().ok()).collect()
}

fn word(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.into())
}

fn setup_of(ulat: f64, tau_half: f64, steps: u64) -> Option<(Setup, Vec<fieldcraft_lbm::Obstacle>)> {
    let dx = 0.05;
    let dt = ulat * dx / 0.1;
    let nu = tau_half / 3.0 * dx * dx / dt;
    let mut d = domain([0.0; 3], [2.0, 1.0, 1.0], dx);
    d.faces.x_min = FaceCondition::Velocity { velocity: [0.1, 0.02, 0.0], profile: Profile::Parabolic };
    d.faces.x_max = FaceCondition::Pressure { pressure: 0.0, outflow: Outflow::Uniform };
    d.faces.z_max = FaceCondition::Slip {};
    if word("STUDY_YFACES", "wall") == "slip" {
        d.faces.y_min = FaceCondition::Slip {};
        d.faces.y_max = FaceCondition::Slip {};
    }
    d.acceleration = [0.0, 0.0, -0.01];
    d.obstacles = vec!["PlateLow".into(), "PlateHigh".into()];
    let slot = word("STUDY_SLOT", "0.1").parse::<f64>().unwrap_or(0.1);
    let bodies = vec![
        obstacle("PlateLow", cuboid([0.6, 0.0, 0.0], [0.8, 1.0, 0.5 - slot / 2.0]), &CUBOID_FACES),
        obstacle("PlateHigh", cuboid([0.6, 0.0, 0.5 + slot / 2.0], [0.8, 1.0, 1.0]), &CUBOID_FACES),
    ];
    let inflow_ratio = word("STUDY_INFLOW", "3").parse().unwrap_or(3.0);
    let sa = match word("STUDY_MODEL", "sa").as_str() {
        "sa_ddes" => Turbulence::SpalartAllmaras { prandtl: 0.9, wall_model: None, inflow_ratio, ddes: Some(Ddes { c_des: 0.65 }) },
        "vreman" => Turbulence::Vreman { constant: 0.18, prandtl: 0.9, wall_model: None },
        "smag" => Turbulence::Smagorinsky { constant: 0.17, prandtl: 0.9, van_driest: None, wall_model: None },
        _ => Turbulence::SpalartAllmaras { prandtl: 0.9, wall_model: None, inflow_ratio, ddes: None },
    };
    let sigma = match word("STUDY_SIGMA", "0.99").as_str() {
        "dynamic" => HrrSigma::Dynamic,
        v => HrrSigma::Value(v.parse().unwrap_or(0.99)),
    };
    let collision = match word("STUDY_COLLISION", "hrr").as_str() {
        "regularized" => Collision::Regularized,
        "mrt" => Collision::Mrt,
        "trt" => Collision::Trt,
        _ => Collision::Hrr,
    };
    let p = Procedure::Fluid {
        domain: "Flow".into(),
        end_time: None,
        steps: Some(steps),
        time_step: Some(dt),
        frames: 2,
        magic: None,
        sigma: Some(sigma),
        collision: Some(collision),
        turbulence: Some(sa),
        steady: None,
    };
    if word("STUDY_LAYER", "0") == "1" {
        d.absorbing = Some(AbsorbingLayer { thickness: None, strength: None, filter_time: None });
    }
    let m = model(1.0, nu, d, p);
    match Setup::new(&m, None, &bodies, &Limits::default()) {
        Ok(s) => Some((s, bodies)),
        Err(e) => {
            println!("setup refused: {e}");
            None
        }
    }
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
            let mut history: Vec<f64> = Vec::new();
            while done < steps {
                if let Err(e) = Backend::advance(&mut a, 100).and_then(|()| Backend::advance(&mut b, 100)) {
                    println!("  steps {}..{}: advance failed: {e}", done + 1, done + 100);
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
                let (mut dr, mut du, mut at, mut umax_at) = (0.0f64, 0.0f64, 0usize, (0.0f64, 0usize));
                for &c in &fluid {
                    let e = (fa.density[c] - fb.density[c]).abs();
                    if e > dr {
                        at = c;
                    }
                    let u = fa.velocity[c].iter().map(|v| v * v).sum::<f64>().sqrt() / scale;
                    if u > umax_at.0 {
                        umax_at = (u, c);
                    }
                    dr = dr.max(e);
                    du = du.max((0..3).map(|k| ((fa.velocity[c][k] - fb.velocity[c][k]) / scale).powi(2)).sum::<f64>().sqrt());
                }
                let ratio = if last > 0.0 { dr / last } else { f64::NAN };
                if done > 300 && ratio.is_finite() {
                    peak_ratio = peak_ratio.max(ratio);
                }
                println!(
                    "  step {done}: umax {ua:.4} / {ub:.4}; max |d_rho| {dr:.3e}; max |d_u| {du:.3e}; growth over 100 steps {ratio:.2}x; nu_t/nu max {:.2}; max d_rho at {:?}, max |u| at {:?}",
                    a.eddy_viscosity_ratio(),
                    a.grid().centre(at).map(|v| (v * 100.0).round() / 100.0),
                    a.grid().centre(umax_at.1).map(|v| (v * 100.0).round() / 100.0)
                );
                last = dr;
                history.push(dr);
            }
            // The score: the last half of the run (at most 2000 steps) gives the mean growth per 100 steps.
            let window = history.len().div_ceil(2).clamp(1, 20);
            let start = history.len().saturating_sub(window + 1);
            let growth = match (history.get(start), history.last()) {
                (Some(a), Some(b)) if *a > 0.0 && history.len() > start + 1 => (b / a).powf(1.0 / (history.len() - 1 - start) as f64),
                _ => f64::NAN,
            };
            let class = if verdict != "ran all steps" {
                "FAIL"
            } else if last < 1e-9 && growth <= 1.0 {
                "HOLD"
            } else if growth <= 1.05 {
                "SATURATED"
            } else {
                "GROWING"
            };
            println!(
                "  SCORE u_lat {ulat} tau-1/2 {tau_half:e}: {class} (stopped at step {done} of {steps}; last max |d_rho| {last:.3e}; mean growth per 100 steps over the last {} steps {growth:.3})",
                100 * history.len().saturating_sub(start + 1)
            );
            println!("  -> {verdict} at step {done}; peak growth per 100 steps after step 300: {peak_ratio:.2}x; last max |d_rho| {last:.3e}");
        }
    }
}
