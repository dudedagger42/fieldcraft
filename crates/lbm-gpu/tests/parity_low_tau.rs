//! CPU/device parity at a very low relaxation time, τ − ½ = 2.4e-4, on generic geometry.
//!
//! The channel with a ball, a parabolic inlet, a pressure outlet, a slip face and a body force
//! (every kind of neighbour a cell can have) on the hybrid collision (σ 0.99) with the
//! Spalart–Allmaras model, 500 steps. The time step is chosen so that τ⁺ − ½ is 2.4e-4 with a
//! peak lattice speed near 0.10; the viscosity follows. Both backends advance 100 steps at a time
//! and the fields are compared after each, so the drift shows how it grows rather than only where
//! it ends. The pass bar is the backend's existing one (verification.rs): max |Δu| < 2e-5 lattice and
//! max |Δρ| < 1e-6 after every chunk (`PARITY_DU`, `PARITY_DRHO` override). Two cases: a slotted
//! channel (a plate with a two-cell slot) and a ball in a channel.
#[path = "../../lbm/tests/common/mod.rs"]
mod common;

use common::*;
use fieldcraft_gpu::select::{Policy, gpu_required};
use fieldcraft_lbm::backend::Backend;
use fieldcraft_lbm::{Limits, Setup};
use fieldcraft_lbm_gpu::{GpuFlow, Options};
use fieldcraft_model::{Collision, FaceCondition, HrrSigma, Outflow, Procedure, Profile, Turbulence};

const STEPS: u64 = 500;
const CHUNK: u64 = 100;
/// τ⁺ − ½: `PARITY_TAU`, 2.4e-4 by default (2.9e-4 is the 0.12-peak fallback at 0.5 mm).
fn tau_half() -> f64 {
    std::env::var("PARITY_TAU").ok().and_then(|v| v.parse().ok()).unwrap_or(2.4e-4)
}

fn env(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

/// `slotted`: a plate across the channel with a slot two cells wide (10 % of the section; the jet through it
/// is ten times the inlet's lattice speed once established; above an inlet of about 0.02 the CPU itself goes
/// unstable within 500 steps at this τ) instead of the ball.
fn case(slotted: bool) -> (fieldcraft_model::Model, Vec<fieldcraft_lbm::Obstacle>, f64, f64) {
    let dx = 0.05;
    // The peak lattice speed of the inlet (0.1 m/s, parabolic) is u dt / dx; the ball speeds the flow
    // up by about half again, so 0.08 at the inlet is about 0.12 at the ball's shoulder.
    let u_lat_inlet = if slotted { env("PARITY_ULAT_SLOT", 0.012) } else { env("PARITY_ULAT", 0.08) };
    let dt = u_lat_inlet * dx / 0.1;
    let nu = tau_half() / 3.0 * dx * dx / dt;
    let mut d = domain([0.0; 3], [2.0, 1.0, 1.0], dx);
    d.faces.x_min = FaceCondition::Velocity { velocity: [0.1, 0.02, 0.0], profile: Profile::Parabolic };
    d.faces.x_max = FaceCondition::Pressure { pressure: 0.0, outflow: Outflow::Uniform };
    d.faces.z_max = FaceCondition::Slip {};
    d.acceleration = [0.0, 0.0, -0.01];
    let bodies = if slotted {
        d.obstacles = vec!["PlateLow".into(), "PlateHigh".into()];
        vec![
            obstacle("PlateLow", cuboid([0.6, 0.0, 0.0], [0.8, 1.0, 0.45]), &CUBOID_FACES),
            obstacle("PlateHigh", cuboid([0.6, 0.0, 0.55], [0.8, 1.0, 1.0]), &CUBOID_FACES),
        ]
    } else {
        d.obstacles = vec!["Ball".into()];
        vec![obstacle("Ball", sphere([0.7, 0.5, 0.45], 0.2, 3), &["skin"])]
    };
    let sa = Turbulence::SpalartAllmaras { prandtl: 0.9, wall_model: None, inflow_ratio: 3.0, ddes: None };
    let p = Procedure::Fluid {
        domain: "Flow".into(),
        end_time: None,
        steps: Some(STEPS),
        time_step: Some(dt),
        frames: 2,
        magic: None,
        sigma: Some(HrrSigma::Value(0.99)),
        collision: Some(Collision::Hrr),
        turbulence: Some(sa),
        steady: None,
    };
    let m = model(1.0, nu, d, p);
    (m, bodies, dt, nu)
}

/// The setup is valid at this τ and the CPU holds 500 steps (no device involved).
#[test]
fn the_cases_are_valid_and_the_cpu_holds_500_steps_at_tau_minus_half_2_4e_minus_4() {
    for slotted in [false, true] {
        cpu_holds(slotted);
    }
}

fn cpu_holds(slotted: bool) {
    let (m, obstacles, _, _) = case(slotted);
    let setup = Setup::new(&m, None, &obstacles, &Limits::default()).unwrap();
    assert_eq!(setup.steps, STEPS);
    assert!((setup.tau_plus - 0.5 - tau_half()).abs() < 1e-9);
    let mut cpu = fieldcraft_lbm::Flow::new(&setup, &obstacles).unwrap();
    Backend::advance(&mut cpu, STEPS).unwrap();
    let f = Backend::fields(&mut cpu).unwrap();
    let umax = f.velocity.iter().map(|u| u.iter().map(|v| v * v).sum::<f64>().sqrt()).fold(0.0, f64::max) / setup.velocity_scale();
    println!("slotted {slotted}: CPU after {STEPS} steps: max |u| {umax:.4} lattice");
    let rho: Vec<f64> = (0..f.density.len()).filter(|&c| Backend::is_fluid(&cpu, c)).map(|c| f.density[c]).collect();
    assert!(rho.iter().all(|r| r.is_finite() && *r > 0.8 && *r < 1.2) && umax < 0.3, "umax {umax}, ρ {:?}..{:?}", rho.iter().cloned().fold(f64::INFINITY, f64::min), rho.iter().cloned().fold(0.0, f64::max));
}

#[test]
#[ignore = "local: run on the machine whose GPU is to be checked (cargo test -p fieldcraft-lbm-gpu --release --test parity_low_tau -- --ignored --nocapture)"]
fn the_cpu_and_the_device_agree_at_tau_minus_half_2_4e_minus_4() {
    for slotted in [true, false] {
        println!("--- {}", if slotted { "slotted channel" } else { "ball in a channel" });
        parity(slotted);
    }
}

fn parity(slotted: bool) {
    let (m, obstacles, dt, nu) = case(slotted);
    let setup = Setup::new(&m, None, &obstacles, &Limits::default()).unwrap();
    assert_eq!(setup.steps, STEPS);
    assert!((setup.tau_plus - 0.5 - tau_half()).abs() < 1e-9, "τ⁺ − ½ = {}", setup.tau_plus - 0.5);
    let mut cpu = fieldcraft_lbm::Flow::new(&setup, &obstacles).unwrap();
    let mut gpu = match GpuFlow::with_options(&cpu, Policy::from_env().unwrap(), Options::default()) {
        Ok(g) => g,
        Err(e) if !gpu_required() && e.to_string().contains("no GPU device") => {
            println!("skipping: {e}");
            return;
        }
        Err(e) => panic!("{e}"),
    };
    let scale = setup.velocity_scale();
    println!(
        "device {}; {} cells; τ⁺ − ½ {:.3e}, τ⁻ − ½ {:.3e}, dt {dt:.4} s, ν {nu:.3e}, velocity scale {scale:.3} m/s per lattice unit",
        gpu.adapter_name(),
        cpu.grid().len(),
        setup.tau_plus - 0.5,
        setup.tau_minus - 0.5
    );
    let (tol_u, tol_r) = (env("PARITY_DU", 2e-5), env("PARITY_DRHO", 1e-6));
    let (mut worst_u, mut worst_r) = (0.0f64, 0.0f64);
    let mut done = 0;
    while done < STEPS {
        Backend::advance(&mut cpu, CHUNK).unwrap();
        Backend::advance(&mut gpu, CHUNK).unwrap();
        done += CHUNK;
        let (a, b) = (Backend::fields(&mut cpu).unwrap(), Backend::fields(&mut gpu).unwrap());
        assert_eq!(a.velocity.len(), b.velocity.len());
        let finite = a.velocity.iter().chain(&b.velocity).all(|u| u.iter().all(|v| v.is_finite())) && a.density.iter().chain(&b.density).all(|r| r.is_finite());
        assert!(finite, "step {done}: a field is not finite (cpu all finite: {})", a.velocity.iter().all(|u| u.iter().all(|v| v.is_finite())));
        let umax = a.velocity.iter().map(|u| u.iter().map(|v| v * v).sum::<f64>().sqrt()).fold(0.0, f64::max) / scale;
        let (mut du, mut drho, mut sq, mut n) = (0.0f64, 0.0f64, 0.0f64, 0usize);
        for c in 0..a.velocity.len() {
            if !Backend::is_fluid(&cpu, c) {
                continue;
            }
            let e = [0, 1, 2].map(|k| (a.velocity[c][k] - b.velocity[c][k]) / scale);
            let e2 = e.iter().map(|v| v * v).sum::<f64>();
            du = du.max(e2.sqrt());
            sq += e2;
            n += 1;
            drho = drho.max((a.density[c] - b.density[c]).abs());
        }
        let rms = (sq / n.max(1) as f64).sqrt();
        let ratio = du / umax.max(1e-12);
        worst_u = worst_u.max(du);
        worst_r = worst_r.max(drho);
        let nt = Backend::nu_tilde(&mut gpu).unwrap().zip(Backend::nu_tilde(&mut cpu).unwrap()).map_or(f64::NAN, |(g, c)| g.iter().zip(&c).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max));
        println!("step {done}: max |u| {umax:.4} lattice; max |Δu| {du:.3e} ({:.2e} of max), rms {rms:.3e}; max |Δρ| {drho:.3e}; max |Δν̃| {nt:.3e}", ratio);
    }
    println!("worst over the chunks: max |Δu| {worst_u:.3e} (bar {tol_u:.0e}), max |Δρ| {worst_r:.3e} (bar {tol_r:.0e})");
    assert!(worst_u < tol_u && worst_r < tol_r, "above the existing GPU tolerances: |Δu| {worst_u:.3e}, |Δρ| {worst_r:.3e}");
}
