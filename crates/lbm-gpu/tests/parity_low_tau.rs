//! CPU/device parity at a very low relaxation time, τ − ½ = 2.4e-4, on generic geometry.
//!
//! The channel with a ball, a parabolic inlet, a pressure outlet, a slip face and a body force
//! (every kind of neighbour a cell can have) on the hybrid collision (σ 0.99) with the
//! Spalart–Allmaras model, 500 steps. The time step is chosen so that τ⁺ − ½ is 2.4e-4 with a
//! peak lattice speed near 0.10; the viscosity follows. Both backends advance 100 steps at a time
//! and the fields are compared after each, so the drift shows how it grows rather than only where
//! it ends. Two cases: a slotted channel (a plate with a two-cell slot) and a ball in a channel.
//!
//! What is checked, and why not the fields at the end. The slotted flow is weakly unstable at this τ:
//! a perturbation of any size grows by about 4× per 100 steps once the jet through the slot has
//! formed (step 250 on; measured on the CPU alone, seeds of 1e-12 and 1e-7 in the density of one
//! cell give the same curve, and at τ − ½ = 1e-2 the same seed decays). The device runs in single
//! precision, whose round-off (about 6e-8 relative, in every cell) seeds those modes, so the two
//! backends' fields part at that rate however accurate the kernels are: max |Δρ| is 2e-7 at step
//! 300 and 1e-5 at step 500. The populations are already stored as deviations from the weights, so
//! shifting them would not help. Therefore:
//! - until step `GATED_THROUGH` (300) the pointwise bar of the backend (verification.rs) holds after every
//!   chunk: max |Δu| < 2e-5 lattice and max |Δρ| < 1e-6 (`PARITY_DU`, `PARITY_DRHO` override);
//! - later chunks are reported, not gated;
//! - at the end the integral quantities are gated, which round-off does not amplify: the flow rate
//!   through the plane of the slot (the plane through the ball's centre), the force on the
//!   obstacles, the kinetic energy and the mean density, each within a relative `PARITY_INTEGRAL`
//!   (1e-4);
//! - both cases always run, then the failures are reported together;
//! - a second ignored test seeds the device twice apart by 1e-7 in one cell's density and reports
//!   how the two device runs part (compare with the CPU pair printed beside it).
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
/// The pointwise bar holds through this step; later chunks are only reported.
const GATED_THROUGH: u64 = 300;
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
    assert!(
        rho.iter().all(|r| r.is_finite() && *r > 0.8 && *r < 1.2) && umax < 0.3,
        "umax {umax}, ρ {:?}..{:?}",
        rho.iter().cloned().fold(f64::INFINITY, f64::min),
        rho.iter().cloned().fold(0.0, f64::max)
    );
}

#[test]
#[ignore = "local: run on the machine whose GPU is to be checked (cargo test -p fieldcraft-lbm-gpu --release --test parity_low_tau -- --ignored --nocapture)"]
fn the_cpu_and_the_device_agree_at_tau_minus_half_2_4e_minus_4() {
    let mut failures = Vec::new();
    for slotted in [true, false] {
        println!("--- {}", if slotted { "slotted channel" } else { "ball in a channel" });
        if let Err(e) = parity(slotted) {
            failures.push(format!("{}: {e}", if slotted { "slotted channel" } else { "ball in a channel" }));
        }
    }
    assert!(
        failures.is_empty(),
        "{}",
        failures.join(
            "
"
        )
    );
}

/// What round-off does not amplify, in lattice units: the flow rate through `plane` (the fluid cells of one
/// layer across the channel at x ≈ 0.68: the slot of the plate, the ball's shoulder), the force on the obstacles
/// (last step's momentum, along x), the kinetic energy and the mean density of the fluid cells.
fn integrals(b: &mut dyn Backend, plane: &[usize], scale: f64) -> Result<[f64; 4], String> {
    let f = b.fields().map_err(|e| e.to_string())?;
    let forces = b.forces().map_err(|e| e.to_string())?;
    let (mut ke, mut rho, mut n) = (0.0, 0.0, 0usize);
    for c in (0..f.density.len()).filter(|&c| b.is_fluid(c)) {
        let u2 = f.velocity[c].iter().map(|v| (v / scale).powi(2)).sum::<f64>();
        ke += 0.5 * f.density[c] * u2;
        rho += f.density[c];
        n += 1;
    }
    let flow = plane.iter().map(|&c| f.velocity[c][0] / scale).sum::<f64>();
    let force = b.groups().iter().zip(&forces).filter(|(g, _)| g.body.is_some()).map(|(_, v)| v[0]).sum::<f64>();
    Ok([flow, force, ke, rho / n.max(1) as f64])
}

const INTEGRAL_NAMES: [&str; 4] = ["flow rate", "obstacle force", "kinetic energy", "mean density"];

/// The fluid cells of the layer at x ≈ 0.68 (the cell centres 0.675).
fn plane_cells(cpu: &fieldcraft_lbm::Flow) -> Vec<usize> {
    (0..cpu.grid().len()).filter(|&c| Backend::is_fluid(cpu, c) && (0.65..0.70).contains(&cpu.grid().centre(c)[0])).collect()
}

/// `Err` carries what failed the gates; a machine without a device skips (`Ok`).
fn parity(slotted: bool) -> Result<(), String> {
    let (m, obstacles, dt, nu) = case(slotted);
    let setup = Setup::new(&m, None, &obstacles, &Limits::default()).map_err(|e| e.to_string())?;
    assert_eq!(setup.steps, STEPS);
    assert!((setup.tau_plus - 0.5 - tau_half()).abs() < 1e-9, "τ⁺ − ½ = {}", setup.tau_plus - 0.5);
    let mut cpu = fieldcraft_lbm::Flow::new(&setup, &obstacles).map_err(|e| e.to_string())?;
    let mut gpu = match GpuFlow::with_options(&cpu, Policy::from_env().map_err(|e| e.to_string())?, Options::default()) {
        Ok(g) => g,
        Err(e) if !gpu_required() && e.to_string().contains("no GPU device") => {
            println!("skipping: {e}");
            return Ok(());
        }
        Err(e) => return Err(e.to_string()),
    };
    let scale = setup.velocity_scale();
    let plane = plane_cells(&cpu);
    println!(
        "device {}; {} cells; τ⁺ − ½ {:.3e}, τ⁻ − ½ {:.3e}, dt {dt:.4} s, ν {nu:.3e}, velocity scale {scale:.3} m/s per lattice unit; {} cells in the plane",
        gpu.adapter_name(),
        cpu.grid().len(),
        setup.tau_plus - 0.5,
        setup.tau_minus - 0.5,
        plane.len()
    );
    let (tol_u, tol_r, tol_i) = (env("PARITY_DU", 2e-5), env("PARITY_DRHO", 1e-6), env("PARITY_INTEGRAL", 1e-4));
    let (mut worst_u, mut worst_r) = (0.0f64, 0.0f64);
    let mut failures = Vec::new();
    let mut done = 0;
    while done < STEPS {
        Backend::advance(&mut cpu, CHUNK).map_err(|e| e.to_string())?;
        Backend::advance(&mut gpu, CHUNK).map_err(|e| e.to_string())?;
        done += CHUNK;
        let (a, b) = (Backend::fields(&mut cpu).map_err(|e| e.to_string())?, Backend::fields(&mut gpu).map_err(|e| e.to_string())?);
        assert_eq!(a.velocity.len(), b.velocity.len());
        let finite = a.velocity.iter().chain(&b.velocity).all(|u| u.iter().all(|v| v.is_finite()))
            && a.density.iter().chain(&b.density).all(|r| r.is_finite());
        if !finite {
            failures.push(format!("step {done}: a field is not finite"));
            break;
        }
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
        let gated = done <= GATED_THROUGH;
        if gated {
            worst_u = worst_u.max(du);
            worst_r = worst_r.max(drho);
            if du >= tol_u || drho >= tol_r {
                failures.push(format!("step {done}: pointwise |Δu| {du:.3e} (bar {tol_u:.0e}), |Δρ| {drho:.3e} (bar {tol_r:.0e})"));
            }
        }
        let nt = Backend::nu_tilde(&mut gpu)
            .ok()
            .flatten()
            .zip(Backend::nu_tilde(&mut cpu).ok().flatten())
            .map_or(f64::NAN, |(g, c)| g.iter().zip(&c).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max));
        println!(
            "step {done} ({}): max |u| {umax:.4} lattice; max |Δu| {du:.3e} ({:.2e} of max), rms {rms:.3e}; max |Δρ| {drho:.3e}; max |Δν̃| {nt:.3e}",
            if gated { "gated" } else { "reported" },
            du / umax.max(1e-12)
        );
    }
    let (ci, gi) = (integrals(&mut cpu, &plane, scale)?, integrals(&mut gpu, &plane, scale)?);
    for k in 0..4 {
        let rel = (ci[k] - gi[k]).abs() / ci[k].abs().max(1e-12);
        println!("step {STEPS}: {} cpu {:.6e} device {:.6e}, relative difference {rel:.2e} (bar {tol_i:.0e})", INTEGRAL_NAMES[k], ci[k], gi[k]);
        if !(rel < tol_i) {
            failures.push(format!("{}: relative difference {rel:.2e} (bar {tol_i:.0e})", INTEGRAL_NAMES[k]));
        }
    }
    println!("worst pointwise through step {GATED_THROUGH}: max |Δu| {worst_u:.3e} (bar {tol_u:.0e}), max |Δρ| {worst_r:.3e} (bar {tol_r:.0e})");
    if failures.is_empty() { Ok(()) } else { Err(failures.join("; ")) }
}

/// The mechanism check: the slotted case twice from the same state, the second with `SEED` added to the density of
/// one fluid cell, on the CPU and on the device. If the instability amplifies round-off, the device pair parts like
/// the CPU pair (by the same factor per chunk), and the device pair's ends reach the size of the CPU/device
/// difference of the main test.
#[test]
#[ignore = "local: run on the machine whose GPU is to be checked (cargo test -p fieldcraft-lbm-gpu --release --test parity_low_tau seed -- --ignored --nocapture)"]
fn a_seed_of_1e_minus_7_grows_alike_on_the_cpu_and_on_the_device() {
    const SEED: f64 = 1e-7;
    let (m, obstacles, _, _) = case(true);
    let setup = Setup::new(&m, None, &obstacles, &Limits::default()).unwrap_or_else(|e| panic!("{e}"));
    let scale = setup.velocity_scale();
    let (mut a, mut b) = match (fieldcraft_lbm::Flow::new(&setup, &obstacles), fieldcraft_lbm::Flow::new(&setup, &obstacles)) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(e), _) | (_, Err(e)) => panic!("{e}"),
    };
    let f0 = Backend::fields(&mut a).unwrap_or_else(|e| panic!("{e}"));
    let fluid: Vec<usize> = (0..f0.density.len()).filter(|&c| Backend::is_fluid(&a, c)).collect();
    let Some(&seed) = fluid.get(fluid.len() / 2) else { panic!("no fluid cell") };
    let state = |c: usize, e: f64| (f0.density[c] + if c == seed { e } else { 0.0 }, [0, 1, 2].map(|k| f0.velocity[c][k] / scale));
    a.set_equilibrium(|c| state(c, 0.0));
    b.set_equilibrium(|c| state(c, SEED));
    let policy = Policy::from_env().unwrap_or_else(|e| panic!("{e}"));
    let (mut ga, mut gb) = match (GpuFlow::with_options(&a, policy, Options::default()), GpuFlow::with_options(&b, policy, Options::default())) {
        (Ok(x), Ok(y)) => (x, y),
        (Err(e), _) | (_, Err(e)) if !gpu_required() && e.to_string().contains("no GPU device") => {
            println!("skipping: {e}");
            return;
        }
        (Err(e), _) | (_, Err(e)) => panic!("{e}"),
    };
    let part = |x: &mut dyn Backend, y: &mut dyn Backend| -> f64 {
        let (fx, fy) = (x.fields().unwrap_or_else(|e| panic!("{e}")), y.fields().unwrap_or_else(|e| panic!("{e}")));
        fluid.iter().map(|&c| (fx.density[c] - fy.density[c]).abs()).fold(0.0, f64::max)
    };
    let mut last = (0.0f64, 0.0f64);
    for done in (CHUNK..=STEPS).step_by(CHUNK as usize) {
        for x in [&mut a as &mut dyn Backend, &mut b, &mut ga, &mut gb] {
            x.advance(CHUNK).unwrap_or_else(|e| panic!("{e}"));
        }
        let (c, g) = (part(&mut a, &mut b), part(&mut ga, &mut gb));
        println!(
            "step {done}: max |Δρ| of the pair: cpu {c:.3e}, device {g:.3e} (growth since the last chunk: cpu {:.2}x, device {:.2}x)",
            c / last.0.max(1e-30),
            g / last.1.max(1e-30)
        );
        assert!(c.is_finite() && g.is_finite(), "step {done}: not finite");
        last = (c, g);
    }
}
