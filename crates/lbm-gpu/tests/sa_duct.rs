//! The Spalart–Allmaras RANS square duct at Re_b ≈ 2205 (Re_τ ≈ 151), on the CPU and the GPU.
//!
//! Fully developed RANS flow in a straight duct does not change along it, so the run needs a few
//! cells along x (periodic), driven by a body force. The duct is 2h × 2h with halfway no-slip box
//! faces on y and z. Units: the half-width h = 1 and the perimeter-mean friction velocity u_τ = 1,
//! ν = 1/Re_τ. The force balance a (2h)² = τ_w (8h) gives a = 2 u_τ²/h. SA is resolved to the wall
//! (no wall model), so the first cell centres sit at y⁺ ≈ ½ Re_τ / N with N cells per half-width.
//!
//! What a run reads, once settled: U_b⁺ (the flow rate over (2h)²), f = 8/U_b⁺², Re_b = U_b h/ν, and
//! U_c/U_b (the centreline over the bulk velocity). The run starts from the product of two 1/7-law
//! profiles at the expected U_b⁺. It is settled when U_b moves by less than [`SETTLED`] between
//! windows of [`WINDOW`] steps.
//!
//! **References.**
//! - The gate: the DNS of S. Pirozzoli, D. Modesti, P. Orlandi, F. Grasso, *Turbulence and
//!   secondary motions in square duct flow*, J. Fluid Mech., 2018 (doi:10.1017/jfm.2018.66;
//!   arXiv:1707.03638). Their case A (their table 1) is this flow, Gavrilakis' case
//!   (J. Fluid Mech. 244:101–129, 1992): 2 h u_b/ν = 4410, so U_b h/ν = 2205; Re_τ* = h u_τ*/ν = 151
//!   with u_τ* the perimeter-mean friction velocity; C_f = 9.38 × 10⁻³, so f = 4 C_f = 0.03752 and
//!   U_b⁺ = √(2/C_f) = 14.60. The run is set at Re_τ 150.9, so f is compared with the DNS directly.
//! - Beside it: Jones's correlation for the square duct (O. C. Jones, *An improvement in the
//!   calculation of turbulent friction in rectangular ducts*, J. Fluids Eng. 98:173–180, 1976),
//!   1/√f = 2 log₁₀(2.25 Re_b √f) − 0.8 with f = 8 u_τ²/U_b², at the run's own Re_b. It gives
//!   f = 0.03748 at Re_b 2205, within 0.1 % of the DNS. A. Pinelli, M. Uhlmann, A. Sekimoto and
//!   G. Kawahara (J. Fluid Mech. 644:107–122, 2010, doi:10.1017/S0022112009992242) find their DNS
//!   over Re_b 1077–3500 in good agreement with Jones but publish no friction values; Pirozzoli et
//!   al. find the hydraulic-diameter law and Jones's correction equally accurate for the square
//!   duct. Neither paper quotes a tolerance for the correlation.
//!
//! **What SA cannot do here.** SA is a linear (Boussinesq) eddy-viscosity model, so it produces no
//! secondary flow of the second kind: Prandtl's corner vortices come from the anisotropy of the
//! normal stresses. The mean velocity stays low in the corners, so U_c/U_b and the wall shear along
//! an edge are reported, not gated. The bulk friction is less sensitive.

#[path = "../../lbm/tests/common/mod.rs"]
mod common;

use common::cases::*;
use common::*;
use fieldcraft_gpu::select::Policy;
use fieldcraft_lbm::backend::Backend;
use fieldcraft_lbm_gpu::GpuFlow;
use fieldcraft_model::{Collision, FaceCondition, HrrSigma, Procedure, Turbulence};

/// Steps per settling window and the relative change of U_b between windows.
const WINDOW: u64 = 5000;
const SETTLED: f64 = 2e-5;
/// Gavrilakis' and Pirozzoli et al.'s case A: Re_b = U_b h/ν 2205; Re_τ follows from Jones's
/// correlation there (150.9, the DNS's 151).
const RE_B: f64 = 2205.0;
/// Pirozzoli et al.'s case A: C_f = 2 (u_τ*/U_b)² and Re_τ* (see the module docs).
const DNS_CF: f64 = 9.38e-3;
const DNS_RE_TAU: f64 = 151.0;

/// Jones's f at `re_b` (U_b h/ν), by fixed-point iteration (it converges in a few steps).
fn jones(re_b: f64) -> f64 {
    let mut f: f64 = 0.03;
    for _ in 0..100 {
        f = (2.0 * (2.25 * re_b * f.sqrt()).log10() - 0.8).powi(-2);
    }
    f
}

/// A duct run's parameters.
#[derive(Debug, Clone, Copy)]
struct Duct {
    re_tau: f64,
    /// Cells per half-width.
    cells: usize,
    lattice_bulk: f64,
    max_steps: u64,
    gpu: bool,
    /// Stop once settled; otherwise run `max_steps` exactly.
    settle: bool,
}

#[derive(Debug, Clone)]
struct DuctResult {
    bulk: f64,
    centre: f64,
    min_nu_tilde: f64,
    settled: bool,
    steps: u64,
    backend: String,
}

impl Duct {
    /// The run, or `None` when the GPU was asked for and there is none.
    fn run(&self) -> Option<DuctResult> {
        let dx = 1.0 / self.cells as f64;
        let mut d = domain([0.0; 3], [4.0 * dx, 2.0, 2.0], dx);
        d.faces.x_min = FaceCondition::Periodic {};
        d.faces.x_max = FaceCondition::Periodic {};
        d.acceleration = [2.0, 0.0, 0.0];
        let bulk_guess = jones(RE_B).recip().sqrt() * 8f64.sqrt();
        let dt = self.lattice_bulk * dx / bulk_guess;
        let sa = Turbulence::SpalartAllmaras { prandtl: 0.9, wall_model: None, inflow_ratio: 3.0, ddes: None };
        let mut p = collide(self.max_steps as f64 * dt, dt, None, Collision::Hrr);
        if let Procedure::Fluid { turbulence, sigma, .. } = &mut p {
            *turbulence = Some(sa);
            *sigma = Some(HrrSigma::Value(0.99));
        }
        let m = model(1.0, 1.0 / self.re_tau, d, p);
        let (setup, mut flow) = prepare(&m, &[]);
        let grid = *flow.grid();
        let scale = setup.velocity_scale();
        // The product of two 1/7 laws has a mean of (7/8)² of its peak.
        let peak = bulk_guess / scale * 64.0 / 49.0;
        flow.set_equilibrium(|c| {
            let [_, y, z] = grid.centre(c);
            let (wy, wz) = ((1.0 - (y - 1.0).abs()).max(0.0), (1.0 - (z - 1.0).abs()).max(0.0));
            (1.0, [peak * (wy * wz).powf(1.0 / 7.0), 0.0, 0.0])
        });
        let mut backend: Box<dyn Backend> = if self.gpu {
            let policy = Policy::from_env().ok()?;
            Box::new(GpuFlow::new(&flow, policy).ok()?)
        } else {
            Box::new(flow)
        };
        let fluid: Vec<usize> = (0..grid.len()).filter(|&c| backend.is_fluid(c)).collect();
        let area = 4.0;
        let bulk = |fields: &fieldcraft_lbm::backend::Fields| {
            fluid.iter().map(|&c| fields.velocity.get(c).map_or(0.0, |u| u[0] * scale)).sum::<f64>() * dx * dx / 4.0 / area
        };
        let (mut steps, mut previous, mut settled, mut ub) = (0u64, f64::NAN, false, f64::NAN);
        while steps < self.max_steps {
            backend.advance(WINDOW).unwrap();
            steps += WINDOW;
            ub = bulk(&backend.fields().unwrap());
            let change = (ub / previous - 1.0).abs();
            previous = ub;
            if change < SETTLED {
                settled = true;
                if self.settle {
                    break;
                }
            }
        }
        let fields = backend.fields().unwrap();
        let near: Vec<usize> = fluid
            .iter()
            .copied()
            .filter(|&c| {
                let [_, y, z] = grid.centre(c);
                (y - 1.0).abs() < 0.75 * dx && (z - 1.0).abs() < 0.75 * dx
            })
            .collect();
        let uc = near.iter().map(|&c| fields.velocity.get(c).map_or(0.0, |u| u[0] * scale)).sum::<f64>() / near.len().max(1) as f64;
        let nu_tilde = backend.nu_tilde().unwrap().unwrap_or_default();
        let min_nu_tilde = fluid.iter().map(|&c| nu_tilde.get(c).copied().unwrap_or(f64::NAN)).fold(f64::INFINITY, f64::min);
        Some(DuctResult { bulk: ub, centre: uc / ub, min_nu_tilde, settled, steps, backend: backend.name() })
    }
}

/// Jones's correlation at Re_b 2205 gives Re_τ ≈ 151 and the DNS's friction within 0.2 %.
#[test]
fn jones_s_correlation_matches_the_dns_edge_in_wall_units() {
    let f = jones(RE_B);
    let re_tau = RE_B * (f / 8.0).sqrt();
    assert!((f - 0.03748).abs() < 1e-5, "{f}");
    assert!((re_tau - DNS_RE_TAU).abs() < 0.5, "Re_τ {re_tau}");
    assert!((f / (4.0 * DNS_CF) - 1.0).abs() < 2e-3, "Jones {f}, DNS {}", 4.0 * DNS_CF);
}

/// The environment variable that runs the duct study (docs/verification.md, "Long studies").
const STUDY: &str = "FIELDCRAFT_DUCT_STUDY";

/// The duct at Re_τ ≈ 151, SA resolved to the wall on the hybrid collision (σ 0.99),
/// `FIELDCRAFT_DUCT_CELLS` cells per half-width (100 by default), run until it settles, on the GPU
/// with `FIELDCRAFT_DUCT_GPU=1`, else on the CPU. Gates: f within 5 % of Pirozzoli et al.'s DNS;
/// settled, ν̃ ≥ 0. Jones at the run's Re_b and U_c/U_b are reported. It runs only when asked:
/// `FIELDCRAFT_DUCT_STUDY=1 FIELDCRAFT_DUCT_GPU=1 cargo test --release -p fieldcraft-lbm-gpu --test
/// sa_duct -- --ignored --nocapture study` (about two minutes on an RTX 4070 Laptop GPU; hours on
/// the CPU). The bounded regression below checks its start on every heavy run.
///
/// Measured when written: on the CPU at 10 cells per h (first cell y⁺ 7.5, 60 000 steps, 59 s on
/// 4 threads), f +70 % against the DNS, too coarse for SA resolved to the wall; on the RTX 4070
/// Laptop GPU (in-place streaming, 160 000 cells) at 100 cells per h (y⁺ 0.75): 270 000 steps in
/// 107 s, U_b⁺ 14.867, Re_b 2244, **f 0.03620, −3.52 % against the
/// DNS** (0.03752; −2.95 % against Jones at Re_b 2244, 0.03730), U_c/U_b 1.352.
#[test]
#[ignore = "minutes on a GPU: set FIELDCRAFT_DUCT_STUDY=1 (docs/verification.md, \"Long studies\")"]
fn study_the_sa_square_duct_against_jones_s_friction() {
    if std::env::var_os(STUDY).is_none_or(|v| v != "1") {
        println!("the duct study runs only with {STUDY}=1: skipped");
        return;
    }
    let cells = std::env::var("FIELDCRAFT_DUCT_CELLS").ok().and_then(|v| v.parse().ok()).unwrap_or(100);
    let gpu = std::env::var("FIELDCRAFT_DUCT_GPU").is_ok_and(|v| v == "1");
    let re_tau = RE_B * (jones(RE_B) / 8.0).sqrt();
    let max_steps = std::env::var("FIELDCRAFT_DUCT_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(2_000_000);
    let duct = Duct { re_tau, cells, lattice_bulk: 0.1, max_steps, gpu, settle: true };
    let started = std::time::Instant::now();
    let Some(r) = duct.run() else {
        println!("skipping: no GPU");
        return;
    };
    let f = 8.0 / (r.bulk * r.bulk);
    let re_b = r.bulk * re_tau;
    let (dns, reference) = (4.0 * DNS_CF, jones(re_b));
    println!(
        "SA square duct on {}, {cells} cells per h (first cell y⁺ ≈ {:.2}), HRR σ 0.99: U_b⁺ {:.3}, Re_b {re_b:.0}; f {f:.5} (DNS {dns:.5}, {:+.2} %; Jones {reference:.5}, {:+.2} %); \
         U_c/U_b {:.3}; min ν̃ {:.2e}; settled {} after {} steps, {:.0} s",
        r.backend,
        0.5 * re_tau / cells as f64,
        r.bulk,
        100.0 * (f / dns - 1.0),
        100.0 * (f / reference - 1.0),
        r.centre,
        r.min_nu_tilde,
        r.settled,
        r.steps,
        started.elapsed().as_secs_f64()
    );
    let mut score = Score::default();
    score.push(Gate::new("SA duct", &format!("f, {cells} cells per h"), f, dns, 0.05));
    println!("{}", score.table());
    assert!(r.settled, "not settled in {} steps", r.steps);
    assert!(r.min_nu_tilde >= 0.0, "ν̃ {}", r.min_nu_tilde);
    assert!(score.passes());
}

/// The study above, bounded and on the CPU: 10 cells per half-width for 10 000 steps (the start,
/// far from settled), U_b⁺ and U_c/U_b frozen at their first measurement. The CPU run is
/// deterministic, so the band of 10⁻⁹ (relative) is round-off room only. About ten seconds on four threads.
#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn a_short_sa_duct_keeps_the_study_s_start() {
    // U_b⁺ and U_c/U_b, measured when frozen (2026-11-08, Windows x86_64; the same bits on one and
    // four threads).
    const FROZEN: (f64, f64) = (1.1760130275441869e1, 1.4119552064598744e0);
    let re_tau = RE_B * (jones(RE_B) / 8.0).sqrt();
    let duct = Duct { re_tau, cells: 10, lattice_bulk: 0.1, max_steps: 10_000, gpu: false, settle: false };
    let r = duct.run().unwrap();
    println!("short SA duct, 10 cells per h, 10 000 steps: U_b⁺ {:e}, U_c/U_b {:e}", r.bulk, r.centre);
    assert_eq!(r.steps, 10_000);
    assert!(r.min_nu_tilde >= 0.0, "ν̃ {}", r.min_nu_tilde);
    assert!(
        (r.bulk / FROZEN.0 - 1.0).abs() < 1e-9 && (r.centre / FROZEN.1 - 1.0).abs() < 1e-9,
        "U_b⁺ {} U_c/U_b {} (frozen {FROZEN:?})",
        r.bulk,
        r.centre
    );
}
