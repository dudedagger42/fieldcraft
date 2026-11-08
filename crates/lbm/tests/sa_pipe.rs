//! The Spalart–Allmaras RANS pipe at Re_b ≈ 5300: a first 3D accuracy check on a curved
//! (staircase) wall, against DNS.
//!
//! Fully developed flow in a straight round pipe does not change along it, so a RANS run needs
//! one cell along x (periodic), driven by a body force: a pipe of radius R = 1 along x, units
//! u_τ = R = 1, ν = 1/Re_τ, and a = 2 u_τ²/R, so that the wall shear of the nominal circle,
//! a R / 2, is u_τ². The wall is a staircase: the cells whose centres lie outside the circle are
//! a tube obstacle, with halfway bounce-back on their links (the same build as
//! `verification.rs::pipe`, Poiseuille on a staircase wall). SA is resolved to the wall (no wall
//! model), so the first cell centres sit at y⁺ ≈ ½ Re_τ/k with k cells per radius.
//!
//! What a run reads, after it settles:
//! - U_b⁺, the flow rate over the nominal area π R² (the cells' own area is reported beside it);
//!   f = 8/U_b⁺² and C_f = 2/U_b⁺²;
//! - U_c/U_b, the centreline over the bulk velocity;
//! - the wall drag the tube reports (momentum exchange) over the body force on the fluid cells,
//!   1 when fully developed.
//!
//! The run starts from a 1/7-law profile at the expected U_b⁺. It is settled when U_b moves by less
//! than [`SETTLED`] between two windows of [`WINDOW`] steps and the drag balances the body force to
//! [`BALANCE`].
//!
//! **References.**
//! - J. G. M. Eggels, F. Unger, M. H. Weiss, J. Westerweel, R. J. Adrian, R. Friedrich, F. T. M.
//!   Nieuwstadt, *Fully developed turbulent pipe flow: a comparison between direct numerical
//!   simulation and experiment*, J. Fluid Mech. 268:175–209, 1994 (doi:10.1017/S002211209400131X).
//!   Their published radial DNS profile at Re_b = 5300 (4TU.ResearchData,
//!   doi:10.4121/uuid:3e4aee44-0857-40e3-8492-1cb37ac1e189, `dnsEggels5300.asc`) is in wall units
//!   (the total shear stress is 1 at the wall). Its area integral over its 96 radial cells gives
//!   U_b⁺ = 14.730 (Re_τ = Re_b / (2 U_b⁺) = 179.9, f = 0.03687), and its first cell gives
//!   U_c⁺ = 19.310, U_c/U_b = 1.311. Those are the gate's references. The dataset is published
//!   under 4TU.ResearchData's General Terms of Use; only these numbers are used, and the file is not
//!   in the repository.
//! - G. K. El Khoury, P. Schlatter, A. Noorani, P. F. Fischer, G. Brethouwer, A. V. Johansson,
//!   *Direct numerical simulation of turbulent pipe flow at moderately high Reynolds numbers*,
//!   Flow Turbul. Combust. 91:475–495, 2013 (doi:10.1007/s10494-013-9482-8): u_τ/U_b = 0.0683 at
//!   Re_b 5300, Re_τ 181 (U_b⁺ 14.64), as tabulated by a later paper; reported beside the gate.
//! - Blasius' f = 0.316 Re_b^(−1/4) at the run's own Re_b = 2 U_b⁺ Re_τ, reported.

mod common;

use common::cases::*;
use common::*;
use fieldcraft_lbm::backend::Backend;
use fieldcraft_model::{Collision, FaceCondition, HrrSigma, Procedure, Turbulence};

/// Eggels et al.'s DNS at Re_b 5300 (see the module docs): U_b⁺, U_c/U_b and Re_τ.
const EGGELS_BULK: f64 = 14.730;
const EGGELS_CENTRE: f64 = 1.311;
const EGGELS_RE_TAU: f64 = 179.9;
/// El Khoury et al. at Re_b 5300: u_τ/U_b.
const EL_KHOURY_FRICTION: f64 = 0.0683;

/// Steps per settling window, relative change of U_b between windows, and the drag balance.
const WINDOW: u64 = 2000;
const SETTLED: f64 = 2e-5;
const BALANCE: f64 = 1e-3;

/// A pipe run's parameters.
#[derive(Debug, Clone, Copy)]
struct Pipe {
    re_tau: f64,
    /// Cells per radius.
    cells: usize,
    /// U_b in lattice units (about), which sets the time step with the expected U_b⁺.
    lattice_bulk: f64,
    collision: Collision,
    sigma: Option<f64>,
    max_steps: u64,
    /// Stop once settled; otherwise run `max_steps` exactly.
    settle: bool,
}

/// A run's state when it stopped.
#[derive(Debug, Clone)]
struct PipeResult {
    /// U_b⁺ on the nominal area π R², and on the fluid cells' own area.
    bulk: f64,
    bulk_cells: f64,
    /// U_c / U_b (nominal).
    centre: f64,
    /// The tube's wall drag over the body force on the fluid.
    balance: f64,
    min_nu_tilde: f64,
    settled: bool,
    steps: u64,
}

impl Pipe {
    fn run(&self) -> PipeResult {
        let dx = 1.0 / self.cells as f64;
        let half = 1.0 + 2.0 * dx;
        let mut d = domain([0.0, -half, -half], [dx, half, half], dx);
        d.faces.x_min = FaceCondition::Periodic {};
        d.faces.x_max = FaceCondition::Periodic {};
        d.acceleration = [2.0, 0.0, 0.0];
        d.obstacles = vec!["Tube".into()];
        let tube = obstacle("Tube", tube(-1.0, dx + 1.0, [0.0, 0.0], 1.0, 3.0 * half, 2048), &["inner", "outer", "start", "end"]);
        let dt = self.lattice_bulk * dx / EGGELS_BULK;
        let sa = Turbulence::SpalartAllmaras { prandtl: 0.9, wall_model: None, inflow_ratio: 3.0, ddes: None };
        let mut p = collide(self.max_steps as f64 * dt, dt, None, self.collision);
        if let Procedure::Fluid { turbulence, sigma, .. } = &mut p {
            *turbulence = Some(sa);
            *sigma = self.sigma.map(HrrSigma::Value);
        }
        let m = model(1.0, 1.0 / self.re_tau, d, p);
        let (setup, mut flow) = prepare(&m, std::slice::from_ref(&tube));
        let grid = *flow.grid();
        let scale = setup.velocity_scale();
        // The 1/7 law at the expected U_b⁺: U_b = (49/60) U_c.
        let centre_guess = EGGELS_BULK / scale * 60.0 / 49.0;
        flow.set_equilibrium(|c| {
            let [_, y, z] = grid.centre(c);
            let wall = (1.0 - (y * y + z * z).sqrt()).max(0.0);
            (1.0, [centre_guess * wall.powf(1.0 / 7.0), 0.0, 0.0])
        });
        let fluid: Vec<usize> = (0..grid.len()).filter(|&c| flow.moments(c).is_some()).collect();
        let area_cells = fluid.len() as f64 * dx * dx;
        let nominal = std::f64::consts::PI;
        let measure = |flow: &fieldcraft_lbm::Flow| {
            let q: f64 = fluid.iter().map(|&c| flow.moments(c).map_or(0.0, |(_, u)| u[0] * scale)).sum::<f64>() * dx * dx;
            let wall: f64 =
                flow.groups().iter().zip(flow.forces()).filter(|(g, _)| g.body.is_some()).map(|(_, f)| f[0]).sum::<f64>() * setup.force_scale();
            let drive = 2.0 * area_cells * dx;
            (q, wall / drive)
        };
        let (mut steps, mut previous, mut settled) = (0u64, f64::NAN, false);
        let (mut q, mut balance) = (f64::NAN, f64::NAN);
        while steps < self.max_steps {
            flow.advance(WINDOW).unwrap();
            steps += WINDOW;
            (q, balance) = measure(&flow);
            let change = (q / previous - 1.0).abs();
            previous = q;
            if change < SETTLED && (balance - 1.0).abs() < BALANCE {
                settled = true;
                if self.settle {
                    break;
                }
            }
        }
        // The centreline: the mean of the cells nearest the axis.
        let near: Vec<usize> = fluid
            .iter()
            .copied()
            .filter(|&c| {
                let [_, y, z] = grid.centre(c);
                (y * y + z * z).sqrt() < 0.75 * dx
            })
            .collect();
        let uc = near.iter().map(|&c| flow.moments(c).map_or(0.0, |(_, u)| u[0] * scale)).sum::<f64>() / near.len().max(1) as f64;
        let nu_tilde = flow.spalart_allmaras().unwrap().nu_tilde();
        let min_nu_tilde = fluid.iter().map(|&c| nu_tilde[c]).fold(f64::INFINITY, f64::min);
        PipeResult { bulk: q / nominal, bulk_cells: q / area_cells, centre: uc / (q / nominal), balance, min_nu_tilde, settled, steps }
    }
}

/// The reference numbers: Eggels' U_b⁺ and Re_τ agree with Re_b 5300, and El Khoury's friction
/// velocity and Blasius' law lie within 1 % of Eggels' f.
#[test]
fn the_pipe_references_agree_with_each_other() {
    assert!((2.0 * EGGELS_BULK * EGGELS_RE_TAU - 5300.0).abs() < 5.0, "Re_b = 2 U_b⁺ Re_τ");
    let f = 8.0 / (EGGELS_BULK * EGGELS_BULK);
    assert!((f - 0.03687).abs() < 1e-5, "{f}");
    let el_khoury = 8.0 * EL_KHOURY_FRICTION * EL_KHOURY_FRICTION;
    let blasius = 0.316 * 5300f64.powf(-0.25);
    assert!((el_khoury / f - 1.0).abs() < 0.015 && (blasius / f - 1.0).abs() < 0.01, "{el_khoury} {blasius} {f}");
}

/// The environment variable that runs the pipe study (docs/verification.md, "Long studies").
const STUDY: &str = "FIELDCRAFT_PIPE_STUDY";

/// The pipe at Eggels' Re_τ 179.9 on the hybrid collision (σ 0.99), SA resolved to the wall on a
/// staircase tube, `FIELDCRAFT_PIPE_CELLS` cells per radius (100 by default), run until it
/// settles. Gates: f within 5 % of Eggels' DNS; settled, with ν̃ ≥ 0. U_c/U_b, El Khoury's and
/// Blasius' f are reported. About an hour on eight threads at 100 cells per radius, so it runs only
/// when asked: `FIELDCRAFT_PIPE_STUDY=1 cargo test --release -p fieldcraft-lbm --test sa_pipe --
/// --ignored --nocapture study`. The bounded regression below checks its start on every heavy run.
///
/// Measured when written (CPU, 8 threads; the 25-cell run on 4):
///
/// | cells per R (first cell y⁺) | steps | time | U_b⁺ | f against Eggels | U_c/U_b (Eggels 1.311) |
/// |---|---|---|---|---|---|
/// | 25 (3.6) | 168 000 | 357 s | 13.278 | +23.1 % | 1.309 |
/// | 50 (1.8) | 236 000 | 1237 s | 14.379 | +4.94 % | 1.298 |
/// | 100 (0.9) | 270 000 | 3372 s | 14.777 | −0.64 % | 1.299 |
///
/// SA resolved to the wall needs the first cell at y⁺ ≲ 1 on this staircase wall. At 100 cells per
/// R the friction is within 1 % of the DNS (Blasius at the run's Re_b 5317: f 0.03701), and the
/// tube's drag balances the body force to 1e-3. The error falls with the first cell's y⁺, so it is
/// the near-wall resolution, not the staircase.
#[test]
#[ignore = "an hour: set FIELDCRAFT_PIPE_STUDY=1 (docs/verification.md, \"Long studies\")"]
fn study_the_sa_pipe_at_re_b_5300_against_the_dns() {
    if std::env::var_os(STUDY).is_none_or(|v| v != "1") {
        println!("the pipe study runs only with {STUDY}=1: skipped");
        return;
    }
    let cells = std::env::var("FIELDCRAFT_PIPE_CELLS").ok().and_then(|v| v.parse().ok()).unwrap_or(100);
    let pipe =
        Pipe { re_tau: EGGELS_RE_TAU, cells, lattice_bulk: 0.05, collision: Collision::Hrr, sigma: Some(0.99), max_steps: 1_000_000, settle: true };
    let started = std::time::Instant::now();
    let r = pipe.run();
    let f = 8.0 / (r.bulk * r.bulk);
    let re_b = 2.0 * r.bulk * EGGELS_RE_TAU;
    println!(
        "SA pipe, {cells} cells per R (first cell y⁺ ≈ {:.2}), HRR σ 0.99: U_b⁺ {:.3} (Eggels {EGGELS_BULK}; {:+.2} %), on the cells' area {:.3}; f {f:.5}          (Eggels {:.5} {:+.2} %, El Khoury {:.5}, Blasius at Re_b {re_b:.0} {:.5}); U_c/U_b {:.3} (Eggels {EGGELS_CENTRE}); drag / drive {:.5};          min ν̃ {:.2e}; settled {} after {} steps, {:.0} s",
        0.5 * EGGELS_RE_TAU / cells as f64,
        r.bulk,
        100.0 * (r.bulk / EGGELS_BULK - 1.0),
        r.bulk_cells,
        8.0 / (EGGELS_BULK * EGGELS_BULK),
        100.0 * (f * EGGELS_BULK * EGGELS_BULK / 8.0 - 1.0),
        8.0 * EL_KHOURY_FRICTION * EL_KHOURY_FRICTION,
        0.316 * re_b.powf(-0.25),
        r.centre,
        r.balance,
        r.min_nu_tilde,
        r.settled,
        r.steps,
        started.elapsed().as_secs_f64()
    );
    let mut score = Score::default();
    score.push(Gate::new("SA pipe", &format!("f, {cells} cells per R"), f, 8.0 / (EGGELS_BULK * EGGELS_BULK), 0.05));
    println!("{}", score.table());
    assert!(r.settled, "not settled in {} steps", r.steps);
    assert!(r.min_nu_tilde >= 0.0, "ν̃ {}", r.min_nu_tilde);
    assert!(score.passes());
}

/// The study above, bounded: 16 cells per radius for 10 000 steps (the start, far from settled),
/// its U_b⁺ and U_c/U_b frozen at their first measurement. The run is deterministic, so the band
/// of 10⁻⁹ (relative) is round-off room only, and any change to the collision, SA, the staircase
/// wall or the measuring fails it. About a minute on four threads.
#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn a_short_sa_pipe_keeps_the_study_s_start() {
    // U_b⁺ and U_c/U_b, measured when frozen (2026-11-08, Windows x86_64; the same bits on one and
    // four threads).
    const FROZEN: (f64, f64) = (1.3434975838292615e1, 1.3634084915151985e0);
    let pipe =
        Pipe { re_tau: EGGELS_RE_TAU, cells: 16, lattice_bulk: 0.05, collision: Collision::Hrr, sigma: Some(0.99), max_steps: 10_000, settle: false };
    let r = pipe.run();
    println!("short SA pipe, 16 cells per R, 10 000 steps: U_b⁺ {:e}, U_c/U_b {:e}", r.bulk, r.centre);
    assert_eq!(r.steps, 10_000);
    assert!(r.min_nu_tilde >= 0.0, "ν̃ {}", r.min_nu_tilde);
    assert!(
        (r.bulk / FROZEN.0 - 1.0).abs() < 1e-9 && (r.centre / FROZEN.1 - 1.0).abs() < 1e-9,
        "U_b⁺ {} U_c/U_b {} (frozen {FROZEN:?})",
        r.bulk,
        r.centre
    );
}
