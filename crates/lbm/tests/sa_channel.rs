//! The Spalart–Allmaras RANS channel at Re_τ 1000 (board task C7), and the laminar channel that
//! checks its measuring pipeline.
//!
//! Fully developed plane channel flow is one-dimensional in the mean, so a RANS run is a column:
//! one cell periodic in x and z, walls at y = 0 and 2δ, driven by a body force a = u_τ²/δ.
//! Units are u_τ = δ = 1 and ν = 1/Re_τ.
//!
//! What a run reads, each a time mean over the averaging window:
//! - **carried** wall stress: the momentum the flow hands to the walls, from the closed-box
//!   budget of `support/wall_balance.rs` (#829), the body impulse less the change of the
//!   populations' momentum, per wall area;
//! - **applied**: the momentum exchange the walls report (`Flow::forces`), per area;
//! - **requested**: what the wall model asked for, ρ u_τ² per wall face: with SA, the wall
//!   function's u_τ at each first fluid node (`Flow::sa_wall_function`, Spalding's law on Han's
//!   on-node bounce, Musker's under the Musker wall); with a subgrid model under Musker's wall,
//!   Musker's request through `wall_balance::musker_request`;
//! - U_b, the bulk velocity, and C_f = τ_w / (½ ρ U_b²) from the carried stress.
//!
//! Settling ([`GATED`], common::cases): a flow-through is 2πδ/U_b, the length of the usual channel
//! box at the bulk speed. Two checks must both hold on means over the averaging window:
//! - the carried C_f moves by less than `drift` from the window before;
//! - the applied wall drag balances the body force to within `drift`, so a slow, steady approach to
//!   the fully developed state can't pass for settled.
//!
//! The averaging window is a flow-through, and at least two periods of the column's acoustic mode
//! between its walls (2 N_y / c_s steps).
//!
//! The reference is Lee & Moser's DNS at Re_τ = 1000.512: U_b⁺ = 1/0.0500256 = 19.990 and
//! C_f = 2 (u_τ/U_b)² = 5.005 × 10⁻³. The values come from the bulk velocity and friction velocity
//! in the header of their published mean profile, LM_Channel_1000_mean_prof.dat
//! (<https://turbulence.oden.utexas.edu/channel2015/>; M. Lee and R. D. Moser, *Direct numerical
//! simulation of turbulent channel flow up to Re_τ ≈ 5200*, J. Fluid Mech. 774:395–415, 2015,
//! doi:10.1017/jfm.2015.268).

mod common;
#[path = "support/wall_balance.rs"]
mod wall_balance;

use common::cases::*;
use common::*;
use fieldcraft_lbm::Flow;
use fieldcraft_lbm::statistics::Blocks;
use fieldcraft_model::{Collision, FaceCondition, Procedure, Turbulence, WallModel};

/// Lee & Moser (2015), Re_τ = 1000.512: U_b⁺ and C_f on the bulk velocity (see the module docs).
const LEE_MOSER_1000_BULK: f64 = 1.0 / 0.0500256;
const LEE_MOSER_1000_CF: f64 = 2.0 * 0.0500256 * 0.0500256;

/// A channel run's parameters.
#[derive(Debug, Clone)]
struct Channel {
    re_tau: f64,
    /// Cells per half-height δ.
    cells: usize,
    /// The expected U_b⁺, which sets the time step with `lattice_bulk`.
    bulk_guess: f64,
    /// U_b in lattice units (about).
    lattice_bulk: f64,
    collision: Collision,
    turbulence: Option<Turbulence>,
    /// The walls on lattice nodes (the on-node bounce: one boundary node layer more) rather than
    /// halfway between cells.
    on_node: bool,
    max_steps: u64,
    sample: u64,
    settling: Settling,
}

/// A channel run's time means over its last averaging window.
#[derive(Debug, Clone)]
struct ChannelResult {
    /// U_b / u_τ, with u_τ from the carried stress.
    bulk: f64,
    /// C_f from the carried, applied and requested wall stress, on the measured U_b.
    carried: f64,
    applied: f64,
    requested: Option<f64>,
    /// The applied wall drag over the body force (1 when fully developed).
    balance: f64,
    /// Every step's momentum budget closed (`wall_balance::closed_step`).
    budget_ok: bool,
    /// The smallest ν̃ of the run's last step (SA only).
    min_nu_tilde: Option<f64>,
    /// The carried C_f's block statistics over the whole run after the minimum flow-throughs,
    /// one block per averaging window.
    blocks: Blocks,
    settled: bool,
    steps: u64,
}

impl Channel {
    fn spacing(&self) -> f64 {
        1.0 / self.cells as f64
    }

    fn model(&self) -> fieldcraft_model::Model {
        let dx = self.spacing();
        let height = if self.on_node { 2.0 + dx } else { 2.0 };
        let mut d = domain([0.0; 3], [dx, height, dx], dx);
        for f in [&mut d.faces.x_min, &mut d.faces.x_max, &mut d.faces.z_min, &mut d.faces.z_max] {
            *f = FaceCondition::Periodic {};
        }
        d.acceleration = [1.0, 0.0, 0.0];
        d.initial_velocity = [self.bulk_guess, 0.0, 0.0];
        let dt = self.lattice_bulk * dx / self.bulk_guess;
        let mut p = collide(self.max_steps as f64 * dt, dt, None, self.collision);
        if let Procedure::Fluid { turbulence, .. } = &mut p {
            *turbulence = self.turbulence;
        }
        model(1.0, 1.0 / self.re_tau, d, p)
    }

    /// Run to settling or the step cap.
    fn run(&self) -> ChannelResult {
        let (setup, mut flow) = prepare(&self.model(), &[]);
        let grid = *flow.grid();
        let rows = grid.cells[1];
        let a = flow.acceleration();
        let nu = (setup.tau_plus - 0.5) / 3.0;
        // Wall faces per wall: one cell in x and z.
        let area = 2.0 * (grid.cells[0] * grid.cells[2]) as f64;
        let musker = matches!(self.turbulence, Some(Turbulence::Smagorinsky { wall_model: Some(WallModel::Musker), .. }))
            || matches!(self.turbulence, Some(Turbulence::SpalartAllmaras { wall_model: Some(WallModel::Musker), .. }));
        // The bulk velocity: the mean over the rows between the walls (the on-node walls' rows,
        // on the walls themselves, at half weight).
        let bulk = |flow: &Flow| {
            let row = |j: usize| {
                (0..grid.cells[0] * grid.cells[2])
                    .map(|k| flow.moments(grid.index([k % grid.cells[0], j, k / grid.cells[0]])).map_or(0.0, |(_, u)| u[0]))
                    .sum::<f64>()
                    / (grid.cells[0] * grid.cells[2]) as f64
            };
            let sum: f64 = (0..rows).map(|j| if self.on_node && (j == 0 || j + 1 == rows) { 0.5 * row(j) } else { row(j) }).sum();
            sum / if self.on_node { (rows - 1) as f64 } else { rows as f64 }
        };
        let acoustic = 2.0 * rows as f64 * 3f64.sqrt();
        let bulk_lattice = self.lattice_bulk;
        let flow_through = 2.0 * std::f64::consts::PI * self.cells as f64 / bulk_lattice;
        let window = ((flow_through.max(2.0 * acoustic) / self.sample as f64).round() as usize).max(1);
        let min_steps = (self.settling.min_flow_throughs * flow_through).ceil() as u64;
        let mut blocks = Blocks::new(window, min_steps / self.sample).unwrap();
        // Per sample interval: [carried, applied, requested, body, U_b] summed, then averaged.
        let mut history: Vec<[f64; 5]> = Vec::new();
        let mut sums = [0.0; 5];
        let mut any_requested = false;
        let mut budget_ok = true;
        let mut p = wall_balance::population_momentum(&flow);
        let mut steps = 0;
        let mean = |rows: &[[f64; 5]]| -> [f64; 5] { [0, 1, 2, 3, 4].map(|c| rows.iter().map(|r| r[c]).sum::<f64>() / rows.len().max(1) as f64) };
        let cf = |stress: f64, ub: f64, rho: f64| 2.0 * stress / (rho * ub * ub);
        let rho = flow.mass() / (0..grid.len()).filter(|&c| flow.is_fluid(c)).count() as f64;
        let mut settled = false;
        while steps < self.max_steps {
            flow.step().unwrap();
            steps += 1;
            let next = wall_balance::population_momentum(&flow);
            let m = flow.mass();
            let wall: [f64; 3] = [0, 1, 2].map(|k| flow.forces().iter().map(|f| f[k]).sum());
            let density = |cell: usize| flow.moments(cell).map_or(f64::NAN, |(r, _)| r);
            // With SA under a wall model, the law's ρ u_τ² at each first fluid node (along the flow,
            // +x); otherwise Musker's request where the wall model is Musker.
            let requested = match (flow.sa_wall_function(), flow.walls()) {
                ((Some(_), samples), _) => [samples.iter().map(|&(cell, _, u_tau)| density(cell) * u_tau * u_tau).sum(), 0.0, 0.0],
                (_, Some(walls)) if musker => walls
                    .wall_cells()
                    .filter(|w| w.4.is_some())
                    .filter_map(|(cell, q, distance, _, _)| {
                        let target = walls.modelled(cell)?.0;
                        let normal = fieldcraft_lbm::lattice::C[q].map(|v| f64::from(v) * 0.5 / distance);
                        wall_balance::musker_request(target, normal, distance, nu).ok().map(|r| r.map(|r| r * density(cell)))
                    })
                    .fold([0.0; 3], |s, r| [0, 1, 2].map(|k| s[k] + r[k])),
                _ => [0.0; 3],
            };
            any_requested |= requested[0] != 0.0;
            let report =
                wall_balance::closed_step(p, next, a.map(|a| a * m), wall, requested.map(|r| r / area), area, m, 1e-12, f64::EPSILON).unwrap();
            budget_ok &= report.budget_ok;
            p = next;
            for (s, v) in sums.iter_mut().zip([report.carried[0], report.applied[0], requested[0] / area, a[0] * m, 0.0]) {
                *s += v;
            }
            if steps % self.sample != 0 {
                continue;
            }
            let n = self.sample as f64;
            let row = [sums[0] / n, sums[1] / n, sums[2] / n, sums[3] / n, bulk(&flow)];
            sums = [0.0; 5];
            assert!(row.iter().all(|v| v.is_finite()), "the channel is not finite after {steps} steps: {row:?}");
            // Past MAX_BLOCKS the stream refuses more samples; the statistics keep what they hold.
            blocks.push(cf(row[0], row[4], rho)).ok();
            history.push(row);
            let carried: Vec<Vec<f64>> = history.iter().map(|r| vec![cf(r[0], r[4], rho)]).collect();
            let last = mean(&history[history.len().saturating_sub(window)..]);
            let balanced = (last[1] * area / last[3] - 1.0).abs() <= self.settling.drift;
            if balanced && self.settling.settled(&carried, window, steps as f64 / flow_through) {
                settled = true;
                break;
            }
        }
        let last = mean(&history[history.len().saturating_sub(window)..]);
        let u_tau = (last[0] / rho).sqrt();
        ChannelResult {
            bulk: last[4] / u_tau,
            carried: cf(last[0], last[4], rho),
            applied: cf(last[1], last[4], rho),
            requested: any_requested.then(|| cf(last[2], last[4], rho)),
            balance: last[1] * area / last[3],
            budget_ok,
            min_nu_tilde: flow
                .spalart_allmaras()
                .map(|f| (0..grid.len()).filter(|&c| flow.is_fluid(c)).filter_map(|c| f.nu_tilde().get(c).copied()).fold(f64::INFINITY, f64::min)),
            blocks,
            settled,
            steps,
        }
    }
}

fn report(name: &str, r: &ChannelResult, bulk_reference: f64, cf_reference: f64) {
    let a = r.blocks.acceptance();
    println!(
        "{name}: {} steps, settled {}; U_b⁺ {:.4} (ref {bulk_reference:.4}, {:+.2} %); C_f carried {:.5e} (ref {cf_reference:.5e}, {:+.2} %), applied {:.5e}, requested {}; drag / body force {:.6}; budget closed {}; min ν̃ {:?}; {} blocks, half-width {:.2e}, accepted {}",
        r.steps,
        r.settled,
        r.bulk,
        100.0 * (r.bulk / bulk_reference - 1.0),
        r.carried,
        100.0 * (r.carried / cf_reference - 1.0),
        r.applied,
        r.requested.map_or("—".into(), |c| format!("{c:.5e}")),
        r.balance,
        r.budget_ok,
        r.min_nu_tilde,
        r.blocks.means().len(),
        r.blocks.half_width(),
        a.accepted,
    );
}

/// The measuring pipeline on the laminar channel: plane Poiseuille flow at Re_τ = 10, where
/// U_b⁺ = Re_τ/3 and C_f = 2/U_b⁺² = 18/Re_τ² exactly. On 16 cells per δ with halfway walls (TRT)
/// the discretisation error is small next to the 0.5 % asked of U_b⁺ and the carried C_f; the
/// budget closes every step and applied matches carried. The pipeline's settling is the SA
/// channel's own, [`GATED`].
#[test]
fn a_laminar_channel_reads_poiseuille_friction() {
    let re_tau = 10.0;
    let ch = Channel {
        re_tau,
        cells: 16,
        bulk_guess: re_tau / 3.0,
        lattice_bulk: 0.05,
        collision: Collision::Trt,
        turbulence: None,
        on_node: false,
        max_steps: 200_000,
        sample: 50,
        settling: GATED,
    };
    let r = ch.run();
    let (bulk, cf) = (re_tau / 3.0, 18.0 / (re_tau * re_tau));
    report("laminar channel, Re_τ 10, 16 cells per δ", &r, bulk, cf);
    assert!(r.settled, "the laminar channel did not settle in {} steps", r.steps);
    assert!(r.budget_ok);
    assert!((r.bulk / bulk - 1.0).abs() < 5e-3, "U_b⁺ {}", r.bulk);
    assert!((r.carried / cf - 1.0).abs() < 5e-3, "C_f {}", r.carried);
    assert!((r.applied / r.carried - 1.0).abs() < 1e-3, "applied {} vs carried {}", r.applied, r.carried);
}

/// The reference numbers are Lee & Moser's header values (u_τ = 0.0500256 at U_b = 1).
#[test]
fn the_lee_moser_reference_is_their_published_bulk_and_friction_velocity() {
    assert!((LEE_MOSER_1000_BULK - 19.9898).abs() < 1e-4, "{LEE_MOSER_1000_BULK}");
    assert!((LEE_MOSER_1000_CF - 5.00512e-3).abs() < 1e-8, "{LEE_MOSER_1000_CF}");
}

/// The model with wall model `wall`, inflow ν̃/ν 3 (the initial field).
fn sa(wall: WallModel) -> Turbulence {
    Turbulence::SpalartAllmaras { prandtl: 0.9, wall_model: Some(wall), inflow_ratio: 3.0, ddes: None }
}

/// With SA under a wall model the harness reads the wall function's request, ρ u_τ² at each first
/// fluid node: positive, finite and of the carried stress's order after a few hundred steps on
/// Han's on-node wall and on Musker's halfway one (a smoke test of the reading, not the gate).
/// Under a second in release.
#[test]
fn the_sa_wall_function_s_request_is_read() {
    for (wall, on_node, cells) in [(WallModel::SpaldingBounce, true, 20), (WallModel::Musker, false, 10)] {
        let ch = Channel {
            re_tau: 1000.0,
            cells,
            bulk_guess: LEE_MOSER_1000_BULK,
            lattice_bulk: 0.1,
            collision: Collision::Mrt,
            turbulence: Some(sa(wall)),
            on_node,
            max_steps: 400,
            sample: 100,
            settling: GATED,
        };
        let r = ch.run();
        report(&format!("SA {wall:?}, {cells} cells per δ, 400 steps"), &r, LEE_MOSER_1000_BULK, LEE_MOSER_1000_CF);
        let requested = r.requested.unwrap_or(f64::NAN);
        assert!(requested > 0.0 && requested.is_finite(), "{wall:?}: requested {requested}");
        assert!(requested / r.carried > 0.1 && requested / r.carried < 10.0, "{wall:?}: requested {requested}, carried {}", r.carried);
        // The on-node wall's per-step budget leaves out its boundary nodes and is not exact in
        // the start-up; the halfway wall's closes every step.
        assert!(on_node || r.budget_ok, "{wall:?}");
    }
}
