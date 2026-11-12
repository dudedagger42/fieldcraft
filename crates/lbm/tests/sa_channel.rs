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
//! - U_b, the bulk velocity, and C_f = τ_w / (½ ρ U_b²) from the carried stress. On on-node walls
//!   the interval between each wall and its first fluid node is integrated from the wall law
//!   (common::cases::WallInterval, board WL3 P1), with the trapezoid from u = 0 reported beside it.
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

/// Lee & Moser (2015), Re_τ = 182.088: u_τ = 0.0637309 at U_b = 1 (ν = 3.5 × 10⁻⁴, δ = 1), from
/// the header of LM_Channel_0180_mean_prof.dat (same source as the Re_τ 1000 values).
const LEE_MOSER_180_RE_TAU: f64 = 182.088;
const LEE_MOSER_180_BULK: f64 = 1.0 / 0.0637309;
const LEE_MOSER_180_CF: f64 = 2.0 * 0.0637309 * 0.0637309;
/// The centreline U⁺ of Lee & Moser's Re_τ 182 profile, as the gate's window rounds it (17.75 to
/// 18.65 is 18.2 ± 2.5 %; the tabulated mean at y = δ reads about 18.27).
const LEE_MOSER_180_CENTRE: f64 = 18.2;

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
    /// The hybrid collision's fixed σ (`None`: its default, or another collision).
    sigma: Option<f64>,
    turbulence: Option<Turbulence>,
    /// The walls on lattice nodes (the on-node bounce: one boundary node layer more) rather than
    /// halfway between cells.
    on_node: bool,
    /// The walls as two obstacle slabs one cell thick in a box periodic in y, one row more at each
    /// end, rather than the y box faces (Han's on-node wall on obstacles, board WL1): the node rows
    /// sit where the box faces' would.
    slabs: bool,
    max_steps: u64,
    sample: u64,
    settling: Settling,
}

/// A channel run's time means over its last averaging window.
#[derive(Debug, Clone)]
struct ChannelResult {
    /// U_b / u_τ, with u_τ from the carried stress. On on-node walls the interval between each
    /// wall and its first node is integrated from the wall law (common::cases::WallInterval,
    /// board WL3 P1); `bulk_trapezoid` is the same with the trapezoid from u = 0 at the wall.
    bulk: f64,
    bulk_trapezoid: f64,
    /// The centreline velocity over u_τ (the mean of the two middle rows when there is no row on
    /// the centreline).
    centre: f64,
    /// C_f from the carried, applied and requested wall stress, on the measured U_b (the law's);
    /// `carried_trapezoid` the carried C_f on the trapezoid's U_b.
    carried: f64,
    carried_trapezoid: f64,
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
    /// The mean u over u_τ at the first three open rows (the half-way wall's y = ½, 1½, 2½ cells),
    /// the friction velocity and the viscosity (lattice units).
    profile: [f64; 3],
    u_tau: f64,
    nu: f64,
}

impl Channel {
    fn spacing(&self) -> f64 {
        1.0 / self.cells as f64
    }

    /// Whether the walls are the half-way wall (`*_bounce_surface`): two obstacle slabs two cells
    /// thick (`slabs`) whose faces are the walls at y = 0 and 2, the open rows exactly 2 cells per δ.
    fn surface(&self) -> bool {
        let wall = match self.turbulence {
            Some(Turbulence::SpalartAllmaras { wall_model, .. } | Turbulence::Smagorinsky { wall_model, .. }) => wall_model,
            _ => None,
        };
        matches!(wall, Some(WallModel::SpaldingBounceSurface | WallModel::SpalartAllmarasBounceSurface))
    }

    /// Cell rows of obstacle slab at each end of the box (0 without slabs).
    fn slab_rows(&self) -> usize {
        if !self.slabs {
            0
        } else if self.surface() {
            2
        } else {
            1
        }
    }

    /// The obstacle slabs, with `slabs`.
    fn obstacles(&self) -> Vec<fieldcraft_lbm::Obstacle> {
        let dx = self.spacing();
        let height = if self.on_node { 2.0 + dx } else { 2.0 };
        if !self.slabs {
            return Vec::new();
        }
        let t = self.slab_rows() as f64 * dx;
        vec![
            obstacle("Floor", cuboid([-1.0, -t, -1.0], [1.0 + dx, 0.0, 1.0 + dx]), &CUBOID_FACES),
            obstacle("Roof", cuboid([-1.0, height, -1.0], [1.0 + dx, height + t, 1.0 + dx]), &CUBOID_FACES),
        ]
    }

    fn model(&self) -> fieldcraft_model::Model {
        let dx = self.spacing();
        let height = if self.on_node { 2.0 + dx } else { 2.0 };
        let t = self.slab_rows() as f64 * dx;
        let mut d = if self.slabs { domain([0.0, -t, 0.0], [dx, height + t, dx], dx) } else { domain([0.0; 3], [dx, height, dx], dx) };
        for f in [&mut d.faces.x_min, &mut d.faces.x_max, &mut d.faces.z_min, &mut d.faces.z_max] {
            *f = FaceCondition::Periodic {};
        }
        if self.slabs {
            d.faces.y_min = FaceCondition::Periodic {};
            d.faces.y_max = FaceCondition::Periodic {};
            d.obstacles = self.obstacles().iter().map(|o| o.body.clone()).collect();
        }
        d.acceleration = [1.0, 0.0, 0.0];
        d.initial_velocity = [self.bulk_guess, 0.0, 0.0];
        let dt = self.lattice_bulk * dx / self.bulk_guess;
        let mut p = collide(self.max_steps as f64 * dt, dt, None, self.collision);
        if let Procedure::Fluid { turbulence, sigma, .. } = &mut p {
            *turbulence = self.turbulence;
            *sigma = self.sigma.map(fieldcraft_model::HrrSigma::Value);
        }
        model(1.0, 1.0 / self.re_tau, d, p)
    }

    /// Run to settling or the step cap.
    fn run(&self) -> ChannelResult {
        let (setup, mut flow) = prepare(&self.model(), &self.obstacles());
        let grid = *flow.grid();
        // The rows between the walls (the slabs' rows left out).
        let first = self.slab_rows();
        let rows = grid.cells[1] - 2 * first;
        let a = flow.acceleration();
        let nu = (setup.tau_plus - 0.5) / 3.0;
        // Wall faces per wall: one cell in x and z.
        let area = 2.0 * (grid.cells[0] * grid.cells[2]) as f64;
        let musker = matches!(self.turbulence, Some(Turbulence::Smagorinsky { wall_model: Some(WallModel::Musker), .. }))
            || matches!(self.turbulence, Some(Turbulence::SpalartAllmaras { wall_model: Some(WallModel::Musker), .. }));
        // The bulk velocity: the mean over the rows between the walls (the on-node walls' rows,
        // on the walls themselves, at half weight), `[with the wall intervals from the wall law,
        // with the trapezoid]`: the two differ on on-node walls only (WL3 P1).
        let sa_law = matches!(self.turbulence, Some(Turbulence::SpalartAllmaras { wall_model: Some(WallModel::SpalartAllmarasBounce), .. }));
        let interval = |u1: f64| if sa_law { WallInterval::new_sa_law(u1, 1.0, nu) } else { WallInterval::new(u1, 1.0, nu) };
        let bulk = |flow: &Flow| -> [f64; 2] {
            let row = |j: usize| {
                (0..grid.cells[0] * grid.cells[2])
                    .map(|k| flow.moments(grid.index([k % grid.cells[0], j + first, k / grid.cells[0]])).map_or(0.0, |(_, u)| u[0]))
                    .sum::<f64>()
                    / (grid.cells[0] * grid.cells[2]) as f64
            };
            let sum: f64 = (0..rows).map(|j| if self.on_node && (j == 0 || j + 1 == rows) { 0.5 * row(j) } else { row(j) }).sum();
            let height = if self.on_node { (rows - 1) as f64 } else { rows as f64 };
            // Each wall's interval, one cell (lattice units) from its node to its first fluid node.
            let law: f64 = if self.on_node {
                [row(1), row(rows - 2)].iter().filter_map(|&u1| interval(u1)).map(|w| w.law[0] - w.trapezoid[0]).sum()
            } else {
                0.0
            };
            [(sum + law) / height, sum / height]
        };
        let centre = |flow: &Flow| -> f64 {
            let row = |j: usize| {
                (0..grid.cells[0] * grid.cells[2])
                    .map(|k| flow.moments(grid.index([k % grid.cells[0], j + first, k / grid.cells[0]])).map_or(0.0, |(_, u)| u[0]))
                    .sum::<f64>()
                    / (grid.cells[0] * grid.cells[2]) as f64
            };
            if rows % 2 == 1 { row(rows / 2) } else { 0.5 * (row(rows / 2 - 1) + row(rows / 2)) }
        };
        let plane = |flow: &Flow, j: usize| -> f64 {
            (0..grid.cells[0] * grid.cells[2])
                .map(|k| flow.moments(grid.index([k % grid.cells[0], j + first, k / grid.cells[0]])).map_or(0.0, |(_, u)| u[0]))
                .sum::<f64>()
                / (grid.cells[0] * grid.cells[2]) as f64
        };
        let mut planes: Vec<[f64; 3]> = Vec::new();
        let acoustic = 2.0 * rows as f64 * 3f64.sqrt();
        let bulk_lattice = self.lattice_bulk;
        let flow_through = 2.0 * std::f64::consts::PI * self.cells as f64 / bulk_lattice;
        let window = ((flow_through.max(2.0 * acoustic) / self.sample as f64).round() as usize).max(1);
        let min_steps = (self.settling.min_flow_throughs * flow_through).ceil() as u64;
        let mut blocks = Blocks::new(window, min_steps / self.sample).unwrap();
        // Per sample interval: [carried, applied, requested, body, U_b (law), U_b (trapezoid)]
        // summed, then averaged.
        let mut history: Vec<[f64; 7]> = Vec::new();
        let mut sums = [0.0; 7];
        let mut any_requested = false;
        let mut budget_ok = true;
        let mut p = wall_balance::population_momentum(&flow);
        let mut steps = 0;
        let mean =
            |rows: &[[f64; 7]]| -> [f64; 7] { [0, 1, 2, 3, 4, 5, 6].map(|c| rows.iter().map(|r| r[c]).sum::<f64>() / rows.len().max(1) as f64) };
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
            for (s, v) in sums.iter_mut().zip([report.carried[0], report.applied[0], requested[0] / area, a[0] * m, 0.0, 0.0, 0.0]) {
                *s += v;
            }
            if steps % self.sample != 0 {
                continue;
            }
            let n = self.sample as f64;
            let [law, trapezoid] = bulk(&flow);
            let row = [sums[0] / n, sums[1] / n, sums[2] / n, sums[3] / n, law, trapezoid, centre(&flow)];
            sums = [0.0; 7];
            assert!(row.iter().all(|v| v.is_finite()), "the channel is not finite after {steps} steps: {row:?}");
            // Past MAX_BLOCKS the stream refuses more samples; the statistics keep what they hold.
            blocks.push(cf(row[0], row[4], rho)).ok();
            history.push(row);
            planes.push([plane(&flow, 0), plane(&flow, 1), plane(&flow, 2)]);
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
        let tail = &planes[planes.len().saturating_sub(window)..];
        let profile = [0, 1, 2].map(|k| tail.iter().map(|p| p[k]).sum::<f64>() / tail.len().max(1) as f64 / u_tau);
        ChannelResult {
            profile,
            u_tau,
            nu,
            bulk: last[4] / u_tau,
            bulk_trapezoid: last[5] / u_tau,
            centre: last[6] / u_tau,
            carried: cf(last[0], last[4], rho),
            carried_trapezoid: cf(last[0], last[5], rho),
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
        "{name}: {} steps, settled {}; U_b⁺ {:.4} (ref {bulk_reference:.4}, {:+.2} %; trapezoid {:.4}, {:+.2} %); C_f carried {:.5e} (ref {cf_reference:.5e}, {:+.2} %; trapezoid {:+.2} %), applied {:.5e}, requested {}; drag / body force {:.6}; budget closed {}; min ν̃ {:?}; {} blocks, half-width {:.2e}, accepted {}",
        r.steps,
        r.settled,
        r.bulk,
        100.0 * (r.bulk / bulk_reference - 1.0),
        r.bulk_trapezoid,
        100.0 * (r.bulk_trapezoid / bulk_reference - 1.0),
        r.carried,
        100.0 * (r.carried / cf_reference - 1.0),
        100.0 * (r.carried_trapezoid / cf_reference - 1.0),
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
        slabs: false,
        max_steps: 200_000,
        sample: 50,
        settling: GATED,
        sigma: None,
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

/// The wall interval of an on-node wall (common::cases::WallInterval): its closed forms equal a
/// fine midpoint quadrature of Spalding's profile (inverted by bisection at each point) to 1e-9
/// relative over y₁⁺ 1–500. In the viscous sublayer the law is linear, so it equals the trapezoid
/// there. Unusable inputs give none.
#[test]
fn the_wall_interval_integrates_the_law_the_wall_applies() {
    use fieldcraft_lbm::wall::{BOUNCE_B, BOUNCE_KAPPA};
    let (k, e) = (BOUNCE_KAPPA, (-BOUNCE_KAPPA * BOUNCE_B).exp());
    let y_of = |u: f64| u + e * ((k * u).exp() - 1.0 - k * u - (k * u).powi(2) / 2.0 - (k * u).powi(3) / 6.0);
    let u_of = |y: f64| {
        let (mut lo, mut hi) = (0.0, y.max(1.0));
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if y_of(mid) < y { lo = mid } else { hi = mid }
        }
        0.5 * (lo + hi)
    };
    let nu = 1e-3;
    for y_plus in [1.0, 5.0, 11.0, 31.0, 50.0, 83.0, 200.0, 500.0] {
        let u_tau = 0.05;
        let y1 = y_plus * nu / u_tau;
        let u1 = u_tau * u_of(y_plus);
        let w = WallInterval::new(u1, y1, nu).unwrap();
        assert!((w.u_tau / u_tau - 1.0).abs() < 1e-12, "u_τ {} vs {u_tau}", w.u_tau);
        let n = 20_000;
        let h = y_plus / n as f64;
        let (mut i1, mut i2) = (0.0, 0.0);
        for j in 0..n {
            let u = u_of((j as f64 + 0.5) * h);
            i1 += u * h;
            i2 += u * u * h;
        }
        let want = [nu * i1, nu * u_tau * i2];
        for c in 0..2 {
            assert!((w.law[c] / want[c] - 1.0).abs() < 1e-7, "y⁺ {y_plus}: {:?} vs {want:?}", w.law);
        }
        println!("y⁺₁ {y_plus}: ∫u dy law / trapezoid {:.5}", w.law[0] / w.trapezoid[0]);
        if y_plus <= 1.0 {
            assert!((w.law[0] / w.trapezoid[0] - 1.0).abs() < 1e-3, "the sublayer is linear");
        }
    }
    for (u1, y1, nu) in [(0.0, 1.0, 1e-3), (f64::NAN, 1.0, 1e-3), (1.0, -1.0, 1e-3), (1.0, 1.0, 0.0), (1.0, f64::INFINITY, 1e-3)] {
        assert!(WallInterval::new(u1, y1, nu).is_none(), "{u1} {y1} {nu}");
    }
}

/// The reference numbers are Lee & Moser's header values (u_τ = 0.0500256 at U_b = 1).
#[test]
fn the_lee_moser_reference_is_their_published_bulk_and_friction_velocity() {
    assert!((LEE_MOSER_1000_BULK - 19.9898).abs() < 1e-4, "{LEE_MOSER_1000_BULK}");
    assert!((LEE_MOSER_1000_CF - 5.00512e-3).abs() < 1e-8, "{LEE_MOSER_1000_CF}");
    assert!((LEE_MOSER_180_BULK - 15.6910).abs() < 1e-4, "{LEE_MOSER_180_BULK}");
    assert!((LEE_MOSER_180_CF - 8.12326e-3).abs() < 1e-8, "{LEE_MOSER_180_CF}");
    assert!((LEE_MOSER_180_RE_TAU * 3.5e-4 - 0.0637309).abs() < 1e-6, "Re_τ ν = u_τ δ");
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
    for (wall, on_node, cells) in
        [(WallModel::SpaldingBounce, true, 20), (WallModel::SpalartAllmarasBounce, true, 20), (WallModel::Musker, false, 10)]
    {
        let ch = Channel {
            re_tau: 1000.0,
            cells,
            bulk_guess: LEE_MOSER_1000_BULK,
            lattice_bulk: 0.1,
            collision: Collision::Mrt,
            turbulence: Some(sa(wall)),
            on_node,
            slabs: false,
            max_steps: 400,
            sample: 100,
            settling: GATED,
            sigma: None,
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

/// One run of the C7 pair at Re_τ 1000: SA on MRT under `wall`, `cells` per δ.
fn c7(wall: WallModel, on_node: bool, cells: usize, max_steps: u64) -> ChannelResult {
    c7_on(wall, on_node, false, cells, max_steps)
}

/// [`c7`], with the walls as obstacle slabs when `slabs`.
fn c7_on(wall: WallModel, on_node: bool, slabs: bool, cells: usize, max_steps: u64) -> ChannelResult {
    c7_with(Collision::Mrt, None, wall, on_node, slabs, cells, max_steps)
}

/// [`c7_on`] with the collision `collision` (and the hybrid collision's fixed σ).
fn c7_with(collision: Collision, sigma: Option<f64>, wall: WallModel, on_node: bool, slabs: bool, cells: usize, max_steps: u64) -> ChannelResult {
    Channel {
        re_tau: 1000.0,
        cells,
        bulk_guess: LEE_MOSER_1000_BULK,
        lattice_bulk: 0.1,
        collision,
        sigma,
        turbulence: Some(sa(wall)),
        on_node,
        slabs,
        max_steps,
        sample: 100,
        settling: GATED,
    }
    .run()
}

/// Board task C7: the Spalart–Allmaras RANS channel at Re_τ 1000 on MRT, on Han, Ooka &
/// Kikumoto's on-node wall (Spalding's law, primary) and on Musker's halfway wall (control), each on
/// two grids with the first fluid node at y⁺ ≈ 50 and 31. Gates, on the primary: carried C_f within
/// 5 % of Lee & Moser's 5.005 × 10⁻³, the wall function's request within 1 % of the carried stress.
/// Every run: settled under [`GATED`] with the drag balancing the body force, ν̃ ≥ 0 (so ν_t ≥ 0).
///
/// Measured when written (MRT, 4 threads, 35 s for all four):
///
/// | wall | cells per δ (y⁺₁) | steps | U_b⁺ | C_f carried | requested / carried |
/// |---|---|---|---|---|---|
/// | Han on-node | 20 (50) | 93 700 | 19.79 (−1.0 %) | 5.104e-3 (+2.0 %) | 1.00001 |
/// | Han on-node | 32 (31) | 218 900 | 19.78 (−1.0 %) | 5.110e-3 (+2.1 %) | 1.00000 |
///
/// Those Han U_b⁺ took the interval between each on-node wall and its first node as a trapezoid
/// from u = 0. The gate now integrates it from the wall law
/// (common::cases::WallInterval) and reports the trapezoid beside it. Rerun (FIELDCRAFT_THREADS=4,
/// 42 s with the obstacle pair): Han 20 cells per δ U_b⁺ 19.979 (−0.05 %), C_f 5.011e-3 (+0.11 %;
/// trapezoid +1.99 %); 32 cells per δ U_b⁺ 19.868 (−0.61 %), C_f 5.067e-3 (+1.23 %; trapezoid
/// +2.09 %); the same settling steps. SA's own error against Lee & Moser is +1.43 % (a steady 1D
/// wall-resolved SA channel at Re_τ 1000): the 20-cell figure is nearer DNS than SA itself because the wall
/// law's and the first interval's errors still cancel in part.
/// | Musker halfway | 10 (50) | 137 700 | 22.57 (+12.9 %) | 3.927e-3 (−21.5 %) | 1.383 |
/// | Musker halfway | 16 (31) | 239 300 | 23.83 (+19.2 %) | 3.521e-3 (−29.6 %) | 1.541 |
///
/// The control fails both gates, as Musker's halfway wall does under LES (docs/verification.md,
/// "Requested, applied and carried wall stress"): the stress it requests (Musker's u_τ, 8 % above
/// Lee & Moser here) is not what its reconstruction carries, and the gap grows as the grid is
/// refined. Its budget closes every step, so the gap is in the wall, not the measure. It is
/// recorded here, not gated, until the wall layer's own task takes it up. On Han's on-node wall the
/// carried stress is the request to 10⁻⁵ once settled; its per-step #829 budget is not exact
/// during the start-up (its control volume leaves out the boundary nodes), so it is not asserted.
#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn sa_rans_channel_at_re_tau_1000_meets_lee_and_moser() {
    let runs = [
        (WallModel::SpaldingBounce, true, 20),
        (WallModel::SpaldingBounce, true, 32),
        (WallModel::Musker, false, 10),
        (WallModel::Musker, false, 16),
    ];
    let mut score = Score::default();
    let mut failures = Vec::new();
    for (wall, on_node, cells) in runs {
        let r = c7(wall, on_node, cells, 2_000_000);
        let y_plus = if on_node { 1000.0 / cells as f64 } else { 500.0 / cells as f64 };
        let name = format!("SA {wall:?}, {cells} cells per δ (first node y⁺ {y_plus:.0})");
        report(&name, &r, LEE_MOSER_1000_BULK, LEE_MOSER_1000_CF);
        let ratio = r.requested.unwrap_or(f64::NAN) / r.carried;
        if on_node {
            score.push(Gate::new("C7 channel", &format!("{name}: C_f carried"), r.carried, LEE_MOSER_1000_CF, 0.05));
            score.push(Gate::new("C7 channel", &format!("{name}: requested / carried"), ratio, 1.0, 0.01));
        } else {
            println!(
                "control, not gated: C_f carried {:+.1} % against Lee & Moser, requested / carried {ratio:.3}",
                100.0 * (r.carried / LEE_MOSER_1000_CF - 1.0)
            );
            assert!(r.budget_ok, "{name}: the halfway wall's budget closes every step");
        }
        if !r.settled {
            failures.push(format!("{name}: not settled in {} steps", r.steps));
        }
        if !r.min_nu_tilde.is_some_and(|v| v >= 0.0) {
            failures.push(format!("{name}: ν̃ below 0 ({:?})", r.min_nu_tilde));
        }
    }
    println!("{}", score.table());
    assert!(failures.is_empty(), "{failures:?}");
    assert!(score.passes());
}

/// The SA channel at Lee & Moser's low Reynolds number, Re_τ 182, the range of a pipe at
/// Re_b 5300 (Re_τ 181, `sa_pipe.rs`) and a square duct at Re_τ ≈ 150. It measures the model's own
/// error there, on MRT:
/// - **gated:** SA resolved to the wall (no wall model, halfway walls), first cell centre at
///   y⁺ ≈ 1 and ½: carried C_f within 5 % of Lee & Moser's 8.123 × 10⁻³;
/// - **pinned, not gated:** Han's on-node wall at 10 and 20 cells per δ (first node y⁺ 18 and 9).
///   It over-predicts C_f by 6–7 % at this Reynolds number (+0.1 to +1.2 % at Re_τ 1000, the test
///   above), while its request equals the carried stress (1.00001): the wall carries what it asks
///   for, so the error is the wall-function closure at low Re_τ (Spalding's law at the first node
///   fed by SA's profile). Each run must stay within 0.5 percentage points of its measured error,
///   so a change that moves it, better or worse, fails and comes to review.
///
/// Every run settled under [`GATED`] with ν̃ ≥ 0. Measured when written (MRT, 4 threads, 484 s):
///
/// | wall | cells per δ (first node y⁺) | steps | U_b⁺ (LM 15.691) | C_f carried | requested / carried |
/// |---|---|---|---|---|---|
/// | resolved, halfway | 91 (1.0) | 95 300 | 15.803 (+0.71 %) | −1.41 % | – |
/// | resolved, halfway | 182 (0.5) | 750 100 | 15.889 (+1.26 %) | −2.48 % | – |
/// | Han on-node | 10 (18.2) | 54 400 | 15.166 (−3.35 %) | +7.04 % | 1.00001 |
/// | Han on-node | 20 (9.1) | 115 200 | 15.224 (−2.98 %) | +6.23 % | 1.00001 |
///
/// SA's own error at Re_τ 180 is thus about −2.5 % in C_f, the floor any wall-resolved SA run at
/// this Reynolds number starts from.
#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn sa_rans_channel_at_re_tau_180_meets_lee_and_moser() {
    let resolved = Turbulence::SpalartAllmaras { prandtl: 0.9, wall_model: None, inflow_ratio: 3.0, ddes: None };
    // The wall, on-node or not, cells per δ, and Han's measured C_f error (pinned).
    let han = sa(WallModel::SpaldingBounce);
    let runs = [(resolved, false, 91usize, None), (resolved, false, 182, None), (han, true, 10, Some(0.0704)), (han, true, 20, Some(0.0623))];
    let mut score = Score::default();
    let mut failures = Vec::new();
    for (turbulence, on_node, cells, pinned) in runs {
        let r = Channel {
            re_tau: LEE_MOSER_180_RE_TAU,
            cells,
            bulk_guess: LEE_MOSER_180_BULK,
            lattice_bulk: 0.1,
            collision: Collision::Mrt,
            sigma: None,
            turbulence: Some(turbulence),
            on_node,
            slabs: false,
            max_steps: 2_000_000,
            sample: 100,
            settling: GATED,
        }
        .run();
        let y_plus = if on_node { LEE_MOSER_180_RE_TAU / cells as f64 } else { 0.5 * LEE_MOSER_180_RE_TAU / cells as f64 };
        let wall = if on_node { "Han on-node" } else { "resolved, halfway" };
        let name = format!("SA {wall}, {cells} cells per δ (first node y⁺ {y_plus:.1})");
        report(&name, &r, LEE_MOSER_180_BULK, LEE_MOSER_180_CF);
        let error = r.carried / LEE_MOSER_180_CF - 1.0;
        match pinned {
            None => score.push(Gate::new("SA channel 180", &format!("{name}: C_f carried"), r.carried, LEE_MOSER_180_CF, 0.05)),
            Some(measured) => {
                println!("pinned, not gated: C_f carried {:+.2} % against Lee & Moser (measured {:+.2} %)", 100.0 * error, 100.0 * measured);
                if (error - measured).abs() > 0.005 {
                    failures.push(format!("{name}: C_f error {:+.2} % left its pinned band {:+.2} ± 0.5 %", 100.0 * error, 100.0 * measured));
                }
            }
        }
        if on_node {
            let ratio = r.requested.unwrap_or(f64::NAN) / r.carried;
            score.push(Gate::new("SA channel 180", &format!("{name}: requested / carried"), ratio, 1.0, 0.01));
        } else if !r.budget_ok {
            failures.push(format!("{name}: the halfway wall's budget did not close"));
        }
        if !r.settled {
            failures.push(format!("{name}: not settled in {} steps", r.steps));
        }
        if !r.min_nu_tilde.is_some_and(|v| v >= 0.0) {
            failures.push(format!("{name}: ν̃ below 0 ({:?})", r.min_nu_tilde));
        }
    }
    println!("{}", score.table());
    assert!(failures.is_empty(), "{failures:?}");
    assert!(score.passes());
}

/// The SA channel at Re_τ 180 on the half-way wall (`*_bounce_surface`): the walls are the faces of two
/// obstacle slabs two cells thick, the open rows exactly 2 cells per δ. The law reads the second open
/// cell off each wall (y = 1½; y⁺ 27 and 14 on 10 and 20 cells per δ), and ν̃ is imposed at the first
/// two open cells. U_b⁺ is the mean over the open rows (midpoint). **Pinned, not gated** (each
/// within 0.5 points of its measured error, so a change that moves it comes to review); Han's on-node
/// wall is +7.04 % and +6.23 % at the same cells per δ.
///
/// | law | cells per δ | steps | U_b⁺ (LM 15.691) | C_f carried |
/// |---|---|---|---|---|
/// | Spalding | 10 | 61 000 | 15.034 (−4.18 %) | +8.93 % |
/// | Spalding | 20 | 118 300 | 15.097 (−3.79 %) | +8.02 % |
/// | Spalart–Allmaras | 10 | 55 200 | 15.201 (−3.12 %) | +6.55 % |
/// | Spalart–Allmaras | 20 | 21 400 | 15.796 (+0.67 %) | −1.32 % |
///
/// The same law at the first open cell (y = ½) gave −14.42 % and +5.62 % for the SA law and swung
/// 10–20 points with the grid: the first cell sits in the buffer layer. At Re_τ 1000 (50 and 80 cells
/// per δ) the SA law gives +1.78 % and +0.69 % on MRT, and −1.17 % and −1.70 % on the hybrid collision
/// at σ 0.95 (which at Re_τ 180 gives −1.03 % and −4.92 %). Only the SA law is validated; the
/// Spalding variant is a contrast and 8–9 % high.
#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn sa_rans_channel_at_re_tau_180_on_the_half_way_wall() {
    let mut failures = Vec::new();
    for (wall, cells, pinned) in [
        (WallModel::SpaldingBounceSurface, 10usize, 0.0893),
        (WallModel::SpaldingBounceSurface, 20, 0.0802),
        (WallModel::SpalartAllmarasBounceSurface, 10, 0.0655),
        (WallModel::SpalartAllmarasBounceSurface, 20, -0.0132),
    ] {
        let r = Channel {
            re_tau: LEE_MOSER_180_RE_TAU,
            cells,
            bulk_guess: LEE_MOSER_180_BULK,
            lattice_bulk: 0.1,
            collision: Collision::Mrt,
            sigma: None,
            turbulence: Some(sa(wall)),
            on_node: false,
            slabs: true,
            max_steps: 2_000_000,
            sample: 100,
            settling: GATED,
        }
        .run();
        let name = format!("SA {wall:?}, {cells} cells per δ (first open cell y⁺ {:.1})", 0.5 * LEE_MOSER_180_RE_TAU / cells as f64);
        report(&name, &r, LEE_MOSER_180_BULK, LEE_MOSER_180_CF);
        let error = r.carried / LEE_MOSER_180_CF - 1.0;
        println!("{name}: C_f carried {:+.2} % against Lee & Moser (pinned {:+.2} %)", 100.0 * error, 100.0 * pinned);
        // The mean u⁺ at the first three open cells against the SA law at their own y⁺: the second
        // (the cell the law reads) sits on it by construction, the first is where the model's error is.
        for (k, u) in r.profile.iter().enumerate() {
            let y_plus = (k as f64 + 0.5) * r.u_tau / r.nu;
            let law = fieldcraft_lbm::sa_law::u_plus(y_plus);
            println!("{name}: open cell {} y⁺ {y_plus:.1} u⁺ {u:.2} (SA law {law:.2}, {:+.1} %)", k + 1, 100.0 * (u / law - 1.0));
            if k == 1 && matches!(wall, WallModel::SpalartAllmarasBounceSurface) && (u / law - 1.0).abs() > 0.02 {
                failures.push(format!("{name}: the sampled cell's u⁺ {u:.2} is off the law's {law:.2}"));
            }
        }
        if (error - pinned).abs() > 0.005 {
            failures.push(format!("{name}: C_f error {:+.2} % left its pinned band {:+.2} ± 0.5 %", 100.0 * error, 100.0 * pinned));
        }
        if !r.settled {
            failures.push(format!("{name}: not settled in {} steps", r.steps));
        }
        if !r.min_nu_tilde.is_some_and(|v| v >= 0.0) {
            failures.push(format!("{name}: ν̃ below 0 ({:?})", r.min_nu_tilde));
        }
        let ratio = r.requested.unwrap_or(f64::NAN) / r.carried;
        if (ratio - 1.0).abs() > 0.01 {
            failures.push(format!("{name}: requested / carried {ratio}"));
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}

/// C7's SA channel at Re_τ 1000 on Han's on-node wall, its walls built from obstacle slabs instead
/// of box faces. The
/// gates: carried C_f within 1 % of the box-face run's, requested / carried within 1 %; the two
/// runs are in fact the same run (the same steps, the same carried C_f to the bit), since the
/// slabs' node rows are flat nodes at the box faces' heights (`han_obstacles.rs` checks every
/// population over 200 steps).
#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn sa_rans_channel_on_obstacle_walls_is_the_box_face_channel() {
    let cells = 20;
    let boxed = c7_on(WallModel::SpaldingBounce, true, false, cells, 2_000_000);
    let slabbed = c7_on(WallModel::SpaldingBounce, true, true, cells, 2_000_000);
    report("SA Han on-node, box faces, 20 cells per δ", &boxed, LEE_MOSER_1000_BULK, LEE_MOSER_1000_CF);
    report("SA Han on-node, obstacle slabs, 20 cells per δ", &slabbed, LEE_MOSER_1000_BULK, LEE_MOSER_1000_CF);
    let ratio = slabbed.requested.unwrap_or(f64::NAN) / slabbed.carried;
    let mut score = Score::default();
    score.push(Gate::new("WL1 obstacle channel", "C_f carried, slabs vs box faces", slabbed.carried, boxed.carried, 0.01));
    score.push(Gate::new("WL1 obstacle channel", "requested / carried, slabs", ratio, 1.0, 0.01));
    println!("{}", score.table());
    assert!(boxed.settled && slabbed.settled, "settled: box {} slabs {}", boxed.settled, slabbed.settled);
    assert_eq!((boxed.steps, boxed.carried.to_bits()), (slabbed.steps, slabbed.carried.to_bits()), "the same run");
    assert!(slabbed.min_nu_tilde.is_some_and(|v| v >= 0.0), "{:?}", slabbed.min_nu_tilde);
    assert!(score.passes());
}

/// C7 on the collision 2DZP runs on (board decision after the 2DZP inlet probes): the hybrid
/// collision at fixed σ 0.99, on Han's on-node wall, 20 and 32 cells per δ. The same gates as on
/// MRT: carried C_f within 5 % of Lee & Moser on the wall-law U_b, the request within 1 % of the
/// carried stress, settled, ν̃ ≥ 0.
///
/// Measured when written (4 threads, 46 s for both), wall-law U_b with the trapezoid beside it:
///
/// | cells per δ (y⁺₁) | steps | U_b⁺ | C_f carried | requested / carried |
/// |---|---|---|---|---|
/// | 20 (50) | 128 000 | 19.890 (−0.50 %; trapezoid −1.42 %) | 5.056e-3 (+1.01 %; trapezoid +2.91 %) | 1.00001 |
/// | 32 (31) | 255 200 | 19.734 (−1.28 %; trapezoid −1.70 %) | 5.136e-3 (+2.61 %; trapezoid +3.48 %) | 1.00000 |
///
/// Against MRT (+0.11 % and +1.23 %) the hybrid collision adds about 1 % on the coarse grid and
/// 1.4 % on the fine one, inside the gate on both.
#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn sa_rans_channel_on_the_hybrid_collision_meets_lee_and_moser() {
    let mut score = Score::default();
    let mut failures = Vec::new();
    for cells in [20, 32] {
        let r = c7_with(Collision::Hrr, Some(0.99), WallModel::SpaldingBounce, true, false, cells, 2_000_000);
        let name = format!("SA Han on HRR σ 0.99, {cells} cells per δ (first node y⁺ {:.0})", 1000.0 / cells as f64);
        report(&name, &r, LEE_MOSER_1000_BULK, LEE_MOSER_1000_CF);
        score.push(Gate::new("C7 channel, HRR", &format!("{name}: C_f carried"), r.carried, LEE_MOSER_1000_CF, 0.05));
        score.push(Gate::new("C7 channel, HRR", &format!("{name}: requested / carried"), r.requested.unwrap_or(f64::NAN) / r.carried, 1.0, 0.01));
        if !r.settled {
            failures.push(format!("{name}: not settled in {} steps", r.steps));
        }
        if !r.min_nu_tilde.is_some_and(|v| v >= 0.0) {
            failures.push(format!("{name}: ν̃ below 0 ({:?})", r.min_nu_tilde));
        }
    }
    println!("{}", score.table());
    assert!(failures.is_empty(), "{failures:?}");
    assert!(score.passes());
}

/// The Spalart–Allmaras wall law (`spalart_allmaras_bounce`) on a periodic SA channel at
/// Re_τ 182, 12 and 24 cells per half-height (the first fluid node at y⁺ 15 and 7.6), on MRT and
/// on the hybrid collision at σ 0.99. Spalding's law is biased low in the buffer layer, so the
/// same runs under `spalding_bounce` read the centreline velocity 3 to 5 % off (the finding is in
/// docs/verification.md); the gates are the centreline U⁺ within 2.5 % of Lee & Moser (the
/// window 17.75 to 18.65), carried C_f within 3.5 %, the request within 1 % of the carried
/// stress, settled, ν̃ ≥ 0.
///
/// Measured when written (3 threads, 5 s for the four):
///
/// | collision | cells per δ (y⁺₁) | centre U⁺ (ref 18.2) | C_f carried (ref 8.123e-3) |
/// |---|---|---|---|
/// | MRT | 12 (15.2) | 18.472 (+1.50 %) | 7.959e-3 (−2.02 %) |
/// | MRT | 24 (7.6) | 18.401 (+1.10 %) | 8.005e-3 (−1.45 %) |
/// | HRR σ 0.99 | 12 (15.2) | 18.246 (+0.25 %) | 8.159e-3 (+0.44 %) |
/// | HRR σ 0.99 | 24 (7.6) | 18.102 (−0.54 %) | 8.314e-3 (+2.35 %) |
#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn sa_law_channel_at_re_tau_180_meets_lee_and_moser_at_12_and_24_cells() {
    let mut score = Score::default();
    let mut failures = Vec::new();
    for (collision, sigma) in [(Collision::Mrt, None), (Collision::Hrr, Some(0.99))] {
        for cells in [12, 24] {
            let r = Channel {
                re_tau: LEE_MOSER_180_RE_TAU,
                cells,
                bulk_guess: LEE_MOSER_180_BULK,
                lattice_bulk: 0.1,
                collision,
                sigma,
                turbulence: Some(sa(WallModel::SpalartAllmarasBounce)),
                on_node: true,
                slabs: false,
                max_steps: 2_000_000,
                sample: 100,
                settling: GATED,
            }
            .run();
            let name = format!("SA law, {collision:?}, {cells} cells per δ (first node y⁺ {:.1})", LEE_MOSER_180_RE_TAU / cells as f64);
            report(&name, &r, LEE_MOSER_180_BULK, LEE_MOSER_180_CF);
            println!("{name}: centre U⁺ {:.3} ({:+.2} % of {LEE_MOSER_180_CENTRE})", r.centre, 100.0 * (r.centre / LEE_MOSER_180_CENTRE - 1.0));
            score.push(Gate::new("SA law channel 180", &format!("{name}: centre U⁺"), r.centre, LEE_MOSER_180_CENTRE, 0.025));
            score.push(Gate::new("SA law channel 180", &format!("{name}: C_f carried"), r.carried, LEE_MOSER_180_CF, 0.035));
            score.push(Gate::new(
                "SA law channel 180",
                &format!("{name}: requested / carried"),
                r.requested.unwrap_or(f64::NAN) / r.carried,
                1.0,
                0.01,
            ));
            if !r.settled {
                failures.push(format!("{name}: not settled in {} steps", r.steps));
            }
            if !r.min_nu_tilde.is_some_and(|v| v >= 0.0) {
                failures.push(format!("{name}: ν̃ below 0 ({:?})", r.min_nu_tilde));
            }
        }
    }
    println!("{}", score.table());
    assert!(failures.is_empty(), "{failures:?}");
    assert!(score.passes());
}

/// C7 re-baselined with the Spalart–Allmaras wall law: the channel at Re_τ 1000 on MRT, 20 and 32
/// cells per δ (the first node at y⁺ 50 and 31), the same gates as the Spalding-law C7 above.
///
/// Measured when written (3 threads, 10 s for both): 20 cells per δ, 136 800 steps, U_b⁺ 19.838
/// (−0.76 %), C_f 5.082e-3 (+1.54 %), requested / carried 1.00001; 32 cells per δ, 195 000 steps,
/// U_b⁺ 19.929 (−0.30 %), C_f 5.036e-3 (+0.61 %), requested / carried 1.00000. Against the
/// Spalding law's +0.11 % and +1.23 % the new law is the nearer on the fine grid and 1.4 % further
/// on the coarse one, both inside the gate; the law's gain is at the buffer-layer first nodes
/// (Re_τ 182), not here.
#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn sa_law_channel_at_re_tau_1000_meets_lee_and_moser() {
    let mut score = Score::default();
    let mut failures = Vec::new();
    for cells in [20, 32] {
        let r = c7(WallModel::SpalartAllmarasBounce, true, cells, 2_000_000);
        let name = format!("SA law on MRT, {cells} cells per δ (first node y⁺ {:.0})", 1000.0 / cells as f64);
        report(&name, &r, LEE_MOSER_1000_BULK, LEE_MOSER_1000_CF);
        score.push(Gate::new("C7 channel, SA law", &format!("{name}: C_f carried"), r.carried, LEE_MOSER_1000_CF, 0.05));
        score.push(Gate::new("C7 channel, SA law", &format!("{name}: requested / carried"), r.requested.unwrap_or(f64::NAN) / r.carried, 1.0, 0.01));
        if !r.settled {
            failures.push(format!("{name}: not settled in {} steps", r.steps));
        }
        if !r.min_nu_tilde.is_some_and(|v| v >= 0.0) {
            failures.push(format!("{name}: ν̃ below 0 ({:?})", r.min_nu_tilde));
        }
    }
    println!("{}", score.table());
    assert!(failures.is_empty(), "{failures:?}");
    assert!(score.passes());
}

/// Lee & Moser's Re_τ 543.496 channel (header: ν = 1e-4, u_τ = 5.43496e-2, Re_b = 2 δ U_b / ν =
/// 20 000, so U_b = 1 in units of u_τ / 0.0543496 and C_f = 2 (u_τ / U_b)²); the centreline U⁺ of
/// their mean profile is 21.00. Not used to choose the wall law or its constants.
const LEE_MOSER_550_RE_TAU: f64 = 543.496;
const LEE_MOSER_550_BULK: f64 = 1.0 / 0.0543496;
const LEE_MOSER_550_CF: f64 = 2.0 * 0.0543496 * 0.0543496;
const LEE_MOSER_550_CENTRE: f64 = 21.0;

/// A check of the Spalart–Allmaras wall law at a third Reynolds number, Re_τ 543, one the law was
/// not built or tuned on (Re_τ 182 and 1000 were) but run by the session that built it, so a report
/// and not a validation: MRT and the hybrid collision at σ 0.99, 16 and
/// 24 cells per δ (the first node at y⁺ 34 and 23). Gates: centreline U⁺ within 2.5 % of Lee &
/// Moser, carried C_f within 5 %, the request within 1 % of the carried stress.
///
/// Run once, as written (3 threads, 7 s), no change made after it:
///
/// | collision | cells per δ (y⁺₁) | centre U⁺ (ref 21.0) | C_f carried (ref 5.908e-3) |
/// |---|---|---|---|
/// | MRT | 16 (34) | 20.657 (−1.63 %) | 5.959e-3 (+0.86 %) |
/// | MRT | 24 (23) | 20.801 (−0.95 %) | 5.862e-3 (−0.77 %) |
/// | HRR σ 0.99 | 16 (34) | 20.603 (−1.89 %) | 5.992e-3 (+1.43 %) |
/// | HRR σ 0.99 | 24 (23) | 20.735 (−1.26 %) | 5.911e-3 (+0.06 %) |
#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn sa_law_channel_at_re_tau_550_meets_lee_and_moser() {
    let mut score = Score::default();
    let mut failures = Vec::new();
    for (collision, sigma) in [(Collision::Mrt, None), (Collision::Hrr, Some(0.99))] {
        for cells in [16, 24] {
            let r = Channel {
                re_tau: LEE_MOSER_550_RE_TAU,
                cells,
                bulk_guess: LEE_MOSER_550_BULK,
                lattice_bulk: 0.1,
                collision,
                sigma,
                turbulence: Some(sa(WallModel::SpalartAllmarasBounce)),
                on_node: true,
                slabs: false,
                max_steps: 2_000_000,
                sample: 100,
                settling: GATED,
            }
            .run();
            let name = format!("SA law, {collision:?}, {cells} cells per δ (first node y⁺ {:.0})", LEE_MOSER_550_RE_TAU / cells as f64);
            report(&name, &r, LEE_MOSER_550_BULK, LEE_MOSER_550_CF);
            println!("{name}: centre U⁺ {:.3} ({:+.2} % of {LEE_MOSER_550_CENTRE})", r.centre, 100.0 * (r.centre / LEE_MOSER_550_CENTRE - 1.0));
            score.push(Gate::new("SA law channel 550", &format!("{name}: centre U⁺"), r.centre, LEE_MOSER_550_CENTRE, 0.025));
            score.push(Gate::new("SA law channel 550", &format!("{name}: C_f carried"), r.carried, LEE_MOSER_550_CF, 0.05));
            score.push(Gate::new(
                "SA law channel 550",
                &format!("{name}: requested / carried"),
                r.requested.unwrap_or(f64::NAN) / r.carried,
                1.0,
                0.01,
            ));
            if !r.settled {
                failures.push(format!("{name}: not settled in {} steps", r.steps));
            }
            if !r.min_nu_tilde.is_some_and(|v| v >= 0.0) {
                failures.push(format!("{name}: ν̃ below 0 ({:?})", r.min_nu_tilde));
            }
        }
    }
    println!("{}", score.table());
    assert!(failures.is_empty(), "{failures:?}");
    assert!(score.passes());
}

/// The held-out Re_τ 543.496 channel (Lee & Moser) on the half-way wall, SA law, the hybrid collision at
/// σ 0.95 (the duct's), 20 cells per δ (the first open cell at y⁺ 13.6, the sampled second at 40.8). Run
/// once, with nothing changed after it; gate: carried C_f within 3 %, request = carried within 1 %.
#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn sa_rans_channel_at_re_tau_550_on_the_half_way_wall() {
    let r = Channel {
        re_tau: LEE_MOSER_550_RE_TAU,
        cells: 20,
        bulk_guess: LEE_MOSER_550_BULK,
        lattice_bulk: 0.1,
        collision: Collision::Hrr,
        sigma: Some(0.95),
        turbulence: Some(sa(WallModel::SpalartAllmarasBounceSurface)),
        on_node: false,
        slabs: true,
        max_steps: 3_000_000,
        sample: 100,
        settling: GATED,
    }
    .run();
    report("held-out Re_τ 550, half-way wall, HRR σ 0.95, 20 cells per δ", &r, LEE_MOSER_550_BULK, LEE_MOSER_550_CF);
    let error = r.carried / LEE_MOSER_550_CF - 1.0;
    println!("Re_τ 550: C_f carried {:+.2} % against Lee & Moser; centre U⁺ {:.2} (ref {LEE_MOSER_550_CENTRE})", 100.0 * error, r.centre);
    assert!(r.settled, "not settled in {} steps", r.steps);
    assert!(error.abs() < 0.03, "C_f error {:+.2} %", 100.0 * error);
    assert!((r.requested.unwrap_or(f64::NAN) / r.carried - 1.0).abs() < 0.01);
}
