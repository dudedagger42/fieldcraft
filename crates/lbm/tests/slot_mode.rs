//! A plate with a slot, driven through, for the wall layer's odd-even mode (the bed is common/slot_bed.rs): the flow is held at the
//! slot's mean speed `U_LAT` by a feedback on the body force. The study reads, after the run, how much the flow alternates cell to cell
//! along x downstream of the plate (the second difference of the wall row's u_x and of the slot's flux over the slot's mean speed) and
//! the wall rows' smallest u_x inside the plate.
//!
//! The mode: with σ = 1 alone in the wall layer (pure regularisation there) a slot of 8 cells or fewer carries a standing cell-to-cell
//! mode along x (6 cells: 73 times the alternation of the plain hybrid collision; 8 cells: 8 times; absent from 12 cells). The rule
//! damps the layer's density mode too ([`fieldcraft_lbm::WALL_LAYER_RULE_BULK`]) and the mode is gone; `a_slot_of_6_and_8_cells_carries_no_wall_layer_mode`
//! gates that.
//!
//! Study environment: `SLOT` (default 8 cells), `TAU_MINUS_HALF` (default 2.4e-3·SLOT/8), `U_LAT` (default 0.047: Re 470 at 8 cells),
//! `SIGMA` (default 0.95), `STEPS` (default 8,000), `WALL_SIGMA1`, `WALL_MRT`, `WALL_CORNER`, `WALL_BULK`, `RULE_BULK` (the study knobs).
//!
//! `cargo test --release -p fieldcraft-lbm --test slot_mode -- --ignored --nocapture slot_mode`

mod common;

use common::slot_bed::{LX, PLATE, THICK, build};
use fieldcraft_lbm::Flow;
use fieldcraft_lbm::backend::Backend;

fn env(name: &str, default: f64) -> f64 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// The study knobs of a run (`None`: the flow's own).
#[derive(Clone, Copy, Default)]
struct Knobs {
    wall_sigma1: Option<u8>,
    wall_mrt: bool,
    wall_corner: bool,
    wall_bulk: f64,
}

/// What a run reads: `[wall-row alternation, flux alternation, smallest wall-row u_x inside the plate, largest |u|]`, over U.
fn run(slot: usize, tau: f64, u_lat: f64, sigma: f64, steps: u64, knobs: Knobs) -> Result<[f64; 4], String> {
    let bed = build(slot, tau, u_lat, sigma)?;
    let mut flow = Flow::new(&bed.setup, &bed.obstacles).map_err(|e| e.to_string())?;
    if let Some(mode) = knobs.wall_sigma1 {
        flow.set_wall_layer_sigma_one(mode);
    }
    flow.set_wall_layer_mrt(knobs.wall_mrt);
    flow.set_wall_corner_relax(knobs.wall_corner);
    flow.set_wall_layer_bulk(knobs.wall_bulk);
    if let Some(share) = std::env::var("RULE_BULK").ok().and_then(|v| v.parse().ok()) {
        flow.set_wall_rule_bulk(share);
    }
    let grid = *flow.grid();
    let u_at = |flow: &Flow, x: usize, y: usize| flow.moments(grid.index([x, y, 0])).map_or(f64::NAN, |(_, u)| u[0]);
    let rows = slot..2 * slot;
    let mut a = bed.g_lat;
    let mut done = 0;
    while done < steps {
        Backend::advance(&mut flow, 100).map_err(|e| format!("stop at step {done}: {e}"))?;
        done += 100;
        let mean = rows.clone().map(|y| u_at(&flow, PLATE, y)).sum::<f64>() / slot as f64;
        if !mean.is_finite() {
            return Err(format!("not finite at step {done}"));
        }
        // hold the slot's mean speed: a bounded correction of the force each 100 steps
        a *= (1.0 + 0.5 * (u_lat / mean.max(1e-3) - 1.0)).clamp(0.8, 1.25);
        Backend::set_acceleration(&mut flow, [a, 0.0, 0.0]).map_err(|e| e.to_string())?;
    }
    let mean = rows.clone().map(|y| u_at(&flow, PLATE, y)).sum::<f64>() / slot as f64;
    let wall_row = |x: usize| u_at(&flow, x, slot);
    let flux = |x: usize| rows.clone().map(|y| u_at(&flow, x, y)).sum::<f64>();
    let xs = PLATE + THICK + 1..LX - 1;
    let alt = |f: &dyn Fn(usize) -> f64, norm: f64| xs.clone().map(|x| (f(x + 1) - 2.0 * f(x) + f(x - 1)).abs()).fold(0.0, f64::max) / norm;
    let inside = (PLATE..PLATE + THICK).map(|x| wall_row(x).min(u_at(&flow, x, 2 * slot - 1))).fold(f64::INFINITY, f64::min);
    let peak = (0..grid.len()).filter_map(|c| flow.moments(c)).map(|(_, u)| u[0].abs().max(u[1].abs())).fold(0.0, f64::max);
    Ok([alt(&wall_row, mean), alt(&flux, mean * slot as f64), inside / mean, peak / mean])
}

/// The default wall layer (the σ = 1 rule with its density-mode damping) keeps the alternation of the plain hybrid collision on slots of
/// 6 and 8 cells, where the rule without the damping had 73 and 8 times as much.
#[test]
fn a_slot_of_6_and_8_cells_carries_no_wall_layer_mode() {
    for slot in [6usize, 8] {
        let tau = 2.4e-3 * slot as f64 / 8.0;
        let off = run(slot, tau, 0.047, 0.95, 6000, Knobs { wall_sigma1: Some(0), ..Knobs::default() });
        let rule = run(slot, tau, 0.047, 0.95, 6000, Knobs::default());
        let (off, rule) = (off.unwrap_or_else(|e| panic!("slot {slot}, rule off: {e}")), rule.unwrap_or_else(|e| panic!("slot {slot}: {e}")));
        for k in 0..2 {
            assert!(rule[k] < 2.0 * off[k] + 1e-4, "slot {slot}: alternation {k} {:.3e} with the rule, {:.3e} without (off)", rule[k], off[k]);
        }
    }
}

#[test]
#[ignore = "print-only study: see the module docs"]
fn slot_mode() {
    let slot = env("SLOT", 8.0) as usize;
    let (tau, u_lat, sigma, steps) = (env("TAU_MINUS_HALF", 2.4e-3 * slot as f64 / 8.0), env("U_LAT", 0.047), env("SIGMA", 0.95), env("STEPS", 8000.0) as u64);
    let sigma_mode = env("WALL_SIGMA1", f64::NAN);
    let knobs = Knobs {
        wall_sigma1: sigma_mode.is_finite().then_some(sigma_mode as u8),
        wall_mrt: env("WALL_MRT", 0.0) != 0.0,
        wall_corner: env("WALL_CORNER", 0.0) != 0.0,
        wall_bulk: env("WALL_BULK", 0.0),
    };
    let label = format!(
        "SLOT {slot} tau-1/2 {tau:e} u {u_lat} sigma {sigma} WALL_SIGMA1 {} WALL_MRT {} WALL_CORNER {} WALL_BULK {}",
        std::env::var("WALL_SIGMA1").unwrap_or_else(|_| "default".into()),
        env("WALL_MRT", 0.0),
        env("WALL_CORNER", 0.0),
        env("WALL_BULK", 0.0)
    );
    match run(slot, tau, u_lat, sigma, steps, knobs) {
        Ok([wall, flux, inside, peak]) => println!("SLOT-MODE {label}: wall-row alternation {wall:.4e} flux alternation {flux:.4e} smallest wall u inside the plate/U {inside:+.4} largest |u|/U {peak:.3}"),
        Err(e) => println!("SLOT-MODE {label}: {e}"),
    }
}
