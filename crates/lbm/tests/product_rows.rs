//! Print-only rows: the aligned channel (8.00 and 16.00 cells wide) and the square duct (8 and 16 cells) under a chosen
//! collision and relaxation time, on interpolated walls with the mass correction off, against the exact flow rate. No frozen
//! values and no assertions: the lines are read, not gated.
//!
//! Environment: `CASE` (`channel` or `duct`; both when unset), `CELLS` (comma list of widths, default 8,16), `COLLISION` = `hrr` (default) or `mrt`; `SIGMA` (default 0.95) is the hybrid collision's sigma (1.0 may be refused by the setup: the row then prints REFUSED); `TAU_MINUS_HALF` (default 5e-3: lattice viscosity is a
//! third of it); `SCAN_STEPS` caps a row's steps (default four times its minimum). A row runs at least 1.5 W² / (tau - 1/2)
//! steps (W the width in cells) and then until the flow rate changes by under 0.05 % over the last third of the run, or the cap.
//!
//! `COLLISION=hrr TAU_MINUS_HALF=5e-3 cargo test --release -p fieldcraft-lbm --test product_rows -- --ignored --nocapture product_rows`
//!
//! One line per row: `PRODUCT collision, tau-1/2, case, cells, steps, flow-rate error, drift of the last third, mean-density
//! drift per 10 000 steps, largest nu_t/nu`.

mod common;

use common::*;
use fieldcraft_lbm::backend::Backend;
use fieldcraft_lbm::{Limits, Obstacle, Setup};
use fieldcraft_model::{Collision, FaceCondition, HrrSigma, Procedure, WallTreatment};

/// The exact mean speed in lattice units: the force, and the reference the flow rate is read against, follow it (`U_LAT`, default 0.02).
fn u_lat() -> f64 {
    env("U_LAT", 0.02)
}
const PERIODIC: FaceCondition = FaceCondition::Periodic {};

fn env(name: &str, default: f64) -> f64 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// The plane channel of `w` cells (walls on cell faces, as the width sweep's 8.00 and 16.00) or the square duct of side `w` cells.
fn build(duct: bool, w: usize, nu_lat: f64, mrt: bool, hrr_sigma: f64) -> Result<(Setup, Vec<Obstacle>, [f64; 3], f64, f64), String> {
    let dx = 1.0 / w as f64;
    let dt = nu_lat * dx * dx;
    let wf = w as f64;
    let (obstacles, lo, hi, span) = if duct {
        let (ho, span) = (wf * dx, 4.0 * dx);
        let (lo, hi) = (-2.0 * dx, ho + 2.0 * dx);
        let big = 1.0;
        let o = vec![
            obstacle("Floor", cuboid([-big, lo, -big], [span + big, 0.0, ho + big]), &CUBOID_FACES),
            obstacle("Roof", cuboid([-big, ho, -big], [span + big, hi, ho + big]), &CUBOID_FACES),
            obstacle("Left", cuboid([-big, -big, lo], [span + big, ho + big, 0.0]), &CUBOID_FACES),
            obstacle("Right", cuboid([-big, -big, ho], [span + big, ho + big, hi]), &CUBOID_FACES),
        ];
        (o, [0.0, lo, lo], [span, hi, hi], span)
    } else {
        let ny = (wf + 4.0).ceil() as usize;
        let (lx, ly, lz) = (8.0 * dx, ny as f64 * dx, 4.0 * dx);
        let ho = wf * dx;
        let o = vec![obstacle("Wall0", cuboid([-3.0 * lx, ho, -dx], [3.0 * lx, ly, lz + dx]), &CUBOID_FACES)];
        (o, [0.0, 0.0, 0.0], [lx, ly, lz], lx)
    };
    let _ = span;
    let mut d = domain(lo, hi, dx);
    for f in [&mut d.faces.x_min, &mut d.faces.x_max, &mut d.faces.y_min, &mut d.faces.y_max, &mut d.faces.z_min, &mut d.faces.z_max] {
        *f = PERIODIC;
    }
    d.obstacles = obstacles.iter().map(|o| o.body.clone()).collect();
    // the body force that gives the exact mean speed U_LAT (plane channel: 12 nu U / H^2; duct: U nu / (c H^2), c from the series)
    let g_lat = if duct { u_lat() * nu_lat / (duct_coefficient() * wf * wf) } else { 12.0 * nu_lat * u_lat() / (wf * wf) };
    d.acceleration = [g_lat * dx / (dt * dt), 0.0, 0.0];
    let mut p = with_walls(collide(1e6 * dt, dt, None, if mrt { Collision::Mrt } else { Collision::Hrr }), WallTreatment::Interpolated);
    if let Procedure::Fluid { sigma, .. } = &mut p
        && !mrt
    {
        *sigma = Some(HrrSigma::Value(hrr_sigma));
    }
    let m = model(1.0, 1.0, d, p);
    let setup = Setup::new(&m, None, &obstacles, &Limits::default()).map_err(|e| format!("{e}"))?;
    // the flow rate reference per column of cells along x: the channel's width times the mean speed, or the duct's area times it
    let (along, reference) = if duct { (hi_cells(hi[0], dx) * 1.0, u_lat() * wf * wf) } else { (hi_cells(hi[0], dx) * 4.0, u_lat() * wf) };
    Ok((setup, obstacles, [0.0; 3], along, reference))
}

fn hi_cells(len: f64, dx: f64) -> f64 {
    (len / dx).round()
}

/// The flow rate coefficient of the square duct, per H^4 at body force 1 and viscosity 1 (Boussinesq's series).
fn duct_coefficient() -> f64 {
    let pi = std::f64::consts::PI;
    let sum: f64 = (0..200).map(|m| 2 * m + 1).map(|n| ((n as f64) * pi / 2.0).tanh() / (n as f64).powi(5)).sum();
    (1.0 - 192.0 / pi.powi(5) * sum) / 12.0
}

/// The first chunk (its end step) whose change in the flow rate exceeds 10 times the median change of its neighbours (up to
/// two chunks each side), with a floor of 1e-6 so that a noiseless stretch does not flag rounding.
fn first_jump(hist: &[(u64, f64)]) -> Option<u64> {
    let d: Vec<f64> = hist.windows(2).map(|w| (w[1].1 - w[0].1).abs()).collect();
    (0..d.len()).find_map(|i| {
        let mut nb: Vec<f64> = (i.saturating_sub(2)..(i + 3).min(d.len())).filter(|&j| j != i).map(|j| d[j]).collect();
        nb.sort_by(f64::total_cmp);
        let med = nb.get(nb.len() / 2).copied().unwrap_or(0.0);
        (d[i] > 1e-6 && d[i] > 10.0 * med).then(|| hist[i + 1].0)
    })
}

/// A second jump flag, looking backwards only: the first chunk whose change exceeds 5 times the median of the previous five
/// changes (at least three before it; floor 1e-6). The next chunk's movement cannot hide it. An extra tag, not part of the verdict.
fn first_jump_back(hist: &[(u64, f64)]) -> Option<u64> {
    let d: Vec<f64> = hist.windows(2).map(|w| (w[1].1 - w[0].1).abs()).collect();
    (3..d.len()).find_map(|i| {
        let mut prev: Vec<f64> = d[i.saturating_sub(5)..i].to_vec();
        prev.sort_by(f64::total_cmp);
        let med = prev.get(prev.len() / 2).copied().unwrap_or(0.0);
        (d[i] > 1e-6 && d[i] > 5.0 * med).then(|| hist[i + 1].0)
    })
}

/// max - min of the flow rate over the history points after `from` (the last third of the run).
fn range_after(hist: &[(u64, f64)], from: u64) -> f64 {
    let v: Vec<f64> = hist.iter().filter(|(s, _)| *s > from).map(|h| h.1).collect();
    v.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - v.iter().cloned().fold(f64::INFINITY, f64::min)
}

fn flow_rate(flow: &mut fieldcraft_lbm::Flow, fluid: &[bool], along: f64, reference: f64) -> (f64, f64) {
    let f = Backend::fields(flow).expect("fields");
    let sum: f64 = (0..fluid.len()).filter(|&c| fluid[c]).map(|c| f.velocity[c][0]).sum();
    let (s, k) = f.density.iter().zip(fluid).filter(|(_, fl)| **fl).fold((0.0, 0usize), |(s, k), (r, _)| (s + r, k + 1));
    (sum / along / reference - 1.0, s / k.max(1) as f64)
}

#[test]
#[ignore = "print-only: reads COLLISION, TAU_MINUS_HALF and SCAN_STEPS"]
fn product_rows() {
    let mrt = std::env::var("COLLISION").is_ok_and(|v| v == "mrt");
    let tau_mh = env("TAU_MINUS_HALF", 5e-3);
    let sigma = env("SIGMA", 0.95);
    let label = if mrt { "mrt".to_string() } else { format!("hrr sigma {sigma}") };
    let nu_lat = tau_mh / 3.0;
    // CASE: `channel`, `duct` or unset (both); CELLS: a comma list of widths in cells (default 8,16).
    let only = std::env::var("CASE").ok();
    let widths: Vec<usize> = std::env::var("CELLS")
        .ok()
        .map(|v| v.split(',').filter_map(|t| t.trim().parse().ok()).filter(|&w| (2..=128).contains(&w)).collect())
        .filter(|v: &Vec<usize>| !v.is_empty())
        .unwrap_or_else(|| vec![8, 16]);
    for (name, duct) in [("channel", false), ("duct", true)] {
        if only.as_deref().is_some_and(|c| c != name) {
            continue;
        }
        for &w in &widths {
            let (setup, obstacles, _, along, reference) = match build(duct, w, nu_lat, mrt, sigma) {
                Ok(v) => v,
                Err(e) => {
                    println!("PRODUCT collision {label} tau-1/2 {tau_mh:e} case {name} cells {w}: REFUSED by the setup: {e}");
                    continue;
                }
            };
            let mut flow = fieldcraft_lbm::Flow::new(&setup, &obstacles).expect("flow");
            let n = flow.grid().len();
            let fluid: Vec<bool> = (0..n).map(|c| Backend::is_fluid(&flow, c)).collect();
            let min_steps = (1.5 * (w * w) as f64 / tau_mh).ceil() as u64;
            let cap = env("SCAN_STEPS", 4.0 * min_steps as f64) as u64;
            let chunk = (min_steps / 40).max(250);
            let (_, rho0) = flow_rate(&mut flow, &fluid, along, reference);
            let mut hist: Vec<(u64, f64)> = Vec::new();
            let (mut done, mut max_ratio) = (0u64, 0.0f64);
            let mut settled = false;
            let mut drift_last_third = f64::NAN;
            let mut rho = rho0;
            while done < cap {
                let chunk = chunk.min(cap - done);
                if Backend::advance(&mut flow, chunk).is_err() {
                    println!("PRODUCT collision {label} tau-1/2 {tau_mh:e} case {name} cells {w}: STOPPED by the backend at step {done}");
                    break;
                }
                done += chunk;
                max_ratio = max_ratio.max(flow.eddy_viscosity_ratio());
                let (rate, r) = flow_rate(&mut flow, &fluid, along, reference);
                rho = r;
                hist.push((done, rate));
                if !rate.is_finite() {
                    break;
                }
                if done >= min_steps {
                    let t = done * 2 / 3;
                    if let Some(&(_, back)) = hist.iter().rev().find(|(s, _)| *s <= t) {
                        drift_last_third = (rate - back).abs() / (1.0 + rate).abs();
                        let jump_in_third = first_jump(&hist).filter(|j| *j > t);
                        if drift_last_third < 5e-4 && range_after(&hist, t) < 5e-4 && jump_in_third.is_none() {
                            settled = true;
                            break;
                        }
                    }
                }
            }
            // the history, at most 40 points: step:flow-rate error (%)
            let every = (hist.len() / 40).max(1);
            let series: Vec<String> = hist.iter().step_by(every).map(|(st, r)| format!("{st}:{:+.4}", 100.0 * r)).collect();
            println!("PRODUCT-HISTORY collision {label} tau-1/2 {tau_mh:e} case {name} cells {w}: {}", series.join(" "));
            let last = hist.last().map_or(f64::NAN, |h| h.1);
            let third = hist.iter().rev().find(|(s, _)| *s <= done * 2 / 3).map_or(f64::NAN, |h| h.1);
            let last_jump = first_jump(&hist).filter(|j| *j > done * 2 / 3);
            let status = if settled {
                "settled: under 0.05 %"
            } else if done < min_steps {
                "below the minimum steps"
            } else if last_jump.is_some() {
                "NOT settled (jump)"
            } else if range_after(&hist, done * 2 / 3) >= 5e-4 {
                "NOT settled (oscillating)"
            } else {
                "NOT settled"
            };
            let jumps = first_jump(&hist).map_or(String::new(), |j| format!(" [first JUMP at step {j}]"));
            let jumps = jumps + &first_jump_back(&hist).map_or(String::new(), |j| format!(" [first JUMP-BACK (5x the previous five) at step {j}]"));
            let range = 100.0 * range_after(&hist, done * 2 / 3);
            println!(
                "PRODUCT collision {label} tau-1/2 {tau_mh:e} U_LAT {} case {name} cells {w} steps {done} (minimum {min_steps}, cap {cap}): flow-rate error {:+.4} % (at 2/3 of the run {:+.4} %), change over the last third {:.4} %, max-min over it {range:.4} % ({status}){jumps}, range under the 1 % bar: {}, mean-density drift {:+.3e} per 10000 steps, largest nu_t/nu {max_ratio:.3e}{}",
                u_lat(),
                100.0 * last,
                100.0 * third,
                100.0 * drift_last_third,
                if range < 1.0 { "yes" } else { "NO" },
                (rho - rho0) / done.max(1) as f64 * 1e4,
                if max_ratio > 1e-3 { "  VOID: the model is not idle" } else { "" }
            );
        }
    }
}
