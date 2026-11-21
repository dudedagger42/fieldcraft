//! The rounded slot with one ratio-2 strip against a uniform fine run. CPU only, local (ignored).
//!
//! A plate 0.2 m thick across a channel (x 0 to 2.5 m, z 0 to 1 m, four cells periodic over y), a slot 0.2 m wide in it (z 0.4 to
//! 0.6) whose slot-side corners are rounded (ROUND, the radius over the slot width, 0.3 by default). The inlet is uniform at
//! x = 0, the outlet a pressure face at x = 2.5, the z faces slip, no absorbing layer, no wall model, the hybrid recursive
//! regularised collision, walls interpolated with the mass correction on. Two runs of one case: the base grid with `SLOT_BASE`
//! cells across the slot (default 6) and one strip over the plate and its jet (x 0.4 to 1.8 m), against the uniform grid of
//! twice the cells (the strip's resolution everywhere), both at the same physical viscosity and lattice speed.
//!
//! Reported for both: the plate's pressure drop K = dp / (0.5 u_nom^2) from the density at x = 0.5 and x = 1.4 (dp = d rho / 3),
//! the volume flux through five sections (x = 0.5, 0.7, 1.0, 1.4, 2.1 m), out/in, the throat Reynolds number and the drift of K
//! over the last third of the run. The gates: K within 2 % and the section fluxes within 0.3 % between the two runs.
//!
//! Environment: SAMPLE (20, steps between samples), REPORT (2000, steps between printed samples), WINDOW (20000), SLOT_BASE (6), STEPS (base steps, 40000), TAU (tau - 1/2 of the base grid, 5.1e-3), ROUND (0.3), ULAT (0.014),
//! QUAD (1: quadratic wall interpolation), FIX (local | spread | off, default spread), REPORT (base steps between samples,
//! 1000), CASE (both | refined | uniform), SIGMA (the hybrid collision's sigma, 0.95).
#[path = "common/mod.rs"]
mod common;

use common::*;
use fieldcraft_lbm::interpolated::{MassFix, WallOrder};
use fieldcraft_lbm::refine::{Refined, Strip};
use fieldcraft_lbm::{Flow, Limits, Obstacle};
use fieldcraft_mesh::surface::BoundarySurface;
use fieldcraft_model::{Collision, FaceCondition, HrrSigma, Model, Outflow, Procedure, Profile};

fn env(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

const LENGTH: f64 = 2.5;
const SLOT: f64 = 0.2;
const PLATE: [f64; 2] = [0.6, 0.8];
/// The strip's extent along x (metres); its columns are these over the base cell size.
const STRIP_X: [f64; 2] = [0.4, 1.8];
/// The strip's x range (m): `STRIP_X` unless STRIP_END moves its end.
fn strip_x() -> [f64; 2] {
    [STRIP_X[0], env("STRIP_END", STRIP_X[1])]
}
const SECTIONS: [f64; 5] = [0.5, 0.7, 1.0, 1.4, 2.1];

/// A plate as a prism in (x, z) over y whose slot-side corners are rounded by radius `r` (16 chords each). `slot_above`: the
/// plate lies below the slot (its slot-side face is its top, z = `z[1]`).
fn rounded_plate(x: [f64; 2], z: [f64; 2], y: [f64; 2], r: f64, slot_above: bool) -> BoundarySurface {
    let m = 16usize;
    let (z_far, z_slot) = if slot_above { (z[0], z[1]) } else { (z[1], z[0]) };
    let s = if slot_above { 1.0 } else { -1.0 };
    let mut poly: Vec<[f64; 2]> = vec![[x[0], z_far], [x[1], z_far]];
    for k in 0..=m {
        let a = 0.5 * std::f64::consts::PI * k as f64 / m as f64;
        poly.push([x[1] - r + r * a.cos(), z_slot - s * r + s * r * a.sin()]);
    }
    for k in 0..=m {
        let a = 0.5 * std::f64::consts::PI * (1.0 + k as f64 / m as f64);
        poly.push([x[0] + r + r * a.cos(), z_slot - s * r + s * r * a.sin()]);
    }
    if !slot_above {
        poly.reverse();
    }
    let n = poly.len();
    let mut positions: Vec<[f64; 3]> = poly.iter().map(|p| [p[0], y[0], p[1]]).collect();
    positions.extend(poly.iter().map(|p| [p[0], y[1], p[1]]));
    let (mut tris, mut tags) = (Vec::new(), Vec::new());
    for k in 1..n as u32 - 1 {
        tris.push([0, k, k + 1]);
        tags.push(0);
        tris.push([n as u32, n as u32 + k, n as u32 + k + 1]);
        tags.push(1);
    }
    for k in 0..n as u32 {
        let k1 = (k + 1) % n as u32;
        tris.push([k, k1, n as u32 + k1]);
        tris.push([k, n as u32 + k1, n as u32 + k]);
        tags.extend([2, 2]);
    }
    let centre = [0.5 * (x[0] + x[1]), 0.5 * (y[0] + y[1]), 0.5 * (z[0] + z[1])];
    surface(positions, tris, tags, move |c, _| [c[0] - centre[0], c[1] - centre[1], c[2] - centre[2]])
}

/// The case at `cells` cells across the slot (`base` is the base grid's count, which fixes the viscosity and lattice speed).
fn case(cells: f64, steps: u64, base: f64) -> (Model, Vec<Obstacle>) {
    let (u_in, tau_half) = (env("ULAT", 0.014), env("TAU", 5.1e-3));
    // The base grid's lattice viscosity is (tau - 1/2) / 3 with u_in lattice units at 0.1 m/s; acoustic scaling gives the fine
    // grid the same physical viscosity and speed at half the cell size and time step.
    let dx_base = SLOT / base;
    let dt_base = u_in * dx_base / 0.1;
    let nu = tau_half / 3.0 * dx_base * dx_base / dt_base;
    let dx = SLOT / cells;
    let dt = u_in * dx / 0.1;
    let width = 4.0 * dx_base;
    let mut d = domain([0.0, 0.0, 0.0], [LENGTH, width, 1.0], dx);
    d.faces.x_min = FaceCondition::Velocity { velocity: [0.1, 0.0, 0.0], profile: Profile::Uniform };
    d.faces.x_max = FaceCondition::Pressure { pressure: 0.0, outflow: Outflow::Uniform };
    d.faces.y_min = FaceCondition::Periodic {};
    d.faces.y_max = FaceCondition::Periodic {};
    d.faces.z_min = FaceCondition::Slip {};
    d.faces.z_max = FaceCondition::Slip {};
    d.obstacles = vec!["PlateLow".into(), "PlateHigh".into()];
    let round = env("ROUND", 0.3);
    let r = round * SLOT;
    let faces = ["x_min", "x_max", "wall"];
    let obstacles = if round > 0.0 {
        vec![
            obstacle("PlateLow", rounded_plate(PLATE, [-1.0, 0.4], [-1.0, 2.0], r, true), &faces),
            obstacle("PlateHigh", rounded_plate(PLATE, [0.6, 2.0], [-1.0, 2.0], r, false), &faces),
        ]
    } else {
        vec![
            obstacle("PlateLow", cuboid([PLATE[0], -1.0, -1.0], [PLATE[1], 2.0, 0.4]), &CUBOID_FACES),
            obstacle("PlateHigh", cuboid([PLATE[0], -1.0, 0.6], [PLATE[1], 2.0, 2.0]), &CUBOID_FACES),
        ]
    };
    let p = Procedure::Fluid {
        domain: "Flow".into(),
        end_time: None,
        steps: Some(steps),
        time_step: Some(dt),
        frames: 2,
        magic: None,
        sigma: Some(HrrSigma::Value(env("SIGMA", 0.95))),
        collision: Some(Collision::Hrr),
        turbulence: None,
        steady: None,
    };
    (model(1.0, nu, d, p), obstacles)
}

fn fix() -> MassFix {
    match std::env::var("FIX").as_deref() {
        Ok("local") => MassFix::Local,
        Ok("off") => MassFix::Off,
        _ => MassFix::Spread,
    }
}

/// One sample of a run: base step, K, the five section fluxes, out/in.
type Sample = (u64, f64, [f64; 5], f64, f64, f64);

/// The base-cell density and momentum fields of a run (`bx` x `by` x `bz` base cells; solid cells give zeros).
type Field = Vec<(f64, [f64; 3])>;

/// The run: `refined` (base grid plus strip) or the uniform grid of twice the cells.
fn run(refined: bool) -> Vec<Sample> {
    let base = env("SLOT_BASE", 6.0);
    let steps = env("STEPS", 40_000.0) as u64;
    let every = (env("SAMPLE", 20.0) as u64).max(1);
    let print_every = (env("REPORT", 2000.0) as u64).max(1);
    let order = if env("QUAD", 0.0) > 0.5 { WallOrder::Quadratic } else { WallOrder::Linear };
    let dx = SLOT / base;
    let mut r: Option<Refined> = None;
    let mut f: Option<Flow> = None;
    let [bx, by, bz]: [usize; 3];
    if refined {
        let (m, obstacles) = case(base, steps, base);
        let first = (strip_x()[0] / dx).round() as usize;
        let last = (strip_x()[1] / dx).round() as usize;
        let mut rr = Refined::new(&m, None, &obstacles, &Limits::default(), Strip { first, last, linear: true }).unwrap_or_else(|e| panic!("{e}"));
        rr.interpolate_walls_with(&obstacles, order, fix()).unwrap_or_else(|e| panic!("{e}"));
        bx = rr.coarse().grid().cells[0];
        by = rr.coarse().grid().cells[1];
        bz = rr.coarse().grid().cells[2];
        r = Some(rr);
    } else {
        let (m, obstacles) = case(2.0 * base, 2 * steps, base);
        let (setup, mut flow) = prepare(&m, &obstacles);
        flow.interpolate_walls_with(&setup, &obstacles, order).unwrap_or_else(|e| panic!("{e}"));
        flow.set_wall_mass_correction(fix()).unwrap_or_else(|e| panic!("{e}"));
        bx = flow.grid().cells[0] / 2;
        by = flow.grid().cells[1] / 2;
        bz = flow.grid().cells[2] / 2;
        f = Some(flow);
    }
    // The base-cell fields: composite moments for the strip run, the 2 x 2 x 2 averages of the fine cells for the uniform one.
    let fields = |r: &Option<Refined>, f: &Option<Flow>| -> Field {
        if let Some(r) = r {
            return r.composite_moments();
        }
        let Some(flow) = f else { return Vec::new() };
        let g = *flow.grid();
        let mut acc = vec![(0.0, [0.0; 3], 0.0); bx * by * bz];
        for c in 0..g.len() {
            let Some((rho, u)) = flow.moments(c) else { continue };
            let [x, y, z] = g.coords(c);
            let b = (z / 2 * by + y / 2) * bx + x / 2;
            let Some(a) = acc.get_mut(b) else { continue };
            a.0 += rho;
            for (s, v) in a.1.iter_mut().zip(u) {
                *s += rho * v;
            }
            a.2 += 1.0;
        }
        acc.iter().map(|&(rho, j, n)| if n > 0.0 && rho > 0.0 { (rho / n, j.map(|v| v / rho)) } else { (0.0, [0.0; 3]) }).collect()
    };
    let col = |x: f64| ((x / dx) as usize).min(bx - 1);
    let u_in = env("ULAT", 0.014);
    let u_nom = u_in / (SLOT / 1.0); // the inlet speed over the open fraction of the section (slot 0.2 of height 1)
    let mut samples = Vec::new();
    // BAND_WINDOWS "4000-6000,40000-42000": step ranges in which the total and per-band masses are kept at every sample.
    let windows: Vec<(u64, u64)> = std::env::var("BAND_WINDOWS")
        .unwrap_or_default()
        .split(',')
        .filter_map(|w| w.split_once('-').and_then(|(a, b)| Some((a.trim().parse().ok()?, b.trim().parse().ok()?))))
        .collect();
    let mut bands: Vec<(usize, u64, [f64; 5], f64)> = Vec::new();
    if let Some(r) = &r {
        let (c, f) = (r.coarse(), r.fine());
        println!(
            "LEVELS refined: base {:?} sigma {:?} omegas {:?} tau+ {:.6}; fine {:?} sigma {:?} omegas {:?} tau+ {:.6}",
            c.collision(),
            c.hybrid_sigma(),
            c.omegas(),
            1.0 / c.omegas().0,
            f.collision(),
            f.hybrid_sigma(),
            f.omegas(),
            1.0 / f.omegas().0
        );
    }
    if let Some(f) = &f {
        println!("LEVELS uniform: {:?} sigma {:?} omegas {:?} tau+ {:.6}", f.collision(), f.hybrid_sigma(), f.omegas(), 1.0 / f.omegas().0);
    }
    for step in 1..=steps {
        if let Some(r) = r.as_mut() {
            r.step().unwrap_or_else(|e| panic!("{e}"));
        } else if let Some(f) = f.as_mut() {
            for _ in 0..2 {
                f.step().unwrap_or_else(|e| panic!("{e}"));
            }
        }
        if step % every == 0 || step == steps {
            let mom = fields(&r, &f);
            let mean_rho = |x: usize| -> f64 {
                let (mut s, mut c) = (0.0, 0.0);
                for z in 0..bz {
                    for y in 0..by {
                        if let Some(v) = mom.get((z * by + y) * bx + x)
                            && v.0 > 0.0
                        {
                            s += v.0;
                            c += 1.0;
                        }
                    }
                }
                s / f64::max(c, 1.0)
            };
            let flux = |x: usize| -> f64 {
                let mut s = 0.0;
                for z in 0..bz {
                    for y in 0..by {
                        if let Some(v) = mom.get((z * by + y) * bx + x) {
                            s += v.0 * v.1[0];
                        }
                    }
                }
                s
            };
            let dp = (mean_rho(col(0.5)) - mean_rho(col(1.4))) / 3.0;
            let k = dp / (0.5 * u_nom * u_nom);
            let fl = SECTIONS.map(|x| flux(col(x)));
            let mass: f64 = mom.iter().map(|c| c.0).sum();
            let inflow = flux(col(0.2)).abs().max(1e-30);
            assert!(mom.iter().all(|c| c.0.is_finite() && c.1.iter().all(|v| v.is_finite())), "non-finite state at step {step}");
            let name = if refined { "refined" } else { "uniform" };
            if step % print_every == 0 || step == steps {
                println!("SAMPLE {name} step {step}: K {k:.4} fluxes {:?} out/in {:.5}", fl.map(|v| (v * 1e5).round() / 1e5), fl[4] / inflow);
                println!(
                    "MASS {name} step {step}: total {mass:.6} (base cells), flux at x 0.1 {:.5}, at the strip's faces 0.4 {:.5} and 1.8 {:.5}",
                    flux(col(0.1)),
                    flux(col(strip_x()[0])),
                    flux(col(strip_x()[1]))
                );
            }
            if let Some(w) = windows.iter().position(|&(lo, hi)| step > lo && step <= hi) {
                // Fluid mass in x bands of the box, to see where the box mode lives.
                let edges = [0.0, 0.4, 1.0, 1.8, 2.3];
                let mut b = [0.0; 5];
                for z in 0..bz {
                    for y in 0..by {
                        for x in 0..bx {
                            let rho = mom.get((z * by + y) * bx + x).map_or(0.0, |v| v.0);
                            let xm = (x as f64 + 0.5) * dx;
                            let band = edges.iter().rposition(|&e| xm >= e).unwrap_or(0);
                            b[band] += rho;
                        }
                    }
                }
                bands.push((w, step, b, mass));
            }
            samples.push((step, k, fl, fl[4] / inflow, mass, flux(col(0.1))));
        }
    }
    let name = if refined { "refined" } else { "uniform" };
    let mut previous: Option<(u64, f64)> = None;
    for (w, &(lo, hi)) in windows.iter().enumerate() {
        let part: Vec<&(usize, u64, [f64; 5], f64)> = bands.iter().filter(|b| b.0 == w).collect();
        if part.len() < 3 {
            continue;
        }
        let n = part.len() as f64;
        let mean = part.iter().map(|b| b.3).sum::<f64>() / n;
        let (low, high) = part.iter().fold((f64::MAX, f64::MIN), |(l, h), b| (l.min(b.3), h.max(b.3)));
        let amp = 0.5 * (high - low);
        let ups: Vec<u64> = (1..part.len()).filter(|&i| part[i - 1].3 < mean && part[i].3 >= mean).map(|i| part[i].1).collect();
        let period = match (ups.first(), ups.last()) {
            (Some(&a), Some(&z)) if ups.len() > 1 => (z - a) as f64 / (ups.len() - 1) as f64,
            _ => f64::NAN,
        };
        let band_amp: Vec<String> = (0..5)
            .map(|i| {
                let m = part.iter().map(|b| b.2[i]).sum::<f64>() / n;
                let (l, h) = part.iter().fold((f64::MAX, f64::MIN), |(l, h), b| (l.min(b.2[i]), h.max(b.2[i])));
                format!("{:.2} %", 100.0 * 0.5 * (h - l) / m)
            })
            .collect();
        print!(
            "WINDOW {name} steps {lo}-{hi}: total mass mean {mean:.1}, amplitude {amp:.1} ({:.3} %), period {period:.1} steps ({} crossings); band amplitudes x 0-0.4 / 0.4-1.0 / 1.0-1.8 / 1.8-2.3 / 2.3-2.5: {band_amp:?}",
            100.0 * amp / mean,
            ups.len()
        );
        if let Some((s0, a0)) = previous {
            let rate = (amp / a0).ln() / ((lo + hi) as f64 / 2.0 - s0 as f64) * 1000.0;
            print!("; growth {:+.3} % per 1000 steps since the previous window", 100.0 * rate);
        }
        println!();
        previous = Some(((lo + hi) / 2, amp));
    }
    samples
}

/// The window of a run's samples that ends at the last step and holds a whole number of K's oscillation periods
/// within the last `WINDOW` steps (20000): the samples from its first upward crossing of the window mean to its
/// last one. Returns the mean K, the mean fluxes and out/in over it, the period in steps, the periods the window
/// covers and the drift (the mean K of the last 10000 steps against the 10000 before, over the window's mean K).
fn tail(s: &[Sample]) -> (f64, f64, [f64; 5], f64, f64, usize, f64, f64) {
    let last = s.last().map_or(0, |x| x.0);
    let width = env("WINDOW", 20_000.0) as u64;
    let w: Vec<&Sample> = s.iter().filter(|x| x.0 + width > last).collect();
    let mean = w.iter().map(|x| x.1).sum::<f64>() / w.len().max(1) as f64;
    let ups: Vec<usize> = (1..w.len()).filter(|&i| w[i - 1].1 < mean && w[i].1 >= mean).collect();
    let (from, to) = match (ups.first(), ups.last()) {
        (Some(&a), Some(&b)) if b > a => (a, b),
        _ => (0, w.len()),
    };
    let periods = ups.len().saturating_sub(1);
    let period = if periods > 0 { (w[to].0 - w[from].0) as f64 / periods as f64 } else { f64::NAN };
    let part = &w[from..to.max(from + 1).min(w.len())];
    let n = part.len().max(1) as f64;
    let k = part.iter().map(|x| x.1).sum::<f64>() / n;
    let mut fl = [0.0; 5];
    for x in part {
        for (a, v) in fl.iter_mut().zip(x.2) {
            *a += v / n;
        }
    }
    let half = |lo: u64, hi: u64| {
        let v: Vec<f64> = s.iter().filter(|x| x.0 > lo && x.0 <= hi).map(|x| x.1).collect();
        v.iter().sum::<f64>() / v.len().max(1) as f64
    };
    let mass = part.iter().map(|x| x.4).sum::<f64>() / n;
    let f01 = part.iter().map(|x| x.5).sum::<f64>() / n;
    let drift = (half(last.saturating_sub(10_000), last) - half(last.saturating_sub(20_000), last.saturating_sub(10_000))) / k.abs().max(1e-30);
    (k, drift, fl, part.iter().map(|x| x.3).sum::<f64>() / n, period, periods, mass, f01)
}

#[test]
#[ignore = "local: CPU run, minutes to an hour"]
fn the_rounded_slot_with_a_strip_matches_the_uniform_fine_run() {
    let which = std::env::var("CASE").unwrap_or_else(|_| "both".into());
    let base = env("SLOT_BASE", 6.0);
    let (tau, u_in) = (env("TAU", 5.1e-3), env("ULAT", 0.014));
    let re = u_in / SLOT * base / (tau / 3.0); // u_nom x the slot's base cells / the base grid's lattice viscosity
    println!(
        "CASE rounded slot: base {base} cells across the slot (strip {} to {} m), uniform {}; ROUND {}; tau-1/2 {tau} (base), ULAT {u_in}; throat Re {re:.0}; walls {}, mass fix {:?}; no wall-layer sigma rule on this line",
        strip_x()[0],
        strip_x()[1],
        2.0 * base,
        env("ROUND", 0.3),
        if env("QUAD", 0.0) > 0.5 { "quadratic" } else { "linear" },
        fix()
    );
    let a = (which != "uniform").then(|| run(true));
    let b = (which != "refined").then(|| run(false));
    if let Some(one) = a.as_ref().or(b.as_ref()).filter(|_| a.is_none() || b.is_none()) {
        let (k, d, f, o, p, n, m, i) = tail(one);
        println!(
            "RESULT single run: K {k:.4} (drift of the last 10000 steps {:.2} %), period {p:.0} steps ({n} periods), total mass {m:.3}, flux at x 0.1 {i:.5}, out/in {o:.5}",
            100.0 * d
        );
        println!("RESULT fluxes {:?}; over the flux at x 0.1 {:?}", f.map(|v| (v * 1e5).round() / 1e5), f.map(|v| (v / i * 1e5).round() / 1e5));
    }
    if let (Some(a), Some(b)) = (&a, &b) {
        let (ka, da, fa, oa, pa, na, ma, ia) = tail(a);
        let (kb, db, fb, ob, pb, nb, mb, ib) = tail(b);
        let kdiff = (ka - kb) / kb;
        // Each section's flux over the same run's flux at x 0.1, so the inlet density cancels.
        let fdiff: Vec<f64> = (0..5).map(|i| (fa[i] / ia) / (fb[i] / ib) - 1.0).collect();
        let settled = da.abs() <= 0.005 && db.abs() <= 0.005;
        println!(
            "RESULT K refined {ka:.4} (drift of the last 10000 steps {:.2} %), uniform {kb:.4} (drift {:.2} %): difference {:+.2} % (gate 2 %)",
            100.0 * da,
            100.0 * db,
            100.0 * kdiff
        );
        println!(
            "RESULT oscillation period: refined {pa:.0} steps ({na} periods in the window), uniform {pb:.0} steps ({nb} periods); means over whole periods"
        );
        println!(
            "RESULT whole-period means: total mass refined {ma:.3} uniform {mb:.3} ({:+.3} %); flux at x 0.1 refined {ia:.5} uniform {ib:.5} ({:+.3} %)",
            100.0 * (ma - mb) / mb,
            100.0 * (ia - ib) / ib
        );
        println!("RESULT fluxes refined {:?} uniform {:?}", fa.map(|v| (v * 1e5).round() / 1e5), fb.map(|v| (v * 1e5).round() / 1e5));
        println!(
            "RESULT flux differences over the flux at x 0.1, at x 0.5 0.7 1.0 1.4 and 2.1 (2.1 is past the strip and not gated): {:?} % (gate 0.3 %); out/in refined {oa:.5} uniform {ob:.5}",
            fdiff.iter().map(|v| (v * 1e4).round() / 1e2).collect::<Vec<_>>()
        );
        println!(
            "RESULT gates: {}",
            if !settled {
                "not read: K still drifts by more than 0.5 % over the last 10000 steps".to_string()
            } else {
                format!(
                    "K {} flux {}",
                    if kdiff.abs() <= 0.02 { "PASS" } else { "FAIL" },
                    if fdiff.iter().take(4).all(|v| v.abs() <= 0.003) { "PASS" } else { "FAIL" }
                )
            }
        );
    }
}
