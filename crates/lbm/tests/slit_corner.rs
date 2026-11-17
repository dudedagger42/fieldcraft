//! A driven test bed for the dead-end corner of a slit: how strongly does the corner cell amplify a disturbance, and does the
//! disturbance decay or grow afterwards? Print-only, `#[ignore]`: the lines are read, not gated.
//!
//! Geometry (3D, periodic along z over 4 cells, so it is a 2D case): a plane channel `CELLS` cells wide (default 8) driven along x by a
//! body force, with a slit `SLIT_W` cells wide (default 2) and `SLIT_D` cells deep (default 30) cut into its upper wall, closed at the
//! far end. The slit's closed end has two corner cells, each with two solid axis neighbours (a side wall and the end). The channel flow
//! passes the slit's mouth. `FILL="x,y;x,y;..."` makes the listed cells (cell coordinates) solid, for the variants that fill cells
//! next to the corner; the corner cells are `(4, W + D - 1)` and `(5, W + D - 1)` for a 2-wide slit (printed).
//!
//! The drive: after `SETTLE_STEPS` (default 20,000) the flow is read as the baseline, then for `PULSE_STEPS` (default 300) the body force
//! carries an extra `PULSE_AMP / PULSE_STEPS` along x (default amplitude 1e-3 lattice speed: the free-flow velocity change the pulse
//! would cause), then the run continues `AFTER_STEPS` (default 20,000) more. The pulse and the read-out are the reusable part: a corner
//! remedy is scored by the lines below with the same pulse.
//!
//! Read-out, every `SAMPLE` (default 10) steps: the largest |rho - rho_baseline| over the corner cells (called the corner deviation),
//! and whether that cell alternates cell to cell: along some axis both its fluid neighbours' deviations have the opposite sign and
//! at least half its size (2 dx alternation). The lines:
//! - `SLIT-CORNER ... amplification A`: the largest corner deviation over the whole run divided by `PULSE_AMP`, and the step it occurs at;
//! - `rate`: the corner deviation's envelope (the maximum over each 500-step block) after the pulse ends, as the natural log of the last
//!   block over the first block, per 1,000 steps (positive: growing, negative: decaying), with the envelope blocks printed;
//! - `SLIT-HISTORY`: the corner deviation every 100 steps (`*` marks an alternating read);
//! - the run's stop (the solver's own, a speed over 0.3 or a non-finite number), if any.
//!
//! `GEOM3=1` selects the 3D variant: a floor (two solid layers; the slit's x = 4 row one layer lower), side walls that widen into a wedge
//! above z = 5, a lid and a closed end. `FILL="x,y,z;..."` then fills single cells (z optional). Probes: `site` (5, 34, 2) and `corner`.
//!
//! Environment: `TAU_MINUS_HALF` (default 1.8e-4), `PEAK` (peak lattice speed of the channel's parabola, default 0.11; the mean is 2/3 of it
//! (parabola), `SIGMA` (default 0.95), `WALL_SIGMA1` (0 or 1, default 0), `SA_SEED` (default 0: the model's trivial solution).
//!
//! `cargo test --release -p fieldcraft-lbm --test slit_corner -- --ignored --nocapture slit_corner`

mod common;

use common::*;
use fieldcraft_lbm::backend::Backend;
use fieldcraft_lbm::{Flow, Limits, Obstacle, Setup};
use fieldcraft_model::{Collision, FaceCondition, HrrSigma, Procedure, Turbulence};

const PERIODIC: FaceCondition = FaceCondition::Periodic {};

fn env(name: &str, default: f64) -> f64 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// `FILL="x,y;x,y"`: cells (coordinates) made solid.
fn fill_cells() -> Vec<(usize, usize, Option<usize>)> {
    std::env::var("FILL")
        .unwrap_or_default()
        .split(';')
        .filter_map(|p| {
            let mut it = p.split(',').map(|v| v.trim().parse::<usize>().ok());
            let (a, b) = (it.next()??, it.next()??);
            Some((a, b, it.next().flatten()))
        })
        .collect()
}

/// The vertical layout of the `GEOM3` variant: cells in z, the floor's solid layers, and the layer the slit's walls start to widen at.
const NZ: usize = 12;
const FLOOR: usize = 2;
const WEDGE_AT: usize = 5;

struct Case {
    setup: Setup,
    obstacles: Vec<Obstacle>,
    /// The probes: a name and the cells it reads (the largest deviation of them).
    probes: Vec<(&'static str, Vec<[usize; 3]>)>,
    /// The acceleration (lattice units) the flow is driven with.
    base_acceleration: f64,
}

/// The channel with the slit, in lattice units with dx = 1/w.
fn build(
    w: usize,
    slit_w: usize,
    slit_d: usize,
    tau_minus_half: f64,
    peak: f64,
    sigma: f64,
    fills: &[(usize, usize, Option<usize>)],
    three_d: bool,
) -> Result<Case, String> {
    let nu_lat = tau_minus_half / 3.0;
    let dx = 1.0 / w as f64;
    let dt = nu_lat * dx * dx;
    let wf = w as f64;
    let ny = w + 4 + slit_d;
    let nz = if three_d { NZ } else { 4 };
    let inlet = if three_d { env("INLET_ROW", 0.0) } else { 0.0 };
    let (lx, ly, lz) = (16.0 * dx, ny as f64 * dx, nz as f64 * dx);
    let ho = wf * dx;
    let (x0, x1) = (4.0 * dx, (4 + slit_w) as f64 * dx);
    let mut obstacles = Vec::new();
    if three_d {
        // GEOM3: the slit has a floor (the layers below z = FLOOR are solid; the slit's x = 4 row is one layer lower), full-height side
        // walls up to z = WEDGE_AT, then walls that retreat one cell per two layers (the wedge), a lid on the domain, a closed end
        let mut names = Vec::new();
        let mut add = |name: String, lo: [f64; 3], hi: [f64; 3]| {
            let n: &'static str = Box::leak(name.into_boxed_str());
            names.push(obstacle(n, cuboid(lo, hi), &CUBOID_FACES));
        };
        let big = 3.0 * lx;
        let zz = |k: usize| k as f64 * dx;
        if inlet > 0.0 {
            // the slit's x = 4 floor row (z = 0 under the lowered z = 1 row) is a velocity inlet pointing up into the slit
            let (xa, xb, ya, yc) = (x0, x0 + dx, ho, ho + slit_d as f64 * dx);
            add("Floor0Left".into(), [-big, -dx, -dx], [xa, ly + dx, zz(1)]);
            add("Floor0Right".into(), [xb, -dx, -dx], [big, ly + dx, zz(1)]);
            add("Floor0Front".into(), [xa, -dx, -dx], [xb, ya, zz(1)]);
            add("Floor0Back".into(), [xa, yc, -dx], [xb, ly + dx, zz(1)]);
            add("Inlet".into(), [xa, ya, -dx], [xb, yc, zz(1)]);
        } else {
            add("Floor0".into(), [-big, -dx, -dx], [big, ly + dx, zz(1)]);
        }
        add("FloorUnderChannel".into(), [-big, -dx, zz(1)], [big, ho, zz(FLOOR)]);
        add("FloorSlitWall".into(), [x1, ho, zz(1)], [big, ly + dx, zz(FLOOR)]);
        add("FloorSlitLeftWall".into(), [-big, ho, zz(1)], [x0, ly + dx, zz(FLOOR)]);
        if inlet <= 0.0 {
            add("Lid".into(), [-big, -dx, zz(nz - 1)], [big, ly + dx, zz(nz) + dx]);
        }
        let top = (env("FREE_TOP", (nz - 1) as f64) as usize).clamp(FLOOR + 1, nz - 1);
        for k in FLOOR..top {
            let extra = if k >= WEDGE_AT { ((k - WEDGE_AT) / 2 + 1) as f64 * dx } else { 0.0 };
            add(format!("WallLeft{k}"), [-big, ho, zz(k)], [x0 - extra, ly + dx, zz(k + 1)]);
            add(format!("WallRight{k}"), [x1 + extra, ho, zz(k)], [big, ly + dx, zz(k + 1)]);
            add(format!("WallCap{k}"), [x0 - extra, ho + slit_d as f64 * dx, zz(k)], [x1 + extra, ly + dx, zz(k + 1)]);
        }
        obstacles = names;
    } else {
        obstacles.push(obstacle("WallLeft", cuboid([-3.0 * lx, ho, -dx], [x0, ly, lz + dx]), &CUBOID_FACES));
        obstacles.push(obstacle("WallRight", cuboid([x1, ho, -dx], [3.0 * lx, ly, lz + dx]), &CUBOID_FACES));
        obstacles.push(obstacle("WallCap", cuboid([x0, ho + slit_d as f64 * dx, -dx], [x1, ly, lz + dx]), &CUBOID_FACES));
    }
    for (k, &(cx, cy, cz)) in fills.iter().enumerate() {
        let (a, b) = (cx as f64 * dx, cy as f64 * dx);
        let name: &'static str =
            ["Fill0", "Fill1", "Fill2", "Fill3", "Fill4", "Fill5", "Fill6", "Fill7"].get(k).copied().ok_or("at most 8 fill cells")?;
        let (z0, z1) = match cz {
            Some(z) => (z as f64 * dx, (z + 1) as f64 * dx),
            None => (-dx, lz + dx),
        };
        obstacles.push(obstacle(name, cuboid([a, b, z0], [a + dx, b + dx, z1]), &CUBOID_FACES));
    }
    let mut d = domain([0.0; 3], [lx, ly, lz], dx);
    for f in [&mut d.faces.x_min, &mut d.faces.x_max, &mut d.faces.y_min, &mut d.faces.y_max, &mut d.faces.z_min, &mut d.faces.z_max] {
        *f = PERIODIC;
    }
    if inlet > 0.0 {
        // an open top (a pressure outlet) takes the inlet's flow; the floor below is a wall
        d.faces.z_min = FaceCondition::Wall {};
        d.faces.z_max = FaceCondition::Pressure { pressure: 0.0, outflow: Default::default() };
        d.surfaces.push(fieldcraft_model::SurfaceBoundary {
            face: fieldcraft_model::GeometryRef::face("Inlet", "z_max").map_err(|e| e.to_string())?,
            condition: fieldcraft_model::SurfaceCondition::MovingWall { velocity: [0.0, 0.0, inlet * dx / dt] },
        });
        d.moving_wall_ramp = Some(env("INLET_RAMP", 2000.0) * dt);
    }
    d.obstacles = obstacles.iter().map(|o| o.body.clone()).collect();
    // the body force of a plane channel for the mean speed (2/3 of the peak): 12 nu U / H^2
    let mean = peak * 2.0 / 3.0;
    // with FREE_TOP the walls stop below the lid and a free layer of height h carries the stream: the force is for that layer
    let free = (nz - 1).saturating_sub(env("FREE_TOP", (nz - 1) as f64) as usize);
    let hf = if three_d && free > 0 { (free + 1) as f64 } else { wf };
    let g_lat = 12.0 * nu_lat * mean / (hf * hf);
    d.acceleration = [g_lat * dx / (dt * dt), 0.0, 0.0];
    // MRT_ALL=1: the control with MRT everywhere (sigma unused)
    let mrt_all = env("MRT_ALL", 0.0) != 0.0;
    let mut p = collide(1e6 * dt, dt, None, if mrt_all { Collision::Mrt } else { Collision::Hrr });
    if let Procedure::Fluid { sigma: s, turbulence, .. } = &mut p {
        if !mrt_all {
            *s = Some(HrrSigma::Value(sigma));
        }
        if tau_minus_half < 5e-3 - 1e-12 {
            *turbulence = Some(Turbulence::SpalartAllmaras { prandtl: 0.9, wall_model: None, inflow_ratio: env("SA_SEED", 0.0), ddes: None });
        }
    }
    let m = model(1.0, 1.0, d, p);
    let setup = Setup::new(&m, None, &obstacles, &Limits::default()).map_err(|e| e.to_string())?;
    let yb = w + slit_d - 1;
    let probes = if three_d {
        vec![("site", vec![[5, yb - 3, FLOOR]]), ("corner", vec![[5, yb, FLOOR], [4, yb, FLOOR - 1]])]
    } else {
        vec![("site", vec![[5, yb - 3, 0]]), ("corner", vec![[4, yb, 0], [5, yb, 0]])]
    };
    Ok(Case { setup, obstacles, probes, base_acceleration: g_lat })
}

/// One read of the corner: the largest |rho - baseline| over the corner cells, its signed value, and the alternation flag.
fn read_corner(flow: &mut Flow, fluid: &[bool], corners: &[[usize; 3]], baseline: &[f64]) -> Option<(f64, bool)> {
    let f = Backend::fields(flow).ok()?;
    let g = *flow.grid();
    let dev = |c: usize| f.density.get(c).copied().unwrap_or(f64::NAN) - baseline.get(c).copied().unwrap_or(f64::NAN);
    let (mut best, mut best_dev) = (None::<[usize; 3]>, 0.0f64);
    for &p in corners {
        let c = g.index(p);
        if !fluid.get(c).copied().unwrap_or(false) {
            continue;
        }
        let d = dev(c);
        if !d.is_finite() {
            return Some((f64::NAN, false));
        }
        if best.is_none() || d.abs() > best_dev.abs() {
            best = Some(p);
            best_dev = d;
        }
    }
    let p = best?;
    let mut alternates = false;
    for a in 0..3 {
        let n = g.cells[a];
        let mut vals = [f64::NAN; 2];
        for (k, up) in [false, true].into_iter().enumerate() {
            let mut q = p;
            q[a] = if up { (p[a] + 1) % n } else { (p[a] + n - 1) % n };
            let ci = g.index(q);
            if fluid.get(ci).copied().unwrap_or(false) {
                vals[k] = dev(ci);
            }
        }
        alternates |= vals.iter().all(|v| v * best_dev < 0.0 && v.abs() >= 0.5 * best_dev.abs());
    }
    Some((best_dev, alternates))
}

#[test]
#[ignore = "print-only study: see the module docs (cargo test --release -p fieldcraft-lbm --test slit_corner -- --ignored --nocapture)"]
fn slit_corner() {
    let (w, slit_w, slit_d) = (env("CELLS", 8.0) as usize, env("SLIT_W", 2.0) as usize, env("SLIT_D", 30.0) as usize);
    let (tau, peak, sigma) = (env("TAU_MINUS_HALF", 1.8e-4), env("PEAK", 0.11), env("SIGMA", 0.95));
    let (settle, pulse_steps, after) = (env("SETTLE_STEPS", 20_000.0) as u64, env("PULSE_STEPS", 300.0) as u64, env("AFTER_STEPS", 20_000.0) as u64);
    let (amp, sample) = (env("PULSE_AMP", 1e-3), (env("SAMPLE", 10.0) as u64).max(1));
    let wall_sigma1 = env("WALL_SIGMA1", 0.0) as u8;
    let fills = fill_cells();
    let three_d = env("GEOM3", 0.0) != 0.0;
    let label =
        format!("{} tau-1/2 {tau:e} peak {peak} sigma {sigma} WALL_SIGMA1 {wall_sigma1} fill {fills:?}", if three_d { "GEOM3" } else { "2D" });
    let case = match build(w, slit_w, slit_d, tau, peak, sigma, &fills, three_d) {
        Ok(c) => c,
        Err(e) => {
            println!("SLIT-CORNER {label}: setup refused: {e}");
            return;
        }
    };
    let mut flow = match Flow::new(&case.setup, &case.obstacles) {
        Ok(f) => f,
        Err(e) => {
            println!("SLIT-CORNER {label}: refused: {e}");
            return;
        }
    };
    flow.set_wall_layer_sigma_one(wall_sigma1);
    flow.set_wall_layer_bulk(env("WALL_BULK", 0.0));
    flow.set_wall_layer_mrt(env("WALL_MRT", 0.0) != 0.0);
    flow.set_wall_corner_relax(env("WALL_CORNER", 0.0) != 0.0);
    flow.set_wall_tau_floor(env("WALL_TAU_FLOOR", 0.0));
    let n = flow.grid().len();
    let fluid: Vec<bool> = (0..n).map(|c| Backend::is_fluid(&flow, c)).collect();
    println!(
        "SLIT-GEOMETRY {label}: grid {:?}, probes {:?}, fluid {:?}",
        flow.grid().cells,
        case.probes,
        case.probes.iter().map(|(_, ps)| ps.iter().map(|&p| fluid[flow.grid().index(p)]).collect::<Vec<_>>()).collect::<Vec<_>>()
    );
    let a0 = flow.acceleration();
    let mut stop: Option<String> = None;
    let mut done = 0u64;
    while done < settle && stop.is_none() {
        let c = 1000.min(settle - done);
        if let Err(e) = Backend::advance(&mut flow, c) {
            stop = Some(format!("solver stop after step {done}: {e}"));
        }
        done += c;
    }
    let baseline: Vec<f64> = match Backend::fields(&mut flow) {
        Ok(f) if stop.is_none() => f.density,
        _ => vec![1.0; n],
    };
    let pulse_a = [a0[0] + amp / pulse_steps.max(1) as f64, a0[1], a0[2]];
    let total = pulse_steps + after;
    let mut t = 0u64;
    let np = case.probes.len();
    let mut hist: Vec<Vec<(u64, f64, bool)>> = vec![Vec::new(); np];
    while t < total && stop.is_none() {
        let c = sample.min(total - t);
        let a = if t < pulse_steps { pulse_a } else { a0 };
        if let Err(e) = Backend::set_acceleration(&mut flow, a) {
            stop = Some(format!("acceleration refused: {e}"));
            break;
        }
        if let Err(e) = Backend::advance(&mut flow, c) {
            stop = Some(format!("solver stop at time {t} after the settle: {e}"));
            break;
        }
        t += c;
        for (i, (_, cells)) in case.probes.iter().enumerate() {
            match read_corner(&mut flow, &fluid, cells, &baseline) {
                Some((d, alt)) => {
                    if !d.is_finite() {
                        stop = Some(format!("non-finite density at time {t}"));
                        break;
                    }
                    if let Some(h) = hist.get_mut(i) {
                        h.push((t, d, alt));
                    }
                }
                None => {
                    stop = Some(format!("could not read the fields at time {t}"));
                    break;
                }
            }
        }
    }
    let stop_text = stop.unwrap_or_else(|| "no stop".to_string());
    for ((name, _), h) in case.probes.iter().zip(&hist) {
        let (mut peak_dev, mut peak_at, mut alt_reads) = (0.0f64, 0u64, 0usize);
        for &(s, d, alt) in h {
            alt_reads += usize::from(alt);
            if d.abs() > peak_dev {
                peak_dev = d.abs();
                peak_at = s;
            }
        }
        let series: Vec<String> =
            h.iter().filter(|(s, ..)| s % 100 == 0).map(|(s, d, alt)| format!("{s}:{d:+.3e}{}", if *alt { "*" } else { "" })).collect();
        println!(
            "SLIT-HISTORY {label} probe {name} (time since the pulse start: deviation):
{}",
            series.join(" ")
        );
        let blocks: Vec<f64> = h
            .iter()
            .filter(|(s, ..)| *s > pulse_steps)
            .fold(Vec::<(u64, f64)>::new(), |mut acc, (s, d, _)| {
                let b = (s - pulse_steps - 1) / 500;
                if acc.last().map(|x| x.0) == Some(b) {
                    if let Some(l) = acc.last_mut() {
                        l.1 = l.1.max(d.abs());
                    }
                } else {
                    acc.push((b, d.abs()));
                }
                acc
            })
            .into_iter()
            .map(|(_, v)| v)
            .collect();
        let rate = match (blocks.first(), blocks.last()) {
            (Some(&f), Some(&l)) if blocks.len() > 1 && f > 0.0 && l > 0.0 => ((l / f).ln() / ((blocks.len() - 1) as f64 * 0.5)).to_string(),
            _ => "n/a".to_string(),
        };
        println!(
            "SLIT-CORNER {label} probe {name}: amplification {:.3} (peak deviation {peak_dev:.3e} at time {peak_at}, pulse {amp:e}); rate per 1000 steps {rate} (positive grows); alternating reads {alt_reads} of {}; envelope blocks of 500 steps {:?}; {stop_text}",
            peak_dev / amp,
            h.len(),
            blocks.iter().map(|b| format!("{b:.2e}")).collect::<Vec<_>>()
        );
    }
    let _ = case.base_acceleration;
}

/// The largest density difference between two axis-neighbouring fluid cells in the slit-tip region, with the pair, and the region's
/// largest speed.
fn pair_gap(flow: &mut Flow, fluid: &[bool], yb: usize) -> Option<(f64, [usize; 3], [usize; 3], f64, f64)> {
    let f = Backend::fields(flow).ok()?;
    let top = env("FREE_TOP", 99.0) as usize;
    let mut inside = 0.0f64;
    let g = *flow.grid();
    let (mut best, mut pa, mut pb, mut speed) = (0.0f64, [0usize; 3], [0usize; 3], 0.0f64);
    for x in 1..10.min(g.cells[0]) {
        for y in yb.saturating_sub(6)..=yb.min(g.cells[1] - 1) {
            for z in 0..g.cells[2] {
                let a = [x, y, z];
                let ca = g.index(a);
                if !fluid.get(ca).copied().unwrap_or(false) {
                    continue;
                }
                let ra = f.density.get(ca).copied().unwrap_or(f64::NAN);
                if let Some(u) = f.velocity.get(ca) {
                    let m = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
                    if z >= top {
                        speed = speed.max(m);
                    } else {
                        inside = inside.max(m);
                    }
                }
                for axis in 0..3 {
                    let mut b = a;
                    b[axis] += 1;
                    if b[axis] >= g.cells[axis] {
                        continue;
                    }
                    let cb = g.index(b);
                    if !fluid.get(cb).copied().unwrap_or(false) {
                        continue;
                    }
                    let d = (ra - f.density.get(cb).copied().unwrap_or(f64::NAN)).abs();
                    if !d.is_finite() {
                        return Some((f64::NAN, a, b, speed, inside));
                    }
                    if d > best {
                        (best, pa, pb) = (d, a, b);
                    }
                }
            }
        }
    }
    Some((best, pa, pb, speed, inside))
}

/// A sustained stream at the slit mouth (GEOM3 or 2D): the body force ramps from zero to the target over `RAMP_STEPS` (default 1,000) at
/// the angle `STREAM_ANGLE` (degrees from x, default 0: across the mouth; the y component points into the mouth), then holds for
/// `HOLD_STEPS` (default 9,000). Every `SAMPLE` (default 50) steps the largest density gap between neighbouring fluid cells in the
/// slit-tip region (y within six cells of the closed end) is logged with the pair's cells and the region's largest speed. `PEAK` is the
/// nominal peak speed of the plane-channel parabola the force would give.
#[test]
#[ignore = "print-only study: see the module docs (cargo test --release -p fieldcraft-lbm --test slit_corner -- --ignored --nocapture slit_stream)"]
fn slit_stream() {
    let (w, slit_w, slit_d) = (env("CELLS", 8.0) as usize, env("SLIT_W", 2.0) as usize, env("SLIT_D", 30.0) as usize);
    let (tau, peak, sigma) = (env("TAU_MINUS_HALF", 1.8e-4), env("PEAK", 0.11), env("SIGMA", 0.95));
    let (ramp, hold, sample) = (env("RAMP_STEPS", 1000.0) as u64, env("HOLD_STEPS", 9000.0) as u64, (env("SAMPLE", 50.0) as u64).max(1));
    let angle = env("STREAM_ANGLE", 0.0).to_radians();
    let wall_sigma1 = env("WALL_SIGMA1", 1.0) as u8;
    let fills = fill_cells();
    let three_d = env("GEOM3", 1.0) != 0.0;
    let label = format!(
        "{} stream peak {peak} angle {:.0} deg inlet row {} tau-1/2 {tau:e} sigma {sigma} WALL_SIGMA1 {wall_sigma1} WALL_BULK {} WALL_MRT {} MRT_ALL {} fill {fills:?}",
        if three_d { "GEOM3" } else { "2D" },
        angle.to_degrees(),
        env("INLET_ROW", 0.0),
        env("WALL_BULK", 0.0),
        env("WALL_MRT", 0.0),
        env("MRT_ALL", 0.0)
    );
    let case = match build(w, slit_w, slit_d, tau, peak, sigma, &fills, three_d) {
        Ok(c) => c,
        Err(e) => {
            println!("SLIT-STREAM {label}: setup refused: {e}");
            return;
        }
    };
    let mut flow = match Flow::new(&case.setup, &case.obstacles) {
        Ok(f) => f,
        Err(e) => {
            println!("SLIT-STREAM {label}: refused: {e}");
            return;
        }
    };
    flow.set_wall_layer_sigma_one(wall_sigma1);
    flow.set_wall_layer_bulk(env("WALL_BULK", 0.0));
    flow.set_wall_layer_mrt(env("WALL_MRT", 0.0) != 0.0);
    flow.set_wall_corner_relax(env("WALL_CORNER", 0.0) != 0.0);
    flow.set_wall_tau_floor(env("WALL_TAU_FLOOR", 0.0));
    let n = flow.grid().len();
    let fluid: Vec<bool> = (0..n).map(|c| Backend::is_fluid(&flow, c)).collect();
    // the census of concave corners: fluid cells with 3 or more solid axis faces (a wrap counts across a periodic box)
    {
        let grid = *flow.grid();
        let mut census = [0usize; 7];
        let mut listed: Vec<([usize; 3], usize)> = Vec::new();
        for cell in (0..n).filter(|&c| fluid[c]) {
            let p = grid.coords(cell);
            let mut faces = 0;
            for axis in 0..3 {
                for up in [false, true] {
                    let mut q = p;
                    q[axis] = if up { (p[axis] + 1) % grid.cells[axis] } else { (p[axis] + grid.cells[axis] - 1) % grid.cells[axis] };
                    if !fluid.get(grid.index(q)).copied().unwrap_or(true) {
                        faces += 1;
                    }
                }
            }
            census[faces] += 1;
            if faces >= 3 && listed.len() < 24 {
                listed.push((p, faces));
            }
            // PROBE_FACES="x,y,z;...": the number of solid faces of the named fluid cells
            if let Ok(list) = std::env::var("PROBE_FACES") {
                let named = list.split(';').any(|t| {
                    let v: Vec<usize> = t.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                    v.as_slice() == p.as_slice()
                });
                if named {
                    println!("SLIT-STREAM probe {p:?}: {faces} solid axis faces");
                }
            }
        }
        println!("SLIT-STREAM {label}: fluid cells by solid axis faces 0..6: {census:?}; first cells with 3+ (cell, faces): {listed:?}");
    }
    let yb = w + slit_d - 1;
    let g0 = flow.acceleration()[0];
    let (mut t, mut stop) = (0u64, None::<String>);
    let mut hist: Vec<(u64, f64, [usize; 3], [usize; 3], f64, f64)> = Vec::new();
    while t < ramp + hold && stop.is_none() {
        let s = (t as f64 / ramp.max(1) as f64).min(1.0);
        // STREAM_MODE 0: the force ramps up to the plane-channel force and holds (the flow develops over a diffusion time, far slower than
        // the run); STREAM_MODE 1 (default): a constant force PEAK / RAMP_STEPS for RAMP_STEPS steps accelerates the free layer to about PEAK,
        // then the force is zero and the stream coasts (it decays over tens of thousands of steps)
        let a = if env("STREAM_MODE", 1.0) as u32 == 1 {
            let m = if t < ramp { peak / ramp.max(1) as f64 } else { 0.0 };
            [m * angle.cos(), m * angle.sin(), 0.0]
        } else {
            [g0 * s * angle.cos(), g0 * s * angle.sin(), 0.0]
        };
        if let Err(e) = Backend::set_acceleration(&mut flow, a) {
            stop = Some(format!("acceleration refused: {e}"));
            break;
        }
        let c = sample.min(ramp + hold - t);
        if let Err(e) = Backend::advance(&mut flow, c) {
            stop = Some(format!("solver stop after step {t}: {e}"));
            break;
        }
        t += c;
        match pair_gap(&mut flow, &fluid, yb) {
            Some((d, a, b, sp, si)) if d.is_finite() => hist.push((t, d, a, b, sp, si)),
            _ => stop = Some(format!("non-finite density or unreadable fields at step {t}")),
        }
    }
    println!("SLIT-STREAM-LOG {label} (step: gap pair, largest speed in the free layer (z >= FREE_TOP) and below it, every {sample} steps):");
    for (s, d, a, b, sp, si) in &hist {
        println!("  {s}: {d:.3e} {a:?}-{b:?} stream {sp:.4} in-slit {si:.4}");
    }
    // growth over 150 steps (three samples at the default spacing) once the gap is large enough to matter
    let lag = (150 / sample).max(1) as usize;
    let mut worst = (0.0f64, 0u64);
    let mut doublings = 0usize;
    for i in lag..hist.len() {
        let (Some(now), Some(before)) = (hist.get(i), hist.get(i - lag)) else { continue };
        if before.1 >= 1e-3 {
            let r = now.1 / before.1;
            if r > worst.0 {
                worst = (r, now.0);
            }
            doublings += usize::from(r >= 1.8);
        }
    }
    let (gmax, gat) = hist.iter().fold((0.0f64, 0u64), |m, h| if h.1 > m.0 { (h.1, h.0) } else { m });
    let last = hist.last().map(|h| h.1).unwrap_or(0.0);
    println!(
        "SLIT-STREAM {label}: largest gap {gmax:.3e} at step {gat}, last {last:.3e}; windows of 150 steps with gap >= 1e-3 and growth >= 1.8x: {doublings}, largest growth {:.2}x at step {}; {}",
        worst.0,
        worst.1,
        stop.unwrap_or_else(|| "no stop".to_string())
    );
}
