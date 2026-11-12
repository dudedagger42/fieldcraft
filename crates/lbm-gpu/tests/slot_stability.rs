//! Local harness (not for the gate): a slotted channel on a uniform fine grid, run on the device in chunks and
//! compared with the CPU, to see how the hybrid collision (σ 0.95) and the f32 device hold at a very low τ.
//! The plate across the channel has a slot `SLOT_CELLS` cells wide (0.2 m, so dx = 0.2 / SLOT_CELLS); the box faces
//! y min/max and z min are walls with the named wall law (`LAW`: spalding or sa), z max is slip.
//! Environment: SLOT_CELLS (4), LAW (spalding), LAYER (1: the default absorbing layer), STEPS (20000), CHUNK (500), TAU (2.4e-4), ULAT (inlet mean lattice
//! speed, 0.014), LAYER_CELLS (the layer's thickness in cells of this grid; hold the physical thickness: 10/20/40 at
//! 4/8/16 cells), ZMAX (wall: z max a wall), TAU (tau+-1/2; hold Re: 2.4e-4/4.8e-4/9.6e-4 at 4/8/16 cells), CPU (1: compare with the CPU each chunk; 0: device only), GPU (1).
// A local harness: index loops over the grid and nested ifs read better than iterator chains here.
#![allow(clippy::needless_range_loop, clippy::collapsible_if)]
#[path = "../../lbm/tests/common/mod.rs"]
mod common;

use common::*;
use fieldcraft_gpu::select::Policy;
use fieldcraft_lbm::backend::Backend;
use fieldcraft_lbm::{Limits, Setup};
use fieldcraft_lbm_gpu::{GpuFlow, Options};
use fieldcraft_model::{Collision, FaceCondition, HrrSigma, Outflow, Procedure, Profile, Turbulence, WallModel};
use std::time::Instant;

fn env(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

#[test]
#[ignore = "local: GPU run, approved per run"]
fn slotted_channel_stability() {
    let cells = env("SLOT_CELLS", 4.0);
    let law = std::env::var("LAW").unwrap_or_else(|_| "spalding".into());
    let steps = env("STEPS", 20000.0) as u64;
    let chunk = env("CHUNK", 500.0) as u64;
    let tau_half = env("TAU", 2.4e-4);
    let u_in = env("ULAT", 0.014);
    let use_cpu = env("CPU", 1.0) > 0.5;
    let use_gpu = env("GPU", 1.0) > 0.5;
    let dx = 0.2 / cells;
    let dt = u_in * dx / 0.1;
    let nu = tau_half / 3.0 * dx * dx / dt;
    let mut d = domain([0.0; 3], [2.0, 1.0, 1.0], dx);
    d.faces.x_min = FaceCondition::Velocity { velocity: [0.1, 0.02, 0.0], profile: Profile::Parabolic };
    d.faces.x_max = FaceCondition::Pressure { pressure: 0.0, outflow: Outflow::Uniform };
    d.faces.z_max = FaceCondition::Slip {};
    if std::env::var("ZMAX").is_ok_and(|v| v == "wall") {
        d.faces.z_max = FaceCondition::Wall {};
    }
    if std::env::var("SPAN").is_ok_and(|v| v == "periodic") {
        d.faces.y_min = FaceCondition::Periodic {};
        d.faces.y_max = FaceCondition::Periodic {};
    }
    d.acceleration = [0.0, 0.0, -0.01];
    if env("LAYER", 1.0) > 0.5 {
        let thickness = std::env::var("LAYER_CELLS").ok().and_then(|v| v.parse::<f64>().ok()).map(|c| c * dx);
        d.absorbing = Some(fieldcraft_model::AbsorbingLayer { thickness, strength: None, filter_time: None });
    }
    d.obstacles = vec!["PlateLow".into(), "PlateHigh".into()];
    let obstacles = vec![
        obstacle("PlateLow", cuboid([0.6, 0.0, 0.0], [0.8, 1.0, 0.4]), &CUBOID_FACES),
        obstacle("PlateHigh", cuboid([0.6, 0.0, 0.6], [0.8, 1.0, 1.0]), &CUBOID_FACES),
    ];
    let wall_model = match law.as_str() {
        "sa" => Some(WallModel::SpalartAllmarasBounce),
        "none" => None,
        _ => Some(WallModel::SpaldingBounce),
    };
    let sa = Turbulence::SpalartAllmaras { prandtl: 0.9, wall_model, inflow_ratio: 3.0, ddes: None };
    let p = Procedure::Fluid {
        domain: "Flow".into(),
        end_time: None,
        steps: Some(steps),
        time_step: Some(dt),
        frames: 2,
        magic: None,
        sigma: Some(HrrSigma::Value(0.95)),
        collision: Some(Collision::Hrr),
        turbulence: Some(sa),
        steady: None,
    };
    let m = model(1.0, nu, d, p);
    let setup = match Setup::new(&m, None, &obstacles, &Limits::default()) {
        Ok(s) => s,
        Err(e) => {
            println!("RESULT setup refused: {e}");
            return;
        }
    };
    let mut cpu = fieldcraft_lbm::Flow::new(&setup, &obstacles).unwrap();
    let n = cpu.grid().len();
    let [nx, ny, nz] = cpu.grid().cells;
    let scale = setup.velocity_scale();
    println!(
        "CASE wall law {law}; slot {cells} cells; grid {nx}x{ny}x{nz} = {n} cells; tau+-1/2 {:.3e}; tau- -1/2 {:.3e}; sigma 0.95 HRR; dt {dt:.5} s; velocity scale {scale:.4} m/s; steps {steps}; chunk {chunk}; layer {}",
        setup.tau_plus - 0.5,
        setup.tau_minus - 0.5,
        std::env::var("LAYER_CELLS").map_or("default 10 cells".to_string(), |c| format!("{c} cells = {:.4} m", c.parse::<f64>().unwrap_or(0.0) * dx))
    );
    let mut gpu = if use_gpu {
        match GpuFlow::with_options(&cpu, Policy::from_env().unwrap(), Options::default()) {
            Ok(g) => {
                println!("DEVICE {}", g.adapter_name());
                Some(g)
            }
            Err(e) => {
                println!("RESULT device refused: {e}");
                return;
            }
        }
    } else {
        None
    };
    let (mut done, mut cpu_ok, mut gpu_ok) = (0u64, use_cpu, use_gpu);
    let (mut worst_u, mut worst_r) = (0.0f64, 0.0f64);
    let (mut t_gpu, mut t_cpu) = (0.0f64, 0.0f64);
    let fluid: Vec<bool> = (0..n).map(|c| Backend::is_fluid(&cpu, c)).collect();
    let grid = *cpu.grid();
    let print_every = env("PRINT", 2000.0) as u64;
    let diag = env("DIAG", 0.0) > 0.5;
    let (i_in, i_out) = ((0.3 / dx) as usize, nx - 1 - (0.3 / dx) as usize);
    // the two pressure planes sit at x = 0.5 m and 0.9 m on every grid: the mean of the cell layers either side
    let (i_up, i_dn) = (((0.5 / dx).round() as usize).max(1), ((0.9 / dx).round() as usize).max(1));
    println!(
        "PLANES pressure planes at x = {:.4} m and {:.4} m (mean of cell layers {}/{} and {}/{}); the plate spans x 0.6 to 0.8 m",
        dx * i_up as f64,
        dx * i_dn as f64,
        i_up - 1,
        i_up,
        i_dn - 1,
        i_dn
    );
    // per sample: (step, peak |u| lattice, rho min, rho max, rho mean over fluid, flux in, flux out, plate dp (lattice, (rho_up - rho_dn)/3))
    type S = (u64, f64, f64, f64, f64, f64, f64, f64);
    let stats = |f: &fieldcraft_lbm::backend::Fields, step: u64| -> Option<S> {
        let (mut um, mut rmin, mut rmax, mut rsum, mut cnt) = (0.0f64, f64::INFINITY, 0.0f64, 0.0f64, 0usize);
        for c in 0..n {
            if !fluid[c] {
                continue;
            }
            let (u, r) = (f.velocity[c], f.density[c]);
            if !(r.is_finite() && u.iter().all(|v| v.is_finite())) {
                return None;
            }
            um = um.max(u.iter().map(|v| v * v).sum::<f64>().sqrt());
            rmin = rmin.min(r);
            rmax = rmax.max(r);
            rsum += r;
            cnt += 1;
        }
        let plane = |i: usize, g: &dyn Fn(usize) -> f64| -> f64 {
            let (mut sum, mut k0) = (0.0, 0usize);
            for k in 0..nz {
                for j in 0..ny {
                    let c = grid.index([i, j, k]);
                    if fluid[c] {
                        sum += g(c);
                        k0 += 1;
                    }
                }
            }
            sum / k0.max(1) as f64
        };
        let area = (ny * nz) as f64;
        let fl = |i: usize| plane(i, &|c| f.density[c] * f.velocity[c][0]) * area / scale;
        let at = |i: usize| 0.5 * (plane(i - 1, &|c| f.density[c]) + plane(i, &|c| f.density[c]));
        let dp = (at(i_up) - at(i_dn)) / 3.0;
        Some((step, um / scale, rmin, rmax, rsum / cnt as f64, fl(i_in), fl(i_out), dp))
    };
    let (mut sc, mut sg): (Vec<S>, Vec<S>) = (vec![], vec![]);
    let mut rho_diff: Vec<(u64, f64)> = vec![];
    let mut last_good: Option<(u64, [usize; 3], f64, [usize; 3], f64)> = None;
    while done < steps && (cpu_ok || gpu_ok) {
        let c = chunk.min(steps - done);
        if gpu_ok {
            let t = Instant::now();
            if let Some(g) = gpu.as_mut() {
                if let Err(e) = Backend::advance(g, c) {
                    println!("EVENT device stopped in the chunk ending at step {}: {e}", done + c);
                    gpu_ok = false;
                }
            }
            t_gpu += t.elapsed().as_secs_f64();
        }
        if cpu_ok {
            let t = Instant::now();
            if let Err(e) = Backend::advance(&mut cpu, c) {
                println!("EVENT cpu stopped in the chunk ending at step {}: {e}", done + c);
                if let Some((st, im, um, ir, dr)) = last_good {
                    println!("EVENT last finite step {st}: max|u| {um:.4} at cell {im:?}; max|rho-1| {dr:.4} at cell {ir:?}; grid {nx}x{ny}x{nz}");
                }
                cpu_ok = false;
            }
            t_cpu += t.elapsed().as_secs_f64();
        }
        done += c;
        let fa = if cpu_ok { Backend::fields(&mut cpu).ok() } else { None };
        let fb = if gpu_ok { gpu.as_mut().and_then(|g| Backend::fields(g).ok()) } else { None };
        if diag {
            if let Some(f) = fa.as_ref() {
                let (mut um, mut im, mut dr, mut ir) = (0.0f64, [0; 3], 0.0f64, [0; 3]);
                for k in 0..nz {
                    for j in 0..ny {
                        for i in 0..nx {
                            let c = grid.index([i, j, k]);
                            if !fluid[c] {
                                continue;
                            }
                            let v = f.velocity[c].iter().map(|v| v * v).sum::<f64>().sqrt() / scale;
                            if v > um {
                                um = v;
                                im = [i, j, k];
                            }
                            let d = (f.density[c] - 1.0).abs();
                            if d > dr {
                                dr = d;
                                ir = [i, j, k];
                            }
                        }
                    }
                }
                last_good = Some((done, im, um, ir, dr));
                if done % 20 == 0 && done >= 400 {
                    println!("diag step {done}: max|u| {um:.4} at {im:?}; max|rho-1| {dr:.4} at {ir:?}");
                }
            }
        }
        let sa = fa.as_ref().and_then(|f| stats(f, done));
        let sb = fb.as_ref().and_then(|f| stats(f, done));
        if fa.is_some() && sa.is_none() {
            cpu_ok = false;
            println!("EVENT cpu field not finite at step {done}");
        }
        if fb.is_some() && sb.is_none() {
            gpu_ok = false;
            println!("EVENT gpu field not finite at step {done}");
        }
        if let Some(x) = sa {
            sc.push(x);
        }
        if let Some(x) = sb {
            sg.push(x);
        }
        let mut line = format!("step {done}:");
        if let (Some(a), Some(b), Some(fa), Some(fb)) = (sa, sb, fa.as_ref(), fb.as_ref()) {
            let (mut du, mut dr) = (0.0f64, 0.0f64);
            for c in 0..n {
                if fluid[c] {
                    let e = (0..3).map(|k| ((fa.velocity[c][k] - fb.velocity[c][k]) / scale).powi(2)).sum::<f64>().sqrt();
                    du = du.max(e);
                    dr = dr.max((fa.density[c] - fb.density[c]).abs());
                }
            }
            worst_u = worst_u.max(du);
            worst_r = worst_r.max(dr);
            rho_diff.push((done, b.4 - a.4));
            line += &format!(" max|du| {du:.2e} max|drho| {dr:.2e}; mean rho gpu-cpu {:.3e}; peak u cpu {:.4} gpu {:.4};", b.4 - a.4, a.1, b.1);
        } else if let Some(a) = sa.or(sb) {
            line += &format!(" peak u {:.4} rho {:.4}..{:.4} flux out/in {:.4}", a.1, a.2, a.3, a.6 / a.5);
        }
        if done % print_every == 0 {
            if let Some(f) = fa.as_ref().or(fb.as_ref()) {
                let mut rim = 0.0f64;
                for k in 0..nz {
                    for j in 0..ny {
                        for i in 0..2.min(nx) {
                            let near = j < 2 || j + 2 >= ny || k < 2 || k + 2 >= nz;
                            let c = grid.index([i, j, k]);
                            if near && fluid[c] && f.density[c].is_finite() {
                                rim = rim.max((f.density[c] - 1.0).abs());
                            }
                        }
                    }
                }
                line += &format!(" inlet-rim max|rho-1| {rim:.4};");
            }
        }
        if done % print_every == 0 || (done <= 1000 && done % 100 == 0 && use_gpu && use_cpu) {
            println!("{line} gpu s {t_gpu:.2} cpu s {t_cpu:.2}");
        }
    }
    // windows: the second half, and its two quarters
    let win = |v: &[S], lo: u64, hi: u64| -> Option<String> {
        let w: Vec<&S> = v.iter().filter(|s| s.0 > lo && s.0 <= hi).collect();
        if w.is_empty() {
            return None;
        }
        let m = |g: &dyn Fn(&S) -> f64| w.iter().map(|s| g(s)).sum::<f64>() / w.len() as f64;
        let (um, fi, fo, dp) = (m(&|s| s.1), m(&|s| s.5), m(&|s| s.6), m(&|s| s.7));
        let rms = (w.iter().map(|s| (s.1 - um).powi(2)).sum::<f64>() / w.len() as f64).sqrt();
        let (r0, r1) = (w.iter().map(|s| s.2).fold(f64::INFINITY, f64::min), w.iter().map(|s| s.3).fold(0.0, f64::max));
        Some(format!(
            "steps {lo}..{hi} ({} samples): mean out/in {:.5}, plate dp {:.5e} (lattice), K = dp/(0.5 U_in^2) {:.3}, peak u mean {um:.5} rms {rms:.5}, peak/U_in {:.4}, rho {r0:.4}..{r1:.4}",
            w.len(),
            fo / fi,
            dp,
            dp / (0.5 * u_in * u_in),
            um / u_in
        ))
    };
    let (h, q) = (steps / 2, steps / 4);
    for (name, v) in [("CPU", &sc), ("GPU", &sg)] {
        for (lo, hi) in [(h, steps), (h, h + q), (h + q, steps)] {
            if let Some(l) = win(v, lo, hi) {
                println!("AVG {name} {l}");
            }
        }
    }
    if !rho_diff.is_empty() {
        let worst = rho_diff.iter().fold((0u64, 0.0f64), |a, b| if b.1.abs() > a.1.abs() { *b } else { a });
        let pick = |st: u64| rho_diff.iter().find(|d| d.0 >= st).map_or(f64::NAN, |d| d.1);
        println!(
            "MASS domain-mean rho gpu-cpu: step {} {:.3e}; {} {:.3e}; {} {:.3e}; {} {:.3e}; {} {:.3e}; end {:.3e}; largest {:.3e} at step {}",
            steps / 20,
            pick(steps / 20),
            steps / 10,
            pick(steps / 10),
            steps / 4,
            pick(steps / 4),
            steps / 2,
            pick(steps / 2),
            3 * steps / 4,
            pick(3 * steps / 4),
            rho_diff.last().map_or(f64::NAN, |d| d.1),
            worst.1,
            worst.0
        );
    }
    println!(
        "RESULT wall law {law}, layer {}, slot {cells} cells, {n} cells, inlet mean {u_in}: reached step {done} of {steps}; GPU {} CPU {}; worst max|du| {worst_u:.3e} lattice, worst max|drho| {worst_r:.3e}; GPU wall time {t_gpu:.1} s ({:.0} M cell updates/s); CPU wall time {t_cpu:.1} s ({:.1} M cell updates/s)",
        env("LAYER", 1.0) > 0.5,
        if gpu_ok { "held" } else { "stopped" },
        if cpu_ok { "held" } else { "stopped" },
        if t_gpu > 0.0 { n as f64 * done as f64 / t_gpu / 1e6 } else { 0.0 },
        if t_cpu > 0.0 { n as f64 * done as f64 / t_cpu / 1e6 } else { 0.0 }
    );
}
