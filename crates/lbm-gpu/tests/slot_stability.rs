//! Local harness (not for the gate): a slotted channel on a uniform fine grid, run on the device in chunks and
//! compared with the CPU, to see how the hybrid collision (σ 0.95) and the f32 device hold at a very low τ.
//! Fields are in lattice units (velocity and density as the solver holds them), so peak u and U_in are comparable.
//! The plate across the channel has a slot `SLOT_CELLS` cells wide (0.2 m, so dx = 0.2 / SLOT_CELLS); the box faces
//! y min/max and z min are walls with the named wall law (`LAW`: spalding or sa), z max is slip.
//! Environment: SURFACE (1: wall-node rows exactly 0.2 m apart across the slot and through the plate; 2: the open rows exactly 0.2 m, wall nodes outside them), TURB (sa, vreman, none), SLOT_CELLS (4), LAW (spalding), LAYER (1: the default absorbing layer), STEPS (20000), CHUNK (500), TAU (2.4e-4), ULAT (inlet mean lattice
//! speed, 0.014), LAYER_CELLS (the layer's thickness in cells of this grid; hold the physical thickness: 10/20/40 at
//! 4/8/16 cells), EXT_UP / EXT_DN (m of box added upstream / downstream; 0.5 and 0.5 give a 0.5 m layer room on every grid), ZMAX (wall: z max a wall), TAU (tau+-1/2; hold Re: 2.4e-4/4.8e-4/9.6e-4 at 4/8/16 cells), CPU (1: compare with the CPU each chunk; 0: device only), GPU (1).
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
    // EXT_UP / EXT_DN (m): the box grows upstream / downstream of the 2 m the plate sits in (x 0 to 2, the plate at 0.6 to
    // 0.8), so a layer of fixed physical thickness stays within a fifth of the box's length on every grid.
    let (ext_up, ext_dn) = (env("EXT_UP", 0.0), env("EXT_DN", 0.0));
    let mut d = domain([-ext_up, 0.0, 0.0], [2.0 + ext_dn, 1.0, 1.0], dx);
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
    // SURFACE=1: the slot and the plate are built so the wall-node rows (the fluid cells next to the plate's solid cells, which
    // the on-node wall law makes wall nodes) are exactly the geometric distance apart: 0.2 m across the slot (cells + 1 rows
    // from wall node to wall node) and 0.2 m through the plate. The solid part of the plate is one cell thinner than the
    // geometry in z (the upper block starts a cell higher) and (cells - 1) cells thick in x.
    let surface = env("SURFACE", 0.0) > 0.5;
    // SURFACE=2: the half-way-true slot: the open (fluid) rows are exactly the geometric 0.2 m (cells rows), the wall-node rows
    // lie outside them (cells + 2 rows from node to node), and the solid part of the plate is 0.2 m thick in x.
    let half_way = env("SURFACE", 0.0) > 1.5;
    // PLATE_T (m, default 0.2): the plate's thickness in x for SURFACE=2, so L/w = PLATE_T / 0.2 (0.1 is half a slot width).
    let plate_t = env("PLATE_T", 0.2);
    let (z_hi, x_hi) = if half_way {
        (0.6 + 2.0 * dx, 0.6 + plate_t)
    } else if surface {
        (0.6 + dx, 0.6 + (cells - 1.0) * dx)
    } else {
        (0.6, 0.8)
    };
    let obstacles = vec![
        obstacle("PlateLow", cuboid([0.6, 0.0, 0.0], [x_hi, 1.0, 0.4]), &CUBOID_FACES),
        obstacle("PlateHigh", cuboid([0.6, 0.0, z_hi], [x_hi, 1.0, 1.0]), &CUBOID_FACES),
    ];
    let wall_model = match law.as_str() {
        "sa" => Some(WallModel::SpalartAllmarasBounce),
        "none" => None,
        _ => Some(WallModel::SpaldingBounce),
    };
    // TURB: sa (default), vreman (a subgrid eddy viscosity that vanishes in laminar shear; the wall law is kept), none
    let sa = match std::env::var("TURB").as_deref() {
        Ok("vreman") => Some(Turbulence::Vreman { constant: 0.17, prandtl: 0.9, wall_model }),
        Ok("none") => None,
        _ => Some(Turbulence::SpalartAllmaras { prandtl: 0.9, wall_model, inflow_ratio: 3.0, ddes: None }),
    };
    let p = Procedure::Fluid {
        domain: "Flow".into(),
        end_time: None,
        steps: Some(steps),
        time_step: Some(dt),
        frames: 2,
        magic: None,
        sigma: Some(HrrSigma::Value(0.95)),
        collision: Some(Collision::Hrr),
        turbulence: sa,
        steady: None,
    };
    let m = model(1.0, nu, d, p);
    // MAX_UPDATES raises the work cap (cell updates) for the long fine-grid runs; the production default stays 1e12.
    let mut limits = Limits::default();
    if let Some(u) = std::env::var("MAX_UPDATES").ok().and_then(|v| v.parse::<f64>().ok()) {
        limits.max_cell_updates = u as u64;
    }
    let setup = match Setup::new(&m, None, &obstacles, &limits) {
        Ok(s) => s,
        Err(e) => {
            panic!("setup refused: {e}");
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
    if surface {
        let (nx0, ny0) = (((0.7 + ext_up) / dx) as usize, 0usize);
        let g = *cpu.grid();
        let col: String = (0..nz).map(|k| if Backend::is_fluid(&cpu, g.index([nx0, ny0, k])) { 'f' } else { 'S' }).collect();
        let kc = ((0.2 / dx) as usize).min(nz - 1);
        let row: String = (((0.5 + ext_up) / dx) as usize..((1.0 + ext_up) / dx) as usize)
            .map(|i| if Backend::is_fluid(&cpu, g.index([i, ny0, kc])) { 'f' } else { 'S' })
            .collect();
        println!("SURFACE z column through the plate (k=0..{nz}; f fluid, S solid or wall node): {col}");
        println!("SURFACE x row through the plate at z {:.3} m, x 0.5 m to 1.0 m (f fluid, S solid or wall node): {row}", (kc as f64 + 0.5) * dx);
    }
    let mut gpu = if use_gpu {
        match GpuFlow::with_options(&cpu, Policy::from_env().unwrap(), Options::default()) {
            Ok(g) => {
                println!("DEVICE {}", g.adapter_name());
                Some(g)
            }
            Err(e) => {
                panic!("device refused: {e}");
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
    // the flux planes at x = 0.3 m and 1.7 m, the pressure planes at x = 0.5 m and 0.9 m, on every grid
    let at = |x: f64| ((x + ext_up) / dx).round() as usize;
    let (i_in, i_out) = (at(0.3), at(1.7));
    // the two pressure planes sit at x = 0.5 m and 0.9 m on every grid: the mean of the cell layers either side
    let (i_up, i_dn) = (at(0.5).max(1), at(0.9).max(1));
    println!(
        "PLANES pressure planes at x = {:.4} m and {:.4} m (mean of cell layers {}/{} and {}/{}); the plate spans x 0.6 to 0.8 m",
        dx * i_up as f64 - ext_up,
        dx * i_dn as f64 - ext_up,
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
        let fl = |i: usize| plane(i, &|c| f.density[c] * f.velocity[c][0]) * area;
        let at = |i: usize| 0.5 * (plane(i - 1, &|c| f.density[c]) + plane(i, &|c| f.density[c]));
        let dp = (at(i_up) - at(i_dn)) / 3.0;
        Some((step, um, rmin, rmax, rsum / cnt as f64, fl(i_in), fl(i_out), dp))
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
                            let v = f.velocity[c].iter().map(|v| v * v).sum::<f64>().sqrt();
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
                    let e = (0..3).map(|k| (fa.velocity[c][k] - fb.velocity[c][k]).powi(2)).sum::<f64>().sqrt();
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
    // DIAG4 reads the final fields from the CPU flow when it ran, else from the device (CPU=0): the same prints either way.
    // The device's nu_tilde may be missing, then the viscosity line reads zero.
    let diag4 = if env("DIAG4", 0.0) <= 0.5 {
        None
    } else if cpu_ok {
        match (Backend::fields(&mut cpu), Backend::nu_tilde(&mut cpu)) {
            (Ok(f), Ok(n)) => Some((f, n, Backend::eddy_viscosity_ratio(&cpu), "cpu")),
            _ => None,
        }
    } else if let (true, Some(g)) = (gpu_ok, gpu.as_mut()) {
        Backend::fields(g).ok().map(|f| (f, Backend::nu_tilde(g).ok().flatten(), Backend::eddy_viscosity_ratio(g), "gpu"))
    } else {
        None
    };
    if let Some((fl, nt, evr, source)) = diag4 {
        let nu_lat = setup.tau_plus - 0.5;
        let nu_lat = nu_lat / 3.0;
        println!("DIAG4 source: {source} fields at step {done}");
        {
            let sec = |i: usize| -> (usize, f64, f64, f64, f64) {
                // open cells, total, mean density, mean u_x over open cells (lattice), max |u| (lattice)
                let (mut open, mut rho, mut ux, mut um) = (0usize, 0.0, 0.0, 0.0f64);
                for k in 0..nz {
                    for j in 0..ny {
                        let c = grid.index([i, j, k]);
                        if fluid[c] {
                            open += 1;
                            rho += fl.density[c];
                            ux += fl.velocity[c][0];
                            um = um.max(fl.velocity[c].iter().map(|v| v * v).sum::<f64>().sqrt());
                        }
                    }
                }
                (open, (ny * nz) as f64, rho / open.max(1) as f64, ux / open.max(1) as f64, um)
            };
            println!(
                "DIAG4 (a) plate plane: the plate occupies x 0.6..0.8 m, slot z 0.4..0.6 m; sections are the mean over fluid cells of the whole section"
            );
            for x in [0.55, 0.65, 0.7, 0.75] {
                let i = at(x);
                let (open, tot, _, ux, um) = sec(i);
                let frac = open as f64 / tot;
                let flux = ux * open as f64 / tot;
                println!(
                    "DIAG4 (a) x {x:.2} (layer {i}): open {open}/{tot} = {frac:.4}; mean u_x over open cells {ux:.5}; section flux/area {flux:.5} lattice; max|u| in section {um:.5}; U_in/fraction (nominal inlet mean {u_in}) {:.5}; measured inlet mean (plane x 0.3) {:.5} -> /fraction {:.5}",
                    u_in / frac,
                    sec(i_in).3,
                    sec(i_in).3 / frac
                );
            }
            let (mut gm, mut gi) = (0.0f64, [0usize; 3]);
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        let c = grid.index([i, j, k]);
                        if fluid[c] {
                            let v = fl.velocity[c].iter().map(|v| v * v).sum::<f64>().sqrt();
                            if v > gm {
                                gm = v;
                                gi = [i, j, k];
                            }
                        }
                    }
                }
            }
            println!("DIAG4 (a) global max|u| {gm:.5} lattice at cell {gi:?} (x {:.3} m)", dx * (gi[0] as f64 + 0.5) - ext_up);
            // K_slot: the plate's pressure drop over 0.5 u_nom^2, u_nom = the volume flow over the nominal opening (0.2 m of the
            // 1 m height, the geometric slot rather than the open-cell count); the flow is the inlet section's, x 0.3 m
            let flow_per_area = {
                let mut q = 0.0;
                for k in 0..nz {
                    for j in 0..ny {
                        let c = grid.index([i_in, j, k]);
                        if fluid[c] {
                            q += fl.density[c] * fl.velocity[c][0];
                        }
                    }
                }
                q / (ny * nz) as f64
            };
            let dens = |i: usize| sec(i).2;
            let dp_end = (0.5 * (dens(i_up - 1) + dens(i_up)) - 0.5 * (dens(i_dn - 1) + dens(i_dn))) / 3.0;
            let u_nom = flow_per_area / 0.2;
            println!(
                "DIAG4 (e) K_slot = dp/(0.5 u_nom^2) = {:.3} (dp {dp_end:.5e}, u_nom {u_nom:.5} = inlet flow per section area {flow_per_area:.5} / 0.2); dp/(0.5 u_peak^2) = {:.3} (peak {gm:.5}); open-cell fraction at the plate {:.4}",
                dp_end / (0.5 * u_nom * u_nom),
                dp_end / (0.5 * gm * gm),
                sec(at(0.7)).0 as f64 / sec(at(0.7)).1
            );
            {
                let i = at(0.7);
                let (open, tot, _, _, _) = sec(i);
                let u_open = flow_per_area / (open as f64 / tot);
                println!(
                    "DIAG4 (f) surface-true: K_w = dp/(0.5 (Q/0.2 m wall-to-wall)^2) = {:.3}; K_fluid (= K_open = K_slot (open fraction / 0.2)^2) = dp/(0.5 (Q/fluid-cell width)^2) = {:.3} (fluid width {:.4} m of the 0.2 wall-to-wall, mean u {u_open:.5}); dp per head of the peak {:.3}",
                    dp_end / (0.5 * u_nom * u_nom),
                    dp_end / (0.5 * u_open * u_open),
                    open as f64 / tot,
                    dp_end / (0.5 * gm * gm)
                );
            }
            if gm > 0.10 {
                println!("DIAG4 (e) WARNING true max|u| {gm:.4} lattice is above 0.10 (Mach {:.3})", gm * 3.0f64.sqrt());
            }
            println!(
                "DIAG4 (b) section-mean density by station (x m, layer, rho-1, jet heads of that drop = 3 drop / u_ref^2 uses u_ref in the line above):"
            );
            let stations = [
                ("inlet face", 0usize),
                ("end of inlet layer (x 0)", at(0.0)),
                ("one slot width upstream (x 0.4)", at(0.4)),
                ("plate plane (x 0.7)", at(0.7)),
                ("one slot width downstream (x 1.0)", at(1.0)),
                ("start of outlet layer (x 2.0)", at(2.0)),
                ("outlet face", nx - 1),
            ];
            for (name, i) in stations {
                let (_, _, rho, _, _) = sec(i);
                println!(
                    "DIAG4 (b) {name:34} layer {i:4} x {:7.4} m  rho-1 {:+.5e}  p-p_out {:+.5e} lattice",
                    dx * (i as f64 + 0.5) - ext_up,
                    rho - 1.0,
                    (rho - sec(nx - 1).2) / 3.0
                );
            }
            for i in [0usize, 1, 2, 5, 10, at(0.0) + 3, at(0.3)] {
                let (open, tot, rho, ux, _) = sec(i);
                println!(
                    "DIAG4 (b) inlet check layer {i}: open {open}/{tot}, mean u_x over open cells {ux:.5}, flux/area {:.5}, rho-1 {:+.4e}",
                    ux * open as f64 / tot,
                    rho - 1.0
                );
            }
            for i in [0usize, 5] {
                let row: Vec<String> = (0..nz).map(|k| format!("{:.4}", fl.velocity[grid.index([i, 0, k])][0])).collect();
                println!("DIAG4 (b) u_x(z) at j=0, layer {i}: {}", row.join(" "));
            }
            for t in 0..=10 {
                let i = at(-0.5 + 0.3 * t as f64).min(nx - 1);
                println!("DIAG4 (b) profile x {:6.3} rho-1 {:+.5e}", dx * (i as f64 + 0.5) - ext_up, sec(i).2 - 1.0);
            }
            let (mut rmax, mut ri) = (0.0f64, [0usize; 3]);
            let (mut jmax, mut ji) = (0.0f64, [0usize; 3]);
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        let c = grid.index([i, j, k]);
                        if !fluid[c] {
                            continue;
                        }
                        let chi = nt.as_ref().and_then(|v| v.get(c)).copied().unwrap_or(0.0) / nu_lat;
                        let fv1 = chi.powi(3) / (chi.powi(3) + 7.1f64.powi(3));
                        let r = chi * fv1;
                        if r > rmax {
                            rmax = r;
                            ri = [i, j, k];
                        }
                        // in the jet: cells downstream of the plate, 0.8 to 1.2 m
                        let x = dx * (i as f64 + 0.5) - ext_up;
                        if (0.8..1.2).contains(&x) && r > jmax {
                            jmax = r;
                            ji = [i, j, k];
                        }
                    }
                }
            }
            let slot_w = cells; // cells
            println!(
                "DIAG4 (c) nu_lat {nu_lat:.3e}; backend eddy_viscosity_ratio (max nu_t/nu, last step) {:.3}; from nu_tilde: max nu_t/nu {rmax:.3} at {ri:?} (x {:.3} m); in the jet region x 0.8..1.2: max {jmax:.3} at {ji:?}; nu_t/(U_in * slot width) there = {:.4e} (U_in {u_in}, slot {slot_w} cells), molecular nu/(U_in * slot width) = {:.4e}",
                evr,
                dx * (ri[0] as f64 + 0.5) - ext_up,
                jmax * nu_lat / (u_in * slot_w),
                nu_lat / (u_in * slot_w)
            );
        }
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
