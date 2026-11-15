//! Local harness (not for the gate): the open-domain check of the interpolated walls. A laminar plane channel along x with a
//! velocity inlet (uniform, the exact mean speed) and a pressure outlet, walls at rest as one obstacle slab whose face cuts the
//! cells (`WIDTH` cells wide, 8.5 by default, on an 8-cell grid), periodic across y and z. Run on the CPU and, when asked, on the device.
//! The global mass correction is defined from the walls alone, so it must not fight the inlet and the outlet; this prints what
//! shows it: the outlet flux over the inlet flux (want 1 within 1e-3), the pressure gradient of the middle third of the channel
//! over the exact one (12 nu U / H^2 per cell, want 1 within 1 %), the mean density in every window (steady) and the size of the
//! correction per 10 000 steps. Fields are in lattice units; the lattice viscosity is `NU` (0.05) and the mean speed `UMEAN` (0.02).
//! Environment: CELLS (8), WIDTH (8.5), NX (64: the channel's length in cells), STEPS (20000), WINDOW (2500), NU, UMEAN, QUAD (1: the
//! quadratic rule, 0 linear), MASSFIX (2: global; 0 off, 1 local), INTERP (1: 0 runs the plain half-way wall), CPU (1), GPU (1).
// A local harness: index loops over the grid and nested ifs read better than iterator chains here.
#![allow(clippy::needless_range_loop, clippy::collapsible_if)]
#[path = "../../lbm/tests/common/mod.rs"]
mod common;

use common::*;
use fieldcraft_gpu::select::Policy;
use fieldcraft_lbm::backend::{Backend, Fields};
use fieldcraft_lbm::interpolated::{MassFix, WallOrder};
use fieldcraft_lbm_gpu::{GpuFlow, Options};
use fieldcraft_model::{Collision, FaceCondition, Outflow, Procedure, Profile};

fn env(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

/// What one window of one backend shows.
struct Window {
    mean_rho: f64,
    flux_ratio: f64,
    /// The gradient over the exact one for the flow the middle plane carries.
    gradient_ratio: f64,
    /// The gradient over the exact one for the nominal mean speed and width.
    nominal_ratio: f64,
    /// The flux through the plane 2 over the one through the middle.
    inlet_share: f64,
}

#[test]
#[ignore = "local: GPU run, approved per run"]
fn interpolated_wall_open_channel() {
    let cells = env("CELLS", 8.0);
    let width = env("WIDTH", 8.5);
    let nx = env("NX", 64.0) as usize;
    let steps = env("STEPS", 20000.0) as u64;
    let window = env("WINDOW", 2500.0) as u64;
    let (nu_lat, u_lat) = (env("NU", 0.05), env("UMEAN", 0.02));
    let (use_cpu, use_gpu) = (env("CPU", 1.0) > 0.5, env("GPU", 1.0) > 0.5);
    let interp = env("INTERP", 1.0) > 0.5;
    // The grid: the cell is 1 / CELLS of a metre, the lattice viscosity fixes the time step (nu = 1 m^2/s).
    let dx = 1.0 / cells;
    let dt = nu_lat * dx * dx;
    let u_phys = u_lat * dx / dt;
    let ny = (width + 4.0).ceil() as usize;
    let nz = 4usize;
    let (lx, ly, lz) = (nx as f64 * dx, ny as f64 * dx, nz as f64 * dx);
    let ho = width * dx;
    // The wall: from the channel's width to the box's far face (periodic across y, so the channel's other side is the face at 0,
    // which lies on a cell face), through the inlet and the outlet.
    let obstacles = vec![obstacle("Wall", cuboid([-2.0 * dx, ho, -dx], [lx + 2.0 * dx, ly, lz + dx]), &CUBOID_FACES)];
    let mut d = domain([0.0; 3], [lx, ly, lz], dx);
    for f in [&mut d.faces.y_min, &mut d.faces.y_max, &mut d.faces.z_min, &mut d.faces.z_max] {
        *f = FaceCondition::Periodic {};
    }
    d.faces.x_min = FaceCondition::Velocity { velocity: [u_phys, 0.0, 0.0], profile: Profile::Uniform };
    d.faces.x_max = FaceCondition::Pressure { pressure: 0.0, outflow: Outflow::Uniform };
    d.initial_velocity = [u_phys, 0.0, 0.0];
    d.obstacles = vec!["Wall".into()];
    let p: Procedure = collide(steps as f64 * dt, dt, None, Collision::Trt);
    let m = model(1.0, 1.0, d, p);
    let setup = fieldcraft_lbm::Setup::new(&m, None, &obstacles, &fieldcraft_lbm::Limits::default()).unwrap_or_else(|e| panic!("setup refused: {e}"));
    let mut cpu = fieldcraft_lbm::Flow::new(&setup, &obstacles).unwrap_or_else(|e| panic!("flow refused: {e}"));
    let mut label = String::from("half-way walls");
    if interp {
        let order = if env("QUAD", 1.0) > 0.5 { WallOrder::Quadratic } else { WallOrder::Linear };
        let r = cpu.interpolate_walls_with(&setup, &obstacles, order).unwrap_or_else(|e| panic!("interpolated walls refused: {e}"));
        let fix = match env("MASSFIX", 2.0) as u32 {
            0 => MassFix::Off,
            1 => MassFix::Local,
            _ => MassFix::Global,
        };
        cpu.set_wall_mass_correction(fix).unwrap_or_else(|e| panic!("{e}"));
        label = format!(
            "interpolated walls ({order:?}, mass fix {fix:?}): {} links, {} interpolated ({} quadratic), {} half way, {} without a second node, {} without a surface",
            r.links, r.interpolated, r.quadratic, r.half_way, r.no_second_node, r.no_surface
        );
    }
    let grid = *cpu.grid();
    let [gx, gy, gz] = grid.cells;
    let fluid: Vec<bool> = (0..gx * gy * gz).map(|c| Backend::is_fluid(&cpu, c)).collect();
    println!(
        "CASE open channel {width} cells wide on a {cells}-cell grid, {gx}x{gy}x{gz} cells, mean speed {u_lat}, viscosity {nu_lat} (relaxation time {}); {label}; steps {steps}, windows of {window}",
        3.0 * nu_lat + 0.5
    );
    let mut gpu = if use_gpu {
        match Policy::from_env().map_err(|e| e.to_string()).and_then(|p| GpuFlow::with_options(&cpu, p, Options::default()).map_err(|e| e.to_string())) {
            Ok(g) => {
                println!("DEVICE {}", g.adapter_name());
                Some(g)
            }
            Err(e) => {
                println!("DEVICE refused: {e}; the run goes on with the CPU");
                None
            }
        }
    } else {
        None
    };
    // The exact pressure gradient as a density gradient per cell: rho = 3 p, dp/dx = -12 nu U / H^2.
    let exact_gradient = -3.0 * 12.0 * nu_lat * u_lat / (width * width);
    let measure = |f: &Fields| -> Window {
        let (mut sum, mut count) = (0.0, 0usize);
        for c in 0..gx * gy * gz {
            if fluid[c] {
                sum += f.density[c];
                count += 1;
            }
        }
        // The flux through the plane i: the sum of rho u_x over its fluid cells.
        let flux = |i: usize| -> f64 {
            let mut s = 0.0;
            for k in 0..gz {
                for j in 0..gy {
                    let c = grid.index([i, j, k]);
                    if fluid[c] {
                        s += f.density[c] * f.velocity[c][0];
                    }
                }
            }
            s
        };
        // The cross-section's mean density at every plane of the middle third, and a least-squares line through them.
        let (lo, hi) = (gx / 3, 2 * gx / 3);
        let mut pts = Vec::new();
        for i in lo..hi {
            let (mut s, mut n) = (0.0, 0usize);
            for k in 0..gz {
                for j in 0..gy {
                    let c = grid.index([i, j, k]);
                    if fluid[c] {
                        s += f.density[c];
                        n += 1;
                    }
                }
            }
            pts.push((i as f64, s / n.max(1) as f64));
        }
        let n = pts.len() as f64;
        let (mx, my) = (pts.iter().map(|p| p.0).sum::<f64>() / n, pts.iter().map(|p| p.1).sum::<f64>() / n);
        let slope = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum::<f64>() / pts.iter().map(|p| (p.0 - mx) * (p.0 - mx)).sum::<f64>();
        // The exact gradient of the flow the channel carries: 12 nu (rho u) / H^2 per cell with rho u the mean over the width at the
        // middle plane (so the inlet's own rows, which the wall at a fractional width does not match, do not enter).
        let carried = flux(gx / 2) / (gz as f64 * width);
        let exact_carried = -3.0 * 12.0 * nu_lat * carried / (width * width);
        Window { mean_rho: sum / count.max(1) as f64, flux_ratio: flux(gx - 3) / flux(2), gradient_ratio: slope / exact_carried, nominal_ratio: slope / exact_gradient, inlet_share: flux(2) / flux(gx / 2) }
    };
    let (mut done, mut last_created) = (0u64, 0.0);
    let (mut last_cpu, mut last_gpu): (Option<Window>, Option<Window>) = (None, None);
    while done < steps {
        let c = window.min(steps - done);
        if use_cpu {
            Backend::advance(&mut cpu, c).unwrap_or_else(|e| panic!("cpu stopped in the window ending at step {}: {e}", done + c));
        }
        if let Some(g) = gpu.as_mut() {
            Backend::advance(g, c).unwrap_or_else(|e| panic!("device stopped in the window ending at step {}: {e}", done + c));
        }
        done += c;
        let fa = if use_cpu { Backend::fields(&mut cpu).ok() } else { None };
        let fb = gpu.as_mut().and_then(|g| Backend::fields(g).ok());
        let (_, created) = cpu.wall_mass_created();
        let per_10k = (created - last_created) / (c as f64) * 1e4;
        last_created = created;
        let show = |who: &str, w: &Option<Window>| match w {
            Some(w) => format!("{who} mean rho {:.9}, outlet/inlet flux {:.5}, gradient/exact {:.4}", w.mean_rho, w.flux_ratio, w.gradient_ratio),
            None => format!("{who} -"),
        };
        let a = fa.as_ref().map(&measure);
        let b = fb.as_ref().map(&measure);
        let parity = match (&fa, &fb) {
            (Some(x), Some(y)) => {
                let mut du = 0.0f64;
                for cell in 0..gx * gy * gz {
                    if fluid[cell] {
                        du = du.max((0..3).map(|k| (x.velocity[cell][k] - y.velocity[cell][k]).abs()).fold(0.0, f64::max));
                    }
                }
                format!("; largest |u cpu - u gpu| {du:.2e}")
            }
            _ => String::new(),
        };
        println!(
            "step {done}: {}; {}; the walls created {:+.3e} of the mass per 10000 steps ({:+.3e} of the total mass){parity}",
            show("cpu", &a),
            show("gpu", &b),
            per_10k,
            per_10k / cpu.fluid_cells() as f64
        );
        last_cpu = a.or(last_cpu);
        last_gpu = b.or(last_gpu);
    }
    // DIAG=1: the flux through every plane of the cpu run, to see where mass appears (rho u_x summed over the plane's fluid cells).
    if env("DIAG", 0.0) > 0.5 {
        if let Ok(f) = Backend::fields(&mut cpu) {
            let flux = |i: usize| -> f64 {
                let mut s = 0.0;
                for k in 0..gz {
                    for j in 0..gy {
                        let c = grid.index([i, j, k]);
                        if fluid[c] {
                            s += f.density[c] * f.velocity[c][0];
                        }
                    }
                }
                s
            };
            let line: Vec<String> = (0..gx).step_by(4).map(|i| format!("{i}:{:.4}", flux(i) / flux(gx / 2))).collect();
            println!("FLUX per plane / flux at the middle: {}", line.join(" "));
        }
    }
    for (who, w) in [("cpu", &last_cpu), ("gpu", &last_gpu)] {
        if let Some(w) = w {
            println!(
                "RESULT {who}: outlet/inlet flux {:.5} (bar 1 +- 1e-3: {}), mid-third pressure gradient {:+.3} % of exact for the flow carried (bar +- 1 %: {}; {:+.3} % of the nominal one), plane 2 flux {:.5} of the middle's, mean density {:.9}",
                w.flux_ratio,
                if (w.flux_ratio - 1.0).abs() <= 1e-3 { "met" } else { "NOT met" },
                100.0 * (w.gradient_ratio - 1.0),
                if (w.gradient_ratio - 1.0).abs() <= 0.01 { "met" } else { "NOT met" },
                100.0 * (w.nominal_ratio - 1.0),
                w.inlet_share,
                w.mean_rho
            );
        }
    }
}
