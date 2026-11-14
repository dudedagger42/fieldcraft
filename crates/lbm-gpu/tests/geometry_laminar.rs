//! Local harness (not for the gate): laminar flows with an exact answer, to see what a staircase wall and a corner do to the
//! flow rate, on the device in chunks and compared with the CPU. Two cases, each its own test:
//!
//! - `rotated_channel_laminar`: Poiseuille flow between two plane walls turned `DIR_A:DIR_B` against the grid (2:1 is 26.57
//!   degrees, 1:1 is 45, 1:0 is the grid-aligned control), the channel width H = `CELLS` cells measured between the wall planes.
//!   The box is periodic, sized so the channel closes on itself after one box diagonal (a box `a m` by `b m` cells); the walls
//!   are slabs turned to the angle, as many copies as the box wraps through. Reference: u(s) = g s (H - s) / (2 nu), flow rate
//!   g H^3 / (12 nu) per unit span, peak g H^2 / (8 nu).
//! - `square_duct_laminar`: Poiseuille flow in a square duct H x H, H = `CELLS` cells, four slabs round it. Reference: the
//!   rectangular-duct series (flow rate g H^4 c / nu with c = 0.0351..., f Re = 56.91 with the Darcy factor), peak at the centre.
//!
//! Both print one line of errors per sample (`step N:`) and a `RESULT` line at the end; nothing asserts, so a bad number never
//! stops the run. All fields are in lattice units; the lattice viscosity is `NU` (0.05) and the exact mean speed 0.02.
//!
//! Environment: CELLS (8: cells across the channel or duct), DIR_A / DIR_B (2, 1: the channel's direction in cells, rotated
//! channel only), PHASE (0: the channel's offset against the grid, in cells, rotated channel only), WALL_MIN (4: the least wall
//! thickness in cells; the box takes the smallest wrap that fits it), STEPS (40 x CELLS^2), CHUNK (500; 20 on the laptop GPU),
//! PRINT (the sample interval in steps, 2000), NU (0.05), COLLISION (trt, the same as the half-way channel test; hrr runs sigma
//! 0.95), TURB (smag, the half-way channel test's Smagorinsky 0.1; none), WALL (none: plain half-way bounce-back; spalding or sa:
//! the half-way wall law of that name, with SA as the turbulence), CPU (1), GPU (1: a refused device is reported and the run
//! goes on with the CPU).
// A local harness: index loops over the grid and nested ifs read better than iterator chains here.
#![allow(clippy::needless_range_loop, clippy::collapsible_if)]
#[path = "../../lbm/tests/common/mod.rs"]
mod common;

use common::*;
use fieldcraft_gpu::select::Policy;
use fieldcraft_lbm::backend::{Backend, Fields};
use fieldcraft_lbm::{Limits, Obstacle, Setup};
use fieldcraft_lbm_gpu::{GpuFlow, Options};
use fieldcraft_mesh::surface::BoundarySurface;
use fieldcraft_model::{Collision, FaceCondition, HrrSigma, Procedure, Turbulence, WallModel};

fn env(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn env_s(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.to_string())
}

const PERIODIC: FaceCondition = FaceCondition::Periodic {};

/// A parallelepiped from `origin` along three edge vectors, faces tagged as [`cuboid`]'s (0 and 1 along the first edge, 2 and 3
/// along the second, 4 and 5 along the third); each triangle turned to face away from the centre.
fn parallelepiped(origin: [f64; 3], e: [[f64; 3]; 3]) -> BoundarySurface {
    let corner = |k: usize| [0, 1, 2].map(|c| origin[c] + (0..3).map(|a| if k >> a & 1 == 1 { e[a][c] } else { 0.0 }).sum::<f64>());
    let positions: Vec<[f64; 3]> = (0..8).map(corner).collect();
    let centre = [0, 1, 2].map(|c| origin[c] + 0.5 * (e[0][c] + e[1][c] + e[2][c]));
    let mut tris = Vec::new();
    let mut tags = Vec::new();
    for axis in 0..3 {
        let (t1, t2) = ((axis + 1) % 3, (axis + 2) % 3);
        for side in 0..2u32 {
            let v = |a: u32, b: u32| ((side as usize) << axis | (a as usize) << t1 | (b as usize) << t2) as u32;
            tris.push([v(0, 0), v(1, 0), v(1, 1)]);
            tris.push([v(0, 0), v(1, 1), v(0, 1)]);
            tags.extend([2 * axis as u32 + side; 2]);
        }
    }
    surface(positions, tris, tags, move |c, _| [c[0] - centre[0], c[1] - centre[1], c[2] - centre[2]])
}

/// The turbulence model and the wall law the knobs name.
fn closure() -> (Option<Turbulence>, Option<WallModel>, String) {
    let wall = env_s("WALL", "none");
    let turb = env_s("TURB", "smag");
    let model = match wall.as_str() {
        "spalding" => Some(WallModel::SpaldingBounceSurface),
        "sa" => Some(WallModel::SpalartAllmarasBounceSurface),
        _ => None,
    };
    let t = if wall == "sa" {
        Some(Turbulence::SpalartAllmaras { prandtl: 0.9, wall_model: model, inflow_ratio: 3.0, ddes: None })
    } else if turb == "none" && model.is_none() {
        None
    } else {
        Some(Turbulence::Smagorinsky { constant: 0.1, prandtl: 0.9, van_driest: None, wall_model: model })
    };
    (t, model, format!("collision {}; turbulence {turb}; wall law {wall}", env_s("COLLISION", "trt")))
}

/// What a run compares: the lattice, the exact answer and the fluid cells.
struct Probe {
    cells: [usize; 3],
    fluid: Vec<bool>,
    /// The exact velocity along the flow at each cell (lattice), and the flow direction there.
    exact: Vec<f64>,
    direction: [f64; 3],
    /// Cells of cross-section times the flow rate's reference (the exact mean speed times the area), in lattice units.
    flow_reference: f64,
    /// The cells along the flow that the sum of `u` is divided by (the box length along the channel, or the span).
    along: f64,
    peak: f64,
}

/// Errors of one field: the flow rate against the exact one, the peak against the exact velocity at the same cell, the largest profile error, the largest cross-flow speed
/// (all as fractions of the exact peak or flow), and the mean speed.
fn errors(p: &Probe, f: &Fields) -> Option<[f64; 5]> {
    let [nx, ny, nz] = p.cells;
    let (mut sum, mut top, mut top_exact, mut worst, mut cross) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for c in 0..nx * ny * nz {
        if !p.fluid[c] {
            continue;
        }
        let u = f.velocity[c];
        if !u.iter().all(|v| v.is_finite()) {
            return None;
        }
        let along = u[0] * p.direction[0] + u[1] * p.direction[1] + u[2] * p.direction[2];
        let side = (u[0] - along * p.direction[0]).powi(2) + (u[1] - along * p.direction[1]).powi(2) + (u[2] - along * p.direction[2]).powi(2);
        sum += along;
        top = top.max(along);
        top_exact = top_exact.max(p.exact[c]);
        worst = worst.max((along - p.exact[c]).abs());
        cross = cross.max(side.sqrt());
    }
    let q = sum / p.along;
    Some([q / p.flow_reference - 1.0, top / top_exact - 1.0, worst / p.peak, cross / p.peak, q / (p.flow_reference / 0.02)])
}

/// Run `cpu` (and the device, when asked) for `steps` in `chunk`s and print the errors at every sample.
fn drive(name: &str, cpu: fieldcraft_lbm::Flow, probe: &Probe, steps: u64, extra: &str) {
    let chunk = env("CHUNK", 500.0) as u64;
    let print_every = env("PRINT", 2000.0) as u64;
    let (use_cpu, use_gpu) = (env("CPU", 1.0) > 0.5, env("GPU", 1.0) > 0.5);
    let mut cpu = cpu;
    let mut gpu = if use_gpu {
        match Policy::from_env()
            .map_err(|e| e.to_string())
            .and_then(|p| GpuFlow::with_options(&cpu, p, Options::default()).map_err(|e| e.to_string()))
        {
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
    println!("CASE {name}; {extra}; exact mean speed 0.02 lattice, peak {:.5}; steps {steps}, chunk {chunk}", probe.peak);
    let (mut done, mut last_print) = (0u64, 0u64);
    let (mut last_cpu, mut last_gpu): (Option<[f64; 5]>, Option<[f64; 5]>) = (None, None);
    let (mut cpu_ok, mut gpu_ok) = (use_cpu, gpu.is_some());
    while done < steps && (cpu_ok || gpu_ok) {
        let c = chunk.min(steps - done);
        if cpu_ok {
            if let Err(e) = Backend::advance(&mut cpu, c) {
                println!("EVENT cpu stopped in the chunk ending at step {}: {e}", done + c);
                cpu_ok = false;
            }
        }
        if gpu_ok {
            if let Some(g) = gpu.as_mut() {
                if let Err(e) = Backend::advance(g, c) {
                    println!("EVENT device stopped in the chunk ending at step {}: {e}", done + c);
                    gpu_ok = false;
                }
            }
        }
        done += c;
        if done - last_print >= print_every || done >= steps {
            last_print = done;
            let a = if cpu_ok { Backend::fields(&mut cpu).ok().and_then(|f| errors(probe, &f)) } else { None };
            let b = if gpu_ok { gpu.as_mut().and_then(|g| Backend::fields(g).ok()).and_then(|f| errors(probe, &f)) } else { None };
            last_cpu = a.or(last_cpu);
            last_gpu = b.or(last_gpu);
            let show = |who: &str, e: Option<[f64; 5]>| match e {
                Some(e) => {
                    format!("{who} flow {:+.3} % peak {:+.3} % profile {:.3} % cross {:.3} %", 100.0 * e[0], 100.0 * e[1], 100.0 * e[2], 100.0 * e[3])
                }
                None => format!("{who} -"),
            };
            println!("step {done}: {}; {}", show("cpu", a), show("gpu", b));
        }
    }
    for (who, e) in [("cpu", last_cpu), ("gpu", last_gpu)] {
        if let Some(e) = e {
            println!(
                "RESULT {name} {who}: flow rate {:+.4} % of exact, peak {:+.4} % of exact, largest profile error {:.4} % of the exact peak, largest cross-flow {:.4} % of it; mean speed {:.6} lattice (exact 0.02)",
                100.0 * e[0],
                100.0 * e[1],
                100.0 * e[2],
                100.0 * e[3],
                e[4]
            );
        }
    }
}

/// A model of the box with the periodic faces and the closure the knobs name, run `steps` at lattice viscosity `nu_lat`.
struct Build {
    dx: f64,
    dt: f64,
    nu_lat: f64,
    u_mean_phys: f64,
}

impl Build {
    fn new(cells: usize) -> Build {
        let (nu, height) = (1.0, 1.0);
        let nu_lat = env("NU", 0.05);
        let dx = height / cells as f64;
        let dt = nu_lat * dx * dx / nu;
        Build { dx, dt, nu_lat, u_mean_phys: 0.02 * dx / dt }
    }

    fn model(&self, lo: [f64; 3], hi: [f64; 3], obstacles: &[Obstacle], accel: [f64; 3], steps: u64) -> (Setup, fieldcraft_lbm::Flow, String) {
        let mut d = domain(lo, hi, self.dx);
        for f in [&mut d.faces.x_min, &mut d.faces.x_max, &mut d.faces.y_min, &mut d.faces.y_max, &mut d.faces.z_min, &mut d.faces.z_max] {
            *f = PERIODIC;
        }
        d.obstacles = obstacles.iter().map(|o| o.body.clone()).collect();
        d.acceleration = accel;
        let hrr = env_s("COLLISION", "trt") == "hrr";
        let mut p = collide(steps as f64 * self.dt, self.dt, None, if hrr { Collision::Hrr } else { Collision::Trt });
        let (turb, _, label) = closure();
        if let Procedure::Fluid { turbulence, sigma, .. } = &mut p {
            *turbulence = turb;
            if hrr {
                *sigma = Some(HrrSigma::Value(0.95));
            }
        }
        let m = model(1.0, 1.0, d, p);
        let setup = Setup::new(&m, None, obstacles, &Limits::default()).unwrap_or_else(|e| panic!("setup refused: {e}"));
        let flow = fieldcraft_lbm::Flow::new(&setup, obstacles).unwrap_or_else(|e| panic!("flow refused: {e}"));
        (setup, flow, label)
    }
}

#[test]
#[ignore = "local: GPU run, approved per run"]
fn rotated_channel_laminar() {
    let h = env("CELLS", 8.0) as usize;
    let (a, b) = (env("DIR_A", 2.0) as usize, env("DIR_B", 1.0) as usize);
    let phase = env("PHASE", 0.0);
    let wall_min = env("WALL_MIN", 4.0);
    let steps = env("STEPS", 40.0 * (h * h) as f64) as u64;
    let g = Build::new(h);
    // The box in cells, the pitch of the wall family (the distance between neighbouring copies of a wall) and the direction.
    let (nx, ny, pitch, dir) = if b == 0 {
        let ny = (h as f64 + wall_min).ceil() as usize;
        (8usize, ny, ny as f64, [1.0, 0.0])
    } else {
        let len = ((a * a + b * b) as f64).sqrt();
        let mut m = 1usize;
        while ((a * b * m) as f64 / len) < h as f64 + wall_min {
            m += 1;
        }
        (a * m, b * m, (a * b * m) as f64 / len, [a as f64 / len, b as f64 / len])
    };
    let nz = 4usize;
    // WIDTH (cells, default CELLS): the channel's width, which may lie between whole cells; the grid stays CELLS.
    let width = env("WIDTH", h as f64);
    let (dx, ho) = (g.dx, width * g.dx);
    let n = [-dir[1], dir[0]];
    let (lx, ly, lz) = (nx as f64 * dx, ny as f64 * dx, nz as f64 * dx);
    let p = pitch * dx;
    let s0 = phase * dx;
    let diag = (lx * lx + ly * ly).sqrt();
    // The wall copies: the strip s in [ho + s0, p + s0] and its shifts by whole boxes along y that reach into the box.
    let mut obstacles = Vec::new();
    for k in -4i32..=4 {
        let (lo, hi) = (ho + s0 + k as f64 * p, p + s0 + k as f64 * p);
        // The box's own range of s is [-p, p] for a turned channel, [0, ly] for the grid-aligned control.
        let (box_lo, box_hi) = if b == 0 { (0.0, ly) } else { (-p, p) };
        if hi < box_lo + 0.5 * dx || lo > box_hi - 0.5 * dx {
            continue;
        }
        let t = 3.0 * diag;
        let origin = [n[0] * lo - dir[0] * t, n[1] * lo - dir[1] * t, -dx];
        let e = [[2.0 * t * dir[0], 2.0 * t * dir[1], 0.0], [n[0] * (hi - lo), n[1] * (hi - lo), 0.0], [0.0, 0.0, lz + 2.0 * dx]];
        let name = format!("Wall{}", k + 4);
        obstacles.push(obstacle(&name, parallelepiped(origin, e), &CUBOID_FACES));
    }
    // The body force along the channel gives the exact mean speed 0.02 lattice: g = 12 nu U / H^2.
    let gmag = 12.0 * 1.0 * g.u_mean_phys / (ho * ho);
    let (setup, flow, label) = g.model([0.0; 3], [lx, ly, lz], &obstacles, [gmag * dir[0], gmag * dir[1], 0.0], steps);
    let grid = *flow.grid();
    let [gx, gy, gz] = grid.cells;
    let fluid: Vec<bool> = (0..gx * gy * gz).map(|c| Backend::is_fluid(&flow, c)).collect();
    // The channel's length in the box: one diagonal for a turned channel, the box length for the control.
    let along_cells = if b == 0 { nx as f64 } else { ((nx * nx + ny * ny) as f64).sqrt() };
    // The exact velocity at each fluid cell from its distance to the wall plane (s in cells, measured from the plane's start).
    let hh = width;
    let g_lat = 12.0 * g.nu_lat * 0.02 / (hh * hh);
    let mut exact = vec![0.0; gx * gy * gz];
    let mut spans = (f64::INFINITY, f64::NEG_INFINITY, 0usize);
    for k in 0..gz {
        for j in 0..gy {
            for i in 0..gx {
                let c = grid.index([i, j, k]);
                if !fluid[c] {
                    continue;
                }
                let (x, y) = ((i as f64 + 0.5) * dx, (j as f64 + 0.5) * dx);
                let s = (n[0] * x + n[1] * y - s0).rem_euclid(p) / dx;
                spans = (spans.0.min(s), spans.1.max(s), spans.2 + 1);
                exact[c] = g_lat / (2.0 * g.nu_lat) * s * (hh - s);
            }
        }
    }
    let probe = Probe {
        cells: [gx, gy, gz],
        fluid,
        exact,
        direction: [dir[0], dir[1], 0.0],
        flow_reference: 0.02 * hh,
        along: along_cells * gz as f64,
        peak: g_lat * hh * hh / (8.0 * g.nu_lat),
    };
    let angle = dir[1].atan2(dir[0]).to_degrees();
    let extra = format!(
        "rotated channel {h} cells wide at {angle:.2} degrees (direction {a}:{b}); box {nx}x{ny}x{nz} cells; wall pitch {pitch:.3} cells, wall thickness {:.3} cells; {} wall copies; phase {phase}; {} fluid cells, their distance from the plane's start {:.3} to {:.3} cells; {label}; setup steps {}",
        pitch - hh,
        obstacles.len(),
        spans.2,
        spans.0,
        spans.1,
        setup.steps
    );
    drive("rotated_channel", flow, &probe, steps, &extra);
}

/// The rectangular-duct series for a square duct of side 1 in a body force 1 and viscosity 1: (the flow rate per H^4, the
/// velocity at (y, z) per H^2 with y, z from the centre in units of the side).
fn duct_flow_rate() -> f64 {
    let pi = std::f64::consts::PI;
    let sum: f64 = (0..200).map(|m| 2 * m + 1).map(|n| ((n as f64) * pi / 2.0).tanh() / (n as f64).powi(5)).sum();
    (1.0 - 192.0 / pi.powi(5) * sum) / 12.0
}

fn duct_velocity(y: f64, z: f64) -> f64 {
    // The side is 1, so a = 1/2; the term is (-1)^((n-1)/2) [1 - cosh(n pi z / 2a) / cosh(n pi / 2)] cos(n pi y / 2a) / n^3.
    let pi = std::f64::consts::PI;
    let a = 0.5;
    let sum: f64 = (0..100)
        .map(|m| 2 * m + 1)
        .map(|n| {
            let nf = n as f64;
            let sign = if (n - 1) / 2 % 2 == 0 { 1.0 } else { -1.0 };
            // cosh ratio written to stay finite at large n.
            let ratio = ((nf * pi * (z.abs() - 0.5) / (2.0 * a)).exp() + (-nf * pi * (z.abs() + 0.5) / (2.0 * a)).exp()) / (1.0 + (-nf * pi).exp());
            sign * (1.0 - ratio) * (nf * pi * y / (2.0 * a)).cos() / nf.powi(3)
        })
        .sum();
    16.0 * a * a / pi.powi(3) * sum
}

#[test]
#[ignore = "local: GPU run, approved per run"]
fn square_duct_laminar() {
    let h = env("CELLS", 8.0) as usize;
    let steps = env("STEPS", 40.0 * (h * h) as f64) as u64;
    let g = Build::new(h);
    let (dx, ho) = (g.dx, h as f64 * g.dx);
    let span = 4.0 * dx;
    let (lo, hi) = (-2.0 * dx, ho + 2.0 * dx);
    // Four slabs round the duct; the two on the sides reach past the floor and roof so the corners are solid.
    let big = 1.0;
    let obstacles = vec![
        obstacle("Floor", cuboid([-big, lo, -big], [span + big, 0.0, ho + big]), &CUBOID_FACES),
        obstacle("Roof", cuboid([-big, ho, -big], [span + big, hi, ho + big]), &CUBOID_FACES),
        obstacle("Left", cuboid([-big, -big, lo], [span + big, ho + big, 0.0]), &CUBOID_FACES),
        obstacle("Right", cuboid([-big, -big, ho], [span + big, ho + big, hi]), &CUBOID_FACES),
    ];
    // The body force gives the exact mean speed 0.02 lattice: U = c g H^2 / nu.
    let c = duct_flow_rate();
    let hh = h as f64;
    let g_lat = 0.02 * g.nu_lat / (c * hh * hh);
    let gmag = g_lat * dx / (g.dt * g.dt);
    let (setup, flow, label) = g.model([0.0, lo, lo], [span, hi, hi], &obstacles, [gmag, 0.0, 0.0], steps);
    let grid = *flow.grid();
    let [gx, gy, gz] = grid.cells;
    let fluid: Vec<bool> = (0..gx * gy * gz).map(|c| Backend::is_fluid(&flow, c)).collect();
    let mut exact = vec![0.0; gx * gy * gz];
    let mut open = 0usize;
    for k in 0..gz {
        for j in 0..gy {
            for i in 0..gx {
                let cell = grid.index([i, j, k]);
                if !fluid[cell] {
                    continue;
                }
                open += 1;
                // The centre of the cell from the duct's centre, in units of the side.
                let (y, z) = (((j as f64 + 0.5) * dx + lo - 0.5 * ho) / ho, ((k as f64 + 0.5) * dx + lo - 0.5 * ho) / ho);
                exact[cell] = duct_velocity(y, z) * g_lat * hh * hh / g.nu_lat;
            }
        }
    }
    // The peak at the centre: the series at 0, 0.
    let peak = duct_velocity(0.0, 0.0) * g_lat * hh * hh / g.nu_lat;
    let probe = Probe { cells: [gx, gy, gz], fluid, exact, direction: [1.0, 0.0, 0.0], flow_reference: 0.02 * hh * hh, along: gx as f64, peak };
    let extra = format!(
        "square duct {h} x {h} cells; box {gx}x{gy}x{gz} cells; {open} fluid cells (the duct holds {} of them); exact flow rate coefficient {c:.6} (f Re Darcy {:.3}); {label}; setup steps {}",
        gx * h * h,
        2.0 / c,
        setup.steps
    );
    drive("square_duct", flow, &probe, steps, &extra);
}
