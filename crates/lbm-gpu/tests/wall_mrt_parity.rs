//! CPU/device parity of the WALL_MRT study knob (MRT in the wall layer of the hybrid collision) on three geometries: the square
//! duct (8 cells, half-way box walls), the 2 x 4 pocket cut into a plane channel's wall, and the driven slit bed (3D floor, wedge walls,
//! free layer, an impulsive stream over the slit). Written for the GPU and `#[ignore]`: run it on the machine whose device is to be
//! checked, one case at a time, and never beside another GPU job.
//!
//! `WALL_MRT=0|1` (default 1) sets the knob on the CPU flow, which the device reads. Each case advances both backends in chunks and prints
//! max |du| (lattice units) and max |drho| over the fluid cells, then the device's field checksum (a hash of the f32 bits of the
//! density and velocity), which is what "off is bit-identical" is read from: run the same case with `WALL_MRT=0` on the commit before
//! the port (the knob call removed) and compare the checksums. Bars as for the sigma = 1 port: |du| < 2e-5, |drho| < 1e-6
//! (`PARITY_DU`, `PARITY_DRHO`).
//!
//! `cargo test -p fieldcraft-lbm-gpu --release --test wall_mrt_parity -- --ignored --nocapture <duct|pocket|bed>`
#[path = "../../lbm/tests/common/mod.rs"]
mod common;

use common::*;
use fieldcraft_gpu::select::Policy;
use fieldcraft_lbm::backend::Backend;
use fieldcraft_lbm::{Limits, Obstacle, Setup};
use fieldcraft_lbm_gpu::{GpuFlow, Options};
use fieldcraft_model::{Collision, FaceCondition, HrrSigma, Procedure, Turbulence};

const PERIODIC: FaceCondition = FaceCondition::Periodic {};

fn env(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

/// What a case hands back: the setup, its obstacles, the steps to run and the body force (lattice units) as a function of the step.
struct Case {
    setup: Setup,
    obstacles: Vec<Obstacle>,
    steps: u64,
    force: Box<dyn Fn(u64) -> [f64; 3]>,
}

fn fluid_procedure(dt: f64, tau_minus_half: f64, sigma: f64) -> Procedure {
    // MRT_ALL=1: the control with the MRT collision everywhere (the knob then does nothing)
    let all = env("MRT_ALL", 0.0) != 0.0;
    let mut p = collide(1e6 * dt, dt, None, if all { Collision::Mrt } else { Collision::Hrr });
    if let Procedure::Fluid { sigma: s, turbulence, .. } = &mut p {
        if !all {
            *s = Some(HrrSigma::Value(sigma));
        }
        if tau_minus_half < 5e-3 - 1e-12 {
            *turbulence = Some(Turbulence::SpalartAllmaras { prandtl: 0.9, wall_model: None, inflow_ratio: 0.0, ddes: None });
        }
    }
    p
}

fn finish(m: fieldcraft_model::Model, obstacles: Vec<Obstacle>, steps: u64, force: Box<dyn Fn(u64) -> [f64; 3]>) -> Result<Case, String> {
    let setup = Setup::new(&m, None, &obstacles, &Limits::default()).map_err(|e| e.to_string())?;
    Ok(Case { setup, obstacles, steps, force })
}

/// The square duct, `w` cells across, 4 cells long (periodic): the fluid cells 1..=w of a box `w + 2` cells across, periodic all round, with four
/// solid slabs for the walls (half-way walls on solid cells, so the wall layer exists).
fn duct(w: usize, tau_minus_half: f64, sigma: f64) -> Result<Case, String> {
    let nu_lat = tau_minus_half / 3.0;
    let dx = 1.0 / w as f64;
    let dt = nu_lat * dx * dx;
    let n = (w + 2) as f64 * dx;
    let big = 3.0 * n;
    let obstacles = vec![
        obstacle("WallYLow", cuboid([-big, -dx, -dx], [big, dx, n + dx]), &CUBOID_FACES),
        obstacle("WallYHigh", cuboid([-big, (w + 1) as f64 * dx, -dx], [big, n + dx, n + dx]), &CUBOID_FACES),
        obstacle("WallZLow", cuboid([-big, -dx, -dx], [big, n + dx, dx]), &CUBOID_FACES),
        obstacle("WallZHigh", cuboid([-big, -dx, (w + 1) as f64 * dx], [big, n + dx, n + dx]), &CUBOID_FACES),
    ];
    let mut d = domain([0.0; 3], [4.0 * dx, n, n], dx);
    for f in [&mut d.faces.x_min, &mut d.faces.x_max, &mut d.faces.y_min, &mut d.faces.y_max, &mut d.faces.z_min, &mut d.faces.z_max] {
        *f = PERIODIC;
    }
    d.obstacles = obstacles.iter().map(|o| o.body.clone()).collect();
    let g_lat = 12.0 * nu_lat * 0.02 / (w * w) as f64;
    d.acceleration = [g_lat * dx / (dt * dt), 0.0, 0.0];
    let m = model(1.0, 1.0, d, fluid_procedure(dt, tau_minus_half, sigma));
    finish(m, obstacles, env("PARITY_STEPS", 2000.0) as u64, Box::new(move |_| [g_lat, 0.0, 0.0]))
}

/// The plane channel (periodic in x and z) with a pocket 2 cells wide and 4 deep cut into the upper wall.
fn pocket(tau_minus_half: f64, sigma: f64) -> Result<Case, String> {
    let w = 8usize;
    let nu_lat = tau_minus_half / 3.0;
    let dx = 1.0 / w as f64;
    let dt = nu_lat * dx * dx;
    let (lx, ly, lz) = (16.0 * dx, (w + 4 + 4) as f64 * dx, 4.0 * dx);
    let ho = w as f64 * dx;
    let (x0, x1, depth) = (4.0 * dx, 6.0 * dx, 4.0 * dx);
    let obstacles = vec![
        obstacle("WallLeft", cuboid([-3.0 * lx, ho, -dx], [x0, ly, lz + dx]), &CUBOID_FACES),
        obstacle("WallRight", cuboid([x1, ho, -dx], [3.0 * lx, ly, lz + dx]), &CUBOID_FACES),
        obstacle("WallCap", cuboid([x0, ho + depth, -dx], [x1, ly, lz + dx]), &CUBOID_FACES),
    ];
    let mut d = domain([0.0; 3], [lx, ly, lz], dx);
    for f in [&mut d.faces.x_min, &mut d.faces.x_max, &mut d.faces.y_min, &mut d.faces.y_max, &mut d.faces.z_min, &mut d.faces.z_max] {
        *f = PERIODIC;
    }
    d.obstacles = obstacles.iter().map(|o| o.body.clone()).collect();
    let g_lat = 12.0 * nu_lat * 0.02 / (w * w) as f64;
    d.acceleration = [g_lat * dx / (dt * dt), 0.0, 0.0];
    let m = model(1.0, 1.0, d, fluid_procedure(dt, tau_minus_half, sigma));
    finish(m, obstacles, env("PARITY_STEPS", 3000.0) as u64, Box::new(move |_| [g_lat, 0.0, 0.0]))
}

/// The driven slit bed: a slit 2 wide and 30 deep cut into the upper wall of the channel, a floor two layers thick (the slit's x = 4 row one
/// layer lower), wedge side walls above z = 5 that retreat one cell per two layers, free layers above z = 8 under a lid, a closed end. An
/// impulsive stream along x over the slit: a constant force `peak / ramp` for `ramp` steps, then none.
fn bed(tau_minus_half: f64, sigma: f64, peak: f64, ramp: u64) -> Result<Case, String> {
    let (w, slit_d, nz, floor, wedge_at, top) = (8usize, 30usize, 12usize, 2usize, 5usize, 8usize);
    let nu_lat = tau_minus_half / 3.0;
    let dx = 1.0 / w as f64;
    let dt = nu_lat * dx * dx;
    let (lx, ly, lz) = (16.0 * dx, (w + 4 + slit_d) as f64 * dx, nz as f64 * dx);
    let ho = w as f64 * dx;
    let (x0, x1) = (4.0 * dx, 6.0 * dx);
    let zz = |k: usize| k as f64 * dx;
    let big = 3.0 * lx;
    let mut obstacles = vec![
        obstacle("Floor0", cuboid([-big, -dx, -dx], [big, ly + dx, zz(1)]), &CUBOID_FACES),
        obstacle("FloorUnderChannel", cuboid([-big, -dx, zz(1)], [big, ho, zz(floor)]), &CUBOID_FACES),
        obstacle("FloorSlitWall", cuboid([x1, ho, zz(1)], [big, ly + dx, zz(floor)]), &CUBOID_FACES),
        obstacle("FloorSlitLeftWall", cuboid([-big, ho, zz(1)], [x0, ly + dx, zz(floor)]), &CUBOID_FACES),
        obstacle("Lid", cuboid([-big, -dx, zz(nz - 1)], [big, ly + dx, zz(nz) + dx]), &CUBOID_FACES),
    ];
    for k in floor..top {
        let extra = if k >= wedge_at { ((k - wedge_at) / 2 + 1) as f64 * dx } else { 0.0 };
        let name = |s: &str| -> &'static str { Box::leak(format!("{s}{k}").into_boxed_str()) };
        obstacles.push(obstacle(name("WallLeft"), cuboid([-big, ho, zz(k)], [x0 - extra, ly + dx, zz(k + 1)]), &CUBOID_FACES));
        obstacles.push(obstacle(name("WallRight"), cuboid([x1 + extra, ho, zz(k)], [big, ly + dx, zz(k + 1)]), &CUBOID_FACES));
        obstacles.push(obstacle(
            name("WallCap"),
            cuboid([x0 - extra, ho + slit_d as f64 * dx, zz(k)], [x1 + extra, ly + dx, zz(k + 1)]),
            &CUBOID_FACES,
        ));
    }
    let mut d = domain([0.0; 3], [lx, ly, lz], dx);
    for f in [&mut d.faces.x_min, &mut d.faces.x_max, &mut d.faces.y_min, &mut d.faces.y_max, &mut d.faces.z_min, &mut d.faces.z_max] {
        *f = PERIODIC;
    }
    d.obstacles = obstacles.iter().map(|o| o.body.clone()).collect();
    let m = model(1.0, 1.0, d, fluid_procedure(dt, tau_minus_half, sigma));
    let a = peak / ramp.max(1) as f64;
    finish(m, obstacles, env("PARITY_STEPS", 1500.0) as u64, Box::new(move |t| if t < ramp { [a, 0.0, 0.0] } else { [0.0; 3] }))
}

/// A hash of the f32 bits of the density and velocity of every cell, for the "off is bit-identical" comparison.
fn checksum(density: &[f64], velocity: &[[f64; 3]]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    let mut mix = |v: f32| {
        for b in v.to_bits().to_le_bytes() {
            h = (h ^ u64::from(b)).wrapping_mul(0x100000001b3);
        }
    };
    for (r, u) in density.iter().zip(velocity) {
        mix(*r as f32);
        u.iter().for_each(|v| mix(*v as f32));
    }
    h
}

fn parity(name: &str, case: Result<Case, String>) -> Result<(), String> {
    let case = case?;
    let mut cpu = fieldcraft_lbm::Flow::new(&case.setup, &case.obstacles).map_err(|e| e.to_string())?;
    let on = env("WALL_MRT", 1.0) != 0.0 && env("MRT_ALL", 0.0) == 0.0;
    cpu.set_wall_layer_mrt(on);
    // WALL_CORNER=1: the corner study (cells with three or more solid axis faces relax to equilibrium), read by the device too
    // WALL_TAU_FLOOR=<f>: the wall layer's floor on tau - 1/2 (with WALL_MRT)
    cpu.set_wall_tau_floor(env("WALL_TAU_FLOOR", 0.0));
    cpu.set_wall_corner_relax(env("WALL_CORNER", 0.0) != 0.0 && env("MRT_ALL", 0.0) == 0.0);
    let mut gpu = GpuFlow::with_options(&cpu, Policy::from_env().map_err(|e| e.to_string())?, Options::default()).map_err(|e| e.to_string())?;
    let (tol_u, tol_r) = (env("PARITY_DU", 2e-5), env("PARITY_DRHO", 1e-6));
    let chunk = env("PARITY_CHUNK", 250.0) as u64;
    let (mut t, mut worst_u, mut worst_r) = (0u64, 0.0f64, 0.0f64);
    println!("WALL_MRT-PARITY {name}: device {}, {} cells, knob {}", gpu.adapter_name(), cpu.grid().len(), if on { "on" } else { "off" });
    while t < case.steps {
        let c = chunk.min(case.steps - t);
        let a = (case.force)(t);
        Backend::set_acceleration(&mut cpu, a).map_err(|e| e.to_string())?;
        Backend::set_acceleration(&mut gpu, a).map_err(|e| e.to_string())?;
        Backend::advance(&mut cpu, c).map_err(|e| e.to_string())?;
        Backend::advance(&mut gpu, c).map_err(|e| e.to_string())?;
        t += c;
        let (fa, fb) = (Backend::fields(&mut cpu).map_err(|e| e.to_string())?, Backend::fields(&mut gpu).map_err(|e| e.to_string())?);
        let (mut du, mut dr, mut at) = (0.0f64, 0.0f64, 0usize);
        for i in 0..fa.velocity.len() {
            if !Backend::is_fluid(&cpu, i) {
                continue;
            }
            let e = (0..3).map(|k| (fa.velocity[i][k] - fb.velocity[i][k]).powi(2)).sum::<f64>().sqrt();
            if e > du {
                (du, at) = (e, i);
            }
            dr = dr.max((fa.density[i] - fb.density[i]).abs());
        }
        println!("WALL_MRT-PARITY {name} step {t}: largest |du| at cell {:?}", cpu.grid().coords(at));
        worst_u = worst_u.max(du);
        worst_r = worst_r.max(dr);
        println!(
            "WALL_MRT-PARITY {name} step {t}: max |du| {du:.3e}, max |drho| {dr:.3e}, device checksum {:016x}",
            checksum(&fb.density, &fb.velocity)
        );
    }
    println!("WALL_MRT-PARITY {name} worst: max |du| {worst_u:.3e} (bar {tol_u:.0e}), max |drho| {worst_r:.3e} (bar {tol_r:.0e})");
    if worst_u < tol_u && worst_r < tol_r { Ok(()) } else { Err(format!("{name}: |du| {worst_u:.3e}, |drho| {worst_r:.3e}")) }
}

#[test]
#[ignore = "GPU: run on the machine whose device is to be checked, alone on it (see the module docs)"]
fn duct_parity() {
    if let Err(e) = parity("duct", duct(8, 5e-3, 0.95)) {
        panic!("{e}");
    }
}

#[test]
#[ignore = "GPU: run on the machine whose device is to be checked, alone on it (see the module docs)"]
fn pocket_parity() {
    if let Err(e) = parity("pocket", pocket(5e-4, 0.95)) {
        panic!("{e}");
    }
}

#[test]
#[ignore = "GPU: run on the machine whose device is to be checked, alone on it (see the module docs)"]
fn bed_parity() {
    if let Err(e) = parity("bed", bed(1.8e-4, 0.95, env("PEAK", 0.11), 1000)) {
        panic!("{e}");
    }
}
