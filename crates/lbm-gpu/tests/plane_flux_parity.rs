//! CPU/device parity of `population_flux` and `open_face_flux`: a driven plane channel (8 cells between half-way walls, periodic along
//! x and z, hybrid collision, 4,000 steps) read through x planes, through the planes of y (the wall and the channel's middle), with a
//! cell mask and through the box's faces, on both backends. Written for the GPU and `#[ignore]`: run it on the machine whose device is
//! to be checked, alone on it. Bars: the flux of a plane differs by under 1e-5 of the channel's flux (f32 populations), and the
//! device call's cost is printed.
//!
//! `cargo test -p fieldcraft-lbm-gpu --release --test plane_flux_parity -- --ignored --nocapture`
#[path = "../../lbm/tests/common/mod.rs"]
mod common;

use std::time::Instant;

use common::*;
use fieldcraft_gpu::select::Policy;
use fieldcraft_lbm::backend::Backend;
use fieldcraft_lbm::{Flow, Limits, Setup};
use fieldcraft_lbm_gpu::{GpuFlow, Options};
use fieldcraft_model::{Collision, FaceCondition, HrrSigma, Procedure};

fn channel() -> (Setup, Vec<fieldcraft_lbm::Obstacle>, f64) {
    let w = 8usize;
    let dx = 1.0 / w as f64;
    let nu_lat = 0.02;
    let dt = nu_lat * dx * dx;
    let (lx, ly, lz) = (8.0 * dx, (w + 4) as f64 * dx, 4.0 * dx);
    let mut d = domain([0.0; 3], [lx, ly, lz], dx);
    for f in [&mut d.faces.x_min, &mut d.faces.x_max, &mut d.faces.y_min, &mut d.faces.y_max, &mut d.faces.z_min, &mut d.faces.z_max] {
        *f = FaceCondition::Periodic {};
    }
    let wall = obstacle("Wall0", cuboid([-3.0 * lx, w as f64 * dx, -dx], [3.0 * lx, ly, lz + dx]), &CUBOID_FACES);
    d.obstacles = vec![wall.body.clone()];
    let g_lat = 12.0 * nu_lat * 0.02 / (w * w) as f64;
    d.acceleration = [g_lat * dx / (dt * dt), 0.0, 0.0];
    let mut p = collide(1e6 * dt, dt, None, Collision::Hrr);
    if let Procedure::Fluid { sigma, .. } = &mut p {
        *sigma = Some(HrrSigma::Value(0.95));
    }
    let m = model(1.0, 1.0, d, p);
    let setup = Setup::new(&m, None, std::slice::from_ref(&wall), &Limits::default()).expect("setup");
    (setup, vec![wall], 0.02 * w as f64 * 4.0)
}

#[test]
#[ignore = "GPU: run on the machine whose device is to be checked, alone on it (see the module docs)"]
fn plane_flux_parity() {
    let (setup, obstacles, reference) = channel();
    let mut cpu = Flow::new(&setup, &obstacles).expect("flow");
    let mut gpu = GpuFlow::with_options(&cpu, Policy::from_env().expect("policy"), Options::default()).expect("device flow");
    for _ in 0..16 {
        Backend::advance(&mut cpu, 250).expect("cpu");
        Backend::advance(&mut gpu, 250).expect("gpu");
    }
    let g = *cpu.grid();
    let n = g.len();
    let row: Vec<bool> = (0..n).map(|c| g.coords(c)[1] == 3).collect();
    let fluid_rows: Vec<bool> = (0..n).map(|c| g.coords(c)[1] < 8).collect();
    let worst = std::cell::Cell::new(0.0f64);
    let check = |what: &str, a: f64, b: f64| {
        let d = (a - b).abs() / reference;
        println!("PLANE-FLUX {what}: cpu {a:+.6e} device {b:+.6e} diff/flux {d:.2e}");
        worst.set(worst.get().max(d));
    };
    for plane in 0..8 {
        check(&format!("x plane {plane}"), cpu.population_flux(0, plane, None).expect("cpu"), gpu.population_flux(0, plane, None).expect("gpu"));
    }
    check("x plane 3, row 3", cpu.population_flux(0, 3, Some(&row)).expect("cpu"), gpu.population_flux(0, 3, Some(&row)).expect("gpu"));
    check("y plane 4 (middle)", cpu.population_flux(1, 4, None).expect("cpu"), gpu.population_flux(1, 4, None).expect("gpu"));
    check(
        "y plane 8 (wall), fluid rows",
        cpu.population_flux(1, 8, Some(&fluid_rows)).expect("cpu"),
        gpu.population_flux(1, 8, Some(&fluid_rows)).expect("gpu"),
    );
    check("z plane 2", cpu.population_flux(2, 2, None).expect("cpu"), gpu.population_flux(2, 2, None).expect("gpu"));
    check("x low face", cpu.open_face_flux(0, false, 0, None).expect("cpu"), gpu.open_face_flux(0, false, 0, None).expect("gpu"));
    check("x high face", cpu.open_face_flux(0, true, 0, None).expect("cpu"), gpu.open_face_flux(0, true, 0, None).expect("gpu"));
    // the cost: a call against a chunk of steps
    let t = Instant::now();
    let calls = 50;
    for _ in 0..calls {
        gpu.population_flux(0, 3, None).expect("gpu");
    }
    let per_call = t.elapsed().as_secs_f64() / f64::from(calls);
    let t = Instant::now();
    Backend::advance(&mut gpu, 50).expect("gpu");
    let per_step = t.elapsed().as_secs_f64() / 50.0;
    println!(
        "PLANE-FLUX-COST: {:.3e} s per call, {:.3e} s per step ({:.2} steps), worst diff/flux {:.2e}",
        per_call,
        per_step,
        per_call / per_step,
        worst.get()
    );
    // the readback does not change the flow: the same chunk afterwards matches the CPU
    Backend::advance(&mut cpu, 50).expect("cpu");
    check("x plane 3 after 50 more steps", cpu.population_flux(0, 3, None).expect("cpu"), gpu.population_flux(0, 3, None).expect("gpu"));
    assert!(worst.get() < 1e-5, "{}", worst.get());
}

/// A 24 x 12 cell open box (one cell deep, periodic in z): a velocity inlet at x min, a convective pressure outlet at x max, no-slip
/// floor and roof. The planes beside the open layers read the same on both backends and the open layers are refused.
#[test]
#[ignore = "GPU: run on the machine whose device is to be checked, alone on it (see the module docs)"]
fn open_channel_plane_flux_parity() {
    use fieldcraft_model::{Outflow, Profile};
    let dx = 1.0 / 12.0;
    let mut d = domain([0.0; 3], [2.0, 1.0, dx], dx);
    d.faces.x_min = FaceCondition::Velocity { velocity: [0.5, 0.0, 0.0], profile: Profile::Uniform };
    d.faces.x_max = FaceCondition::Pressure { pressure: 0.0, outflow: Outflow::Convective };
    d.faces.y_min = FaceCondition::Wall {};
    d.faces.y_max = FaceCondition::Wall {};
    d.faces.z_min = FaceCondition::Periodic {};
    d.faces.z_max = FaceCondition::Periodic {};
    let dt = 0.05 * dx / 0.5;
    let m = model(1.0, 0.02, d, collide(1e6 * dt, dt, None, Collision::Hrr));
    let setup = Setup::new(&m, None, &[], &Limits::default()).expect("setup");
    let mut cpu = Flow::new(&setup, &[]).expect("flow");
    let mut gpu = GpuFlow::with_options(&cpu, Policy::from_env().expect("policy"), Options::default()).expect("device flow");
    for _ in 0..8 {
        Backend::advance(&mut cpu, 250).expect("cpu");
        Backend::advance(&mut gpu, 250).expect("gpu");
    }
    let g = *cpu.grid();
    let n = g.len();
    let mut worst = 0.0f64;
    let reference = cpu.population_flux(0, 12, None).expect("cpu").abs();
    for plane in [2usize, 6, 12, 18, g.cells[0] - 2] {
        let (a, b) = (cpu.population_flux(0, plane, None).expect("cpu"), gpu.population_flux(0, plane, None).expect("gpu"));
        println!("OPEN-PLANE-FLUX x plane {plane}: cpu {a:+.6e} device {b:+.6e}");
        worst = worst.max((a - b).abs() / reference);
    }
    for high in [false, true] {
        let (a, b) = (cpu.open_face_flux(0, high, 1, None).expect("cpu"), gpu.open_face_flux(0, high, 1, None).expect("gpu"));
        println!("OPEN-PLANE-FLUX face high={high}: cpu {a:+.6e} device {b:+.6e}");
        worst = worst.max((a - b).abs() / reference);
    }
    let inside: Vec<bool> = (0..n).map(|c| g.coords(c)[0] >= 1 && g.coords(c)[0] + 1 < g.cells[0]).collect();
    let (a, b) = (cpu.population_flux(0, 1, Some(&inside)).expect("cpu"), gpu.population_flux(0, 1, Some(&inside)).expect("gpu"));
    worst = worst.max((a - b).abs() / reference);
    assert!(cpu.population_flux(0, 1, None).is_err() && gpu.population_flux(0, 1, None).is_err());
    assert!(cpu.open_face_flux(0, false, 0, None).is_err() && gpu.open_face_flux(0, false, 0, None).is_err());
    println!("OPEN-PLANE-FLUX worst diff/flux {worst:.2e}");
    assert!(worst < 1e-5, "{worst}");
}
