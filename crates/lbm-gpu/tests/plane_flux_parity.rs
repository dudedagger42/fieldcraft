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
/// floor and roof. The planes beside the open layers read the same on both backends, with the open arrival cells left out and counted.
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
    let (a, b) = (cpu.population_flux_parts(0, 1, None).expect("cpu"), gpu.population_flux_parts(0, 1, None).expect("gpu"));
    assert!(a.skipped > 0 && a.skipped == b.skipped, "{} {}", a.skipped, b.skipped);
    worst = worst.max((a.flux - b.flux).abs() / reference);
    println!("OPEN-PLANE-FLUX worst diff/flux {worst:.2e}");
    assert!(worst < 1e-5, "{worst}");
}

/// A 3D open duct as the duct runs call it: inlet at x min, pressure outlet at x max, no-slip walls all round, a cell mask (a
/// window of the plane), repeated calls between odd-sized chunks. Both backends must agree at every print and the values must move.
#[test]
#[ignore = "GPU: run on the machine whose device is to be checked, alone on it (see the module docs)"]
fn open_duct_plane_flux_over_time() {
    use fieldcraft_model::{Outflow, Profile};
    let dx = 1.0 / 12.0;
    let mut d = domain([0.0; 3], [2.0, 1.0, 0.75], dx);
    d.faces.x_min = FaceCondition::Velocity { velocity: [0.5, 0.0, 0.0], profile: Profile::Uniform };
    d.faces.x_max = FaceCondition::Pressure { pressure: 0.0, outflow: Outflow::Convective };
    for f in [&mut d.faces.y_min, &mut d.faces.y_max, &mut d.faces.z_min, &mut d.faces.z_max] {
        *f = FaceCondition::Wall {};
    }
    let dt = 0.05 * dx / 0.5;
    let m = model(1.0, 0.02, d, collide(1e6 * dt, dt, None, Collision::Hrr));
    let setup = Setup::new(&m, None, &[], &Limits::default()).expect("setup");
    let mut cpu = Flow::new(&setup, &[]).expect("flow");
    let mut gpu = GpuFlow::with_options(&cpu, Policy::from_env().expect("policy"), Options::default()).expect("device flow");
    let g = *cpu.grid();
    let n = g.len();
    let window: Vec<bool> = (0..n).map(|c| (2..g.cells[1] - 2).contains(&g.coords(c)[1]) && (2..g.cells[2] - 2).contains(&g.coords(c)[2])).collect();
    let mut worst = 0.0f64;
    let mut seen: Vec<f64> = Vec::new();
    let mut reference = 0.0f64;
    for chunk in 0..24 {
        let steps = 37 + 2 * chunk as u64; // odd, so the device alternates its storage phase
        Backend::advance(&mut cpu, steps).expect("cpu");
        Backend::advance(&mut gpu, steps).expect("gpu");
        for (what, a, b) in [
            ("x plane 6", cpu.population_flux(0, 6, None), gpu.population_flux(0, 6, None)),
            ("x plane 12 window", cpu.population_flux(0, 12, Some(&window)), gpu.population_flux(0, 12, Some(&window))),
            ("high face, 1 in, window", cpu.open_face_flux(0, true, 1, Some(&window)), gpu.open_face_flux(0, true, 1, Some(&window))),
            ("y plane 5 window", cpu.population_flux(1, 5, Some(&window)), gpu.population_flux(1, 5, Some(&window))),
        ] {
            let (a, b) = (a.expect("cpu"), b.expect("gpu"));
            if what == "x plane 6" {
                seen.push(b);
                reference = reference.max(a.abs());
            }
            println!("DUCT-FLUX chunk {chunk} {what}: cpu {a:+.6e} device {b:+.6e}");
            worst = worst.max((a - b).abs() / reference.max(1e-9));
        }
    }
    let (a, b) = (cpu.population_flux_parts(1, 5, Some(&window)).expect("cpu"), gpu.population_flux_parts(1, 5, Some(&window)).expect("gpu"));
    assert!(a.skipped > 0 && a.skipped == b.skipped, "{} {}", a.skipped, b.skipped);
    // a passage-style mask: the cells of the plane's high layer only (net flux, both backends)
    let one_layer: Vec<bool> = (0..n).map(|c| g.coords(c)[0] == 12 && (3..g.cells[1] - 3).contains(&g.coords(c)[1])).collect();
    let (a, b) = (cpu.population_flux(0, 12, Some(&one_layer)).expect("cpu"), gpu.population_flux(0, 12, Some(&one_layer)).expect("gpu"));
    let both: Vec<bool> = (0..n).map(|c| (3..g.cells[1] - 3).contains(&g.coords(c)[1])).collect();
    let c2 = cpu.population_flux(0, 12, Some(&both)).expect("cpu");
    println!("DUCT-FLUX one-layer mask: cpu {a:+.6e} device {b:+.6e} both layers {c2:+.6e}");
    assert!((a - c2).abs() < 1e-12 && (a - b).abs() / reference.max(1e-9) < 1e-5 && a.abs() > 0.1, "{a} {b} {c2}");
    let moved = seen.windows(2).filter(|w| (w[0] - w[1]).abs() > 1e-9).count();
    println!("DUCT-FLUX worst diff/flux {worst:.2e}, device value changed in {moved} of {} prints", seen.len() - 1);
    assert!(moved > 10 && worst < 1e-4, "moved {moved}, worst {worst}");
}
