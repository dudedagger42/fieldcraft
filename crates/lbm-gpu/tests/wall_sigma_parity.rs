//! CPU/device parity of the σ = 1 rule's wall layer (σ = 1 and the damped density mode, on by default), on the beds where its layer
//! is a wall of obstacle cells (the slot bed, tests/common/slot_bed.rs) and a box face (a square duct, the SA duct's setup at 8 cells).
//! Both backends advance 500 steps from rest, 100 at a time, with the flow's own force; the fields must agree within the bar of the
//! backend (verification.rs: |Δu| < 2e-5 lattice, |Δρ| < 1e-6) and the rule must act: the CPU flow with the rule on differs from the
//! one with it off by more than that bar. Local: needs a GPU (`cargo test -p fieldcraft-lbm-gpu --release --test wall_sigma_parity --
//! --ignored --nocapture`).
#[path = "../../lbm/tests/common/mod.rs"]
mod common;

use common::slot_bed::build;
use common::*;
use fieldcraft_gpu::select::{Policy, gpu_required};
use fieldcraft_lbm::backend::Backend;
use fieldcraft_lbm::{Flow, Limits, Setup};
use fieldcraft_lbm_gpu::{GpuFlow, Options};
use fieldcraft_model::{Collision, FaceCondition, HrrSigma, Procedure, Turbulence};

const STEPS: u64 = 500;
const CHUNK: u64 = 100;

fn duct() -> (Setup, Vec<fieldcraft_lbm::Obstacle>) {
    let cells = 8.0;
    let dx = 1.0 / cells;
    let mut d = domain([0.0; 3], [4.0 * dx, 2.0, 2.0], dx);
    d.faces.x_min = FaceCondition::Periodic {};
    d.faces.x_max = FaceCondition::Periodic {};
    d.acceleration = [2.0, 0.0, 0.0];
    let bulk_guess = 14.6;
    let dt = 0.1 * dx / bulk_guess;
    let sa = Turbulence::SpalartAllmaras { prandtl: 0.9, wall_model: None, inflow_ratio: 3.0, ddes: None };
    let mut p = collide(STEPS as f64 * dt, dt, None, Collision::Hrr);
    if let Procedure::Fluid { turbulence, sigma, .. } = &mut p {
        *turbulence = Some(sa);
        *sigma = Some(HrrSigma::Value(0.99));
    }
    let m = model(1.0, 1.0 / 151.0, d, p);
    let setup = Setup::new(&m, None, &[], &Limits::default()).expect("the duct's setup");
    (setup, Vec::new())
}

fn slot() -> (Setup, Vec<fieldcraft_lbm::Obstacle>) {
    let bed = build(8, 2.4e-3, 0.047, 0.95).expect("the slot bed");
    (bed.setup, bed.obstacles)
}

fn parity(name: &str, setup: &Setup, obstacles: &[fieldcraft_lbm::Obstacle], corner: bool) -> Result<(), String> {
    let mut cpu = Flow::new(setup, obstacles).map_err(|e| e.to_string())?;
    let mut off = Flow::new(setup, obstacles).map_err(|e| e.to_string())?;
    off.set_wall_layer_sigma_one(0);
    // The corner study on every third layer cell (an explicit set, on both CPU flows; the device reads the same mask), against a flow without it.
    let mut plain = Flow::new(setup, obstacles).map_err(|e| e.to_string())?;
    if corner {
        let mask = cpu.wall_layer_mask();
        let cells: Vec<usize> = (0..mask.len()).filter(|&c| mask.get(c).copied().unwrap_or(false) && c % 3 == 0).collect();
        for f in [&mut cpu, &mut off] {
            f.set_wall_corner_relax(true);
            f.set_wall_corner_cells(Some(cells.clone()));
        }
    }
    let mut gpu = match GpuFlow::with_options(&cpu, Policy::from_env().map_err(|e| e.to_string())?, Options::default()) {
        Ok(g) => g,
        Err(e) if !gpu_required() && e.to_string().contains("no GPU device") => {
            println!("skipping: {e}");
            return Ok(());
        }
        Err(e) => return Err(e.to_string()),
    };
    // `Backend::fields` velocities are lattice units already.
    let layer = cpu.wall_sigma_layer_mask().iter().filter(|&&m| m).count();
    println!("{name}: device {}, {} cells, {layer} in the rule's layer", gpu.adapter_name(), cpu.grid().len());
    assert!(layer > 0, "{name}: the rule's layer is empty");
    let (mut worst_u, mut worst_r) = (0.0f64, 0.0f64);
    let mut done = 0;
    while done < STEPS {
        for b in [&mut cpu as &mut dyn Backend, &mut gpu, &mut off, &mut plain] {
            b.advance(CHUNK).map_err(|e| e.to_string())?;
        }
        done += CHUNK;
        let (a, b) = (Backend::fields(&mut cpu).map_err(|e| e.to_string())?, Backend::fields(&mut gpu).map_err(|e| e.to_string())?);
        for c in (0..a.velocity.len()).filter(|&c| Backend::is_fluid(&cpu, c)) {
            let e = [0, 1, 2].map(|k| a.velocity[c][k] - b.velocity[c][k]);
            worst_u = worst_u.max(e.iter().map(|v| v * v).sum::<f64>().sqrt());
            worst_r = worst_r.max((a.density[c] - b.density[c]).abs());
        }
    }
    let (a, o) = (Backend::fields(&mut cpu).map_err(|e| e.to_string())?, Backend::fields(&mut off).map_err(|e| e.to_string())?);
    let rule = (0..a.velocity.len())
        .filter(|&c| Backend::is_fluid(&cpu, c))
        .map(|c| [0, 1, 2].map(|k| (a.velocity[c][k] - o.velocity[c][k]).abs()).into_iter().fold(0.0, f64::max))
        .fold(0.0, f64::max);
    let p = Backend::fields(&mut plain).map_err(|e| e.to_string())?;
    let corner_effect = (0..a.velocity.len())
        .filter(|&c| Backend::is_fluid(&cpu, c))
        .map(|c| [0, 1, 2].map(|k| (a.velocity[c][k] - p.velocity[c][k]).abs()).into_iter().fold(0.0, f64::max))
        .fold(0.0, f64::max);
    println!(
        "{name}: CPU vs device max |Δu| {worst_u:.3e}, max |Δρ| {worst_r:.3e}; rule on vs off (CPU) max |Δu| {rule:.3e}; corner cells on vs off {corner_effect:.3e}"
    );
    if worst_u >= 2e-5 || worst_r >= 1e-6 {
        return Err(format!("{name}: |Δu| {worst_u:.3e} (bar 2e-5), |Δρ| {worst_r:.3e} (bar 1e-6)"));
    }
    if corner && corner_effect <= 1e-9 {
        return Err(format!("{name}: the corner cells did not act (with vs without max |Δu| {corner_effect:.3e})"));
    }
    if rule <= 1e-9 {
        return Err(format!("{name}: the rule did not act (on vs off max |Δu| {rule:.3e})"));
    }
    Ok(())
}

#[test]
#[ignore = "local: run on the machine whose GPU is to be checked"]
fn the_cpu_and_the_device_agree_on_the_wall_layer_rule() {
    let mut failures = Vec::new();
    for (name, (setup, obstacles), corner) in
        [("slot bed, 8 cells", slot(), false), ("square duct, 8 cells", duct(), false), ("slot bed with corner cells", slot(), true)]
    {
        if let Err(e) = parity(name, &setup, &obstacles, corner) {
            failures.push(e);
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("; "));
}
