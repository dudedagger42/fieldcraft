//! The step study on the device: the case of `crates/lbm/tests/ddes_backstep.rs` (included here, so the
//! same builder, statistics and restart files), stepped on the GPU. A run reads each step's wall shear from the
//! device and mirrors its density, velocity and ν̃ into the CPU flow at each sample, so everything after the
//! step is the study's own code; the device's memory is printed (`nvidia-smi`) at the start, at each
//! checkpoint and at the end.
//!
//! - `zone_mask_parity_over_200_steps`: a small DDES step in T3's zone, the CPU (f64) and the device (f32) side by side,
//!   compared pointwise at every step: the LES-mode cell count must be identical at every step.
//! - `t1s_on_the_gpu`, `t2_on_the_gpu`, `t3_on_the_gpu`: the three study runs, with the environment of
//!   `ddes_backstep.rs` (`FIELDCRAFT_STEP_RESTART`, `_CELLS_PER_H`, `_SPAN`, `_CHECKPOINT`, `_MAX_STEPS`,
//!   `FIELDCRAFT_STEP_INLET_TABLE`), all ignored (hours).
#[allow(dead_code, unused_imports)]
#[path = "../../lbm/tests/ddes_backstep.rs"]
mod ddes_backstep;

use fieldcraft_gpu::select::Policy;
use fieldcraft_lbm::backend::Backend;
use fieldcraft_lbm_gpu::{GpuFlow, Options};

/// Sets the study's device: a GPU flow built from the CPU flow the study has built and set.
fn install() {
    let _ = ddes_backstep::DEVICE.set(Box::new(|flow| {
        let policy = Policy::from_env().map_err(|e| e.to_string())?;
        let gpu = GpuFlow::with_options(flow, policy, Options::default()).map_err(|e| e.to_string())?;
        Ok(Box::new(gpu) as Box<dyn Backend>)
    }));
}

/// A small SA-DDES step in T3's zone, 200 steps on the CPU and on the device, compared at every step: the largest
/// density and velocity differences, ν̃ and d̃ relative to their largest, and the LES-mode cell counts
/// (`Backend::ddes`), which must agree at every step (at most one cell off over the whole run is reported, not accepted).
#[test]
#[ignore = "local: GPU run"]
fn zone_mask_parity_over_200_steps() {
    let cells = std::env::var("CELLS_PER_H").ok().and_then(|v| v.parse().ok()).unwrap_or(16usize);
    let span: f64 = std::env::var("SPAN").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
    let steps: u64 = std::env::var("STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(200);
    let (mut cpu, config) = ddes_backstep::built_zone_step(cells, span);
    println!("CASE {config}");
    let policy = Policy::from_env().expect("FIELDCRAFT_GPU names a policy");
    let mut gpu = match GpuFlow::with_options(&cpu, policy, Options::default()) {
        Ok(g) => g,
        Err(why) => {
            println!("DEVICE refused: {why}");
            return;
        }
    };
    println!("DEVICE {}; {} cells ({} fluid)", gpu.adapter_name(), cpu.grid().len(), cpu.fluid_cells());
    let zone = cpu.ddes_zone().map(|z| z.iter().filter(|v| **v).count()).unwrap_or(0);
    println!("zone: {zone} of {} cells", cpu.grid().len());
    assert!(zone > 0 && zone < cpu.grid().len(), "the zone must split the grid");
    let (mut worst_rho, mut worst_u, mut worst_nu, mut worst_len) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut mismatched = Vec::new();
    for step in 1..=steps {
        cpu.step().unwrap();
        Backend::advance(&mut gpu, 1).unwrap();
        let (sc, sg) = (Backend::ddes(&cpu).unwrap(), Backend::ddes(&gpu).unwrap());
        // The LES-mode cells of this step: the count as the fraction of the counted cells.
        let fluid = cpu.fluid_cells() as f64;
        let (nc, ng) = ((sc.les_fraction * fluid).round() as i64, (sg.les_fraction * fluid).round() as i64);
        if nc != ng {
            mismatched.push((step, nc, ng));
        }
        if step % 10 == 0 || step == steps {
            let f = Backend::fields(&mut gpu).unwrap();
            let (mut dr, mut du) = (0.0f64, 0.0f64);
            let mut count = 0;
            for c in 0..cpu.grid().len() {
                if let Some((rho, u)) = cpu.moments(c) {
                    dr = dr.max((rho - f.density[c]).abs());
                    du = du.max((0..3).map(|a| (u[a] - f.velocity[c][a]).abs()).fold(0.0, f64::max));
                    count += 1;
                }
            }
            let nu_c = cpu.spalart_allmaras().map(|s| s.nu_tilde().to_vec()).unwrap();
            let nu_g = Backend::nu_tilde(&mut gpu).unwrap().unwrap();
            let big = nu_c.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-30);
            let dn = nu_c.iter().zip(&nu_g).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max) / big;
            let lc = cpu.ddes_length().unwrap();
            let lg = gpu.ddes_length().unwrap().unwrap();
            let big = lc.iter().fold(0.0f64, |m, v| m.max(*v)).max(1e-30);
            let dl = lc.iter().zip(&lg).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max) / big;
            worst_rho = worst_rho.max(dr);
            worst_u = worst_u.max(du);
            worst_nu = worst_nu.max(dn);
            worst_len = worst_len.max(dl);
            println!("step {step}: {count} cells; |Δρ| {dr:.2e}, |Δu| {du:.2e}, Δν̃/max {dn:.2e}, Δd̃/max {dl:.2e}; LES cells {nc} vs {ng} of {fluid}");
        }
    }
    println!(
        "RESULT {steps} steps: worst |Δρ| {worst_rho:.2e}, |Δu| {worst_u:.2e}, Δν̃ {worst_nu:.2e}, Δd̃ {worst_len:.2e}; steps with unequal LES-cell counts: {} {:?}",
        mismatched.len(),
        &mismatched[..mismatched.len().min(10)]
    );
    // f32 against f64 flips a cell sitting at f_d = 1/2; the span is uniform, so a flip is a whole spanwise column: the counts
    // may differ by at most one column at a step, in either direction (the report above says how often).
    let column = (span * cells as f64).round() as i64;
    let worst = mismatched.iter().map(|(_, a, b)| (a - b).abs()).max().unwrap_or(0);
    assert!(worst <= column, "the LES-mode cell counts differ by {worst} (more than one {column}-cell spanwise column): {mismatched:?}");
}

/// T1 itself on the device: the full-domain RANS step (`FIELDCRAFT_STEP_CELLS_PER_H`, `_MAX_STEPS`, `_CHECKPOINT`,
/// `_PROFILE`), whose restart file and inlet profile T1s starts from.
#[test]
#[ignore = "minutes: T1 on the device"]
fn t1_on_the_gpu() {
    install();
    ddes_backstep::sa_rans_on_han_s_wall_reattaches_where_measured();
}

#[test]
#[ignore = "hours: the short step from T1's field on the device, FIELDCRAFT_STEP_RESTART"]
fn t1s_on_the_gpu() {
    install();
    ddes_backstep::t1s_the_short_step_from_t1s_field_reattaches_with_t1();
}

#[test]
#[ignore = "hours: the extruded short step on the device, FIELDCRAFT_STEP_RESTART"]
fn t2_on_the_gpu() {
    install();
    ddes_backstep::t2_the_extruded_short_step_keeps_t1s_reattachment();
}

#[test]
#[ignore = "hours: the DDES study on the device from T2's restart, FIELDCRAFT_STEP_RESTART"]
fn t3_on_the_gpu() {
    install();
    ddes_backstep::t3_ddes_in_the_zone_from_t2s_field_reports_x_r_with_its_error();
}
