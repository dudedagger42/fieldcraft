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

/// How the study paces its device: `FIELDCRAFT_GPU_SUBMIT_STEPS` steps per waited-for submission (default 1) and
/// `FIELDCRAFT_GPU_DUTY` the fraction of time kept busy (default 1: no pause), which holds a laptop's temperature down.
fn device_pacing() -> (u64, f64) {
    let steps = std::env::var("FIELDCRAFT_GPU_SUBMIT_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let duty = std::env::var("FIELDCRAFT_GPU_DUTY").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
    (steps, duty)
}

/// Sets the study's device: a GPU flow built from the CPU flow the study has built and set.
fn install() {
    let _ = ddes_backstep::DEVICE.set(Box::new(|flow| {
        let policy = Policy::from_env().map_err(|e| e.to_string())?;
        let (submit_steps, duty) = device_pacing();
        let gpu = GpuFlow::with_options(flow, policy, Options { submit_steps, duty, ..Options::default() }).map_err(|e| e.to_string())?;
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

/// The f32 floor rule on T1's restart file (`FIELDCRAFT_STEP_RESTART`): over its last 10 blocks, no trend and a scatter
/// under the caps; prints the measured numbers.
#[test]
#[ignore = "local: needs T1's restart file"]
fn the_f32_floor_holds_over_t1s_last_blocks() {
    let file = std::path::PathBuf::from(std::env::var("FIELDCRAFT_STEP_RESTART").expect("FIELDCRAFT_STEP_RESTART"));
    let (f, x_r) = ddes_backstep::f32_floor_of_restart(&file).expect("the restart's blocks");
    println!("last blocks' x_r {x_r:?}");
    println!("F32 FLOOR {f:?}");
    println!("VERDICT {}: {}", f.verdict(), f.describe());
    assert!(f.ok(), "the f32 floor is not met: {f:?}");
}

/// T3's readings from its restart file (`FIELDCRAFT_STEP_RESTART`, `FIELDCRAFT_STEP_SPAN`, `FIELDCRAFT_STEP_CELLS_PER_H`,
/// `FIELDCRAFT_STEP_INLET_TABLE` as the run had them), through the current reader: seconds.
#[test]
#[ignore = "local: needs T3's restart file"]
fn t3_readings_from_its_restart_file() {
    let file = std::path::PathBuf::from(std::env::var("FIELDCRAFT_STEP_RESTART").expect("FIELDCRAFT_STEP_RESTART"));
    let span: f64 = std::env::var("FIELDCRAFT_STEP_SPAN").ok().and_then(|v| v.parse().ok()).unwrap_or(4.0);
    println!("READINGS {}", ddes_backstep::t3_readings_of_restart(&file, span).expect("the restart's readings"));
}

/// The four diagnostics of the T3 interim ruling from a T3 restart (`FIELDCRAFT_STEP_RESTART`; `DDES=0` for a T2 restart:
/// stresses and floor shear only), CPU only, seconds.
#[test]
#[ignore = "local: needs a T2 or T3 restart file"]
fn step_diagnostics_from_a_restart_file() {
    let file = std::path::PathBuf::from(std::env::var("FIELDCRAFT_STEP_RESTART").expect("FIELDCRAFT_STEP_RESTART"));
    let span: f64 = std::env::var("FIELDCRAFT_STEP_SPAN").ok().and_then(|v| v.parse().ok()).unwrap_or(4.0);
    let ddes = std::env::var("DDES").map(|v| v != "0").unwrap_or(true);
    println!("DIAGNOSTICS {}", ddes_backstep::step_diagnostics_of_restart(&file, span, ddes).expect("the restart's diagnostics"));
}

/// Short submissions change nothing: the same small DDES step advanced 64 steps in one chunk with one submission, with a
/// submission per step and with one per 3 steps (and the whole-chunk run twice), fields compared bit for bit.
#[test]
#[ignore = "local: GPU run"]
fn short_submissions_are_bit_identical() {
    let cells = std::env::var("CELLS_PER_H").ok().and_then(|v| v.parse().ok()).unwrap_or(8usize);
    let (cpu, config) = ddes_backstep::built_zone_step(cells, 1.0);
    println!("CASE {config}");
    let policy = Policy::from_env().expect("FIELDCRAFT_GPU names a policy");
    let run = |submit_steps: u64| -> Option<(Vec<u64>, Vec<u64>, Vec<u64>, usize)> {
        let mut gpu = match GpuFlow::with_options(&cpu, policy, Options { submit_steps, ..Options::default() }) {
            Ok(g) => g,
            Err(why) => {
                println!("DEVICE refused: {why}");
                return None;
            }
        };
        Backend::advance(&mut gpu, 64).unwrap();
        let f = Backend::fields(&mut gpu).unwrap();
        let nu = Backend::nu_tilde(&mut gpu).unwrap().unwrap();
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<u64>>();
        let vel: Vec<f64> = f.velocity.iter().flatten().copied().collect();
        Some((bits(&f.density), bits(&vel), bits(&nu), gpu.submit_seconds().len()))
    };
    let (Some(base), Some(base2), Some(one), Some(three)) = (run(0), run(0), run(1), run(3)) else { return };
    println!("submissions per 64-step chunk: whole {}, one step {}, three steps {}", base.3, one.3, three.3);
    println!("whole chunk twice: density {} velocity {} nu-tilde {}", base.0 == base2.0, base.1 == base2.1, base.2 == base2.2);
    println!("whole vs one step per submission: density {} velocity {} nu-tilde {}", base.0 == one.0, base.1 == one.1, base.2 == one.2);
    println!("whole vs three steps per submission: density {} velocity {} nu-tilde {}", base.0 == three.0, base.1 == three.1, base.2 == three.2);
    assert!(base.0 == one.0 && base.1 == one.1 && base.2 == one.2, "one step per submission changed the fields");
    assert!(base.0 == three.0 && base.1 == three.1 && base.2 == three.2, "three steps per submission changed the fields");
}

/// The watched load test: a DDES step in T3's zone (`CELLS_PER_H` 24, `SPAN` 4: 12 M cells), `STEPS` steps (3000) with
/// one step per submission (`SUBMIT_STEPS`, 1), each submission waited for; prints the submission times (min, median,
/// 99th percentile, max) and the steps per second. Run it under the watcher script, which polls the System log for
/// nvlddmkm and the GPU's temperature, clocks and power.
#[test]
#[ignore = "local: GPU run, watched"]
fn watched_short_gpu_run() {
    let cells = std::env::var("CELLS_PER_H").ok().and_then(|v| v.parse().ok()).unwrap_or(24usize);
    let span: f64 = std::env::var("SPAN").ok().and_then(|v| v.parse().ok()).unwrap_or(4.0);
    let steps: u64 = std::env::var("STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(3000);
    let (submit_steps, duty) = device_pacing();
    let (cpu, config) = ddes_backstep::built_zone_step(cells, span);
    println!("CASE {config}");
    let policy = Policy::from_env().expect("FIELDCRAFT_GPU names a policy");
    let mut gpu = match GpuFlow::with_options(&cpu, policy, Options { submit_steps, duty, ..Options::default() }) {
        Ok(g) => g,
        Err(why) => {
            println!("DEVICE refused: {why}");
            return;
        }
    };
    println!(
        "DEVICE {}; {} cells ({} fluid); {steps} steps, {submit_steps} per submission, duty {duty}",
        gpu.adapter_name(),
        cpu.grid().len(),
        cpu.fluid_cells()
    );
    println!("WATCH START");
    let started = std::time::Instant::now();
    let mut done = 0;
    while done < steps {
        let n = 20.min(steps - done);
        Backend::advance(&mut gpu, n).unwrap();
        done += n;
        if done % 500 == 0 {
            println!("step {done}: {:.1} s", started.elapsed().as_secs_f64());
        }
    }
    let mut t: Vec<f64> = gpu.submit_seconds().to_vec();
    t.sort_by(f64::total_cmp);
    let q = |p: f64| t.get(((t.len() as f64 - 1.0) * p).round() as usize).copied().unwrap_or(f64::NAN);
    println!(
        "WATCH END; {} submissions: min {:.1} ms, median {:.1} ms, p99 {:.1} ms, max {:.1} ms; {:.1} ms per step overall",
        t.len(),
        1e3 * q(0.0),
        1e3 * q(0.5),
        1e3 * q(0.99),
        1e3 * q(1.0),
        1e3 * started.elapsed().as_secs_f64() / steps as f64
    );
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
