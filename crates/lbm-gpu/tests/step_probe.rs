//! The step study's flow on the device, timed: the case of `crates/lbm/tests/ddes_backstep.rs`
//! (included here, so the same builder) as `FIELDCRAFT_STEP_*` describe it, built on the CPU, copied to the
//! device, and stepped `STEPS` times in chunks of `CHUNK` (a chunk times the cells under about 5e8 keeps the
//! queue short on a laptop GPU: 20 at 12 M cells). It prints the setup times, the cell counts, the seconds per step and the
//! cell updates per second. No shear strips, no statistics: a cost probe, not the study. A device that
//! refuses the flow says why and the test ends.
//!
//! Laptop GPU, short domain, 4 H span, 24 cells per H, 2000 steps:
//! `FIELDCRAFT_STEP_SHORT=1 FIELDCRAFT_STEP_CELLS_PER_H=24 FIELDCRAFT_STEP_SPAN=4 FIELDCRAFT_STEP_MAX_CELLS=20000000 STEPS=2000 CHUNK=20
//! cargo test --release -p fieldcraft-lbm-gpu --test step_probe -- --ignored --nocapture step_probe`.
#[allow(dead_code, unused_imports)]
#[path = "../../lbm/tests/ddes_backstep.rs"]
mod ddes_backstep;

use fieldcraft_gpu::select::Policy;
use fieldcraft_lbm::backend::Backend;
use fieldcraft_lbm_gpu::{GpuFlow, Options};

fn env(k: &str, d: u64) -> u64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

/// The process's peak resident memory (Linux `/proc`); elsewhere the device's own counter is the
/// reading to take (the task manager's dedicated GPU memory).
fn peak_host_memory() -> String {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| s.lines().find_map(|l| l.strip_prefix("VmHWM:").map(|v| v.trim().to_string())))
        .unwrap_or_else(|| "n/a on this system".to_string())
}

#[test]
#[ignore = "local: GPU run, approved per run"]
fn step_probe() {
    let (steps, chunk) = (env("STEPS", 2000), env("CHUNK", 20).max(1));
    let clock = std::time::Instant::now();
    let (cpu, config) = ddes_backstep::built_sa_step();
    let (cells, fluid) = (cpu.grid().len(), cpu.fluid_cells());
    println!("CASE {config}");
    println!("CELLS {cells} ({fluid} fluid); the CPU flow built in {:.1} s", clock.elapsed().as_secs_f64());
    let clock = std::time::Instant::now();
    let device = Policy::from_env().map_err(|e| e.to_string()).and_then(|p| GpuFlow::with_options(&cpu, p, Options::default()).map_err(|e| e.to_string()));
    let mut gpu = match device {
        Ok(g) => g,
        Err(why) => {
            println!("DEVICE refused: {why}");
            return;
        }
    };
    println!("DEVICE {}; copied in {:.1} s; kernels compiled in {:.1} s", gpu.adapter_name(), clock.elapsed().as_secs_f64(), gpu.pipeline_seconds());
    // The device holds the flow now; the CPU copy is no longer needed.
    drop(cpu);
    let clock = std::time::Instant::now();
    let mut done = 0;
    while done < steps {
        let n = chunk.min(steps - done);
        if let Err(e) = Backend::advance(&mut gpu, n) {
            println!("STOP after {done} steps: {e}");
            return;
        }
        done += n;
        if done % (chunk * 25) == 0 || done == steps {
            println!("step {done}: {:.1} s", clock.elapsed().as_secs_f64());
        }
    }
    let secs = clock.elapsed().as_secs_f64();
    println!(
        "RESULT {done} steps in {secs:.1} s: {:.4} s per step, {:.1} M cell updates/s on the fluid cells; host peak memory {}",
        secs / done as f64,
        fluid as f64 * done as f64 / secs / 1e6,
        peak_host_memory()
    );
}
