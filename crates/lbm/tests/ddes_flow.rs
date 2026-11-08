//! DDES on the Spalart–Allmaras model in the flow (`turbulence: {"type": "spalart_allmaras",
//! "ddes": {"c_des": 0.65}}`, P. R. Spalart et al., Theor. Comput. Fluid Dyn. 20:181–195, 2006):
//! each step the gradient that gives the vorticity also gives each cell's length
//! d̃ = d − f_d max(0, d − C_DES Δ), which the destruction term reads instead of d.
//!
//! - Where C_DES Δ covers every wall distance, d̃ = d, and DDES is the Reynolds-averaged model bit for bit.
//! - In a forced channel, d̃ stays in [min(d, C_DES Δ), d] and equals d in the first cells.
//! - The reply carries the LES fraction and the largest f_d.
//! - The result is the same bit for bit on any number of threads.
//! - The memory estimate counts d̃.
//!
//! Each test prints its numbers (`cargo test -p fieldcraft-lbm --test ddes_flow -- --nocapture`).

mod common;

use common::*;
use fieldcraft_lbm::{Choice, Limits, Memory};
use fieldcraft_model::{Collision, Ddes, FaceCondition, Procedure, Turbulence};

/// Spalart–Allmaras at inflow ν̃/ν 3, with DDES at `c_des` or without it.
fn sa(c_des: Option<f64>) -> Turbulence {
    Turbulence::SpalartAllmaras { prandtl: 0.9, wall_model: None, inflow_ratio: 3.0, ddes: c_des.map(|c_des| Ddes { c_des }) }
}

/// `p` with turbulence model `t`.
fn with(mut p: Procedure, t: Turbulence) -> Procedure {
    if let Procedure::Fluid { turbulence, .. } = &mut p {
        *turbulence = Some(t);
    }
    p
}

/// A plane Couette channel 4 cells across (y), one cell along the periodic x and z, with
/// turbulence model `t`: every wall distance is at most 1.5 cells.
fn narrow_couette(t: Turbulence) -> fieldcraft_model::Model {
    let dx = 0.25;
    let mut d = domain([0.0; 3], [dx, 1.0, dx], dx);
    for f in [&mut d.faces.x_min, &mut d.faces.x_max, &mut d.faces.z_min, &mut d.faces.z_max] {
        *f = FaceCondition::Periodic {};
    }
    d.faces.y_max = FaceCondition::MovingWall { velocity: [0.1, 0.0, 0.0] };
    model(1.0, 0.002, d, with(collide(2.0, dx * dx, None, Collision::Trt), t))
}

/// A forced channel two high at Re_τ about 180, six cells per half-height, periodic in x and z
/// (as the Spalart–Allmaras tests').
fn forced_channel(collision: Collision, turbulence: Turbulence) -> fieldcraft_model::Model {
    let dx = 1.0 / 6.0;
    let mut d = domain([0.0; 3], [4.0 * dx, 2.0, 4.0 * dx], dx);
    for f in [&mut d.faces.x_min, &mut d.faces.x_max, &mut d.faces.z_min, &mut d.faces.z_max] {
        *f = FaceCondition::Periodic {};
    }
    d.acceleration = [1.0, 0.0, 0.0];
    d.initial_velocity = [15.7, 0.0, 0.0];
    let dt = 0.1 * dx / 15.7;
    model(1.0, 1.0 / 180.0, d, with(collide(60.0 * dt, dt, None, collision), turbulence))
}

/// With C_DES Δ = 2 cells and no wall distance above 1.5 cells, d̃ = d in every cell, so DDES is
/// the Reynolds-averaged model: the same ν̃, densities and velocities bit for bit.
#[test]
fn where_c_des_delta_covers_every_wall_distance_ddes_is_the_rans_model_bit_for_bit() {
    let run = |t: Turbulence| {
        let (setup, mut flow) = prepare(&narrow_couette(t), &[]);
        run_all(&setup, &mut flow);
        flow
    };
    let (rans, ddes) = (run(sa(None)), run(sa(Some(2.0))));
    let d = ddes.sa_wall_distance().unwrap();
    assert!(d.iter().all(|&d| d <= 1.5), "{d:?}");
    let length = ddes.ddes_length().unwrap();
    for c in 0..ddes.grid().len() {
        if ddes.moments(c).is_some() {
            assert_eq!(length[c].to_bits(), d[c].to_bits(), "cell {c}: d̃ {} d {}", length[c], d[c]);
        }
    }
    let bits = |f: &fieldcraft_lbm::Flow| f.spalart_allmaras().unwrap().nu_tilde().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&rans), bits(&ddes), "ν̃");
    let state =
        |f: &fieldcraft_lbm::Flow| (0..f.grid().len()).map(|c| f.moments(c).map(|(r, u)| (r.to_bits(), u.map(f64::to_bits)))).collect::<Vec<_>>();
    assert_eq!(state(&rans), state(&ddes), "densities and velocities");
    assert!(rans.ddes().is_none());
    let stats = ddes.ddes().unwrap();
    println!("narrow Couette, C_DES 2: DDES equals RANS bit for bit; LES fraction {:.3}, largest f_d {:.3}", stats.les_fraction, stats.max_f_d);
}

/// In a forced channel each fluid cell's d̃ lies in [min(d, C_DES Δ), d], and the first cell off
/// each wall (d = ½ ≤ C_DES Δ) keeps d exactly. ν̃ stays finite, ν_t/ν is positive, and the LES
/// fraction and largest f_d are shares in [0, 1]. Both collisions the case runs on.
#[test]
fn ddes_lengths_stay_between_the_les_and_the_rans_length_in_a_forced_channel() {
    for collision in [Collision::Trt, Collision::Mrt] {
        let (setup, mut flow) = prepare(&forced_channel(collision, sa(Some(0.65))), &[]);
        run_all(&setup, &mut flow);
        let (d, length) = (flow.sa_wall_distance().unwrap(), flow.ddes_length().unwrap());
        let mut shortened = 0;
        for c in 0..flow.grid().len() {
            if flow.moments(c).is_none() {
                continue;
            }
            assert!(length[c] <= d[c] && length[c] >= d[c].min(0.65), "cell {c}: d̃ {} d {}", length[c], d[c]);
            if d[c] == 0.5 {
                assert_eq!(length[c], 0.5, "cell {c}");
            }
            shortened += usize::from(length[c] < d[c]);
        }
        assert!(flow.spalart_allmaras().unwrap().nu_tilde().iter().all(|v| v.is_finite()));
        let ratio = flow.eddy_viscosity_ratio();
        let stats = flow.ddes().unwrap();
        println!(
            "forced channel, {collision:?}, DDES 0.65: largest ν_t/ν {ratio:.4}, LES fraction {:.3}, largest f_d {:.3}, {shortened} cells with d̃ < d",
            stats.les_fraction, stats.max_f_d
        );
        assert!(ratio > 0.0 && ratio.is_finite(), "{ratio}");
        assert!((0.0..=1.0).contains(&stats.les_fraction) && (0.0..=1.0).contains(&stats.max_f_d), "{stats:?}");
    }
}

/// The same result bit for bit on one and on several threads, in a channel with a ball, an inlet,
/// an outlet and a slip face, over several parallel blocks; the solve reports DDES's statistics.
#[test]
fn ddes_results_are_bit_for_bit_the_same_on_one_and_many_threads() {
    let (m, ball) = ball_channel(Collision::Trt, None, Some(sa(Some(0.65))));
    let solve = |threads: usize| {
        fieldcraft_parallel::with_threads(threads, || fieldcraft_lbm::solve_fluid(&m, None, std::slice::from_ref(&ball), &Limits::default()).unwrap())
    };
    let one = solve(1);
    assert!(one.grid.len() > 3 * 4096, "several parallel blocks: {}", one.grid.len());
    let stats = one.stats.ddes;
    println!("DDES, ball channel: largest ν_t/ν {:?}, {stats:?}", one.stats.eddy_viscosity_ratio);
    assert!(stats.is_some_and(|s| (0.0..=1.0).contains(&s.les_fraction) && (0.0..=1.0).contains(&s.max_f_d)), "{stats:?}");
    let seven = solve(7);
    assert_eq!(one, seven);
    assert_eq!(one.stats.ddes, seven.stats.ddes);
    let three = solve(3);
    assert_eq!(one, three);
    assert_eq!(one.stats.ddes, three.stats.ddes);
}

/// The memory estimate counts d̃ (one double per cell) on top of the Reynolds-averaged model, and
/// the setup's estimate agrees with the model's.
#[test]
fn the_memory_estimate_counts_the_ddes_length() {
    let m = |c_des: Option<f64>| narrow_couette(sa(c_des));
    let (rans, ddes) = (Memory::estimate(&m(None), None, Choice::Cpu).unwrap(), Memory::estimate(&m(Some(0.65)), None, Choice::Cpu).unwrap());
    println!("memory of {} cells: {} B with DDES, {} B without", ddes.cells, ddes.total(), rans.total());
    assert_eq!(ddes.total(), rans.total() + ddes.cells * 8, "{ddes:?} vs {rans:?}");
    let (setup, _) = prepare(&m(Some(0.65)), &[]);
    assert_eq!(Memory::of(&setup, Choice::Cpu).unwrap().total(), ddes.total());
}
