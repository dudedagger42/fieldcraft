//! Han, Ooka and Kikumoto's on-node wall-function bounce (`spalding_bounce`) on obstacle walls and
//! where a wall meets an inlet or outlet (agent-hub notes/cfd-wall-obstacles.md).
//!
//! This file starts with the pin: box-face Han channels whose every population, wall force and
//! wall-function sample after 200 steps hashes to the value the box-face-only code gave, so the
//! extension can't move a box-face result by a single bit.

mod common;

use common::*;
use fieldcraft_lbm::Flow;
use fieldcraft_lbm::lattice::Q;
use fieldcraft_model::{Collision, FaceCondition, HrrSigma, Procedure, Turbulence, WallModel};

const PERIODIC: FaceCondition = FaceCondition::Periodic {};

/// FNV-1a over the bits of a stream of numbers.
#[derive(Debug, Clone, Copy)]
struct Fnv(u64);

impl Fnv {
    fn new() -> Fnv {
        Fnv(0xcbf2_9ce4_8422_2325)
    }

    fn add(&mut self, v: f64) {
        for b in v.to_bits().to_le_bytes() {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

/// The hash of a flow's state: every population of every cell, every group's force of the last
/// step, and the wall function's (cell, y, u_τ) samples.
fn state_hash(flow: &Flow) -> u64 {
    let mut h = Fnv::new();
    for cell in 0..flow.grid().len() {
        for q in 0..Q {
            h.add(flow.population(q, cell));
        }
    }
    for f in flow.forces() {
        for v in f {
            h.add(*v);
        }
    }
    for &(cell, y, ut) in flow.sa_wall_function().1 {
        h.add(cell as f64);
        h.add(y);
        h.add(ut);
    }
    h.0
}

/// A perturbed coarse box-face channel (4 cells per half-height, Re_τ 180, periodic in x and z,
/// walls at the y faces), `steps` steps of `turbulence` on `collision`.
fn box_channel(collision: Collision, turbulence: Turbulence, steps: u64) -> Flow {
    use std::f64::consts::PI;
    let (n, re_tau, ub) = (4usize, 180.0, 15.7);
    let dx = 1.0 / n as f64;
    let (lx, lz) = (12.0 * dx, 6.0 * dx);
    let mut d = domain([0.0; 3], [lx, 2.0 + dx, lz], dx);
    for f in [&mut d.faces.x_min, &mut d.faces.x_max, &mut d.faces.z_min, &mut d.faces.z_max] {
        *f = PERIODIC;
    }
    d.acceleration = [1.0, 0.0, 0.0];
    let dt = 0.1 * dx / ub;
    let mut p = collide(steps as f64 * dt, dt, None, collision);
    if let Procedure::Fluid { turbulence: t, sigma, .. } = &mut p {
        *t = Some(turbulence);
        if collision == Collision::Hrr {
            *sigma = Some(HrrSigma::Value(0.99));
        }
    }
    let (setup, mut flow) = prepare(&model(1.0, 1.0 / re_tau, d, p), &[]);
    let grid = *flow.grid();
    let to_lattice = 1.0 / setup.velocity_scale();
    let (kx, kz) = (2.0 * PI / lx, 2.0 * PI / lz);
    flow.set_equilibrium(|cell| {
        let [x, y, z] = grid.centre(cell);
        let eta = y - 1.0 - 0.5 * dx;
        let wall = (1.0 - eta * eta).max(0.0);
        let u = 1.15 * ub * (1.0 - eta.abs().min(1.0).powi(8)) + 2.0 * (2.0 * kz * z).sin() * (kx * x).cos() * wall;
        let v = 1.5 * wall * (2.0 * kz * z).cos() * (3.0 * kx * x).sin();
        let w = 1.5 * wall * eta * (kx * x).sin() * (kz * z).cos();
        (1.0, [u * to_lattice, v * to_lattice, w * to_lattice])
    });
    for _ in 0..steps {
        flow.step().unwrap();
    }
    flow
}

fn smagorinsky() -> Turbulence {
    Turbulence::Smagorinsky { constant: 0.1, prandtl: 0.9, van_driest: None, wall_model: Some(WallModel::SpaldingBounce) }
}

fn spalart_allmaras() -> Turbulence {
    Turbulence::SpalartAllmaras { prandtl: 0.9, wall_model: Some(WallModel::SpaldingBounce), inflow_ratio: 3.0, ddes: None }
}

/// The box-face results of the code before the obstacle extension, bit for bit: Smagorinsky on
/// TRT, MRT and HRR, and Spalart–Allmaras on MRT, 200 steps each.
#[test]
fn box_face_han_results_are_pinned_bit_for_bit() {
    let cases: [(&str, Collision, Turbulence, u64); 4] = [
        ("smagorinsky trt", Collision::Trt, smagorinsky(), 0x0300_7730_e23f_d205),
        ("smagorinsky mrt", Collision::Mrt, smagorinsky(), 0xce34_b201_63e2_d422),
        ("smagorinsky hrr", Collision::Hrr, smagorinsky(), 0xfcfe_6fea_226c_005b),
        ("spalart-allmaras mrt", Collision::Mrt, spalart_allmaras(), 0xb628_ab07_778c_e16d),
    ];
    let mut wrong = Vec::new();
    for (name, collision, turbulence, want) in cases {
        let got = state_hash(&box_channel(collision, turbulence, 200));
        if got != want {
            wrong.push(format!("{name}: {got:#018x} (pinned {want:#018x})"));
        }
    }
    assert!(wrong.is_empty(), "box-face Han results moved:\n{}", wrong.join("\n"));
}
