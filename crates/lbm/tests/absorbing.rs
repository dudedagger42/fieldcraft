//! The absorbing layer at velocity inlets and pressure outlets.
//!
//! This file starts with the pin. Open-face boxes with the layer off: a velocity inlet, a
//! pressure outlet (uniform or convective), a no-slip floor and a slip roof. After 200 steps,
//! every population and every group's force hashes to the value the code gave before the layer
//! existed. So with the layer off, adding it can't move an open-face result by a single bit.

mod common;

use common::*;
use fieldcraft_lbm::Flow;
use fieldcraft_lbm::lattice::Q;
use fieldcraft_model::{Collision, FaceCondition, Outflow, Procedure, Profile, Turbulence, WallModel};

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

/// The hash of a flow's state: every population of every cell and every group's force of the
/// last step.
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
    h.0
}

/// A 2D open box, 48 × 12 cells (one cell deep, periodic in z), run `steps` steps:
/// - a uniform velocity inlet at x min and a pressure outlet at x max (`outflow`);
/// - a no-slip floor and a slip roof.
///
/// It starts from the inlet velocity with a density bump in the middle, so an acoustic pulse
/// runs to both open faces.
fn open_box(collision: Collision, outflow: Outflow, turbulence: Option<Turbulence>, steps: u64) -> Flow {
    let dx = 1.0 / 12.0;
    let (lx, ly) = (4.0, 1.0);
    let mut d = domain([0.0; 3], [lx, ly, dx], dx);
    let u = 1.0;
    d.faces.x_min = FaceCondition::Velocity { velocity: [u, 0.0, 0.0], profile: Profile::Uniform };
    d.faces.x_max = FaceCondition::Pressure { pressure: 0.0, outflow };
    d.faces.y_max = FaceCondition::Slip {};
    d.faces.z_min = FaceCondition::Periodic {};
    d.faces.z_max = FaceCondition::Periodic {};
    let dt = 0.05 * dx / u;
    let mut p = collide(steps as f64 * dt, dt, None, collision);
    if let Procedure::Fluid { turbulence: t, .. } = &mut p {
        *t = turbulence;
    }
    let (setup, mut flow) = prepare(&model(1.0, 0.02, d, p), &[]);
    let grid = *flow.grid();
    let to_lattice = 1.0 / setup.velocity_scale();
    flow.set_equilibrium(|cell| {
        let [x, y, _] = grid.centre(cell);
        let r2 = ((x - 0.5 * lx) / 0.3).powi(2) + ((y - 0.5 * ly) / 0.3).powi(2);
        (1.0 + 1e-3 * (-r2).exp(), [u * to_lattice, 0.0, 0.0])
    });
    for _ in 0..steps {
        flow.step().unwrap();
    }
    flow
}

/// The layer-free results, bit for bit:
/// - laminar on TRT and MRT with a uniform outlet;
/// - TRT with a convective outlet;
/// - Smagorinsky on Han's on-node wall with TRT, whose floor nodes meet the inlet and outlet.
///
/// The last was re-pinned when the inlet cell on Han's node row took the wall's velocity.
#[test]
fn open_faces_without_a_layer_are_pinned_bit_for_bit() {
    let han = Turbulence::Smagorinsky { constant: 0.1, prandtl: 0.9, van_driest: None, wall_model: Some(WallModel::SpaldingBounce) };
    let cases: [(&str, Collision, Outflow, Option<Turbulence>, u64); 4] = [
        ("laminar trt uniform", Collision::Trt, Outflow::Uniform, None, 0x2d98_aaea_28a4_8d4a),
        ("laminar mrt uniform", Collision::Mrt, Outflow::Uniform, None, 0x8588_26fd_d1fa_8def),
        ("laminar trt convective", Collision::Trt, Outflow::Convective, None, 0x0fdb_4d11_38e7_26f3),
        ("han trt uniform", Collision::Trt, Outflow::Uniform, Some(han), 0xc08a_af58_6724_f84b),
    ];
    let mut wrong = Vec::new();
    for (name, collision, outflow, turbulence, want) in cases {
        let got = state_hash(&open_box(collision, outflow, turbulence, 200));
        if got != want {
            wrong.push(format!("{name}: {got:#018x} (pinned {want:#018x})"));
        }
    }
    assert!(wrong.is_empty(), "open-face results moved:\n{}", wrong.join("\n"));
}
