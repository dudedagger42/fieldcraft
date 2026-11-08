//! The absorbing layer at velocity inlets and pressure outlets.
//!
//! This file starts with the pin. Open-face boxes with the layer off: a velocity inlet, a
//! pressure outlet (uniform or convective), a no-slip floor and a slip roof. After 200 steps,
//! every population and every group's force hashes to the value the code gave before the layer
//! existed. So with the layer off, adding it can't move an open-face result by a single bit.

mod common;

use common::*;
use fieldcraft_lbm::lattice::Q;
use fieldcraft_lbm::{Flow, Limits, Setup};
use fieldcraft_model::{AbsorbingLayer, Collision, FaceCondition, Outflow, Procedure, Profile, Turbulence, WallModel};

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
    open_box_with(collision, outflow, turbulence, None, steps)
}

/// [`open_box`] with an absorbing layer.
fn open_box_with(collision: Collision, outflow: Outflow, turbulence: Option<Turbulence>, absorbing: Option<AbsorbingLayer>, steps: u64) -> Flow {
    let dx = 1.0 / 12.0;
    let (lx, ly) = (4.0, 1.0);
    let mut d = domain([0.0; 3], [lx, ly, dx], dx);
    let u = 1.0;
    d.faces.x_min = FaceCondition::Velocity { velocity: [u, 0.0, 0.0], profile: Profile::Uniform };
    d.faces.x_max = FaceCondition::Pressure { pressure: 0.0, outflow };
    d.faces.y_max = FaceCondition::Slip {};
    d.faces.z_min = FaceCondition::Periodic {};
    d.faces.z_max = FaceCondition::Periodic {};
    d.absorbing = absorbing;
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

/// The box's unsteady density: the root mean square over the fluid cells of each cell's density
/// change over the next `steps` steps (half the box's quarter-wave period, so a standing wave's
/// change is largest and the steady parts cancel).
fn unsteady_density(flow: &mut Flow, steps: u64) -> f64 {
    let rho = |flow: &Flow| -> Vec<f64> { (0..flow.grid().len()).filter_map(|c| flow.moments(c).map(|(r, _)| r)).collect() };
    let before = rho(flow);
    for _ in 0..steps {
        flow.step().unwrap();
    }
    let after = rho(flow);
    (before.iter().zip(&after).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / before.len() as f64).sqrt()
}

/// The layer drains the pulse: after it has crossed the box a few times, the unsteady density is
/// well below the layer-free box's, on every collision.
#[test]
fn the_layer_drains_an_acoustic_pulse() {
    // 48 cells long: 8-cell layers at the inlet and the outlet (a sixth of the box each). The
    // quarter-wave period is 4 · 48 √3 ≈ 333 steps.
    let layer = AbsorbingLayer { thickness: 8.0 / 12.0, strength: None, filter_time: None };
    for collision in [Collision::Trt, Collision::Mrt, Collision::Hrr] {
        let free = unsteady_density(&mut open_box(collision, Outflow::Uniform, None, 1200), 166);
        let damped = unsteady_density(&mut open_box_with(collision, Outflow::Uniform, None, Some(layer), 1200), 166);
        println!("{collision:?}: unsteady density {free:.3e} without the layer, {damped:.3e} with it");
        assert!(damped.is_finite() && damped < 0.1 * free, "{collision:?}: {damped:e} vs {free:e}");
    }
}

/// The layer's cells: fluid cells within its thickness of the velocity or pressure faces, without
/// the open faces' own cells or Han's boundary nodes.
#[test]
fn the_layer_leaves_out_open_cells_and_wall_nodes() {
    let han = Turbulence::Smagorinsky { constant: 0.1, prandtl: 0.9, van_driest: None, wall_model: Some(WallModel::SpaldingBounce) };
    let layer = AbsorbingLayer { thickness: 8.0 / 12.0, strength: None, filter_time: None };
    let flow = open_box_with(Collision::Trt, Outflow::Uniform, Some(han), Some(layer), 1);
    let grid = *flow.grid();
    let [nx, ny, _] = grid.cells;
    let cells = flow.absorbing_cells();
    assert!(!cells.is_empty());
    assert!(cells.windows(2).all(|w| w[0] < w[1]));
    for &c in cells {
        let [i, j, _] = grid.coords(c as usize);
        // Depth 1–7 from either open face (depth 0 is the open layer, 8 the layer's inner edge,
        // where the profile is 0); rows 1 and up (row 0 is the floor's nodes).
        let depth = i.min(nx - 1 - i);
        assert!((1..8).contains(&depth), "cell ({i}, {j}) at depth {depth}");
        assert!(j >= 1, "cell ({i}, {j}) is a wall node");
    }
    // Every other qualifying cell is in.
    assert_eq!(cells.len(), 2 * 7 * (ny - 1));
}

/// Hostile layers are refused with a message, never a panic: by the model when it is defined,
/// or by the solver's setup against the grid.
#[test]
fn hostile_layers_are_refused() {
    let dx = 1.0 / 12.0;
    let base = || {
        let mut d = domain([0.0; 3], [4.0, 1.0, dx], dx);
        d.faces.x_min = FaceCondition::Velocity { velocity: [1.0, 0.0, 0.0], profile: Profile::Uniform };
        d.faces.x_max = FaceCondition::Pressure { pressure: 0.0, outflow: Outflow::Uniform };
        d.faces.z_min = FaceCondition::Periodic {};
        d.faces.z_max = FaceCondition::Periodic {};
        d
    };
    let layer = |thickness: f64, strength: Option<f64>, filter_time: Option<f64>| AbsorbingLayer { thickness, strength, filter_time };
    let dt = 0.05 * dx;
    let try_setup = |d: fieldcraft_model::FluidDomain| -> Result<(), String> {
        let mut m = fieldcraft_model::Model::default();
        m.add_material(
            "Fluid",
            fieldcraft_model::Material {
                density: Some(1.0),
                fluid: Some(fieldcraft_model::Fluid::Newtonian { viscosity: 0.02 }),
                ..Default::default()
            },
        )
        .map_err(|e| e.to_string())?;
        m.add_fluid_domain("Flow", d).map_err(|e| e.to_string())?;
        m.add_step(fieldcraft_model::Step::new("Run", collide(10.0 * dt, dt, None, Collision::Trt)), None).map_err(|e| e.to_string())?;
        Setup::new(&m, None, &[], &Limits::default()).map(|_| ()).map_err(|e| e.to_string())
    };
    let mut cases: Vec<(&str, fieldcraft_model::FluidDomain, &str)> = Vec::new();
    for (name, l, says) in [
        ("zero thickness", layer(0.0, None, None), "thickness"),
        ("negative thickness", layer(-1.0, None, None), "thickness"),
        ("NaN thickness", layer(f64::NAN, None, None), "thickness"),
        ("infinite thickness", layer(f64::INFINITY, None, None), "thickness"),
        ("one cell", layer(dx, None, None), "at least 2 cells"),
        ("a quarter of the box", layer(1.0, None, None), "a fifth"),
        ("zero strength", layer(0.5, Some(0.0), None), "strength"),
        ("NaN strength", layer(0.5, Some(f64::NAN), None), "strength"),
        ("strength 5", layer(0.5, Some(5.0), None), "strength"),
        ("negative filter time", layer(0.5, None, Some(-1.0)), "filter time"),
        ("NaN filter time", layer(0.5, None, Some(f64::NAN)), "filter time"),
        ("filter under a step", layer(0.5, None, Some(0.5 * dt)), "time step"),
    ] {
        let mut d = base();
        d.absorbing = Some(l);
        cases.push((name, d, says));
    }
    let mut closed = base();
    closed.faces.x_min = FaceCondition::Wall {};
    closed.faces.x_max = FaceCondition::Wall {};
    closed.absorbing = Some(layer(0.5, None, None));
    cases.push(("no open face", closed, "velocity or pressure face"));
    let mut hot = base();
    hot.thermal = Some(fieldcraft_model::FluidThermal { initial_temperature: 300.0, ..Default::default() });
    hot.absorbing = Some(layer(0.5, None, None));
    cases.push(("heat", hot, "heat"));
    let mut wrong = Vec::new();
    for (name, d, says) in cases {
        match std::panic::catch_unwind(|| try_setup(d)) {
            Ok(Err(e)) if e.contains(says) => {}
            other => wrong.push(format!("{name}: {other:?}")),
        }
    }
    assert!(
        wrong.is_empty(),
        "not refused as expected:
{}",
        wrong.join(
            "
"
        )
    );
    // A sound layer is accepted.
    let mut d = base();
    d.absorbing = Some(layer(0.5, Some(2.0), Some(0.5)));
    assert_eq!(try_setup(d), Ok(()));
}
