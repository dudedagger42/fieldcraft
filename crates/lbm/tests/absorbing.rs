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

/// The laminar flat-plate box (Re 2000 per unit length, Δ = 1/200, U = 0.1 in lattice units,
/// TRT): x from −0.25 to 1, z from −Δ to 2, one cell periodic in y; a uniform velocity inlet, a
/// convective pressure outlet, slip floor and roof, and the plate (0 ≤ x ≤ 1) as solid cells on the
/// floor. It traps a quarter-wave acoustic mode between the inlet and the outlet. `absorbing`
/// puts a layer at both. Run `steps` steps, calling `each` after every one.
fn laminar_plate_box(absorbing: Option<AbsorbingLayer>, steps: u64, mut each: impl FnMut(u64, &Flow)) {
    let h = 1.0 / 200.0;
    let mut d = domain([-0.25, 0.0, -h], [1.0, h, 2.0], h);
    d.faces.x_min = FaceCondition::Velocity { velocity: [1.0, 0.0, 0.0], profile: Profile::Uniform };
    d.faces.x_max = FaceCondition::Pressure { pressure: 0.0, outflow: Outflow::Convective };
    d.faces.y_min = FaceCondition::Periodic {};
    d.faces.y_max = FaceCondition::Periodic {};
    d.faces.z_min = FaceCondition::Slip {};
    d.faces.z_max = FaceCondition::Slip {};
    d.initial_velocity = [1.0, 0.0, 0.0];
    d.absorbing = absorbing;
    let plate = vec![obstacle("Plate", cuboid([0.0, -h, -2.0 * h], [1.0, 2.0 * h, 0.0]), &CUBOID_FACES)];
    d.obstacles = vec!["Plate".into()];
    let dt = 0.1 * h;
    let m = model(1.0, 1.0 / 2000.0, d, collide(steps as f64 * dt, dt, None, Collision::Trt));
    let limits = Limits { max_cell_updates: u64::MAX, max_steps: u64::MAX, ..Limits::default() };
    let setup = Setup::new(&m, None, &plate, &limits).unwrap();
    let mut flow = Flow::new(&setup, &plate).unwrap();
    for k in 1..=steps {
        flow.step().unwrap();
        each(k, &flow);
    }
}

/// The quarter-wave mode of the laminar plate box decays at least tenfold within five periods
/// with the layer on: the inlet's density amplitude (the mean over the first interior column, half
/// its range over one period) in the period after step 2000 + 5 T against the period before step
/// 2000. The period is T = 4 L_x / c_s / (1 − Ma²) = 1786 steps. The layer is 20 cells at both
/// ends (x < −0.15 and x > 0.9), clear of the plate's measured stations (0.3 ≤ x ≤ 0.9). Without
/// the layer the same ratio is printed beside it: the mode barely decays.
#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn the_layer_drains_the_laminar_plate_s_quarter_wave() {
    let h = 1.0 / 200.0;
    let period = (4.0 * 250.0 * 3f64.sqrt() / (1.0 - 3.0 * 0.01)).round() as u64;
    let (early, late) = (2000 - period..2000, 2000 + 5 * period..2000 + 6 * period);
    let amplitude = |layer: Option<AbsorbingLayer>| -> (f64, f64) {
        let (mut a, mut b) = ((f64::INFINITY, f64::NEG_INFINITY), (f64::INFINITY, f64::NEG_INFINITY));
        laminar_plate_box(layer, late.end, |k, flow| {
            let into = |r: &mut (f64, f64)| {
                let grid = flow.grid();
                let [_, _, nz] = grid.cells;
                let rho: Vec<f64> = (0..nz).filter_map(|kz| flow.moments(grid.index([1, 0, kz])).map(|(r, _)| r)).collect();
                let mean = rho.iter().sum::<f64>() / rho.len().max(1) as f64;
                r.0 = r.0.min(mean);
                r.1 = r.1.max(mean);
            };
            if early.contains(&k) {
                into(&mut a);
            } else if late.contains(&k) {
                into(&mut b);
            }
            if k == 1 && layer.is_some() {
                // The layer stays clear of the measured stations.
                let grid = flow.grid();
                for &c in flow.absorbing_cells() {
                    let x = grid.centre(c as usize)[0];
                    assert!(x < -0.15 + 1e-9 || x > 0.9 - 1e-9, "a layer cell at x = {x}");
                }
            }
        });
        (0.5 * (a.1 - a.0), 0.5 * (b.1 - b.0))
    };
    let layer = AbsorbingLayer { thickness: 20.0 * h, strength: None, filter_time: None };
    let (on_early, on_late) = amplitude(Some(layer));
    let (off_early, off_late) = amplitude(None);
    println!(
        "inlet density amplitude, period {period} steps: with the layer {on_early:.3e} → {on_late:.3e} (÷{:.1}), without {off_early:.3e} → {off_late:.3e} (÷{:.2})",
        on_early / on_late,
        off_early / off_late
    );
    assert!(on_late.is_finite() && on_late * 10.0 <= on_early, "the mode decayed only ÷{:.2}", on_early / on_late);
}
