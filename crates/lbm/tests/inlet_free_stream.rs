//! A velocity inlet at small τ − ½, and the absorbing layer that holds it.
//!
//! The case is a free stream with no body:
//! - a uniform velocity inlet at x min and a convective pressure outlet at x max;
//! - a slip floor and roof, one cell deep (periodic in z);
//! - U = 0.1 in lattice units, τ − ½ = 1.2 × 10⁻⁴;
//! - the hybrid collision at fixed σ 0.99 and Spalart–Allmaras without a wall model, as on a
//!   wall-function flat plate. The setup refuses τ this small without a turbulence model; the free
//!   stream's ν̃ = 3 ν gives ν_t ≈ 0.2 ν.
//!
//! At this τ a density mode one cell wide grows along the inlet's first columns (Zou and He's
//! cells and the few behind them), seeded by any pulse that reaches the face. The absorbing layer
//! (H. Xu, P. Sagaut, J. Comput. Phys. 245:14–42, 2013) damps it only near the face, so a thick
//! layer of a fixed length, weak next to the face on a fine grid, lets it grow; the default layer of
//! 10 cells holds it at every spacing.
//!
//! Measured without a layer at Δ = 1/200 with a pulse:
//! - the hybrid collision runs away up to τ − ½ = 3 × 10⁻³ and holds at 10⁻²;
//! - TRT is unstable at 10⁻² and holds at 3 × 10⁻²;
//! - MRT reaches 1.55 U next to the inlet at 10⁻².
//!
//! These tests run the hybrid collision, which holds longest.

mod common;

use common::*;
use fieldcraft_lbm::{Flow, Limits, Setup};
use fieldcraft_model::{AbsorbingLayer, Collision, FaceCondition, HrrSigma, Model, Outflow, Procedure, Profile, Turbulence};

const U_LAT: f64 = 0.1;
const TAU_MINUS_HALF: f64 = 1.2e-4;

/// The default layer: no thickness, so 10 cells.
const DEFAULT_LAYER: AbsorbingLayer = AbsorbingLayer { thickness: None, strength: None, filter_time: None };

/// The free stream's model, `length` × `height` at `cells_per_unit` and τ − ½ = `tau_minus_half`.
fn stream_model(cells_per_unit: f64, length: f64, height: f64, tau_minus_half: f64, absorbing: Option<AbsorbingLayer>, steps: u64) -> Model {
    let dx = 1.0 / cells_per_unit;
    let dt = U_LAT * dx;
    let nu = tau_minus_half / 3.0 * dx * dx / dt;
    let mut d = domain([0.0; 3], [length, height, dx], dx);
    d.faces.x_min = FaceCondition::Velocity { velocity: [1.0, 0.0, 0.0], profile: Profile::Uniform };
    d.faces.x_max = FaceCondition::Pressure { pressure: 0.0, outflow: Outflow::Convective };
    d.faces.y_min = FaceCondition::Slip {};
    d.faces.y_max = FaceCondition::Slip {};
    d.faces.z_min = FaceCondition::Periodic {};
    d.faces.z_max = FaceCondition::Periodic {};
    d.initial_velocity = [1.0, 0.0, 0.0];
    d.absorbing = absorbing;
    let mut p = collide(steps as f64 * dt, dt, None, Collision::Hrr);
    if let Procedure::Fluid { sigma, turbulence, .. } = &mut p {
        *sigma = Some(HrrSigma::Value(0.99));
        *turbulence = Some(Turbulence::SpalartAllmaras { prandtl: 0.9, wall_model: None, inflow_ratio: 3.0, ddes: None });
    }
    model(1.0, nu, d, p)
}

/// The free stream, `length` × `height` at `cells_per_unit`, started at the inlet velocity with
/// each cell's density from `density(x, y)`.
fn free_stream(
    cells_per_unit: f64,
    length: f64,
    height: f64,
    absorbing: Option<AbsorbingLayer>,
    steps: u64,
    density: impl Fn(f64, f64) -> f64,
) -> Flow {
    let (setup, mut flow) = prepare(&stream_model(cells_per_unit, length, height, TAU_MINUS_HALF, absorbing, steps), &[]);
    let grid = *flow.grid();
    let to_lattice = 1.0 / setup.velocity_scale();
    flow.set_equilibrium(|cell| {
        let [x, y, _] = grid.centre(cell);
        (density(x, y), [to_lattice, 0.0, 0.0])
    });
    flow
}

/// The largest |ρ − 1| in the inlet's first four columns.
fn inlet_departure(flow: &Flow) -> f64 {
    let g = *flow.grid();
    let [nx, ny, _] = g.cells;
    let mut worst: f64 = 0.0;
    for i in 0..4.min(nx) {
        for j in 0..ny {
            if let Some((rho, _)) = flow.moments(g.index([i, j, 0])) {
                worst = worst.max((rho - 1.0).abs());
            }
        }
    }
    worst
}

/// The grid-scale part of the inlet column's density: the largest |ρ − ρ̄| along it, ρ̄ its mean.
fn column_spread(flow: &Flow) -> f64 {
    let g = *flow.grid();
    let rho: Vec<f64> = (0..g.cells[1]).filter_map(|j| flow.moments(g.index([0, j, 0])).map(|m| m.0)).collect();
    let mean = rho.iter().sum::<f64>() / rho.len().max(1) as f64;
    rho.iter().map(|r| (r - mean).abs()).fold(0.0, f64::max)
}

/// A density pulse, +1.5 % with a radius of 4 cells, on the floor 0.33 from the inlet (where a
/// flat plate's leading edge would send its start-up pulse from).
fn pulse(cells_per_unit: f64) -> impl Fn(f64, f64) -> f64 {
    let r = 4.0 / cells_per_unit;
    move |x, y| 1.0 + 0.015 * (-((x - 0.33).powi(2) + y * y) / (r * r)).exp()
}

/// Run `steps` steps; the step at which the inlet's first columns departed by more than 0.05 (or
/// the flow went unstable), and the largest departure seen, sampled every 100 steps.
fn run(flow: &mut Flow, steps: u64) -> (Option<u64>, f64) {
    let mut worst: f64 = 0.0;
    for step in 1..=steps {
        if flow.step().is_err() {
            return (Some(step), f64::INFINITY);
        }
        if step % 100 == 0 {
            let d = inlet_departure(flow);
            worst = worst.max(d);
            if d > 0.05 {
                return (Some(step), worst);
            }
        }
    }
    (None, worst)
}

/// At Δ = 1/200, a pulse reaching the inlet seeds the mode:
/// - without a layer it departs within 8000 steps (at 4000 on today's inlet: this pins today's
///   Zou–He inlet, so a better inlet that holds it fails here and must say so);
/// - with the default layer it holds 20 000 steps.
#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn the_default_layer_holds_a_velocity_inlet_at_small_viscosity() {
    let (n, length, height) = (200.0, 2.33, 0.25);
    let (departed, _) = run(&mut free_stream(n, length, height, None, 8000, pulse(n)), 8000);
    assert!(departed.is_some(), "without a layer the inlet held 8000 steps; today's inlet departs at 4000");
    let (departed, worst) = run(&mut free_stream(n, length, height, Some(DEFAULT_LAYER), 20_000, pulse(n)), 20_000);
    assert!(departed.is_none() && worst < 2e-3, "with the default layer the inlet departed at {departed:?} (largest |ρ − 1| {worst:.3e})");
}

/// The same on a fine grid (Δ = 1/2000, 50 cells high), where a layer of a fixed length is 100
/// cells thick:
/// - 0.05 thick at its default strength it lets the inlet go unstable (at step 10 375 today: this
///   pins today's inlet and layer);
/// - the default layer holds 20 000 steps.
#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn the_default_layer_holds_the_inlet_on_a_fine_grid_where_a_thick_one_does_not() {
    let (n, length, height) = (2000.0, 2.33, 0.025);
    let thick = AbsorbingLayer { thickness: Some(0.05), ..DEFAULT_LAYER };
    let (departed, _) = run(&mut free_stream(n, length, height, Some(thick), 20_000, pulse(n)), 20_000);
    assert!(departed.is_some(), "the 100-cell layer held 20 000 steps; today's inlet goes unstable at 10 375");
    let (departed, worst) = run(&mut free_stream(n, length, height, Some(DEFAULT_LAYER), 20_000, pulse(n)), 20_000);
    assert!(departed.is_none() && worst < 2e-3, "with the default layer the inlet departed at {departed:?} (largest |ρ − 1| {worst:.3e})");
}

/// The mode itself, seeded, in a small box (Δ = 1/200, 100 × 20 cells): the inlet column starts at
/// ρ = 1 ± 10⁻³ alternating cell by cell along the face, with no pulse. Without a layer it dips in
/// the start-up transient and then grows (fiftyfold from step 500 to 1000, to 2.2 × 10⁻³ on today's inlet: this pins today's Zou–He inlet, so a better inlet
/// that holds it fails here and must say so); with the default layer it is below 10⁻⁶ at steps 500
/// and 1000 (3.4 × 10⁻⁷ and 1.1 × 10⁻⁷ measured). The fast check of the default; the two runs that
/// let a pulse seed the mode are heavy.
#[test]
fn the_default_layer_damps_the_inlet_s_grid_scale_mode() {
    let n = 200.0;
    let dx = 1.0 / n;
    let seeded = move |x: f64, y: f64| {
        if x < dx {
            let j = (y / dx).floor() as i64;
            1.0 + if j % 2 == 0 { 1e-3 } else { -1e-3 }
        } else {
            1.0
        }
    };
    let spreads = |layer: Option<AbsorbingLayer>| -> [f64; 3] {
        let mut flow = free_stream(n, 0.5, 0.1, layer, 1000, seeded);
        let start = column_spread(&flow);
        let mut at = [start, 0.0, 0.0];
        for step in 1..=1000 {
            flow.step().unwrap();
            match step {
                500 => at[1] = column_spread(&flow),
                1000 => at[2] = column_spread(&flow),
                _ => {}
            }
        }
        at
    };
    let bare = spreads(None);
    println!("no layer: the inlet column's spread {:.3e} at the start, {:.3e} at 500, {:.3e} at 1000", bare[0], bare[1], bare[2]);
    let layered = spreads(Some(DEFAULT_LAYER));
    println!("default layer: {:.3e} at the start, {:.3e} at 500, {:.3e} at 1000", layered[0], layered[1], layered[2]);
    assert!(bare[2] > 10.0 * bare[1], "without a layer the mode did not grow from step 500 to 1000: {bare:?}");
    assert!(layered[1] < 1e-6 && layered[2] < 1e-6, "the default layer left the mode at {layered:?}");
}

/// Setup warns about a velocity inlet with no layer at small τ − ½, where the runs above depart,
/// and is silent with the default layer or at τ − ½ = 0.1.
#[test]
fn a_velocity_inlet_without_a_layer_at_small_viscosity_is_warned_about() {
    let warned = |tau_minus_half: f64, absorbing: Option<AbsorbingLayer>| {
        let setup = Setup::new(&stream_model(200.0, 0.5, 0.1, tau_minus_half, absorbing, 10), None, &[], &Limits::default()).unwrap();
        setup.warnings.iter().any(|w| w.contains("without an absorbing layer"))
    };
    assert!(warned(TAU_MINUS_HALF, None), "no warning for a bare inlet at τ − ½ = {TAU_MINUS_HALF}");
    assert!(!warned(TAU_MINUS_HALF, Some(DEFAULT_LAYER)), "a warning with the default layer");
    assert!(!warned(0.1, None), "a warning at τ − ½ = 0.1");
}
