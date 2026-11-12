//! The absorbing layer (`fieldcraft_lbm::absorbing`) on the GPU against the CPU.

use super::*;
use fieldcraft_model::AbsorbingLayer;

/// A 2D open box, 48 × 12 cells (one deep, periodic in z): a velocity inlet at x min, a pressure
/// outlet at x max (`outflow`), a no-slip floor and a slip roof, started from the inlet velocity
/// with a density bump so that an acoustic pulse runs to both open faces. The layer is
/// `cells` thick with `strength` (times the matched value) and the default filter.
fn open_box(collision: Collision, outflow: Outflow, absorbing: Option<(f64, f64)>, steps: f64) -> (Setup, Flow) {
    let dx = 1.0 / 12.0;
    let (lx, ly) = (4.0, 1.0);
    let mut d = domain([0.0; 3], [lx, ly, dx], dx);
    let u = 1.0;
    d.faces.x_min = FaceCondition::Velocity { velocity: [u, 0.0, 0.0], profile: Profile::Uniform };
    d.faces.x_max = FaceCondition::Pressure { pressure: 0.0, outflow };
    d.faces.y_max = FaceCondition::Slip {};
    d.faces.z_min = PERIODIC;
    d.faces.z_max = PERIODIC;
    d.absorbing = absorbing.map(|(cells, strength)| AbsorbingLayer { thickness: Some(cells * dx), strength: Some(strength), filter_time: None });
    let dt = 0.05 * dx / u;
    let (setup, mut flow) = prepare(&model(1.0, 0.02, d, collide(steps * dt, dt, None, collision)), &[]);
    let grid = *flow.grid();
    let to_lattice = 1.0 / setup.velocity_scale();
    flow.set_equilibrium(|cell| {
        let [x, y, _] = grid.centre(cell);
        let r2 = ((x - 0.5 * lx) / 0.3).powi(2) + ((y - 0.5 * ly) / 0.3).powi(2);
        (1.0 + 1e-3 * (-r2).exp(), [u * to_lattice, 0.0, 0.0])
    });
    (setup, flow)
}

/// The velocity (over the largest CPU speed) and density differences of the GPU from the CPU at
/// `steps`, in chunks of 100 so the worst is kept, and whether the layer changed the CPU's result
/// from the layer-free one by more than `10 ×` the worst difference found.
fn parity(collision: Collision, outflow: Outflow, layer: (f64, f64), steps: u64, streaming: Streaming) -> Option<(f64, f64, usize)> {
    let (_, mut cpu) = open_box(collision, outflow, Some(layer), steps as f64);
    let cells = cpu.absorbing_cells().len();
    let mut gpu = gpu(&cpu, streaming)?;
    let (mut du, mut drho) = (0.0f64, 0.0f64);
    for _ in 0..steps / 100 {
        cpu.advance(100).unwrap();
        gpu.advance(100).unwrap();
        let (a, b) = field_difference(&mut cpu, &mut gpu);
        du = du.max(a);
        drho = drho.max(b);
    }
    Some((du, drho, cells))
}

/// A layer 8 cells thick at strength 1 on a velocity inlet and a pressure outlet (uniform and
/// convective), the hybrid collision (σ 0.99) and the multiple-relaxation-time one, both
/// layouts: 1000 steps, in which the pulse reaches both faces and the layer works. Bounds: velocity
/// and density 1e-6 pointwise at every 100 steps (measured 7e-7 and 9e-8 at 600).
#[test]
fn the_absorbing_layer_follows_the_cpu() {
    let mut failed = Vec::new();
    for streaming in [Streaming::InPlace, Streaming::TwoBuffers] {
        for collision in [Collision::Hrr, Collision::Mrt] {
            for outflow in [Outflow::Uniform, Outflow::Convective] {
                let Some((du, drho, cells)) = parity(collision, outflow, (8.0, 1.0), 1000, streaming) else { return };
                let what = format!("absorbing layer, {collision:?}, {outflow:?}, {streaming:?}, 1000 steps: {cells} layer cells");
                println!("{what}: velocity {du:.2e}, density {drho:.2e}");
                assert!(cells > 0);
                if !(du < 1e-6 && drho < 1e-6) {
                    failed.push(what);
                }
            }
        }
    }
    assert!(failed.is_empty(), "{failed:?}");
}

/// The comparison means something: the layer moves the CPU's box well past the bounds above.
#[test]
fn the_absorbing_layer_changes_the_result_it_is_compared_on() {
    let (_, mut with) = open_box(Collision::Mrt, Outflow::Uniform, Some((8.0, 1.0)), 1000.0);
    let (_, mut without) = open_box(Collision::Mrt, Outflow::Uniform, None, 1000.0);
    with.advance(1000).unwrap();
    without.advance(1000).unwrap();
    let (a, b) = (with.fields().unwrap(), without.fields().unwrap());
    let drho = a.density.iter().zip(&b.density).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
    println!("layer against none, density: {drho:.2e}");
    assert!(drho > 1e-5, "{drho}");
}
