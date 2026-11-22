//! The hybrid collision's wall-layer rule: σ = 1 in the layer of the resolved walls and a small share of the layer's density mode removed
//! (`Flow::set_wall_layer_sigma_one`, on by default). The cases are a 16-row channel as a one-cell column, between two box-face walls and
//! between two obstacle slabs (the same fluid), on the hybrid collision at σ 0.95.

mod common;

use common::*;
use fieldcraft_lbm::Flow;
use fieldcraft_model::{Collision, FaceCondition, HrrSigma, Procedure};

const CELLS: usize = 8;
const STEPS: usize = 200;

/// The channel's flow: the walls are the box's y faces, or two obstacle slabs one cell thick in a box periodic in y.
fn channel(slabs: bool) -> Flow {
    let dx = 1.0 / CELLS as f64;
    let (lo, hi) = if slabs { (-dx, 2.0 + dx) } else { (0.0, 2.0) };
    let mut d = domain([0.0, lo, 0.0], [dx, hi, dx], dx);
    for f in [&mut d.faces.x_min, &mut d.faces.x_max, &mut d.faces.z_min, &mut d.faces.z_max] {
        *f = FaceCondition::Periodic {};
    }
    let obstacles = if slabs {
        d.faces.y_min = FaceCondition::Periodic {};
        d.faces.y_max = FaceCondition::Periodic {};
        d.obstacles = vec!["Floor".into(), "Roof".into()];
        vec![
            obstacle("Floor", cuboid([-1.0, -dx, -1.0], [1.0 + dx, 0.0, 1.0 + dx]), &CUBOID_FACES),
            obstacle("Roof", cuboid([-1.0, 2.0, -1.0], [1.0 + dx, 2.0 + dx, 1.0 + dx]), &CUBOID_FACES),
        ]
    } else {
        d.faces.y_min = FaceCondition::Wall {};
        d.faces.y_max = FaceCondition::Wall {};
        Vec::new()
    };
    d.acceleration = [1.0, 0.0, 0.0];
    d.initial_velocity = [12.0, 0.0, 0.0];
    // Lattice viscosity 0.0125 and a bulk speed near 0.1 (τ − ½ about 0.04).
    let dt = 0.1 * dx / 12.0;
    let mut p = collide(STEPS as f64 * dt, dt, None, Collision::Hrr);
    if let Procedure::Fluid { sigma, .. } = &mut p {
        *sigma = Some(HrrSigma::Value(0.95));
    }
    prepare(&model(1.0, 1.0 / 20.0, d, p), &obstacles).1
}

/// ρ and u of every cell, as bits, folded into one hash.
fn hash(flow: &Flow) -> u64 {
    (0..flow.grid().len())
        .flat_map(|c| flow.moments(c).map_or([f64::NAN; 4], |(r, u)| [r, u[0], u[1], u[2]]))
        .fold(0xcbf2_9ce4_8422_2325u64, |h, x| (h ^ x.to_bits()).wrapping_mul(0x0000_0100_0000_01b3))
}

fn run(slabs: bool, mode: Option<u8>) -> Flow {
    let mut flow = channel(slabs);
    if let Some(mode) = mode {
        flow.set_wall_layer_sigma_one(mode);
    }
    for _ in 0..STEPS {
        flow.step().unwrap();
    }
    flow
}

/// The rule is on by default: the default run is the explicit mode 1 bit for bit, the rule changes the flow, and mode 0 is the collision as it
/// was before the rule (its hash is the one the same run gave before the rule existed).
#[test]
fn the_rule_is_on_by_default_and_mode_zero_is_the_collision_before_it() {
    let flow = channel(true);
    assert_eq!(flow.wall_layer_sigma_one(), 1, "the rule is on by default");
    assert!(flow.wall_sigma_layer_mask().iter().any(|&m| m), "the slabs' walls have a layer");
    let (default, on, off) = (hash(&run(true, None)), hash(&run(true, Some(1))), hash(&run(true, Some(0))));
    println!("hashes: default {default:#018x} on {on:#018x} off {off:#018x}");
    assert_eq!(default, on, "the default is the rule on");
    assert_ne!(on, off, "the rule acts on a half-way wall");
    assert_eq!(
        off, 0xe68c_caf8_0bc4_26d7,
        "the opt-out is the collision before the rule (the hash the same run gave on main before the rule existed)"
    );
}

/// The layer is what the walls are, not how they are described: the channel between box faces and between slabs of obstacle cells differs by
/// round-off with the rule off, and with it on only by the different density-mode shares the two kinds of wall take.
#[test]
fn a_box_face_wall_and_a_slab_wall_agree_to_the_difference_of_their_shares() {
    let gap = |mode: u8| {
        let (a, b) = (run(false, Some(mode)), run(true, Some(mode)));
        let n = 2 * CELLS;
        let (ga, gb) = (*a.grid(), *b.grid());
        let mut worst = 0.0f64;
        let mut peak = 0.0f64;
        for y in 0..n {
            let (p, q) = (a.moments(ga.index([0, y, 0])), b.moments(gb.index([0, y + 1, 0])));
            let (Some((ra, ua)), Some((rb, ub))) = (p, q) else { panic!("row {y} is not fluid") };
            worst = worst.max((ra - rb).abs());
            for k in 0..3 {
                worst = worst.max((ua[k] - ub[k]).abs());
            }
            peak = peak.max(ua[0]);
        }
        (worst, peak)
    };
    let (off, peak) = gap(0);
    let (on, _) = gap(1);
    println!("box face against slab, lattice units: rule off {off:.3e}, rule on {on:.3e} (peak speed {peak:.3e})");
    assert!(peak > 1e-3, "the channel did not accelerate: {peak:e}");
    assert!(off <= 1e-10 * peak.max(1.0), "the same wall differs by {off:e} with the rule off");
    assert!(on <= 1e-3 * peak, "the gap with the rule on is {on:e} (peak {peak:e})");
}
