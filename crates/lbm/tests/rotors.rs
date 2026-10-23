//! Rotors ([`fieldcraft_lbm::rotor`]) against closed forms: circular Couette flow between a
//! turning inner cylinder and a fixed outer one, whose speed is `A r + B / r` (compared over the middle of the gap) and whose torque per
//! length is `4π μ Ω R₁² R₂² / (R₂² − R₁²)` (e.g. L. D. Landau, E. M. Lifshitz, *Fluid
//! Mechanics*, 2nd ed., §18). The voxelised cylinder is not the same at every angle, so its
//! cells are covered and uncovered as it turns, and the test covers that too.
//!
//! Each test prints its numbers (`cargo test --release -p fieldcraft-lbm --test rotors --
//! --nocapture`).

mod common;

use common::*;
use fieldcraft_lbm::Obstacle;
use fieldcraft_model::{FaceCondition, Rotor};

const PERIODIC: FaceCondition = FaceCondition::Periodic {};

/// A ring along z about (`x`, `y`) from radius `r_in` to `r_out`, `m` segments (the tube of
/// `common` turned so that its axis is z). Faces: 0 inner wall, 1 outer wall, 2 and 3 the ends.
fn ring_z(centre: [f64; 2], r_in: f64, r_out: f64, z0: f64, z1: f64, m: usize) -> fieldcraft_mesh::surface::BoundarySurface {
    let mut positions = Vec::new();
    for z in [z0, z1] {
        for radius in [r_in, r_out] {
            for k in 0..m {
                let phi = 2.0 * std::f64::consts::PI * k as f64 / m as f64;
                positions.push([centre[0] + radius * phi.cos(), centre[1] + radius * phi.sin(), z]);
            }
        }
    }
    let id = |e: usize, k: usize, s: usize| ((e * 2 + k) * m + s % m) as u32;
    let mut tris = Vec::new();
    let mut tags = Vec::new();
    for s in 0..m {
        for k in 0..2 {
            tris.push([id(0, k, s), id(1, k, s), id(1, k, s + 1)]);
            tris.push([id(0, k, s), id(1, k, s + 1), id(0, k, s + 1)]);
            tags.extend([k as u32; 2]);
        }
        for e in 0..2 {
            tris.push([id(e, 0, s), id(e, 1, s), id(e, 1, s + 1)]);
            tris.push([id(e, 0, s), id(e, 1, s + 1), id(e, 0, s + 1)]);
            tags.extend([2 + e as u32; 2]);
        }
    }
    surface(positions, tris, tags, move |p, tag| {
        let radial = [p[0] - centre[0], p[1] - centre[1], 0.0];
        match tag {
            0 => radial.map(|c| -c),
            1 => radial,
            2 => [0.0, 0.0, -1.0],
            _ => [0.0, 0.0, 1.0],
        }
    })
}

/// Circular Couette flow, `r1` cells of inner radius and twice that outside, the rim at lattice
/// speed 0.05, run to steady state: the largest error of the speed over the gap against
/// `A r + B / r` (over the rim speed), and the mean torque per length over the last tenth of the
/// run against the closed form's (relative).
fn couette(r1: usize, rim: f64) -> (f64, f64) {
    let (r1, r2) = (r1 as f64, 2.0 * r1 as f64);
    let (nu, rho) = (0.1, 1.0);
    let omega = rim / r1;
    let side = 2.0 * r2 + 4.0;
    let c = [side / 2.0, side / 2.0];
    let mut d = domain([0.0; 3], [side, side, 1.0], 1.0);
    d.faces.z_min = PERIODIC;
    d.faces.z_max = PERIODIC;
    d.obstacles = vec!["Outer".into()];
    d.rotors = vec![Rotor { body: "Inner".into(), origin: [c[0], c[1], 0.0], axis: [0.0, 0.0, 1.0], speed: omega }];
    let m = 8 * r2 as usize;
    let obstacles = [
        Obstacle { body: "Outer".into(), surface: ring_z(c, r2, side, -1.0, 2.0, m), faces: vec![] },
        Obstacle { body: "Inner".into(), surface: cylinder(c, r1, -1.0, 2.0, m), faces: vec![] },
    ];
    let steps = (8.0 * (r2 - r1) * (r2 - r1) / nu) as u64;
    let model = model(rho, nu, d, step(steps as f64, 1.0, None));
    let (_, mut flow) = prepare(&model, &obstacles);
    let mut torque = 0.0;
    let tail = steps / 10;
    let grid = *flow.grid();
    // The velocity, averaged over the tail like the torque: the covered and uncovered cells
    // send small pressure waves through the gap.
    let mut mean = vec![[0.0; 3]; grid.len()];
    for k in 0..steps {
        flow.step().unwrap_or_else(|e| panic!("step {k}: {e}"));
        if k >= steps - tail {
            torque += flow.rotors()[0].axial_torque();
            for (cell, m) in mean.iter_mut().enumerate() {
                if let Some((_, u)) = flow.moments(cell) {
                    for a in 0..3 {
                        m[a] += u[a] / tail as f64;
                    }
                }
            }
        }
    }
    torque /= tail as f64;

    let (a, b) = (-omega * r1 * r1 / (r2 * r2 - r1 * r1), omega * r1 * r1 * r2 * r2 / (r2 * r2 - r1 * r1));
    let mut err: f64 = 0.0;
    for (cell, u) in mean.iter().enumerate() {
        let p = grid.centre(cell);
        let (x, y) = (p[0] - c[0], p[1] - c[1]);
        let r = x.hypot(y);
        // The middle of the gap, away from the walls' staircase.
        if r < r1 + 0.2 * (r2 - r1) || r > r2 - 0.2 * (r2 - r1) {
            continue;
        }
        let u_theta = (-y * u[0] + x * u[1]) / r;
        err = err.max((u_theta - (a * r + b / r)).abs() / rim);
    }
    // The fluid's torque on the cylinder resists its turning; one cell of length.
    let exact = -4.0 * std::f64::consts::PI * rho * nu * omega * r1 * r1 * r2 * r2 / (r2 * r2 - r1 * r1);
    println!(
        "circular Couette, R₁ = {r1} cells, R₂ = {r2}, rim speed {rim}: speed error {err:.3e} of the rim; torque {torque:.5e} vs {exact:.5e} ({:+.2}%)",
        100.0 * (torque / exact - 1.0)
    );
    (err, (torque / exact - 1.0).abs())
}

#[test]
fn a_turning_cylinder_drives_circular_couette_flow() {
    let coarse = couette(8, 0.05);
    let fine = couette(16, 0.05);
    // Measured: speed 4.1e-2 → 1.6e-2 of the rim, torque +5.3 % → +2.2 %. A staircase wall on
    // both sides (the rotor's at half the spacing), so first order in the spacing: both halve.
    assert!(fine.0 < 2e-2 && fine.1 < 3e-2, "{fine:?}");
    assert!(fine.0 < 0.5 * coarse.0 && fine.1 < 0.5 * coarse.1, "{coarse:?} → {fine:?}");
}

/// One closed surface from boxes, each given by its centre, half-extents and a turn about each
/// axis in order (x, then y, then z), as separate shells (an impeller's hub and blades).
fn boxes(parts: &[([f64; 3], [f64; 3], [f64; 3])]) -> fieldcraft_mesh::surface::BoundarySurface {
    let turn = |p: [f64; 3], a: [f64; 3]| {
        let (sx, cx) = a[0].sin_cos();
        let p = [p[0], cx * p[1] - sx * p[2], sx * p[1] + cx * p[2]];
        let (sy, cy) = a[1].sin_cos();
        let p = [cy * p[0] + sy * p[2], p[1], -sy * p[0] + cy * p[2]];
        let (sz, cz) = a[2].sin_cos();
        [cz * p[0] - sz * p[1], sz * p[0] + cz * p[1], p[2]]
    };
    let mut positions = Vec::new();
    let mut tris = Vec::new();
    let mut tags = Vec::new();
    let mut centres: Vec<[f64; 3]> = Vec::new();
    for (centre, half, angles) in parts {
        let base = positions.len() as u32;
        for k in 0..8 {
            let local = [0, 1, 2].map(|a| if k >> a & 1 == 1 { half[a] } else { -half[a] });
            let p = turn(local, *angles);
            positions.push([0, 1, 2].map(|a| centre[a] + p[a]));
        }
        // The six faces, two triangles each (corner k has bit a set on the +a side).
        for (a, b, c, d) in [(0, 2, 6, 4), (1, 5, 7, 3), (0, 4, 5, 1), (2, 3, 7, 6), (0, 1, 3, 2), (4, 6, 7, 5)] {
            tris.push([base + a, base + b, base + c]);
            tris.push([base + a, base + c, base + d]);
            tags.extend([(centres.len()) as u32; 2]);
        }
        centres.push(*centre);
    }
    // Each box is one face, tagged by its index: its triangles face away from its centre.
    surface(positions, tris, tags, move |p, tag| {
        let c = centres[tag as usize];
        [p[0] - c[0], p[1] - c[1], p[2] - c[2]]
    })
}

/// A flat paddle turning about z in a closed box at rest: the fluid spins up with it, resists it
/// (the torque opposes the turning and the shaft does work on the fluid), and the cells it covers
/// and uncovers as it sweeps keep the mass to a small drift.
#[test]
fn a_paddle_stirs_a_closed_box() {
    let (side, nu, tip) = (32.0, 0.05, 0.08);
    let radius = 10.0;
    let omega = tip / radius;
    let c = side / 2.0;
    let mut d = domain([0.0; 3], [side; 3], 1.0);
    d.rotors = vec![Rotor { body: "Paddle".into(), origin: [c, c, c], axis: [0.0, 0.0, 1.0], speed: omega }];
    let paddle = boxes(&[([c, c, c], [radius, 1.5, 6.0], [0.0, 0.0, 0.3])]);
    let obstacles = [Obstacle { body: "Paddle".into(), surface: paddle, faces: vec![] }];
    let steps = 3000u64;
    let m = model(1.0, nu, d, step(steps as f64, 1.0, None));
    let (_, mut flow) = prepare(&m, &obstacles);
    let mass0 = flow.mass();
    let mut torque = 0.0;
    for k in 0..steps {
        flow.step().unwrap_or_else(|e| panic!("step {k}: {e}"));
        if k >= steps / 2 {
            torque += flow.rotors()[0].axial_torque() / (steps / 2) as f64;
        }
    }
    let grid = *flow.grid();
    let mut spin = 0.0;
    for cell in 0..grid.len() {
        if let Some((rho, u)) = flow.moments(cell) {
            let p = grid.centre(cell);
            spin += rho * ((p[0] - c) * u[1] - (p[1] - c) * u[0]);
        }
    }
    // The fluid's mass and the body's reference share of the cells it covers beyond the start's.
    let drift = (flow.mass() + flow.rotors()[0].displaced() - mass0) / mass0;
    let turns = flow.rotors()[0].angle() / (2.0 * std::f64::consts::PI);
    println!(
        "paddle, {turns:.2} turns: mean torque {torque:.4e} (lattice), fluid angular momentum {spin:.4e}, mass drift {drift:.2e}, power to the fluid {:.4e}",
        -torque * omega
    );
    assert!(turns > 1.0, "{turns}");
    assert!(torque < 0.0, "the fluid resists the paddle: {torque}");
    assert!(spin > 0.0, "the fluid turns with it: {spin}");
    // Round-off only (7.8e-12 measured): the covered and refilled cells' difference and Ladd's
    // mass are given back every step.
    assert!(drift.abs() < 1e-10, "{drift}");
}

/// A four-bladed impeller with pitched blades in a periodic duct pumps the fluid along its axis,
/// and the other way round when it turns the other way.
#[test]
fn pitched_blades_pump_along_the_axis_and_reverse_with_the_rotation() {
    let flow_for = |speed_sign: f64| {
        let (side, length, nu, tip, radius) = (28.0, 24.0, 0.05, 0.08, 10.0);
        let c = side / 2.0;
        let mut d = domain([0.0; 3], [side, side, length], 1.0);
        d.faces.z_min = PERIODIC;
        d.faces.z_max = PERIODIC;
        d.obstacles = vec!["Duct".into()];
        d.rotors = vec![Rotor { body: "Impeller".into(), origin: [c, c, length / 2.0], axis: [0.0, 0.0, 1.0], speed: speed_sign * tip / radius }];
        let mut parts = vec![([c, c, length / 2.0], [2.5, 2.5, 3.0], [0.0; 3])];
        for b in 0..4 {
            let phi = b as f64 * std::f64::consts::FRAC_PI_2;
            // A blade along its radius from just outside the hub (shells may not overlap),
            // pitched 45° about it.
            let (inner, half) = (4.0, (radius - 4.0) / 2.0);
            let mid = inner + half;
            parts.push(([c + mid * phi.cos(), c + mid * phi.sin(), length / 2.0], [half, 3.0, 0.75], [std::f64::consts::FRAC_PI_4, 0.0, phi]));
        }
        let obstacles = [
            Obstacle { body: "Duct".into(), surface: ring_z([c, c], radius + 2.0, side, -1.0, length + 1.0, 96), faces: vec![] },
            Obstacle { body: "Impeller".into(), surface: boxes(&parts), faces: vec![] },
        ];
        let steps = 4000u64;
        let m = model(1.0, nu, d, step(steps as f64, 1.0, None));
        let (_, mut flow) = prepare(&m, &obstacles);
        for k in 0..steps {
            flow.step().unwrap_or_else(|e| panic!("step {k}: {e}"));
        }
        let grid = *flow.grid();
        let (mut axial, mut count) = (0.0, 0usize);
        for cell in 0..grid.len() {
            if let Some((_, u)) = flow.moments(cell) {
                axial += u[2];
                count += 1;
            }
        }
        (axial / count as f64, flow.rotors()[0].axial_torque())
    };
    let (forward, t_forward) = flow_for(1.0);
    let (back, t_back) = flow_for(-1.0);
    println!("pitched impeller: mean axial speed {forward:.4e} turning +z, {back:.4e} turning −z (lattice); torque {t_forward:.4e} / {t_back:.4e}");
    assert!(forward.abs() > 1e-3, "it pumps: {forward}");
    assert!(forward * back < 0.0, "the other way round when reversed: {forward} {back}");
    assert!((forward + back).abs() < 0.1 * forward.abs(), "about as much either way: {forward} {back}");
}
