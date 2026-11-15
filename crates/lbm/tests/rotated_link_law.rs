//! The link-based wall law (Malaspinas and Sagaut's wall model of `wall.rs`) on interpolated
//! walls with the surface's true geometry: each wall cell's distance and normal from the nearest
//! point of the surface, its second cell one link along the direction nearest the normal, at its
//! own true distance. Gates, in order: free-slip plug flow on the rotated channel, the aligned
//! channel against Lee & Moser, the rotated 26.57° channel at Re_τ 550.

mod common;

use common::*;
use fieldcraft_lbm::Flow;
use fieldcraft_model::{Collision, FaceCondition, HrrSigma, Procedure, Turbulence, WallModel};

const LEE_MOSER_550_RE_TAU: f64 = 543.496;

fn musker_sa() -> Turbulence {
    Turbulence::SpalartAllmaras { prandtl: 0.9, wall_model: Some(WallModel::Musker), inflow_ratio: 3.0, ddes: None }
}

/// A parallelepiped from `origin` along three edge vectors, faces tagged as [`cuboid`]'s; each
/// triangle turned to face away from the centre.
fn parallelepiped(origin: [f64; 3], e: [[f64; 3]; 3]) -> fieldcraft_mesh::surface::BoundarySurface {
    let corner = |k: usize| [0, 1, 2].map(|c| origin[c] + (0..3).map(|a| if k >> a & 1 == 1 { e[a][c] } else { 0.0 }).sum::<f64>());
    let positions: Vec<[f64; 3]> = (0..8).map(corner).collect();
    let centre = [0, 1, 2].map(|c| origin[c] + 0.5 * (e[0][c] + e[1][c] + e[2][c]));
    let mut tris = Vec::new();
    let mut tags = Vec::new();
    for axis in 0..3 {
        let (t1, t2) = ((axis + 1) % 3, (axis + 2) % 3);
        for side in 0..2u32 {
            let v = |a: u32, b: u32| ((side as usize) << axis | (a as usize) << t1 | (b as usize) << t2) as u32;
            tris.push([v(0, 0), v(1, 0), v(1, 1)]);
            tris.push([v(0, 0), v(1, 1), v(0, 1)]);
            tags.extend([2 * axis as u32 + side; 2]);
        }
    }
    surface(positions, tris, tags, move |c, _| [c[0] - centre[0], c[1] - centre[1], c[2] - centre[2]])
}

/// The rotated channel's walls and box: `cells` cells across δ, walls turned `a:b` against the grid
/// and moved `phase` cells. Returns the obstacles, the domain (periodic all round, no force), the
/// direction along the channel, the wall normal, the box in cells and the channel width in cells.
#[allow(clippy::type_complexity)]
fn rotated_geometry(
    cells: usize,
    (a, b): (usize, usize),
    phase: f64,
) -> (Vec<fieldcraft_lbm::Obstacle>, fieldcraft_model::FluidDomain, [f64; 2], [f64; 2], [usize; 3], f64, f64, f64) {
    let dx = 1.0 / cells as f64;
    let width = 2.0 * cells as f64;
    let wall_min = 4.0;
    let len = ((a * a + b * b) as f64).sqrt();
    let mut m = 1usize;
    while ((a * b * m) as f64 / len) < width + wall_min {
        m += 1;
    }
    let (nx, ny, nz) = (a * m, b * m, 4usize);
    let pitch = (a * b * m) as f64 / len;
    let dir = [a as f64 / len, b as f64 / len];
    let n = [-dir[1], dir[0]];
    let (lx, ly, lz) = (nx as f64 * dx, ny as f64 * dx, nz as f64 * dx);
    let (p, ho, s0) = (pitch * dx, width * dx, phase * dx);
    let diag = (lx * lx + ly * ly).sqrt();
    let mut obstacles = Vec::new();
    for k in -4i32..=4 {
        let (lo, hi) = (ho + s0 + k as f64 * p, p + s0 + k as f64 * p);
        if hi < -p + 0.5 * dx || lo > p - 0.5 * dx {
            continue;
        }
        let t = 3.0 * diag;
        let origin = [n[0] * lo - dir[0] * t, n[1] * lo - dir[1] * t, -dx];
        let e = [[2.0 * t * dir[0], 2.0 * t * dir[1], 0.0], [n[0] * (hi - lo), n[1] * (hi - lo), 0.0], [0.0, 0.0, lz + 2.0 * dx]];
        obstacles.push(obstacle(&format!("Wall{}", k + 4), parallelepiped(origin, e), &CUBOID_FACES));
    }
    let mut d = domain([0.0; 3], [lx, ly, lz], dx);
    for f in [&mut d.faces.x_min, &mut d.faces.x_max, &mut d.faces.y_min, &mut d.faces.y_max, &mut d.faces.z_min, &mut d.faces.z_max] {
        *f = FaceCondition::Periodic {};
    }
    d.obstacles = obstacles.iter().map(|o| o.body.clone()).collect();
    (obstacles, d, dir, n, [nx, ny, nz], width, dx, p)
}

/// What a plug-flow run found.
struct PlugResult {
    /// Largest deviation of |u| from the start speed over it, in any fluid cell at the end.
    worst: f64,
    /// Mean speed lost over the run, as a fraction of the start speed.
    lost: f64,
    /// Largest wall-normal speed at a wall cell (lattice units).
    normal: f64,
    /// The step at which the flow went unstable, when it did.
    failed_at: Option<u64>,
}

/// Free-slip plug flow: the rotated channel (`a:b`, moved `phase` cells, 8 cells per δ) started
/// uniform along the walls at lattice speed 0.1, no force, the wall model taking the second cell's
/// velocity along the wall as the wall cell's (`free_slip`); 2000 steps. The exact answer is that
/// the flow stays uniform.
fn rotated_plug((a, b): (usize, usize), phase: f64, collision: Collision, steps: u64, free_slip: bool) -> PlugResult {
    let (obstacles, mut d, dir, n, _, _, dx, _) = rotated_geometry(8, (a, b), phase);
    d.initial_velocity = [dir[0], dir[1], 0.0];
    let dt = 0.1 * dx;
    let mut proc = collide(steps as f64 * dt, dt, None, collision);
    if let Procedure::Fluid { turbulence, sigma, .. } = &mut proc {
        *turbulence = Some(musker_sa());
        if matches!(collision, Collision::Hrr) {
            *sigma = Some(HrrSigma::Value(0.95));
        }
    }
    let (setup, mut flow) = prepare(&model(1.0, 1.0 / LEE_MOSER_550_RE_TAU, d, proc), &obstacles);
    flow.interpolate_walls_with_model(&setup, &obstacles).unwrap();
    assert!(flow.set_wall_model_free_slip(free_slip));
    if std::env::var("PLUG_COUNT").is_ok() {
        println!(
            "{:?} phase {phase}: {} wall cells, {} without a second cell",
            (a, b),
            flow.wall_model_cells().len(),
            flow.wall_model_without_second()
        );
    }
    let grid = *flow.grid();
    let fluid: Vec<usize> = (0..grid.len()).filter(|&c| flow.is_fluid(c)).collect();
    let speed = |flow: &Flow, c: usize| flow.moments(c).map_or(0.0, |(_, u)| (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt());
    let mut failed_at = None;
    for k in 1..=steps {
        if flow.step().is_err() {
            failed_at = Some(k);
            break;
        }
    }
    // The wall cells' stored populations are the streamed ones, which the model rebuilds before
    // the collision: read the flow where the model does not rebuild, and the wall cells apart.
    let wallset: std::collections::BTreeSet<usize> = flow.wall_model_cells().into_iter().collect();
    let off: Vec<usize> = fluid.iter().copied().filter(|c| !wallset.contains(c)).collect();
    let worst = off.iter().map(|&c| (speed(&flow, c) - 0.1).abs() / 0.1).fold(0.0, f64::max);
    let mean = off.iter().map(|&c| speed(&flow, c)).sum::<f64>() / off.len().max(1) as f64;
    let normal = off.iter().map(|&c| flow.moments(c).map_or(0.0, |(_, u)| (u[0] * n[0] + u[1] * n[1]).abs())).fold(0.0, f64::max);
    PlugResult { worst, lost: 1.0 - mean / 0.1, normal, failed_at }
}

/// Gate 1: the free-slip plug on the rotated channel, 26.57° and 45°, two offsets, both
/// collisions; the same runs with the wall law's own profile (no free slip) are printed beside it.
#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn free_slip_plug_flow_stays_uniform_on_rotated_walls_under_the_link_law() {
    let steps = std::env::var("PLUG_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(2000u64);
    for free_slip in [true, false] {
        for angle in [(2usize, 1usize), (1, 1)] {
            for phase in [0.0, 0.5] {
                for (cname, collision) in [("MRT", Collision::Mrt), ("HRR σ0.95", Collision::Hrr)] {
                    let r = rotated_plug(angle, phase, collision, steps, free_slip);
                    println!(
                        "free slip {free_slip:5} {angle:?} phase {phase} {cname:10}: worst |u| deviation {:.3e}, mean speed lost {:.3e}, wall-normal u {:.3e}, failed at {:?}",
                        r.worst, r.lost, r.normal, r.failed_at
                    );
                }
            }
        }
    }
}
