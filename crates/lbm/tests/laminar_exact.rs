//! Laminar flows with an exact answer, on walls that do not sit on cell faces: how far the interpolated
//! walls (`walls: "interpolated"`, Bouzidi, Firdaouss & Lallemand 2001, linear) are from the closed form.
//!
//! Every case is driven by a body force chosen so the exact mean speed is 0.02 in lattice units, at the
//! lattice viscosity 0.05 (relaxation time 0.65, TRT with the default magic parameter), no turbulence
//! model, in a box periodic on every face, and is run for 10 240 steps (many diffusion times H²/ν: 1 280
//! at 8 cells, 5 120 at 16). The exact flow rate is the reference; the error is the run's flow rate over it, less 1.
//!
//! - **Rotated channel** (`rotated_channel_*`): plane Poiseuille flow between two parallel plane walls turned
//!   26.57° (direction 2:1) or 45° (1:1) against the grid, width H = 8 or 16 cells measured between the wall planes.
//!   The box is sized so the channel closes on itself after one box diagonal; the walls are slabs turned to the
//!   angle. Three offsets of the wall planes against the grid are tried at each angle. Reference (plane Poiseuille
//!   flow, any fluid mechanics text): u(s) = g s (H − s) / (2 ν), flow rate g H³ / (12 ν) per unit depth.
//! - **Width sweep** (`width_sweep_*`): the same channel aligned with the grid and its width changed in
//!   quarter-cell steps from 8.00 to 9.00 cells (16.00 to 17.00 on the finer grid); the wall plane then cuts the
//!   cells at 0, ¼, ½ and ¾ of a cell.
//! - **Square duct** (`square_duct_*`): Poiseuille flow in a square duct of side H = 8 or 16 cells, four slabs
//!   round it, halfway walls. Reference: the classical Fourier series for the rectangular duct (Boussinesq 1868;
//!   in F. M. White, *Viscous Fluid Flow*, the section on flow in a rectangular duct): flow rate g H⁴ c / ν with
//!   c = (1 − (192/π⁵) Σ_{n odd} tanh(nπ/2)/n⁵) / 12, evaluated in this file.
//! - **Open channel** (`open_channel`): the plane channel of width 8.5 cells on the 8-cell grid with a uniform
//!   velocity inlet and a pressure outlet, 64 cells long; the flux through the plane at ⅔ of the length over the
//!   one at ⅓ must be 1, and the pressure gradient of the middle third the exact 12 ν U / H² per cell.
//!
//! **Pass bars:** flow rate within 1 % of exact; open channel flux ratio within 1e-3 of 1 and gradient within 1 %.
//! Each case also prints the mean-density drift per 10 000 steps (interpolated walls do not conserve mass, so a
//! drift is a leak) and the number of links that fell back to half-way bounce-back.
//!
//! **Known misses.** Several cases miss the bar with the linear rule. They are recorded, not hidden: each case's
//! expected error is a frozen value (this code's own result, a band of ±0.05 percentage points), the miss is marked
//! known, and the test fails if a case leaves its band or if a case marked known starts to pass its bar (so a change
//! that fixes one updates the table and says so). The printed `LAMINAR` lines come with `-- --nocapture`.

mod common;

use common::*;
use fieldcraft_lbm::backend::Backend;
use fieldcraft_lbm::{Limits, Obstacle, Setup};
use fieldcraft_mesh::surface::BoundarySurface;
use fieldcraft_model::{Collision, FaceCondition, Outflow, Procedure, Profile, WallTreatment};

const PERIODIC: FaceCondition = FaceCondition::Periodic {};
/// The lattice viscosity, and the exact mean speed in lattice units.
const NU_LAT: f64 = 0.05;
const U_LAT: f64 = 0.02;
const STEPS: u64 = 10_240;
/// The bar on the flow rate, and the band around a frozen error (a fraction of the exact flow).
const BAR: f64 = 0.01;
const BAND: f64 = 5e-4;
const LIN: WallTreatment = WallTreatment::Interpolated;

/// A parallelepiped from `origin` along three edge vectors, faces tagged as [`cuboid`]'s, each triangle
/// turned to face away from the centre.
fn parallelepiped(origin: [f64; 3], e: [[f64; 3]; 3]) -> BoundarySurface {
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

/// What one run measured.
#[derive(Debug, Clone, Copy)]
struct Measured {
    /// The flow rate over the exact one, less 1.
    flow: f64,
    /// The mean density of the fluid cells, change per 10 000 steps (it starts at 1).
    drift: f64,
    /// Links that fell back to half-way bounce-back.
    fallback: u64,
}

/// The periodic box model of `n` cells at `cells` per metre with `obstacles`, the body acceleration `accel` in
/// lattice units and the wall treatment `walls`.
fn periodic_flow(n: [usize; 3], obstacles: &[Obstacle], accel: [f64; 3], walls: WallTreatment, cells: usize) -> fieldcraft_lbm::Flow {
    let dx = 1.0 / cells as f64;
    let dt = NU_LAT * dx * dx;
    let mut d = domain([0.0; 3], n.map(|c| c as f64 * dx), dx);
    for f in [&mut d.faces.x_min, &mut d.faces.x_max, &mut d.faces.y_min, &mut d.faces.y_max, &mut d.faces.z_min, &mut d.faces.z_max] {
        *f = PERIODIC;
    }
    d.obstacles = obstacles.iter().map(|o| o.body.clone()).collect();
    d.acceleration = accel.map(|a| a * dx / (dt * dt));
    let p = with_walls(collide(STEPS as f64 * dt, dt, None, Collision::Trt), walls);
    let m = model(1.0, 1.0, d, p);
    let setup = Setup::new(&m, None, obstacles, &Limits::default()).expect("setup");
    fieldcraft_lbm::Flow::new(&setup, obstacles).expect("flow")
}

/// Run STEPS and measure the flow rate: the sum of the velocity along `direction` over the fluid cells, per cell
/// of length `along` (cells along the channel times the cells deep), over `reference`.
fn measure(mut flow: fieldcraft_lbm::Flow, direction: [f64; 3], along: f64, reference: f64) -> Measured {
    let n = flow.grid().len();
    let fluid: Vec<bool> = (0..n).map(|c| Backend::is_fluid(&flow, c)).collect();
    let mean_rho = |f: &fieldcraft_lbm::backend::Fields| {
        let (s, k) = f.density.iter().zip(&fluid).filter(|(_, fl)| **fl).fold((0.0, 0usize), |(s, k), (r, _)| (s + r, k + 1));
        s / k.max(1) as f64
    };
    let rho0 = mean_rho(&Backend::fields(&mut flow).expect("fields"));
    Backend::advance(&mut flow, STEPS).expect("advance");
    let f = Backend::fields(&mut flow).expect("fields");
    let sum: f64 = (0..n).filter(|&c| fluid[c]).map(|c| (0..3).map(|k| f.velocity[c][k] * direction[k]).sum::<f64>()).sum();
    let links = Backend::wall_links(&flow).unwrap_or_default();
    Measured {
        flow: sum / along / reference - 1.0,
        drift: (mean_rho(&f) - rho0) / STEPS as f64 * 1e4,
        fallback: links.no_second_node + links.no_crossing,
    }
}

/// The plane channel of `width` cells (fractions allowed) turned `a:b` against the grid (b = 0: aligned), its walls
/// offset `phase` cells, on a grid of `cells` per metre.
fn rotated(cells: usize, (a, b): (usize, usize), phase: f64, width: f64) -> Measured {
    let wall_min = 4.0;
    let (nx, ny, pitch, dir) = if b == 0 {
        let ny = (width + wall_min).ceil() as usize;
        (8usize, ny, ny as f64, [1.0, 0.0])
    } else {
        let len = ((a * a + b * b) as f64).sqrt();
        let mut m = 1usize;
        while ((a * b * m) as f64 / len) < width + wall_min {
            m += 1;
        }
        (a * m, b * m, (a * b * m) as f64 / len, [a as f64 / len, b as f64 / len])
    };
    let nz = 4usize;
    let dx = 1.0 / cells as f64;
    let (lx, ly, lz) = (nx as f64 * dx, ny as f64 * dx, nz as f64 * dx);
    let (ho, p, s0) = (width * dx, pitch * dx, phase * dx);
    let nrm = [-dir[1], dir[0]];
    let diag = (lx * lx + ly * ly).sqrt();
    let mut obstacles = Vec::new();
    for k in -4i32..=4 {
        let (lo, hi) = (ho + s0 + k as f64 * p, p + s0 + k as f64 * p);
        let (box_lo, box_hi) = if b == 0 { (0.0, ly) } else { (-p, p) };
        if hi < box_lo + 0.5 * dx || lo > box_hi - 0.5 * dx {
            continue;
        }
        let t = 3.0 * diag;
        let origin = [nrm[0] * lo - dir[0] * t, nrm[1] * lo - dir[1] * t, -dx];
        let e = [[2.0 * t * dir[0], 2.0 * t * dir[1], 0.0], [nrm[0] * (hi - lo), nrm[1] * (hi - lo), 0.0], [0.0, 0.0, lz + 2.0 * dx]];
        obstacles.push(obstacle(&format!("Wall{}", k + 4), parallelepiped(origin, e), &CUBOID_FACES));
    }
    // g = 12 ν U / H² in lattice units gives the exact mean speed U.
    let g_lat = 12.0 * NU_LAT * U_LAT / (width * width);
    let flow = periodic_flow([nx, ny, nz], &obstacles, [g_lat * dir[0], g_lat * dir[1], 0.0], LIN, cells);
    let along = if b == 0 { nx as f64 } else { ((nx * nx + ny * ny) as f64).sqrt() };
    measure(flow, [dir[0], dir[1], 0.0], along * nz as f64, U_LAT * width)
}

/// The flow rate coefficient of the square duct, per H⁴ at body force 1 and viscosity 1.
fn duct_coefficient() -> f64 {
    let pi = std::f64::consts::PI;
    let sum: f64 = (0..200).map(|m| 2 * m + 1).map(|n| ((n as f64) * pi / 2.0).tanh() / (n as f64).powi(5)).sum();
    (1.0 - 192.0 / pi.powi(5) * sum) / 12.0
}

/// The square duct of side `cells` cells: four slabs round it, halfway walls.
fn square_duct(cells: usize) -> Measured {
    let dx = 1.0 / cells as f64;
    let (ho, span) = (cells as f64 * dx, 4.0 * dx);
    let (lo, hi) = (-2.0 * dx, ho + 2.0 * dx);
    let big = 1.0;
    let obstacles = vec![
        obstacle("Floor", cuboid([-big, lo, -big], [span + big, 0.0, ho + big]), &CUBOID_FACES),
        obstacle("Roof", cuboid([-big, ho, -big], [span + big, hi, ho + big]), &CUBOID_FACES),
        obstacle("Left", cuboid([-big, -big, lo], [span + big, ho + big, 0.0]), &CUBOID_FACES),
        obstacle("Right", cuboid([-big, -big, ho], [span + big, ho + big, hi]), &CUBOID_FACES),
    ];
    let dt = NU_LAT * dx * dx;
    let mut d = domain([0.0, lo, lo], [span, hi, hi], dx);
    for f in [&mut d.faces.x_min, &mut d.faces.x_max, &mut d.faces.y_min, &mut d.faces.y_max, &mut d.faces.z_min, &mut d.faces.z_max] {
        *f = PERIODIC;
    }
    d.obstacles = obstacles.iter().map(|o| o.body.clone()).collect();
    let hh = cells as f64;
    let g_lat = U_LAT * NU_LAT / (duct_coefficient() * hh * hh);
    d.acceleration = [g_lat * dx / (dt * dt), 0.0, 0.0];
    let m = model(1.0, 1.0, d, collide(STEPS as f64 * dt, dt, None, Collision::Trt));
    let setup = Setup::new(&m, None, &obstacles, &Limits::default()).expect("setup");
    let flow = fieldcraft_lbm::Flow::new(&setup, &obstacles).expect("flow");
    let gx = flow.grid().cells[0] as f64;
    measure(flow, [1.0, 0.0, 0.0], gx, U_LAT * hh * hh)
}

/// One expected row: its name, the error frozen, whether it is a known miss of the bar, and the drift frozen.
struct Row {
    name: String,
    frozen: f64,
    known: bool,
    drift: f64,
}

fn check(rows: Vec<(Row, Measured)>) {
    let mut failures = Vec::new();
    for (row, m) in &rows {
        let miss = m.flow.abs() > BAR;
        println!(
            "LAMINAR {}: flow rate {:+.4} % of exact (frozen {:+.4} %; {}), mean-density drift {:+.3e} per 10000 steps, {} links at half-way",
            row.name,
            100.0 * m.flow,
            100.0 * row.frozen,
            if miss { "MISSES the 1 % bar" } else { "within the 1 % bar" },
            m.drift,
            m.fallback
        );
        if (m.flow - row.frozen).abs() > BAND {
            failures.push(format!("{}: flow rate error {:+.4} % left its frozen band {:+.4} ± 0.05 %", row.name, 100.0 * m.flow, 100.0 * row.frozen));
        }
        if miss != row.known {
            failures.push(format!("{}: marked known {} but the case {} the bar", row.name, row.known, if miss { "misses" } else { "meets" }));
        }
        if (m.drift - row.drift).abs() > 1e-4 {
            failures.push(format!("{}: drift {:+.3e} left its frozen value {:+.3e} ± 1e-4", row.name, m.drift, row.drift));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The rotated channel on `cells` per metre, rows of (angle, offset, frozen error, known, frozen drift).
fn rotated_rows(cells: usize, table: &[((usize, usize), f64, f64, bool, f64)]) {
    let rows = table
        .iter()
        .map(|&(dir, phase, frozen, known, drift)| {
            let name = format!("rotated channel {}:{}, {cells} cells, offset {phase}", dir.0, dir.1);
            (Row { name, frozen, known, drift }, rotated(cells, dir, phase, cells as f64))
        })
        .collect();
    check(rows);
}

/// The aligned channel of widths `cells` + 0, ¼, ½, ¾, 1 on `cells` per metre, rows of (frozen error, known, frozen drift).
fn width_rows(cells: usize, table: &[(f64, bool, f64); 5]) {
    let rows = table
        .iter()
        .enumerate()
        .map(|(k, &(frozen, known, drift))| {
            let width = cells as f64 + 0.25 * k as f64;
            (Row { name: format!("aligned channel {width} cells wide on {cells} cells"), frozen, known, drift }, rotated(cells, (1, 0), 0.0, width))
        })
        .collect();
    check(rows);
}

// Frozen with develop's linear rule: (angle, offset, error, known, drift).
const ROT_8: [((usize, usize), f64, f64, bool, f64); 6] = [
    ((2, 1), 0.0, -0.006799, false, -3.2970e-02),
    ((2, 1), 0.33, -0.010669, true, 1.2520e-03),
    ((2, 1), 0.67, -0.013132, true, 7.5180e-03),
    ((1, 1), 0.0, -0.022069, true, -7.2010e-03),
    ((1, 1), 0.25, -0.013923, true, -2.0670e-02),
    ((1, 1), 0.5, 0.016006, true, -3.9230e-03),
];
const ROT_16: [((usize, usize), f64, f64, bool, f64); 6] = [
    ((2, 1), 0.0, -0.001215, false, -6.6620e-03),
    ((2, 1), 0.33, -0.002819, false, 1.4220e-03),
    ((2, 1), 0.67, -0.002489, false, 6.1320e-04),
    ((1, 1), 0.0, -0.009315, false, -1.1760e-03),
    ((1, 1), 0.25, -0.008985, false, -1.1820e-03),
    ((1, 1), 0.5, 0.003072, false, -1.4340e-03),
];
const SWEEP_8: [(f64, bool, f64); 5] = [
    (0.007813, false, -4.2340e-13),
    (-0.005419, false, -3.6160e-13),
    (-0.025529, true, -6.1620e-13),
    (0.015103, true, -5.6010e-13),
    (0.006173, false, -6.3710e-13),
];
const SWEEP_16: [(f64, bool, f64); 5] = [
    (0.001953, false, -8.4650e-13),
    (-0.001454, false, -8.7510e-13),
    (-0.007135, false, -8.3640e-13),
    (0.004025, false, -8.7820e-13),
    (0.001730, false, -8.4720e-13),
];

#[test]
fn rotated_channel_8_cells() {
    rotated_rows(8, &ROT_8);
}

#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn rotated_channel_16_cells() {
    rotated_rows(16, &ROT_16);
}

#[test]
fn width_sweep_8_cells() {
    width_rows(8, &SWEEP_8);
}

#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn width_sweep_16_cells() {
    width_rows(16, &SWEEP_16);
}

#[test]
fn square_duct_8_cells() {
    check(vec![(Row { name: "square duct, 8 cells".into(), frozen: 0.014020, known: true, drift: -4.208e-13 }, square_duct(8))]);
}

#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn square_duct_16_cells() {
    check(vec![(Row { name: "square duct, 16 cells".into(), frozen: 0.003671, known: false, drift: -8.055e-13 }, square_duct(16))]);
}

/// The open channel: width 8.5 cells on the 8-cell grid, 64 cells long, uniform inlet at the exact mean speed, pressure
/// outlet, periodic across; read after 20 000 steps. Returns (flux ratio of the ⅔ plane over the ⅓ plane, the middle
/// third's pressure gradient over the exact one for the flow carried less 1, mean density).
fn open_channel() -> (f64, f64, f64) {
    let (cells, width, nx, steps) = (8usize, 8.5f64, 64usize, 20_000u64);
    let dx = 1.0 / cells as f64;
    let dt = NU_LAT * dx * dx;
    let u_phys = U_LAT * dx / dt;
    let ny = (width + 4.0).ceil() as usize;
    let nz = 4usize;
    let (lx, ly, lz) = (nx as f64 * dx, ny as f64 * dx, nz as f64 * dx);
    let ho = width * dx;
    // One wall slab from the channel's width to the box's far face; the channel's other side is the face at 0.
    let obstacles = vec![obstacle("Wall", cuboid([-2.0 * dx, ho, -dx], [lx + 2.0 * dx, ly, lz + dx]), &CUBOID_FACES)];
    let mut d = domain([0.0; 3], [lx, ly, lz], dx);
    for f in [&mut d.faces.y_min, &mut d.faces.y_max, &mut d.faces.z_min, &mut d.faces.z_max] {
        *f = PERIODIC;
    }
    d.faces.x_min = FaceCondition::Velocity { velocity: [u_phys, 0.0, 0.0], profile: Profile::Uniform };
    d.faces.x_max = FaceCondition::Pressure { pressure: 0.0, outflow: Outflow::Uniform };
    d.initial_velocity = [u_phys, 0.0, 0.0];
    d.obstacles = vec!["Wall".into()];
    let p: Procedure = with_walls(collide(steps as f64 * dt, dt, None, Collision::Trt), LIN);
    let m = model(1.0, 1.0, d, p);
    let setup = Setup::new(&m, None, &obstacles, &Limits::default()).expect("setup");
    let mut flow = fieldcraft_lbm::Flow::new(&setup, &obstacles).expect("flow");
    let grid = *flow.grid();
    let [gx, gy, gz] = grid.cells;
    let fluid: Vec<bool> = (0..gx * gy * gz).map(|c| Backend::is_fluid(&flow, c)).collect();
    Backend::advance(&mut flow, steps).expect("advance");
    let f = Backend::fields(&mut flow).expect("fields");
    let plane = |i: usize, g: &dyn Fn(usize) -> f64| -> (f64, usize) {
        let mut s = 0.0;
        let mut n = 0usize;
        for k in 0..gz {
            for j in 0..gy {
                let c = grid.index([i, j, k]);
                if fluid[c] {
                    s += g(c);
                    n += 1;
                }
            }
        }
        (s, n)
    };
    let flux = |i: usize| plane(i, &|c| f.density[c] * f.velocity[c][0]).0;
    let pts: Vec<(f64, f64)> = (gx / 3..2 * gx / 3)
        .map(|i| {
            let (s, n) = plane(i, &|c| f.density[c]);
            (i as f64, s / n.max(1) as f64)
        })
        .collect();
    let n = pts.len() as f64;
    let (mx, my) = (pts.iter().map(|p| p.0).sum::<f64>() / n, pts.iter().map(|p| p.1).sum::<f64>() / n);
    let slope = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum::<f64>() / pts.iter().map(|p| (p.0 - mx) * (p.0 - mx)).sum::<f64>();
    // Density is 3 p in lattice units, so the exact gradient of the flow carried is -3 · 12 ν (ρu) / H² per cell, with ρu the
    // mean over the width at the middle plane.
    let carried = flux(gx / 2) / (gz as f64 * width);
    let exact = -3.0 * 12.0 * NU_LAT * carried / (width * width);
    let total: f64 = f.density.iter().zip(&fluid).filter(|(_, fl)| **fl).map(|(r, _)| r).sum();
    let count = fluid.iter().filter(|fl| **fl).count();
    (flux(2 * gx / 3) / flux(gx / 3), slope / exact - 1.0, total / count.max(1) as f64)
}

/// Frozen with develop's linear rule: (flux ratio, gradient error, mean density).
const OPEN: (f64, f64, f64) = (1.00000, 0.02998, 1.016031);

#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn open_channel_conserves_the_flux_between_interior_planes() {
    let (ratio, gradient, rho) = open_channel();
    println!(
        "LAMINAR open channel: flux ratio {ratio:.5} (bar 1 ± 1e-3: {}), mid-third pressure gradient {:+.3} % of exact (bar ± 1 %: {}), mean density {rho:.6}",
        if (ratio - 1.0).abs() <= 1e-3 { "met" } else { "MISSED" },
        100.0 * gradient,
        if gradient.abs() <= BAR { "met" } else { "MISSED" }
    );
    assert!((ratio - OPEN.0).abs() < 1e-4, "flux ratio {ratio} left its frozen value {}", OPEN.0);
    assert!((gradient - OPEN.1).abs() < BAND, "gradient error {gradient} left its frozen value {}", OPEN.1);
    assert!((rho - OPEN.2).abs() < 1e-4, "mean density {rho} left its frozen value {}", OPEN.2);
}
