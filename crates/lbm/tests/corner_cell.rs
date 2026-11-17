//! A concave three-wall corner beside a stream: does the corner cell's velocity grow? Print-only, `#[ignore]`.
//!
//! Geometry (periodic box, walls as solid slabs, lattice units with dx = 1/8): a duct along x (floor and ceiling at y, wall at z = 0
//! side, `NZ` wide) driven by a body force to a core speed of `STREAM` (default 0.10), with a cavity cut into the z = 0 side wall:
//! `CAV_W` cells along x, the duct's full height in y, `CAV_D` cells deep in z. The cavity's inner corners (`(x0, y0, 2)`, floor/back/side, and
//! the same under the ceiling) have three solid axis faces; the printed census counts the fluid cells by their number of solid faces.
//!
//! Every `SAMPLE` steps the corner cells' |u|, the largest |u| of the stream and the largest |u| anywhere (with its cell) are printed.
//! Environment: `TAU_MINUS_HALF` (default 2e-4), `SIGMA` (0.95), `STREAM` (0.10), `STEPS` (30,000), `SAMPLE` (500), `CAV_W` (6), `CAV_D` (4),
//! `WALL_MRT` (1), `WALL_CORNER` (0), `MRT_ALL` (0), `RAMP_STEPS` (2,000; the stream is held after it by a feedback on the duct's mean speed).
//!
//! `cargo test --release -p fieldcraft-lbm --test corner_cell -- --ignored --nocapture`

mod common;

use common::*;
use fieldcraft_lbm::backend::Backend;
use fieldcraft_lbm::{Flow, Limits, Setup};
use fieldcraft_model::{Collision, FaceCondition, HrrSigma, Procedure, Turbulence};

const PERIODIC: FaceCondition = FaceCondition::Periodic {};

fn env(name: &str, default: f64) -> f64 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

#[test]
#[ignore = "study: prints the corner cell's speed against time (cargo test --release -p fieldcraft-lbm --test corner_cell -- --ignored --nocapture)"]
fn corner_cell() {
    let (nx, ny, nz) = (28usize, 16usize, 16usize);
    let cav_w = env("CAV_W", 6.0) as usize;
    let cav_d = env("CAV_D", 4.0) as usize;
    let wall = 6usize; // thickness of the side-wall slab in z (the cavity is cut into it)
    let (x0, x1) = (10usize, 10 + cav_w);
    let tau = env("TAU_MINUS_HALF", 2e-4);
    let sigma = env("SIGMA", 0.95);
    let stream = env("STREAM", 0.10);
    let steps = env("STEPS", 30_000.0) as u64;
    let sample = (env("SAMPLE", 500.0) as u64).max(1);
    let mrt_all = env("MRT_ALL", 0.0) != 0.0;
    let nu_lat = tau / 3.0;
    let dx = 1.0 / 8.0;
    let dt = nu_lat * dx * dx;
    let (lx, ly, lz) = (nx as f64 * dx, ny as f64 * dx, nz as f64 * dx);
    let c = |i: usize| i as f64 * dx;
    let big = 3.0 * lx;
    let floor_h = 2usize;
    let solid = |name: &str, lo: [f64; 3], hi: [f64; 3]| obstacle(name, cuboid(lo, hi), &CUBOID_FACES);
    let obstacles = vec![
        solid("Floor", [-big, -big, -big], [big, c(floor_h), big]),
        solid("Ceiling", [-big, c(ny - floor_h), -big], [big, big, big]),
        // the side wall z < wall, cut by the cavity (x0..x1, z from 2 to wall)
        solid("WallBack", [-big, -big, -big], [big, big, c(wall - cav_d)]),
        solid("WallLeft", [-big, -big, -big], [c(x0), big, c(wall)]),
        solid("WallRight", [c(x1), -big, -big], [big, big, c(wall)]),
        // the far side
        solid("WallFar", [-big, -big, c(nz - 2)], [big, big, big]),
    ];
    let mut d = domain([0.0; 3], [lx, ly, lz], dx);
    for f in [&mut d.faces.x_min, &mut d.faces.x_max, &mut d.faces.y_min, &mut d.faces.y_max, &mut d.faces.z_min, &mut d.faces.z_max] {
        *f = PERIODIC;
    }
    d.obstacles = obstacles.iter().map(|o| o.body.clone()).collect();
    // the body force of a plane channel of height H: 12 nu U_mean / H^2, with U_mean = 2/3 of the core speed
    let h = (ny - 2 * floor_h) as f64;
    let g_lat = 12.0 * nu_lat * (stream * 2.0 / 3.0) / (h * h);
    let _ = g_lat;
    d.acceleration = [0.0; 3];
    let mut p = collide(1e6 * dt, dt, None, if mrt_all { Collision::Mrt } else { Collision::Hrr });
    if let Procedure::Fluid { sigma: s, turbulence: t, .. } = &mut p {
        if !mrt_all {
            *s = Some(HrrSigma::Value(sigma));
        }
        *t = Some(Turbulence::Vreman { constant: 0.18, prandtl: 0.9, wall_model: None });
    }
    let m = model(1.0, 1.0, d, p);
    let setup = match Setup::new(&m, None, &obstacles, &Limits::default()) {
        Ok(s) => s,
        Err(e) => {
            println!("CORNER setup refused: {e}");
            return;
        }
    };
    let mut flow = match Flow::new(&setup, &obstacles) {
        Ok(f) => f,
        Err(e) => {
            println!("CORNER refused: {e}");
            return;
        }
    };
    flow.set_wall_layer_mrt(env("WALL_MRT", 1.0) != 0.0 && !mrt_all);
    flow.set_wall_corner_relax(env("WALL_CORNER", 0.0) != 0.0);
    flow.set_wall_tau_floor(env("WALL_TAU_FLOOR", 0.0));
    flow.set_wall_tau_floor_ratio(env("WALL_TAU_FLOOR_K", 0.0));
    let grid = *flow.grid();
    let fluid: Vec<bool> = (0..grid.len()).map(|c| Backend::is_fluid(&flow, c)).collect();
    // the census: fluid cells by their number of solid axis faces
    let faces_of = |cell: usize| -> usize {
        let p = grid.coords(cell);
        let mut n = 0;
        for axis in 0..3 {
            for up in [false, true] {
                let mut q = p;
                let v = if up { (p[axis] + 1) % grid.cells[axis] } else { (p[axis] + grid.cells[axis] - 1) % grid.cells[axis] };
                q[axis] = v;
                if !fluid.get(grid.index(q)).copied().unwrap_or(true) {
                    n += 1;
                }
            }
        }
        n
    };
    let mut census = [0usize; 7];
    let mut corners: Vec<[usize; 3]> = Vec::new();
    for cell in (0..grid.len()).filter(|&c| fluid[c]) {
        let n = faces_of(cell);
        census[n] += 1;
        if n >= 3 {
            corners.push(grid.coords(cell));
        }
    }
    println!(
        "CORNER tau-1/2 {tau:e} sigma {sigma} stream {stream} WALL_MRT {} WALL_CORNER {} MRT_ALL {} cavity {cav_w}x{cav_d}: fluid cells by solid faces 0..6: {census:?}; cells with 3+: {corners:?}",
        env("WALL_MRT", 1.0),
        env("WALL_CORNER", 0.0),
        env("MRT_ALL", 0.0)
    );
    // the corner (solid across x, y and z: faces at -x, -y, -z) and the neighbour across from its solid z face (the follower of the A8 pair)
    let probe = [("corner", [x0, floor_h, wall - cav_d]), ("follower", [x0, floor_h, wall - cav_d + 1])];
    for (name, p) in probe {
        println!("CORNER probe {name} {p:?}: {} solid axis faces", faces_of(grid.index(p)));
    }
    let mut done = 0u64;
    let mut stop = None::<String>;
    // the stream is held: the body force accelerates the duct to STREAM in `RAMP_STEPS` steps, then each chunk sets the force that
    // brings the duct's mean speed back to STREAM within the chunk (never negative)
    let ramp = env("RAMP_STEPS", 2_000.0) as u64;
    let mut mean_x = 0.0f64;
    while done < steps && stop.is_none() {
        let a = if done < ramp { stream / ramp as f64 } else { ((stream - mean_x) / sample as f64).max(0.0) };
        if let Err(e) = Backend::set_acceleration(&mut flow, [a, 0.0, 0.0]) {
            stop = Some(format!("acceleration refused: {e}"));
            break;
        }
        if let Err(e) = Backend::advance(&mut flow, sample) {
            stop = Some(format!("stopped by the solver after step {done}: {e}"));
            break;
        }
        done += sample;
        let f = match Backend::fields(&mut flow) {
            Ok(f) => f,
            Err(e) => {
                stop = Some(format!("could not read the fields: {e}"));
                break;
            }
        };
        let speed = |c: usize| {
            let u = f.velocity[c];
            (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt()
        };
        let (mut jet, mut all, mut at) = (0.0f64, 0.0f64, [0usize; 3]);
        let (mut sum_x, mut n_x) = (0.0f64, 0usize);
        let mut finite = true;
        for cell in (0..grid.len()).filter(|&c| fluid[c]) {
            let s = speed(cell);
            finite &= s.is_finite();
            let p = grid.coords(cell);
            if p[2] >= wall {
                jet = jet.max(s);
                sum_x += f.velocity[cell][0];
                n_x += 1;
            }
            if s > all {
                all = s;
                at = p;
            }
        }
        mean_x = sum_x / n_x.max(1) as f64;
        let reads: Vec<String> = probe
            .iter()
            .map(|(n, p)| {
                let u = f.velocity[grid.index(*p)];
                format!("{n} |u| {:.3e} ux {:+.2e} uy {:+.2e} uz {:+.2e}", speed(grid.index(*p)), u[0], u[1], u[2])
            })
            .collect();
        println!("CORNER step {done}: {} | jet mean {mean_x:.4} max {jet:.4} | all max {all:.4} at {at:?}", reads.join(" "));
        if !finite || all > 0.3 {
            stop = Some(format!("speed {all:.3} (or a non-finite number) at step {done}"));
        }
    }
    println!("CORNER end: {}", stop.unwrap_or_else(|| format!("ran {done} steps")));
}
