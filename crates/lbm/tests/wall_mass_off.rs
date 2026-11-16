//! The walls' mass correction is off unless the flow's API switches it on, and off it changes
//! nothing: the populations of a short open-channel run (the geometry of `interpolated_open.rs`,
//! 300 steps, CPU f64) hash to what the code gave before the correction existed (FNV-1a over the
//! bits of every population of every cell). Recorded on the commit before the correction; a
//! change that moves it without meaning to fails here.

mod common;

use common::*;
use fieldcraft_lbm::Obstacle;
use fieldcraft_model::{FaceCondition, Outflow, Profile, WallTreatment};

fn hash_after(walls: WallTreatment) -> u64 {
    let (cells, width, nx, steps) = (8usize, 8.5, 32usize, 300u64);
    let dx = 1.0 / cells as f64;
    let dt = 0.05 * dx * dx;
    let u_phys = 0.02 * dx / dt;
    let ny = (width + 4.0f64).ceil() as usize;
    let nz = 4usize;
    let (lx, ly, lz) = (nx as f64 * dx, ny as f64 * dx, nz as f64 * dx);
    let obstacles: Vec<Obstacle> = vec![obstacle("Wall", cuboid([-2.0 * dx, width * dx, -dx], [lx + 2.0 * dx, ly, lz + dx]), &CUBOID_FACES)];
    let mut d = domain([0.0; 3], [lx, ly, lz], dx);
    for f in [&mut d.faces.y_min, &mut d.faces.y_max, &mut d.faces.z_min, &mut d.faces.z_max] {
        *f = FaceCondition::Periodic {};
    }
    d.faces.x_min = FaceCondition::Velocity { velocity: [u_phys, 0.0, 0.0], profile: Profile::Uniform };
    d.faces.x_max = FaceCondition::Pressure { pressure: 0.0, outflow: Outflow::Uniform };
    d.initial_velocity = [u_phys, 0.0, 0.0];
    d.obstacles = vec!["Wall".into()];
    let proc = with_walls(step(steps as f64 * dt, dt, None), walls);
    let (_, mut flow) = prepare(&model(1.0, 1.0, d, proc), &obstacles);
    for _ in 0..steps {
        flow.step().unwrap();
    }
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for c in 0..flow.grid().len() {
        for q in 0..19 {
            for b in flow.population(q, c).to_bits().to_le_bytes() {
                h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
            }
        }
    }
    h
}

#[test]
fn without_the_correction_the_populations_are_what_they_were() {
    let (lin, quad) = (hash_after(WallTreatment::Interpolated), hash_after(WallTreatment::InterpolatedQuadratic));
    println!("linear {lin:#018x} quadratic {quad:#018x}");
    assert_eq!(lin, LINEAR, "linear {lin:#018x}");
    assert_eq!(quad, QUADRATIC, "quadratic {quad:#018x}");
}

const LINEAR: u64 = 0xc3326ee557b0db00;
const QUADRATIC: u64 = 0xe0f21da528905b5b;
