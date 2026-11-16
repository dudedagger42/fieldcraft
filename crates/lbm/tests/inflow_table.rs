//! A tabulated inlet (`"profile": "table"` and the domain's `inflow_table`): the velocity of each
//! inlet cell is the face's velocity times the table's factor at the cell's coordinate, and with
//! the Spalart–Allmaras model its inflow ν̃ is ν times the table's third column.
//!
//! - A table holding one constant reproduces the uniform inlet bit for bit: velocities, densities
//!   and ν̃, with and without the Spalart–Allmaras model.
//! - Inlet cells read the table (linear between rows, the end rows beyond them): their velocity
//!   and, per cell, their inflow ν̃.

mod common;

use common::*;
use fieldcraft_lbm::Flow;
use fieldcraft_model::fluid_zones::Axis;
use fieldcraft_model::{Collision, FaceCondition, InflowTable, Outflow, Procedure, Profile, Turbulence};

const INFLOW_RATIO: f64 = 3.0;

/// A channel along x, periodic across y, walls at z = 0 and 1, an inlet at x_min with the profile
/// `profile` (and `table`) and a pressure outlet; Spalart–Allmaras with inflow ratio 3 when `sa`.
fn channel(profile: Profile, table: Option<InflowTable>, sa: bool) -> fieldcraft_model::Model {
    let dx = 1.0 / 12.0;
    let mut d = domain([0.0; 3], [2.0, 4.0 * dx, 1.0], dx);
    d.faces.y_min = FaceCondition::Periodic {};
    d.faces.y_max = FaceCondition::Periodic {};
    d.faces.x_min = FaceCondition::Velocity { velocity: [1.0, 0.0, 0.0], profile };
    d.faces.x_max = FaceCondition::Pressure { pressure: 0.0, outflow: Outflow::Uniform };
    d.initial_velocity = [1.0, 0.0, 0.0];
    d.inflow_table = table;
    let dt = 0.05 * dx;
    let mut p: Procedure = collide(300.0 * dt, dt, None, Collision::Trt);
    if let Procedure::Fluid { turbulence, .. } = &mut p
        && sa
    {
        *turbulence = Some(Turbulence::SpalartAllmaras { prandtl: 0.9, wall_model: None, inflow_ratio: INFLOW_RATIO, ddes: None });
    }
    model(1.0, 0.05, d, p)
}

fn table(rows: Vec<[f64; 3]>) -> Option<InflowTable> {
    Some(InflowTable { axis: Axis::Z, rows })
}

/// Densities and velocities as bits, and ν̃ with SA.
fn bits(f: &Flow) -> (Vec<Option<(u64, [u64; 3])>>, Option<Vec<u64>>) {
    let state = (0..f.grid().len()).map(|c| f.moments(c).map(|(r, u)| (r.to_bits(), u.map(f64::to_bits)))).collect();
    (state, f.spalart_allmaras().map(|s| s.nu_tilde().iter().map(|v| v.to_bits()).collect()))
}

/// A table of one constant is the uniform inlet, bit for bit.
#[test]
fn a_constant_table_is_the_uniform_inlet_bit_for_bit() {
    for sa in [false, true] {
        let run = |m: fieldcraft_model::Model| {
            let (setup, mut flow) = prepare(&m, &[]);
            run_all(&setup, &mut flow);
            flow
        };
        let uniform = run(channel(Profile::Uniform, None, sa));
        let tabled = run(channel(Profile::Table, table(vec![[0.0, 1.0, INFLOW_RATIO]]), sa));
        assert_eq!(bits(&uniform), bits(&tabled), "sa {sa}");
        // The same with two equal rows, between which it interpolates.
        let two = run(channel(Profile::Table, table(vec![[0.1, 1.0, INFLOW_RATIO], [0.9, 1.0, INFLOW_RATIO]]), sa));
        assert_eq!(bits(&uniform), bits(&two), "two rows, sa {sa}");
    }
}

/// The inlet cells read the table: velocity along z, and with SA each cell's own ν̃.
#[test]
fn inlet_cells_take_their_velocity_and_nu_tilde_from_the_table() {
    let rows = vec![[0.0, 0.5, 2.0], [0.5, 1.0, 4.0], [1.0, 1.5, 6.0]];
    let m = channel(Profile::Table, table(rows.clone()), true);
    let (setup, mut flow) = prepare(&m, &[]);
    flow.step().unwrap();
    let grid = *flow.grid();
    let t = InflowTable { axis: Axis::Z, rows };
    let nu = (setup.tau_plus - 0.5) / 3.0;
    let u_lat = setup.lattice_velocity;
    let cells = flow.spalart_allmaras().unwrap().inflow_cells().to_vec();
    let mut checked = 0;
    for k in 1..grid.cells[2] - 1 {
        let c = grid.index([0, 1, k]);
        let z = grid.centre(c)[2];
        let (factor, ratio) = t.at(z);
        let (_, u) = flow.moments(c).unwrap();
        assert!((u[0] - u_lat * factor).abs() < 1e-9, "row {k}: u {} want {}", u[0], u_lat * factor);
        let own = cells.iter().find(|e| e.0 as usize == c).unwrap_or_else(|| panic!("row {k} has no inflow ν̃"));
        assert!((own.1 - ratio * nu).abs() <= 1e-15 * (ratio * nu).abs().max(1.0), "row {k}: ν̃ {} want {}", own.1, ratio * nu);
        checked += 1;
    }
    assert!(checked >= 8, "{checked}");
}
