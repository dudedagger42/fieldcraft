//! The σ = 1 rule's layer on a box with a ball: the cells are split by kind (next to the ball, on a box face), the layer leaves out the
//! cells near the open faces (`WALL_LAYER_OPEN_MARGIN`), and the density-mode share follows the kind. Without open faces nothing is left out.
mod common;

use common::*;
use fieldcraft_lbm::{Flow, Limits, Setup, WALL_LAYER_OPEN_MARGIN, WALL_LAYER_RULE_BULK, WALL_LAYER_RULE_BULK_SOLID};
use fieldcraft_model::{Collision, FaceCondition, HrrSigma, Outflow, Procedure, Profile, Turbulence};

fn ball_bed() -> (Setup, Vec<fieldcraft_lbm::Obstacle>) {
    ball_bed_at(0.9)
}

fn ball_bed_at(x: f64) -> (Setup, Vec<fieldcraft_lbm::Obstacle>) {
    let dx = 0.1;
    let dt = 0.08 * dx / 0.1;
    let mut d = domain([0.0; 3], [2.0, 1.0, 1.0], dx);
    d.faces.x_min = FaceCondition::Velocity { velocity: [0.1, 0.0, 0.0], profile: Profile::Uniform };
    d.faces.x_max = FaceCondition::Pressure { pressure: 0.0, outflow: Outflow::Uniform };
    d.faces.z_max = FaceCondition::Slip {};
    d.obstacles = vec!["Ball".into()];
    let bodies = vec![obstacle("Ball", sphere([x, 0.5, 0.5], 0.2, 3), &["skin"])];
    let sa = Turbulence::SpalartAllmaras { prandtl: 0.9, wall_model: None, inflow_ratio: 3.0, ddes: None };
    let p = Procedure::Fluid {
        domain: "Flow".into(),
        end_time: None,
        steps: Some(10),
        time_step: Some(dt),
        frames: 2,
        magic: None,
        sigma: Some(HrrSigma::Value(0.99)),
        collision: Some(Collision::Hrr),
        turbulence: Some(sa),
        steady: None,
    };
    let m = model(1.0, 1e-4, d, p);
    let setup = Setup::new(&m, None, &bodies, &Limits::default()).expect("the ball bed");
    (setup, bodies)
}

#[test]
fn the_layer_is_split_by_kind_and_leaves_out_the_cells_near_the_open_faces() {
    let (setup, bodies) = ball_bed();
    let flow = Flow::new(&setup, &bodies).expect("flow");
    let census = flow.wall_sigma_census();
    let (layer, solid) = (flow.wall_sigma_layer_mask(), flow.wall_sigma_solid_mask());
    let grid = flow.grid();
    let nx = grid.cells[0];
    println!("{census:?}");
    assert!(census.obstacle_only > 0 && census.box_face_only > 0, "{census:?}");
    assert!(census.removed_near_open > 0, "{census:?}");
    let in_layer = layer.iter().filter(|&&m| m).count();
    assert_eq!(in_layer, census.obstacle_only + census.box_face_only + census.both);
    assert_eq!(solid.iter().filter(|&&m| m).count(), census.obstacle_only + census.both);
    for c in (0..layer.len()).filter(|&c| layer[c]) {
        let i = grid.coords(c)[0];
        let inside = i < WALL_LAYER_OPEN_MARGIN || i + WALL_LAYER_OPEN_MARGIN >= nx;
        assert!(!inside || solid[c], "box-face layer cell {c} at x index {i} is within the open-face margin");
    }
    assert!((0..layer.len()).all(|c| !solid[c] || layer[c]), "a solid-share cell outside the layer");
    // The ball's layer cells are all away from the box faces here (its gap to the walls is more than two cells).
    assert_eq!(census.both, 0);
    assert_eq!(census.kept_solid_near_open, 0, "{census:?}");
}

#[test]
fn a_solid_inside_the_margin_keeps_its_layer() {
    // The same bed with the ball moved to x = 0.5: part of its layer lies within the margin of the inlet, and the margin leaves it alone.
    let (setup, bodies) = ball_bed_at(0.5);
    let flow = Flow::new(&setup, &bodies).expect("flow");
    let census = flow.wall_sigma_census();
    println!("{census:?}");
    assert!(census.kept_solid_near_open > 0, "{census:?}");
    assert!(census.removed_near_open > 0, "{census:?}");
    let (layer, solid) = (flow.wall_sigma_layer_mask(), flow.wall_sigma_solid_mask());
    let kept = (0..layer.len()).filter(|&c| solid[c] && flow.grid().coords(c)[0] < WALL_LAYER_OPEN_MARGIN).count();
    assert!(kept > 0 && (0..layer.len()).all(|c| !solid[c] || layer[c]));
}

#[test]
fn a_box_without_open_faces_loses_no_layer_cells() {
    let dx = 0.125;
    let mut d = domain([0.0; 3], [4.0 * dx, 1.0, 1.0], dx);
    d.faces.x_min = FaceCondition::Periodic {};
    d.faces.x_max = FaceCondition::Periodic {};
    d.acceleration = [0.1, 0.0, 0.0];
    let p = collide(100.0 * 0.03, 0.03, None, Collision::Hrr);
    let m = model(1.0, 1e-3, d, p);
    let setup = Setup::new(&m, None, &[], &Limits::default()).expect("the duct");
    let flow = Flow::new(&setup, &[]).expect("flow");
    let census = flow.wall_sigma_census();
    assert!(census.box_face_only > 0 && census.removed_near_open == 0 && census.obstacle_only == 0, "{census:?}");
}

#[test]
fn the_density_mode_shares_are_the_documented_constants() {
    assert!((WALL_LAYER_RULE_BULK - 0.03).abs() < 1e-15 && (WALL_LAYER_RULE_BULK_SOLID - 0.5).abs() < 1e-15);
    assert_eq!(WALL_LAYER_OPEN_MARGIN, 4);
    let (setup, bodies) = ball_bed();
    let flow = Flow::new(&setup, &bodies).expect("flow");
    assert!((flow.wall_rule_bulk() - WALL_LAYER_RULE_BULK).abs() < 1e-15);
}
