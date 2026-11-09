//! A moving-wall ramp is part of what a fluid step reads, so a checkpoint written with one ramp
//! is refused when the run is resumed with another.

use fieldcraft_io::{Geometry, checkpoint, results};
use fieldcraft_model::{Fluid, FluidDomain, Material, Model, Procedure, Step};
use serde_json::json;

fn model(ramp: Option<f64>) -> Model {
    let mut domain: FluidDomain =
        serde_json::from_value(json!({"material": "Fluid", "min": [0.0, 0.0, 0.0], "max": [4.0, 4.0, 4.0], "spacing": 1.0})).unwrap();
    domain.moving_wall_ramp = ramp;
    let mut m = Model::default();
    m.add_material("Fluid", Material { density: Some(1.0), fluid: Some(Fluid::Newtonian { viscosity: 0.1 }), ..Material::default() }).unwrap();
    m.add_fluid_domain("Flow", domain).unwrap();
    let procedure = Procedure::Fluid {
        domain: "Flow".into(),
        end_time: Some(10.0),
        steps: None,
        time_step: Some(1.0),
        frames: 2,
        magic: None,
        sigma: None,
        collision: None,
        turbulence: None,
        steady: None,
    };
    m.add_step(Step::new("Run", procedure), None).unwrap();
    m
}

#[test]
fn a_resume_with_another_moving_wall_ramp_is_refused() {
    let identity = |ramp| checkpoint::identity(results::definition_hash(&model(ramp), "Run", &Geometry::default()).unwrap(), b"", &[]);
    assert_eq!(identity(Some(1.0)), identity(Some(1.0)));
    assert_ne!(identity(Some(1.0)), identity(Some(2.0)));
    assert_ne!(identity(Some(1.0)), identity(None));
}
