//! The `wall_function` wall model: one name for "a wall function with the law of the backend you
//! solve on". On the CPU, with Spalart–Allmaras turbulence, it is the model's own law; with a subgrid
//! model, Spalding's; on the device Spalding's for either until the device's default switches
//! (`DEVICE_WALL_FUNCTION`). The explicit laws mean what they say, no wall model stays resolved
//! no-slip walls, and the model's own law with a subgrid model is still refused.
mod common;

use common::*;
use fieldcraft_lbm::backend::Choice;
use fieldcraft_lbm::setup::{DEVICE_WALL_FUNCTION, resolve_wall_function};
use fieldcraft_lbm::{Limits, Setup};
use fieldcraft_model::{Procedure, Turbulence, WallModel};

fn setup_with(turbulence: Turbulence, choice: Choice) -> fieldcraft_lbm::Result<Setup> {
    let d = domain([0.0; 3], [1.0, 0.5, 0.5], 0.05);
    let mut p = collide(1.0, 0.025, None, fieldcraft_model::Collision::Trt);
    if let Procedure::Fluid { turbulence: t, .. } = &mut p {
        *t = Some(turbulence);
    }
    let m = model(1.0, 0.01, d, p);
    Setup::new_for(&m, None, &[], &Limits::default(), choice)
}

fn sa(wall_model: Option<WallModel>) -> Turbulence {
    Turbulence::SpalartAllmaras { prandtl: 0.9, wall_model, inflow_ratio: 3.0, ddes: None }
}

fn smagorinsky(wall_model: Option<WallModel>) -> Turbulence {
    Turbulence::Smagorinsky { constant: 0.1, prandtl: 0.9, van_driest: None, wall_model }
}

#[test]
fn wall_function_resolves_per_backend_and_model() {
    let resolved = |t: Turbulence, c: Choice| setup_with(t, c).unwrap().wall_model;
    let function = Some(WallModel::WallFunction);
    assert_eq!(resolved(sa(function), Choice::Cpu), Some(WallModel::SpalartAllmarasBounce));
    assert_eq!(resolved(sa(function), Choice::Device), Some(DEVICE_WALL_FUNCTION));
    assert_eq!(DEVICE_WALL_FUNCTION, WallModel::SpaldingBounce, "the device keeps Spalding's law until its default is flipped");
    for c in [Choice::Cpu, Choice::Device] {
        assert_eq!(resolved(smagorinsky(function), c), Some(WallModel::SpaldingBounce), "a subgrid model's law is Spalding's");
    }
    // Nothing else moves: no wall model stays resolved walls, the explicit laws mean what they say.
    for c in [Choice::Cpu, Choice::Device] {
        assert_eq!(resolved(sa(None), c), None);
        assert_eq!(resolved(sa(Some(WallModel::SpaldingBounce)), c), Some(WallModel::SpaldingBounce));
        assert_eq!(resolved(sa(Some(WallModel::SpalartAllmarasBounce)), c), Some(WallModel::SpalartAllmarasBounce));
        assert_eq!(resolved(smagorinsky(Some(WallModel::Musker)), c), Some(WallModel::Musker));
    }
    // `Setup::new` is the CPU's.
    let d = domain([0.0; 3], [1.0, 0.5, 0.5], 0.05);
    let mut p = collide(1.0, 0.025, None, fieldcraft_model::Collision::Trt);
    if let Procedure::Fluid { turbulence: t, .. } = &mut p {
        *t = Some(sa(function));
    }
    assert_eq!(Setup::new(&model(1.0, 0.01, d, p), None, &[], &Limits::default()).unwrap().wall_model, Some(WallModel::SpalartAllmarasBounce));
}

#[test]
fn the_models_own_law_is_still_refused_with_a_subgrid_model() {
    for c in [Choice::Cpu, Choice::Device] {
        let e = setup_with(smagorinsky(Some(WallModel::SpalartAllmarasBounce)), c).err().map(|e| e.to_string());
        assert!(e.as_deref().is_some_and(|e| e.contains("spalart_allmaras_bounce") && e.contains("spalart_allmaras")), "{e:?}");
    }
}

#[test]
fn the_resolution_is_a_pure_function_of_the_request() {
    for (model, sa_model, choice, want) in [
        (WallModel::WallFunction, true, Choice::Cpu, WallModel::SpalartAllmarasBounce),
        (WallModel::WallFunction, false, Choice::Cpu, WallModel::SpaldingBounce),
        (WallModel::WallFunction, false, Choice::Device, WallModel::SpaldingBounce),
        (WallModel::Musker, true, Choice::Cpu, WallModel::Musker),
    ] {
        assert_eq!(resolve_wall_function(model, sa_model, choice), want, "{model:?} {sa_model} {choice:?}");
    }
}
