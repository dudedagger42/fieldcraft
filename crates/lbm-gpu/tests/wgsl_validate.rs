//! `lbm.wgsl` parses and validates with naga (the front end and validator wgpu itself uses), for the override
//! settings the collision variants compile with, so a WGSL error shows up without a device. Headless: no adapter needed.

use wgpu::naga::{
    back::{PipelineConstants, pipeline_constants::process_overrides},
    front::wgsl,
    valid::{Capabilities, ValidationFlags, Validator},
};

const SHADER: &str = include_str!("../src/lbm.wgsl");

fn validate(settings: &[(&str, f64)]) -> Result<(), String> {
    let module = wgsl::parse_str(SHADER).map_err(|e| e.emit_to_string(SHADER))?;
    let info = Validator::new(ValidationFlags::all(), Capabilities::all()).validate(&module).map_err(|e| format!("{e:?}"))?;
    let mut constants = PipelineConstants::default();
    for (k, v) in settings {
        constants.insert((*k).to_string(), *v);
    }
    let (module, _) = process_overrides(&module, &info, None, &constants).map_err(|e| format!("{e:?}"))?;
    Validator::new(ValidationFlags::all(), Capabilities::all()).validate(&module).map_err(|e| format!("{e:?}"))?;
    Ok(())
}

#[test]
fn lbm_wgsl_validates_for_the_collision_variants() {
    let hybrid = [("REGULARIZED", 1.0), ("HRR", 1.0)];
    let cases: Vec<(&str, Vec<(&str, f64)>)> = vec![
        ("defaults", vec![]),
        ("hybrid", hybrid.to_vec()),
        ("hybrid with WALL_SIGMA1 = 0", [&hybrid[..], &[("WALL_SIGMA1", 0.0)]].concat()),
        ("hybrid with WALL_SIGMA1 = 1", [&hybrid[..], &[("WALL_SIGMA1", 1.0)]].concat()),
        ("hybrid, dynamic σ, WALL_SIGMA1 = 1", [&hybrid[..], &[("DYNAMIC", 1.0), ("VREMAN", 1.0), ("WALL_SIGMA1", 1.0)]].concat()),
        ("hybrid, SA, WALL_SIGMA1 = 1", [&hybrid[..], &[("SA", 1.0), ("WALL_SIGMA1", 1.0)]].concat()),
        ("hybrid, on-node, WALL_SIGMA1 = 1", [&hybrid[..], &[("ON_NODE", 1.0), ("WALL_SIGMA1", 1.0)]].concat()),
        ("hybrid with WALL_MRT = 0", [&hybrid[..], &[("WALL_MRT", 0.0)]].concat()),
        ("hybrid with WALL_MRT = 1", [&hybrid[..], &[("WALL_MRT", 1.0)]].concat()),
        ("hybrid, Vreman, WALL_MRT = 1", [&hybrid[..], &[("VREMAN", 1.0), ("WALL_MRT", 1.0)]].concat()),
        ("hybrid, SA, WALL_MRT = 1", [&hybrid[..], &[("SA", 1.0), ("WALL_MRT", 1.0)]].concat()),
        ("hybrid with WALL_CORNER = 1", [&hybrid[..], &[("WALL_CORNER", 1.0)]].concat()),
        ("hybrid, Vreman, WALL_CORNER and WALL_MRT = 1", [&hybrid[..], &[("VREMAN", 1.0), ("WALL_CORNER", 1.0), ("WALL_MRT", 1.0)]].concat()),
        ("hybrid, SA, WALL_CORNER = 1", [&hybrid[..], &[("SA", 1.0), ("WALL_CORNER", 1.0)]].concat()),
        ("MRT everywhere", vec![("MRT", 1.0)]),
        ("MRT everywhere, Vreman", vec![("MRT", 1.0), ("VREMAN", 1.0)]),
        ("MRT everywhere, SA", vec![("MRT", 1.0), ("SA", 1.0)]),
        ("hybrid, on-node, WALL_MRT = 1", [&hybrid[..], &[("ON_NODE", 1.0), ("WALL_MRT", 1.0)]].concat()),
    ];
    let failures: Vec<String> = cases.iter().filter_map(|(name, s)| validate(s).err().map(|e| format!("{name}: {e}"))).collect();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
