//! The scoring of the turbulence benchmark cases (`common::cases`) on inputs with known answers,
//! so the heavy case tests (`sa_flat_plate`, `ddes_backstep`) measure what they claim.

mod common;

use common::cases::*;

/// The reference values in the 2DZP case's `expected.json` are Kármán–Schoenherr's formula at
/// the Re_θ their ids name, to the five digits written there.
#[test]
fn the_2dzp_references_are_karman_schoenherr() {
    let refs = expected("sa-flat-plate-2dzp");
    assert_eq!(refs.len(), 5);
    for (id, value, tolerance) in refs {
        let re: f64 = id.trim_start_matches("cf_re").parse().unwrap();
        assert!((karman_schoenherr(re) - value).abs() < 5e-8, "{id}: {} vs {value}", karman_schoenherr(re));
        assert_eq!(tolerance, 0.05, "{id}: the 15a gate is 5 %");
    }
    // The correlation falls with Re_θ over the gated range.
    assert!(karman_schoenherr(4000.0) > karman_schoenherr(13000.0));
}

#[test]
fn the_backstep_reference_is_the_measured_reattachment() {
    let refs = expected("ddes-backstep-driver-seegmiller");
    let r = refs.iter().find(|(id, ..)| id == "reattachment").unwrap();
    assert_eq!((r.1, r.2), (6.26, 0.1));
}

/// θ and δ* of profiles with closed forms: a linear profile u/U = z/δ has θ = δ/6, δ* = δ/2;
/// the midpoint rule is exact for each cell's mean, and second-order for the profile.
#[test]
fn thicknesses_of_known_profiles() {
    let (delta, n) = (1.0, 400);
    let h = 2.0 * delta / n as f64;
    let u: Vec<f64> = (0..n).map(|j| ((j as f64 + 0.5) * h / delta).min(1.0)).collect();
    assert!((momentum_thickness(&u, h, 1.0) - delta / 6.0).abs() < 1e-5);
    assert!((displacement_thickness(&u, h, 1.0) - delta / 2.0).abs() < 1e-12);
    // A uniform stream has none.
    assert_eq!(momentum_thickness(&[2.0; 10], 0.1, 2.0), 0.0);
    // Blasius: c_f Re_θ = 0.664².
    assert!((BLASIUS_CF_RE_THETA - 0.440896).abs() < 1e-12);
}

#[test]
fn interpolation_and_reattachment() {
    let xs = [0.0, 1.0, 2.0, 4.0];
    let ys = [0.0, 10.0, 20.0, 0.0];
    assert_eq!(interpolate(&xs, &ys, 0.5), Some(5.0));
    assert_eq!(interpolate(&xs, &ys, 3.0), Some(10.0));
    assert_eq!(interpolate(&xs, &ys, 4.5), None);
    assert_eq!(interpolate(&xs, &ys, -0.1), None);

    // A small corner bubble (−, +) before the main one: the reattachment is the crossing after
    // the minimum, 6.25 here.
    let x = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 6.5, 7.0];
    let tau = [-0.1, 0.05, -0.2, -0.5, -1.0, -0.8, -0.4, -0.1, 0.1, 0.4];
    assert!((reattachment(&x, &tau).unwrap() - 6.25).abs() < 1e-12);
    // Attached everywhere, or never reattached.
    assert_eq!(reattachment(&x, &[1.0; 10]), None);
    assert_eq!(reattachment(&x, &[-1.0; 10]), None);
}

#[test]
fn settling_and_the_gate_table() {
    let mut h: Vec<Vec<f64>> = (0..20).map(|k| vec![1.0 + 0.01 * k as f64, 2.0]).collect();
    assert!(!settled(&h, 2, 1e-3));
    h.extend((0..5).map(|_| vec![1.0, 2.0]));
    assert!(settled(&h, 2, 1e-3));
    assert!(!settled(&h, 30, 1e-3), "a window longer than the history is not settled");
    assert!(!settled(&[vec![f64::NAN], vec![f64::NAN]], 1, 1e-3));

    let mut s = Score::default();
    assert!(!s.passes(), "no gates is not a pass");
    s.push(Gate::new("2dzp", "c_f at Re_θ 4000", 0.00320, karman_schoenherr(4000.0), 0.05));
    assert!(s.passes());
    s.push(Gate::new("backstep", "x_r/H", 5.5, 6.26, 0.1));
    assert!(!s.passes());
    let t = s.table();
    assert!(t.contains("| pass |") && t.contains("| FAIL |") && t.contains("1 of 2 gates pass"), "{t}");
    assert!(!Gate::new("c", "q", f64::NAN, 1.0, 0.1).passes());
}
