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
fn interpolation() {
    let xs = [0.0, 1.0, 2.0, 4.0];
    let ys = [0.0, 10.0, 20.0, 0.0];
    assert_eq!(interpolate(&xs, &ys, 0.5), Some(5.0));
    assert_eq!(interpolate(&xs, &ys, 3.0), Some(10.0));
    assert_eq!(interpolate(&xs, &ys, 4.5), None);
    assert_eq!(interpolate(&xs, &ys, -0.1), None);
}

/// The step's floor strip centres, H/8 apart over 0 … 30 H (`ddes_backstep`'s floor), or `per_h`
/// strips per H.
fn stations(per_h: usize) -> Vec<f64> {
    (0..30 * per_h).map(|k| (k as f64 + 0.5) / per_h as f64).collect()
}

/// A shear linear in x through a known zero: the crossing is exact. At 6.25 H (a strip boundary,
/// all values dyadic) bit for bit; at the measured 6.26 H to round-off.
#[test]
fn the_reattachment_of_a_linear_shear_is_exact() {
    let x = stations(8);
    let tau: Vec<f64> = x.iter().map(|&x| (x - 6.25) / 1024.0).collect();
    assert_eq!(reattachment(&x, &tau), Ok(6.25));
    let tau: Vec<f64> = x.iter().map(|&x| 3e-3 * (x - 6.26)).collect();
    let r = reattachment(&x, &tau).unwrap();
    assert!((r - 6.26).abs() <= 4.0 * f64::EPSILON * 6.26, "{r}");
    // A zero exactly on a station is that station.
    let tau: Vec<f64> = x.iter().map(|&x| x - 6.1875).collect();
    assert_eq!(reattachment(&x, &tau), Ok(6.1875));
}

/// A curved shear, τ = (x − ½)(x − 6.26): positive in a corner eddy under the step, negative in
/// the bubble, positive after it. The edge of the corner eddy (+ to −) is not a crossing; the
/// reattachment's error stays within the documented bound Δx² max|τ''| / (8 min|τ'|) and falls
/// at least threefold each time the strips halve (second order; the constant depends on where
/// 6.26 falls between two stations).
#[test]
fn the_reattachment_of_a_curved_shear_is_within_its_interpolation_bound() {
    let (a, b) = (0.5, 6.26);
    let mut errors = Vec::new();
    for per_h in [8, 16, 32] {
        let x = stations(per_h);
        let tau: Vec<f64> = x.iter().map(|&x| (x - a) * (x - b)).collect();
        assert_eq!(crossings(&x, &tau).unwrap().len(), 1, "the corner eddy's edge is not a crossing");
        let error = (reattachment(&x, &tau).unwrap() - b).abs();
        let dx = 1.0 / per_h as f64;
        // τ'' = 2; |τ'| = |2x − a − b| ≥ b − a − 2 Δx on the bracketing interval.
        let bound = dx * dx * 2.0 / (8.0 * (b - a - 2.0 * dx));
        assert!(error > 0.0 && error <= bound, "{per_h} per H: error {error:.3e}, bound {bound:.3e}");
        errors.push(error);
    }
    assert!(errors.windows(2).all(|w| w[1] < w[0] / 3.0), "{errors:?}");
}

/// No crossing (attached everywhere, or never reattached) or several (a second bubble) are
/// refused, with what was found; so are samples that cannot be read.
#[test]
fn the_reattachment_refuses_none_several_and_bad_samples() {
    let x = stations(8);
    assert_eq!(reattachment(&x, &vec![1e-3; x.len()]), Err(NoReattachment::None { lowest: 1e-3 }));
    assert_eq!(reattachment(&x, &vec![-2e-3; x.len()]), Err(NoReattachment::None { lowest: -2e-3 }));
    // Two bubbles: reattaching at 3.25 H and at 10.25 H.
    let tau: Vec<f64> = x
        .iter()
        .map(|&x| {
            if x < 6.0 {
                (x - 3.25) / 64.0
            } else if x < 8.0 {
                -1.0 / 64.0
            } else {
                (x - 10.25) / 64.0
            }
        })
        .collect();
    assert_eq!(reattachment(&x, &tau), Err(NoReattachment::Several(vec![3.25, 10.25])));
    // A mean that touches zero and separates again counts twice too.
    let tau = [-1.0, 0.0, -1.0, 1.0];
    assert_eq!(reattachment(&[0.0, 1.0, 2.0, 3.0], &tau), Err(NoReattachment::Several(vec![1.0, 2.5])));
    for (xs, tau) in [
        (vec![0.0, 1.0, 2.0], vec![-1.0, 1.0]),
        (vec![0.0], vec![-1.0]),
        (vec![0.0, 1.0, 1.0], vec![-1.0, 1.0, 2.0]),
        (vec![0.0, 2.0, 1.0], vec![-1.0, 1.0, 2.0]),
        (vec![0.0, 1.0, 2.0], vec![-1.0, f64::NAN, 2.0]),
        (vec![0.0, f64::INFINITY, 2.0], vec![-1.0, 1.0, 2.0]),
    ] {
        assert!(matches!(reattachment(&xs, &tau), Err(NoReattachment::Input(_))), "{xs:?} {tau:?}");
    }
}

/// The gated cases' settling rule: 2 flow-throughs, then a time-mean drift under 0.1 %. A change
/// here is a change of the gated cases' acceptance.
#[test]
fn the_gated_settling_rule_is_the_boards() {
    assert_eq!(GATED, Settling { min_flow_throughs: 2.0, drift: 1e-3 });
}

#[test]
fn settling_and_the_gate_table() {
    // A stationary oscillation settles once each window holds whole periods of it.
    let h: Vec<Vec<f64>> = (0..40).map(|k| vec![1.0 + 0.05 * (std::f64::consts::TAU * k as f64 / 4.0).sin(), 2.0]).collect();
    assert!(means_settled(&h, 8, 1e-12));
    assert!(!means_settled(&h, 6, 1e-3), "windows of 1.5 periods see the oscillation");
    // A creep does not.
    let creep: Vec<Vec<f64>> = (0..40).map(|k| vec![1.0 + 1e-3 * k as f64]).collect();
    assert!(!means_settled(&creep, 10, 1e-3));
    assert!(means_settled(&creep, 10, 1e-2));
    assert!(!means_settled(&h, 21, 1.0), "two windows longer than the history are not settled");
    assert!(!means_settled(&h, 0, 1.0));
    assert!(!means_settled(&[vec![f64::NAN], vec![f64::NAN]], 1, 1.0));
    assert!(!means_settled(&[vec![1.0], vec![1.0, 2.0]], 1, 1.0), "ragged rows");
    assert!(!GATED.settled(&h, 8, 1.9), "too few flow-throughs");
    assert!(GATED.settled(&h, 8, 2.0));

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
