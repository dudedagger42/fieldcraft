//! The step's reattachment and resolved-stress readings (`common/step_readings.rs`): the furthest
//! downstream crossing, the forward-flow station, the resolved share of the shear stress, and the
//! files the running sums are kept in. Each against numbers worked out by hand.

mod common;

use common::step_readings::*;

#[test]
fn the_furthest_crossing_beyond_one_step_height_is_the_primary_reattachment() {
    // Corner bubble: negative before 0.2 H and again from 1.0 to 1.3 H; the main bubble ends at 5.5 H.
    let x = [0.0, 0.2, 0.6, 1.0, 1.3, 2.0, 5.0, 6.0, 8.0];
    let tau = [-1.0, 1.0, 1.0, -1.0, 1.0, -2.0, -1.0, 1.0, 2.0];
    let got = furthest_crossing(&x, &tau, 1.0);
    assert!(matches!(got, Ok(v) if (v - 5.5).abs() < 1e-12), "{got:?}");
    // Beyond 5.6 H there is none.
    assert!(furthest_crossing(&x, &tau, 5.6).is_err());
    // Hostile input is an error: mismatched lengths, a NaN, stations out of order.
    assert!(furthest_crossing(&x, &tau[..3], 1.0).is_err());
    assert!(furthest_crossing(&x, &[f64::NAN; 9], 1.0).is_err());
    assert!(furthest_crossing(&[1.0, 0.0], &[-1.0, 1.0], 0.0).is_err());
}

fn rows(value: f64, n: usize) -> Vec<Vec<f64>> {
    vec![vec![value; n]; ROWS]
}

#[test]
fn the_forward_flow_station_is_where_the_positive_share_rises_through_a_half() -> Result<(), String> {
    let x: Vec<f64> = (0..11).map(|i| i as f64).collect();
    let mut floor = Floor::new(x)?;
    // Share 0, 0, ... then 0.25, 0.75 between x = 6 and 7: the crossing of 0.5 is at 6.5.
    let share: Vec<f64> = (0..11).map(|i| if i < 7 { if i == 6 { 0.25 } else { 0.0 } } else { 1.0 }).collect();
    let share: Vec<f64> = share.iter().enumerate().map(|(i, &v)| if i == 7 { 0.75 } else { v }).collect();
    floor.add(&rows(0.0, 11), &share)?;
    let got = floor.forward_flow(1.0)?;
    assert!((got - 6.5).abs() < 1e-12, "{got}");
    // Two samples average: another all-ones sample lifts every share by half the difference.
    floor.add(&rows(0.0, 11), &[1.0; 11])?;
    assert_eq!(floor.samples(), 2);
    assert!((floor.positive_share()[6] - 0.625).abs() < 1e-12);
    // Now every share is at least a half: no rise through it.
    assert!(floor.forward_flow(1.0).is_err());
    Ok(())
}

#[test]
fn the_floor_means_its_rows_and_refuses_bad_samples() -> Result<(), String> {
    let mut floor = Floor::new(vec![0.0, 1.0, 2.0])?;
    floor.add(&rows(1.0, 3), &[0.0; 3])?;
    floor.add(&rows(3.0, 3), &[1.0; 3])?;
    assert!(floor.mean_rows().iter().all(|r| r.iter().all(|&v| (v - 2.0).abs() < 1e-15)));
    assert_eq!(floor.forward_flow(0.0).map(|_| ()), Err("no negative → non-negative change beyond 0 H".to_string()));
    assert!(floor.add(&rows(1.0, 2), &[0.0; 3]).is_err(), "a short row");
    assert!(floor.add(&rows(f64::NAN, 3), &[0.0; 3]).is_err(), "a NaN");
    assert!(floor.add(&rows(1.0, 3), &[0.0; 2]).is_err(), "short shares");
    assert_eq!(floor.samples(), 2, "refused samples change nothing");
    assert!(Floor::new(vec![1.0]).is_err());
    assert!(Floor::new(vec![0.0, f64::INFINITY]).is_err());
    Ok(())
}

#[test]
fn the_resolved_share_of_a_known_shear_layer() -> Result<(), String> {
    // U = a z with a = 0.5; w' = −e, u' = s e with e = ±1: ⟨u′w′⟩ = −s, so the resolved stress is
    // s. The modelled one is (ν + ν_t) a. The share is s / (s + (ν + ν_t) a) at every height.
    let (a, s, nu, nu_t) = (0.5, 0.01, 0.002, 0.018);
    let z: Vec<f64> = (0..9).map(|k| k as f64 * 0.25).collect();
    let mut st = Station::new(3.0, z.clone())?;
    for e in [1.0, -1.0, 1.0, -1.0] {
        let u: Vec<f64> = z.iter().map(|&h| a * h + s * e).collect();
        let w: Vec<f64> = vec![-e; z.len()];
        st.add(&u, &w, &vec![nu_t; z.len()])?;
    }
    assert_eq!(st.points(), 4);
    let profile = st.profile(nu)?;
    assert_eq!(profile.len(), z.len() - 2);
    for p in &profile {
        assert!((p.resolved - s).abs() < 1e-12, "resolved {}", p.resolved);
        assert!((p.modelled - (nu + nu_t) * a).abs() < 1e-12, "modelled {}", p.modelled);
    }
    let (at, share) = st.resolved_share(nu, 0.5)?;
    assert!(at >= 0.5);
    let want = s / (s + (nu + nu_t) * a);
    assert!((share - want).abs() < 1e-12, "{share} against {want}");
    // Hostile input.
    assert!(st.add(&z, &z, &z[..2]).is_err());
    assert!(st.add(&vec![f64::NAN; z.len()], &z, &z).is_err());
    assert!(st.profile(0.0).is_err(), "ν must be positive");
    assert!(st.resolved_share(nu, 100.0).is_err(), "no height above 100 H");
    assert!(Station::new(0.0, vec![0.0, 0.0, 1.0]).is_err(), "heights must increase");
    assert!(Station::new(0.0, vec![0.0, 1.0]).is_err(), "three heights at least");
    Ok(())
}

#[test]
fn a_station_without_resolved_stress_reads_a_share_of_zero() -> Result<(), String> {
    // A steady (RANS) field: u and w never change, so ⟨u′w′⟩ = 0 and the share is 0.
    let z: Vec<f64> = (0..5).map(|k| k as f64).collect();
    let mut st = Station::new(1.0, z.clone())?;
    for _ in 0..3 {
        let u: Vec<f64> = z.iter().map(|&h| 0.1 * h).collect();
        st.add(&u, &[0.01; 5], &[0.0; 5])?;
    }
    let (_, share) = st.resolved_share(0.01, 0.0)?;
    assert!(share.abs() < 1e-12, "{share}");
    Ok(())
}

fn sample() -> Result<Readings, String> {
    let mut floor = Floor::new(vec![0.0, 1.0, 2.0, 3.0])?;
    floor.add(&rows(0.5, 4), &[0.0, 0.25, 0.75, 1.0])?;
    let mut st = Station::new(1.0, vec![0.0, 0.5, 1.0])?;
    st.add(&[0.0, 0.1, 0.2], &[0.0, 0.01, 0.02], &[0.0, 0.003, 0.004])?;
    Ok(Readings { floor, stations: vec![st] })
}

#[test]
fn the_readings_round_trip_through_their_file_and_refuse_anything_else() -> Result<(), String> {
    let r = sample()?;
    let bytes = r.to_bytes(4200);
    let (steps, back) = Readings::from_bytes(&bytes)?;
    assert_eq!((steps, &back), (4200, &r));
    // The file on disk, written atomically.
    let dir = std::env::temp_dir().join(format!("step_readings_{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let path = dir.join("t3.readings");
    r.save(&path, 7)?;
    assert_eq!(Readings::load(&path)?, (7, r.clone()));
    assert!(!path.with_extension("readings.tmp").exists());
    std::fs::remove_dir_all(&dir).map_err(|e| e.to_string())?;
    // Truncated at every length, wrong magic, an absurd stored length: errors, never a panic.
    for n in 0..bytes.len() {
        assert!(Readings::from_bytes(&bytes[..n]).is_err(), "truncated to {n} bytes");
    }
    let mut wrong = bytes.clone();
    wrong[0] ^= 0xff;
    assert!(Readings::from_bytes(&wrong).is_err());
    let mut huge = bytes.clone();
    huge[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(Readings::from_bytes(&huge).is_err());
    assert!(Readings::load(std::path::Path::new("no/such/file.readings")).is_err());
    Ok(())
}
