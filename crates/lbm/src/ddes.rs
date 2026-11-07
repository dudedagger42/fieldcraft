//! Delayed detached-eddy simulation (DDES): the shielding function and length scale that switch
//! the Spalart–Allmaras model between RANS near walls and a subgrid model away from them.
//!
//! From P. R. Spalart, S. Deck, M. L. Shur, K. D. Squires, M. Kh. Strelets, A. Travin, *A new
//! version of detached-eddy simulation, resistant to ambiguous grid densities*, Theor. Comput.
//! Fluid Dyn. 20:181–195, 2006 (doi:10.1007/s00162-006-0015-0), their eqs. (1)–(3):
//!
//! - `r_d = (ν_t + ν) / (√(U_ij U_ij) κ² d²)`, with `U_ij = ∂u_i/∂x_j`, κ = 0.41 and `d` the
//!   wall distance ([`crate::distance`]); the paper floors the velocity-gradient norm at 10⁻¹⁰;
//! - `f_d = 1 − tanh((8 r_d)³)`: 0 in the boundary layer (`r_d` ≈ 1 in its log layer), 1 away
//!   from it (`r_d ≪ 1`);
//! - `d̃ = d − f_d max(0, d − C_DES Δ)`, C_DES = 0.65, with Δ = max(Δx, Δy, Δz), one cell on the
//!   uniform lattice (Δ = 1 in lattice units).
//!
//! `d̃` replaces `d` in the Spalart–Allmaras destruction term and nothing else changes: with
//! `f_d = 0` (or `C_DES Δ ≥ d`) the model is RANS exactly; with `f_d = 1` and `d > C_DES Δ` the
//! destruction balances production at `ν̃ ∝ Δ² |S|`, a Smagorinsky-like subgrid model.
//!
//! **No double counting.** The hybrid eddy viscosity is the Spalart–Allmaras model's own
//! `ν_t = ν̃ f_v1`, in the RANS and the LES regions alike; it *replaces* any subgrid model, it is
//! never added to one. The turbulence choice is one variant of the model (Smagorinsky, Vreman or
//! Spalart–Allmaras with or without DDES), so a Smagorinsky or Vreman `ν_t` is never computed in
//! the same run, and the collision relaxes with `ν + ν_t` exactly as it does with Vreman's
//! explicit eddy viscosity. Van Driest's damping belongs to Smagorinsky's length and does not
//! apply (the model damps itself through `f_v1`). The hybrid regularized collision's filter with
//! a fixed σ is numerical dissipation of the non-hydrodynamic moments, not a modelled stress, so it
//! stays. Its dynamic σ is refused with Spalart–Allmaras, with or without DDES: it scales the
//! filter with a model viscosity, and Spalart–Allmaras' ν_t would let the filter's dissipation grow
//! with the RANS eddy viscosity inside the boundary layer.
//!
//! These are pure functions of a cell's values; every one stays finite and in range for any
//! input, hostile included (see each).
//!
//! No code of any flow solver was read; this is written from the paper.

/// Kármán's constant in `r_d`, the paper's.
pub const KAPPA: f64 = 0.41;

/// The DDES constant C_DES of the paper, for Spalart–Allmaras.
pub const C_DES: f64 = 0.65;

/// The floor of the velocity-gradient norm in `r_d`, the paper's.
pub const GRADIENT_FLOOR: f64 = 1e-10;

/// The shielding parameter `r_d` of a cell: `nu_total` = ν_t + ν, `gradient[i][j]` = ∂u_i/∂x_j,
/// `d` the wall distance, all in one consistent set of units (lattice units in the solver).
///
/// A negative or not-finite `nu_total` counts as 0; a `d` that is not positive or not finite
/// (a cell on a wall, or a corrupted value) gives +∞, which [`f_d`] turns into full shielding
/// (RANS); a not-finite gradient norm is floored like a vanishing one.
pub fn r_d(nu_total: f64, gradient: &[[f64; 3]; 3], d: f64) -> f64 {
    let nu = if nu_total.is_finite() && nu_total > 0.0 { nu_total } else { 0.0 };
    let norm = gradient.iter().flatten().map(|g| g * g).sum::<f64>().sqrt();
    let norm = if norm.is_finite() { norm.max(GRADIENT_FLOOR) } else { GRADIENT_FLOOR };
    if !(d.is_finite() && d > 0.0) {
        return f64::INFINITY;
    }
    let r = nu / (norm * KAPPA * KAPPA * d * d);
    if r.is_nan() { f64::INFINITY } else { r }
}

/// The shielding function `f_d = 1 − tanh((8 r_d)³)`, in [0, 1]: 0 (RANS) for a not-a-number
/// `r_d`, 1 for `r_d ≤ 0`.
pub fn f_d(r_d: f64) -> f64 {
    if r_d.is_nan() {
        return 0.0;
    }
    if r_d <= 0.0 {
        return 1.0;
    }
    (1.0 - (8.0 * r_d).powi(3).tanh()).clamp(0.0, 1.0)
}

/// The DDES length `d̃ = d − f_d max(0, d − c_des Δ)`, in `[min(d, c_des Δ), d]` for a finite
/// `d ≥ 0`, a finite `c_des Δ ≥ 0` and `f_d` in [0, 1]. `f_d` is clamped to [0, 1] (not a number:
/// 0, RANS); a not-finite or negative `c_des Δ` leaves `d` (RANS), and so does a not-finite `d`.
/// The result is held in that interval against rounding.
pub fn length(d: f64, delta: f64, f_d: f64, c_des: f64) -> f64 {
    let f = if f_d.is_nan() { 0.0 } else { f_d.clamp(0.0, 1.0) };
    let les = c_des * delta;
    if !(les.is_finite() && les >= 0.0 && d.is_finite()) || d <= les {
        return d;
    }
    (d - f * (d - les)).clamp(les, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// (ν + ν_t, ∂u_i/∂x_j row by row, d, r_d, f_d, d̃ with Δ = 1 and C_DES = 0.65), the last
    /// three evaluated from the paper's formulae in 50-digit decimal arithmetic.
    #[rustfmt::skip]
    const HAND: [(f64, [f64; 9], f64, f64, f64, f64); 20] = [
        (0.01, [0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.5, 23.79535990481856, 0.0, 0.5),
        (0.05, [0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1.5, 13.219644391565867, 0.0, 1.5),
        (0.2, [0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 4.0, 7.4360499702558, 0.0, 4.0),
        (0.0016667, [0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 12.5, 0.006345556216537775, 0.9998691785235133, 0.6515502344963671),
        (0.01, [0.002, 0.03, 0.0, -0.001, -0.002, 0.0005, 0.0, 0.004, 0.0], 40.0, 0.0012223139182647013, 0.9999990649857451, 0.6500367928109315),
        (0.05, [0.002, 0.03, 0.0, -0.001, -0.002, 0.0005, 0.0, 0.004, 0.0], 0.5, 39.11404538447044, 0.0, 0.5),
        (0.2, [0.002, 0.03, 0.0, -0.001, -0.002, 0.0005, 0.0, 0.004, 0.0], 1.5, 17.38402017087575, 0.0, 1.5),
        (0.0016667, [0.002, 0.03, 0.0, -0.001, -0.002, 0.0005, 0.0, 0.004, 0.0], 4.0, 0.020372306075717776, 0.9956709976123865, 0.6645021579985053),
        (0.01, [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 12.5, 0.0038072575847709696, 0.9999717442563038, 0.6503348305628008),
        (0.05, [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 40.0, 0.00185901249256395, 0.9999967105985033, 0.6501294379488961),
        (0.2, [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.5, 47.59071980963712, 0.0, 0.5),
        (0.0016667, [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1.5, 0.04406636261484566, 0.95621616242224, 0.687216261941096),
        (0.01, [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, -0.0015], 4.0, 1.795079378291447, 0.0, 4.0),
        (0.05, [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, -0.0015], 12.5, 0.9190806416852209, 0.0, 12.5),
        (0.2, [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, -0.0015], 40.0, 0.3590158756582894, 5.272109832894492e-21, 40.0),
        (0.0016667, [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, -0.0015], 0.5, 19.14789631870947, 0.0, 0.5),
        (0.0168, [0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 10.0, 0.09994051160023795, 0.5292084505222674, 5.0519009876168),
        (0.02, [0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 10.0, 0.1189767995240928, 0.3025608155310235, 7.17105637478493),
        (0.025, [0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 10.0, 0.148720999405116, 0.06659894927896631, 9.377299824241666),
        (0.012, [0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 10.0, 0.07138607971445568, 0.8158682282151436, 2.3716320661884067),
    ];

    /// Within 10⁻¹⁵ of the value's scale. For d̃ that scale is `d`: `f_d` itself is rounded to
    /// about 10⁻¹⁶, and `d̃ = d − f_d (d − C_DES Δ)` multiplies that by `d − C_DES Δ`.
    fn close(a: f64, b: f64, scale: f64) -> bool {
        (a - b).abs() <= 1e-15 * scale.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn hand_values_match_the_paper() {
        for (k, (nu, g, d, rd, fd, dt)) in HAND.into_iter().enumerate() {
            let grad = [[g[0], g[1], g[2]], [g[3], g[4], g[5]], [g[6], g[7], g[8]]];
            let r = r_d(nu, &grad, d);
            let f = f_d(r);
            let l = length(d, 1.0, f, C_DES);
            assert!(close(r, rd, rd) && close(f, fd, fd) && close(l, dt, d), "point {k}: r_d {r} vs {rd}, f_d {f} vs {fd}, d̃ {l} vs {dt}");
        }
    }

    #[test]
    fn rans_where_les_length_exceeds_the_wall_distance() {
        // C_DES Δ ≥ d: d̃ = d bit for bit, whatever f_d.
        for d in [0.0, 0.1, 0.5, 0.65] {
            for f in [0.0, 0.3, 1.0] {
                assert_eq!(length(d, 1.0, f, C_DES).to_bits(), d.to_bits());
            }
        }
        // f_d = 0: RANS anywhere.
        assert_eq!(length(37.0, 1.0, 0.0, C_DES), 37.0);
        // f_d = 1 far from walls: C_DES Δ.
        assert!((length(37.0, 2.0, 1.0, C_DES) - 1.3).abs() < 1e-14);
    }

    #[test]
    fn stays_in_range_on_hostile_input() {
        let values =
            [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1e300, -1.0, -1e-300, 0.0, 1e-300, 1e-12, 1e-3, 0.5, 1.0, 7.0, 1e6, 1e300, f64::MAX];
        for &nu in &values {
            for &g in &values {
                for &d in &values {
                    let grad = [[g, 0.0, 0.0], [0.0, g, 0.0], [0.0, 0.0, -g]];
                    let r = r_d(nu, &grad, d);
                    assert!(!r.is_nan() && r >= 0.0, "r_d({nu}, {g}, {d}) = {r}");
                    let f = f_d(r);
                    assert!((0.0..=1.0).contains(&f), "f_d({r}) = {f}");
                    if d.is_finite() && d >= 0.0 {
                        for delta in [0.0, 0.5, 1.0, 10.0] {
                            let l = length(d, delta, f, C_DES);
                            let lo = d.min(C_DES * delta);
                            assert!(l >= lo && l <= d, "d̃({d}, {delta}, {f}) = {l}");
                        }
                    }
                    assert!(!length(d, 1.0, f, C_DES).is_nan() || d.is_nan());
                }
            }
        }
        // No gradient and a wall at d = 0: fully shielded.
        assert_eq!(f_d(r_d(1e-3, &[[0.0; 3]; 3], 0.0)), 0.0);
        // No gradient away from a wall: the floor gives a huge r_d, still shielded.
        assert_eq!(f_d(r_d(1e-3, &[[0.0; 3]; 3], 10.0)), 0.0);
        // Hostile f_d and Δ.
        assert_eq!(length(5.0, 1.0, f64::NAN, C_DES), 5.0);
        assert!((length(5.0, 1.0, 2.0, C_DES) - C_DES).abs() < 1e-15);
        assert_eq!(length(5.0, f64::NAN, 1.0, C_DES), 5.0);
        assert_eq!(length(5.0, -1.0, 1.0, C_DES), 5.0);
        assert_eq!(f_d(f64::NAN), 0.0);
        assert_eq!(f_d(-3.0), 1.0);
    }
}
