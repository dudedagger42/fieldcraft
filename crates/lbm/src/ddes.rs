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
//! [`r_d`], [`f_d`] and [`length`] are pure functions of a cell's values; every one stays finite
//! and in range for any input, hostile included (see each). [`length_field`] applies them to every
//! cell of a grid, with each cell's velocity gradient from its neighbours' velocities.
//!
//! No code of any flow solver was read; this is written from the paper.

use fieldcraft_parallel as parallel;

use crate::distance::WallDistance;
use crate::flow::neighbour;
use crate::voxel::Grid;
use crate::{Error, Result};

/// Kármán's constant in `r_d`: Spalart–Allmaras' own ([`crate::sa::KAPPA`], the wall coupling's
/// [`crate::wall::BOUNCE_KAPPA`], 0.41 as the paper takes it), so that the model, DDES and the wall
/// coupling read one κ.
pub const KAPPA: f64 = crate::sa::KAPPA;

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

/// The largest argument `f_d` passes to tanh. tanh(20) is exactly 1 in double precision (1 − tanh x
/// ≈ 2 e^(−2x) is below half an ulp of 1 from x ≈ 18.7), so the cap changes no CPU result; it keeps
/// a device backend's tanh, which some shader compilers evaluate as sinh/cosh and turn into NaN
/// once cosh overflows single precision (x ≳ 89), on a finite argument (the GPU note, G0).
pub const TANH_ARGUMENT_CAP: f64 = 20.0;

/// The shielding function `f_d = 1 − tanh((8 r_d)³)`, in [0, 1]: 0 (RANS) for a not-a-number
/// `r_d`, 1 for `r_d ≤ 0`. The argument of tanh is capped at [`TANH_ARGUMENT_CAP`].
pub fn f_d(r_d: f64) -> f64 {
    if r_d.is_nan() {
        return 0.0;
    }
    if r_d <= 0.0 {
        return 1.0;
    }
    (1.0 - (8.0 * r_d).powi(3).min(TANH_ARGUMENT_CAP).tanh()).clamp(0.0, 1.0)
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

/// One fluid cell's DDES length `d̃` and shielding `f_d`, from its `nu_total` = ν + ν_t, velocity
/// gradient `gradient` (`g[a][b] = ∂_a u_b`; [`r_d`] reads only its norm), wall distance `d`,
/// filter width `delta` and `c_des`: [`length`] of [`f_d`] of [`r_d`]. What [`length_field`] and the
/// flow's Spalart–Allmaras step both apply per cell, the flow with the gradient it already takes
/// for the vorticity.
pub fn cell(nu_total: f64, gradient: &[[f64; 3]; 3], d: f64, delta: f64, c_des: f64) -> (f64, f64) {
    let f = f_d(r_d(nu_total, gradient, d));
    (length(d, delta, f, c_des), f)
}

/// The cells DDES acts in: those whose centres lie in the box `[min, max]` (metres), by the same
/// rule as a porous zone's cells; a box with no cell centre gives no cell, and DDES is then RANS
/// everywhere. Refused for a grid with more cells than can be addressed.
pub fn zone_mask(grid: &Grid, min: [f64; 3], max: [f64; 3]) -> Result<Vec<bool>> {
    let [nx, ny, nz] = grid.cells;
    let n =
        nx.checked_mul(ny).and_then(|v| v.checked_mul(nz)).ok_or_else(|| Error::Limit(format!("{nx} × {ny} × {nz} cells cannot be addressed")))?;
    let mut mask = Vec::new();
    mask.try_reserve_exact(n).map_err(|_| Error::Limit(format!("no memory for {n} DDES zone flags")))?;
    mask.resize(n, false);
    if let Some((lo, hi)) = crate::setup::cell_range(grid, min, max) {
        for z in lo[2]..=hi[2] {
            for y in lo[1]..=hi[1] {
                for x in lo[0]..=hi[0] {
                    if let Some(m) = mask.get_mut(x + nx * (y + ny * z)) {
                        *m = true;
                    }
                }
            }
        }
    }
    Ok(mask)
}

/// The cells of one parallel task of [`length_field`]: fixed, so the result does not depend on the
/// number of threads.
const BLOCK: usize = 4096;

/// What [`length_field`] found: the share of fluid cells in LES mode (`f_d > 0.5`) and the largest
/// `f_d` of any fluid cell (both 0 without fluid cells).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DdesStats {
    pub les_fraction: f64,
    pub max_f_d: f64,
}

impl DdesStats {
    /// The statistics of per-block `([fluid cells, LES cells], largest f_d)`, summed in block order
    /// (the same on any number of threads).
    pub(crate) fn of_blocks(blocks: impl Iterator<Item = ([usize; 2], f64)>) -> DdesStats {
        let (cells, les, max) = blocks.fold((0usize, 0usize, 0.0f64), |(c, l, m), (k, x)| (c + k[0], l + k[1], m.max(x)));
        DdesStats { les_fraction: if cells > 0 { les as f64 / cells as f64 } else { 0.0 }, max_f_d: max }
    }
}

/// Every cell's DDES length `d̃` into `out` (0 in cells that are not fluid), from the wall distance
/// `d`, each cell's `nu_total` = ν + ν_t and the cell velocities `velocity` (the pass before the
/// collision, the field the hybrid collision and Vreman's model read), on `grid` with `fluid` cells
/// and the `periodic` axes; `delta` is the filter width Δ (the cell size: 1 in lattice units on a
/// uniform grid, each level's own under refinement) and `c_des` C_DES. Run by fixed blocks of cells
/// in parallel: the same result on any number of threads.
///
/// Each cell's velocity gradient is [`crate::hrr::gradient`]'s, from the velocities of its six
/// neighbours (centred differences, one-sided where a neighbour is not fluid, across periodic
/// faces): `g[a][b] = ∂_a u_b`, the transpose of the paper's `U_ij = ∂u_i/∂x_j`; `r_d` reads only
/// its norm `√(g_ab g_ab)`, which the transpose does not change.
///
/// Refused when a slice's length differs from the grid's cell count, or `delta` or `c_des` is not a
/// finite number ≥ 0. Hostile cell values stay in range ([`r_d`], [`f_d`], [`length`]).
#[allow(clippy::too_many_arguments)]
pub fn length_field(
    d: &WallDistance,
    nu_total: &[f64],
    velocity: &[[f64; 3]],
    grid: &Grid,
    fluid: &[bool],
    periodic: [bool; 3],
    delta: f64,
    c_des: f64,
    out: &mut [f64],
) -> Result<DdesStats> {
    let [nx, ny, nz] = grid.cells;
    let n =
        nx.checked_mul(ny).and_then(|v| v.checked_mul(nz)).ok_or_else(|| Error::Limit(format!("{nx} × {ny} × {nz} cells cannot be addressed")))?;
    let distance = d.as_slice();
    let lengths = [
        ("wall distances", distance.len()),
        ("viscosities", nu_total.len()),
        ("velocities", velocity.len()),
        ("fluid flags", fluid.len()),
        ("lengths", out.len()),
    ];
    if let Some((what, len)) = lengths.iter().find(|(_, len)| *len != n) {
        return Err(Error::Invalid(format!("{len} {what} for a grid of {n} cells")));
    }
    if !(delta.is_finite() && delta >= 0.0 && c_des.is_finite() && c_des >= 0.0) {
        return Err(Error::Invalid(format!("the DDES filter width {delta} and constant {c_des} must be finite numbers ≥ 0")));
    }
    // Each block: its lengths, its fluid and LES cell counts and its largest f_d.
    let mut parts: Vec<(&mut [f64], [usize; 2], f64)> = out.chunks_mut(BLOCK).map(|c| (c, [0; 2], 0.0)).collect();
    parallel::for_each_mut(&mut parts, |b, (chunk, counts, max)| {
        for (i, slot) in chunk.iter_mut().enumerate() {
            let cell = b * BLOCK + i;
            *slot = 0.0;
            if !fluid.get(cell).copied().unwrap_or(false) {
                continue;
            }
            let dc = distance.get(cell).copied().unwrap_or(0.0);
            let nu = nu_total.get(cell).copied().unwrap_or(0.0);
            let (l, f) = self::cell(nu, &cell_gradient(grid, periodic, fluid, velocity, cell), dc, delta, c_des);
            *slot = l;
            counts[0] += 1;
            counts[1] += usize::from(f > 0.5);
            *max = max.max(f);
        }
    });
    Ok(DdesStats::of_blocks(parts.iter().map(|(_, counts, max)| (*counts, *max))))
}

/// Cell `cell`'s velocity gradient `g[a][b] = ∂_a u_b` ([`crate::hrr::gradient`]) from the
/// velocities of its fluid neighbours: the one gradient the flow's Spalart–Allmaras step takes per
/// cell for both its vorticity and DDES.
pub(crate) fn cell_gradient(grid: &Grid, periodic: [bool; 3], fluid: &[bool], velocity: &[[f64; 3]], cell: usize) -> [[f64; 3]; 3] {
    let p = grid.coords(cell);
    let u = velocity.get(cell).copied().unwrap_or([0.0; 3]);
    let nb: crate::hrr::Neighbours =
        [0, 1, 2].map(|axis| [false, true].map(|up| neighbour(grid, periodic, fluid, p, axis, up).and_then(|c| velocity.get(c).copied())));
    crate::hrr::gradient(u, &nb)
}

#[cfg(test)]
mod tests {

    /// The cap on tanh's argument changes no value: below it f_d is the uncapped formula bit for
    /// bit, and from it on (where double-precision tanh is already exactly 1) f_d is exactly 0,
    /// up to the largest finite r_d.
    #[test]
    fn the_tanh_cap_changes_no_value() {
        assert_eq!(TANH_ARGUMENT_CAP.tanh(), 1.0);
        let uncapped = |r: f64| (1.0 - (8.0 * r).powi(3).tanh()).clamp(0.0, 1.0);
        let mut r = 1e-6;
        while r < 1e300 {
            assert_eq!(f_d(r).to_bits(), uncapped(r).to_bits(), "r_d {r}");
            r *= 1.01;
        }
        for r in [0.35, 1.0, 1e10, f64::MAX, f64::INFINITY] {
            assert_eq!(f_d(r), 0.0, "r_d {r}");
        }
    }
    use super::*;
    use crate::Limits;
    use crate::distance::WallSurface;

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

    fn grid(cells: [usize; 3]) -> Grid {
        Grid { origin: [0.0; 3], spacing: 1.0, cells }
    }

    /// A wall plane at y = −½ under a grid periodic along x and z.
    fn floor(g: &Grid, fluid: &[bool]) -> WallDistance {
        let [nx, _, nz] = g.cells;
        let pts: Vec<[f64; 3]> = (0..nz).flat_map(|k| (0..nx).map(move |i| [i as f64, -0.5, k as f64])).collect();
        WallDistance::new(g, fluid, &WallSurface::from_points(&pts).unwrap(), [true, false, true], &Limits::default()).unwrap()
    }

    /// A deterministic field: velocities and viscosities that vary from cell to cell.
    fn field(g: &Grid) -> (Vec<f64>, Vec<[f64; 3]>) {
        let nu = (0..g.len()).map(|c| 1e-3 * (1.0 + (c % 7) as f64)).collect();
        let u = (0..g.len())
            .map(|c| {
                let [i, j, k] = g.coords(c).map(|v| v as f64);
                [0.01 * j + 0.002 * (i * 0.7).sin(), 0.001 * (k * 0.3).cos(), 0.0005 * i]
            })
            .collect();
        (nu, u)
    }

    #[test]
    fn rans_where_the_les_length_exceeds_every_distance() {
        let g = grid([6, 9, 5]);
        let mut fluid = vec![true; g.len()];
        fluid[17] = false;
        let d = floor(&g, &fluid);
        let (nu, u) = field(&g);
        let max = d.as_slice().iter().copied().fold(0.0, f64::max);
        let mut out = vec![-1.0; g.len()];
        length_field(&d, &nu, &u, &g, &fluid, [true, false, true], max / C_DES + 1.0, C_DES, &mut out).unwrap();
        assert!(out.iter().zip(d.as_slice()).all(|(a, b)| a.to_bits() == b.to_bits()));
        // At C_DES Δ = max d exactly too.
        length_field(&d, &nu, &u, &g, &fluid, [true, false, true], max, 1.0, &mut out).unwrap();
        assert!(out.iter().zip(d.as_slice()).all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    #[test]
    fn same_on_any_number_of_threads() {
        let g = grid([24, 20, 24]);
        let fluid = vec![true; g.len()];
        let d = floor(&g, &fluid);
        let (nu, u) = field(&g);
        let run = |t| {
            parallel::with_threads(t, || {
                let mut out = vec![0.0; g.len()];
                let s = length_field(&d, &nu, &u, &g, &fluid, [true, false, true], 1.0, C_DES, &mut out).unwrap();
                (out, s)
            })
        };
        let one = run(1);
        assert!(one.1.les_fraction > 0.0 && one.1.les_fraction < 1.0, "{:?}", one.1);
        for t in [2, 7] {
            let other = run(t);
            assert!(one.0.iter().zip(&other.0).all(|(a, b)| a.to_bits() == b.to_bits()) && one.1 == other.1, "{t} threads");
        }
    }

    #[test]
    fn stats_equal_a_hand_count() {
        // At rest the gradient norm is floored: ν_total = 0 gives r_d = 0 (f_d = 1), any
        // ν_total > 0 a huge r_d (f_d = 0).
        let g = grid([4, 4, 4]);
        let mut fluid = vec![true; 64];
        fluid[0] = false;
        let d = floor(&g, &fluid);
        let u = vec![[0.0; 3]; 64];
        let mut nu = vec![1e-3; 64];
        for c in [0, 5, 9, 22, 31, 40, 41, 63] {
            nu[c] = 0.0;
        }
        let mut out = vec![0.0; 64];
        let s = length_field(&d, &nu, &u, &g, &fluid, [true, false, true], 1.0, C_DES, &mut out).unwrap();
        // Seven LES cells (cell 0 is solid) of 63 fluid cells.
        assert_eq!(s, DdesStats { les_fraction: 7.0 / 63.0, max_f_d: 1.0 });
        let s = length_field(&d, &[1e-3; 64], &u, &g, &fluid, [true, false, true], 1.0, C_DES, &mut out).unwrap();
        assert_eq!(s, DdesStats { les_fraction: 0.0, max_f_d: 0.0 });
        let s = length_field(&d, &nu, &u, &g, &[false; 64], [true, false, true], 1.0, C_DES, &mut out).unwrap();
        assert_eq!(s, DdesStats::default());
        assert!(out.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn linear_shear_gives_the_hand_r_d() {
        // u = (γ y, 0, 0), γ = 2⁻¹⁰, ν_total = 0.01, a wall at y = −½: r_d = ν / (γ κ² (j + ½)²),
        // evaluated in 50-digit decimal arithmetic for the interior rows j = 1 … 6.
        const HAND: [f64; 6] = [27.073831713926896, 9.746579417013683, 4.972744600517185, 3.008203523769655, 2.013756077895389, 1.4418016889073495];
        let gamma = 1.0 / 1024.0;
        let g = grid([5, 8, 4]);
        let fluid = vec![true; g.len()];
        let d = floor(&g, &fluid);
        let u: Vec<[f64; 3]> = (0..g.len()).map(|c| [gamma * g.coords(c)[1] as f64, 0.0, 0.0]).collect();
        let nu = vec![0.01; g.len()];
        let mut out = vec![0.0; g.len()];
        length_field(&d, &nu, &u, &g, &fluid, [true, false, true], 1.0, C_DES, &mut out).unwrap();
        for c in 0..g.len() {
            let j = g.coords(c)[1];
            let r = r_d(0.01, &cell_gradient(&g, [true, false, true], &fluid, &u, c), d.as_slice()[c]);
            if (1..=6).contains(&j) {
                let hand = HAND[j - 1];
                assert!((r - hand).abs() <= 1e-15 * hand, "row {j}: r_d {r} vs {hand}");
            }
            assert_eq!(out[c].to_bits(), length(d.as_slice()[c], 1.0, f_d(r), C_DES).to_bits());
        }
    }

    #[test]
    fn refuses_mismatched_input_and_stays_in_range_on_hostile_values() {
        let g = grid([4, 5, 4]);
        let n = g.len();
        let fluid = vec![true; n];
        let d = floor(&g, &fluid);
        let (nu, u) = field(&g);
        let mut out = vec![0.0; n];
        let p = [true, false, true];
        assert!(length_field(&d, &nu[1..], &u, &g, &fluid, p, 1.0, C_DES, &mut out).is_err());
        assert!(length_field(&d, &nu, &u[1..], &g, &fluid, p, 1.0, C_DES, &mut out).is_err());
        assert!(length_field(&d, &nu, &u, &g, &fluid[1..], p, 1.0, C_DES, &mut out).is_err());
        assert!(length_field(&d, &nu, &u, &g, &fluid, p, 1.0, C_DES, &mut out[1..]).is_err());
        assert!(length_field(&d, &nu, &u, &grid([4, 5, 5]), &fluid, p, 1.0, C_DES, &mut out).is_err());
        assert!(length_field(&d, &nu, &u, &grid([usize::MAX, 3, 3]), &fluid, p, 1.0, C_DES, &mut out).is_err());
        for (delta, c) in [(f64::NAN, C_DES), (-1.0, C_DES), (1.0, f64::INFINITY), (1.0, -0.1)] {
            assert!(length_field(&d, &nu, &u, &g, &fluid, p, delta, c, &mut out).is_err(), "{delta} {c}");
        }
        // W2's hostile values in every cell's viscosity and velocity.
        let values =
            [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1e300, -1.0, -1e-300, 0.0, 1e-300, 1e-12, 1e-3, 0.5, 1.0, 7.0, 1e6, 1e300, f64::MAX];
        let nu: Vec<f64> = (0..n).map(|c| values[c % 16]).collect();
        let u: Vec<[f64; 3]> = (0..n).map(|c| [values[(3 * c) % 16], values[(5 * c + 1) % 16], values[(7 * c + 2) % 16]]).collect();
        for delta in [0.0, 0.5, 1.0, 10.0] {
            let s = length_field(&d, &nu, &u, &g, &fluid, p, delta, C_DES, &mut out).unwrap();
            assert!((0.0..=1.0).contains(&s.les_fraction) && (0.0..=1.0).contains(&s.max_f_d), "{s:?}");
            for (l, dc) in out.iter().zip(d.as_slice()) {
                assert!(*l >= dc.min(C_DES * delta) && l <= dc, "d̃ {l} for d {dc}");
            }
        }
    }
}
