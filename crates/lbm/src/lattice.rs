//! The D3Q19 velocity set (Qian, d'Humières & Lallemand 1992; Krüger et al. 2017, ch. 3).
//!
//! Directions come in opposite pairs: 1–2, 3–4, …, 17–18, so the opposite of an odd `q` is
//! `q + 1` and of an even `q` (not 0) is `q − 1`.

/// The number of velocities.
pub const Q: usize = 19;

/// The lattice velocities `c_q` in lattice units.
pub const C: [[i32; 3]; Q] = [
    [0, 0, 0],
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
    [1, 1, 0],
    [-1, -1, 0],
    [1, -1, 0],
    [-1, 1, 0],
    [1, 0, 1],
    [-1, 0, -1],
    [1, 0, -1],
    [-1, 0, 1],
    [0, 1, 1],
    [0, -1, -1],
    [0, 1, -1],
    [0, -1, 1],
];

/// The weights `w_q`: 1/3 at rest, 1/18 along the axes, 1/36 along the face diagonals.
pub const W: [f64; Q] = [
    1.0 / 3.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
];

/// The squared speed of sound in lattice units, `c_s² = 1/3`.
pub const CS2: f64 = 1.0 / 3.0;

/// The direction opposite `q`.
#[inline(always)]
pub const fn opposite(q: usize) -> usize {
    if q == 0 {
        0
    } else if q % 2 == 1 {
        q + 1
    } else {
        q - 1
    }
}

/// `c_q` as floating-point numbers.
#[inline(always)]
pub fn c(q: usize) -> [f64; 3] {
    let v = C.get(q).copied().unwrap_or([0; 3]);
    [f64::from(v[0]), f64::from(v[1]), f64::from(v[2])]
}

/// The second-order equilibrium `w_q ρ (1 + 3 c·u + 9/2 (c·u)² − 3/2 u²)`.
#[inline(always)]
pub fn equilibrium(q: usize, rho: f64, u: [f64; 3]) -> f64 {
    let c = c(q);
    let cu = c[0] * u[0] + c[1] * u[1] + c[2] * u[2];
    let uu = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
    W.get(q).copied().unwrap_or(0.0) * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * uu)
}

/// The moving walls' speed factor for time step `step` (1 for the first) of a ramp `ramp_steps`
/// long (the domain's `moving_wall_ramp` over the time step): `½ (1 − cos(π step / ramp_steps))`
/// before its end, exactly 1 from its last step on and without a ramp (`ramp_steps` 0). A
/// standard smooth start (the walls' acceleration is zero at both ends), from no particular
/// paper. It depends only on the step number, so a resumed run continues it and every backend
/// gets the same factors.
pub fn wall_ramp(step: u64, ramp_steps: u64) -> f64 {
    if step >= ramp_steps {
        return 1.0;
    }
    0.5 * (1.0 - (std::f64::consts::PI * step as f64 / ramp_steps as f64).cos())
}

/// The ramp's length in time steps: `ramp` (a time) over `dt`, rounded, at least 1 for any ramp
/// above 0; 0 without one.
pub fn ramp_steps(ramp: Option<f64>, dt: f64) -> u64 {
    match ramp {
        Some(t) if t > 0.0 && dt > 0.0 => {
            let n = (t / dt).round();
            if n.is_finite() { (n as u64).max(1) } else { u64::MAX }
        }
        _ => 0,
    }
}

/// Ladd's moving-wall term for a population bounced back into direction `q` off a wall moving
/// with `u` (lattice units, reference density 1): `2 w_q c_q·u / c_s²` (Ladd 1994).
#[inline(always)]
pub fn moving_wall(q: usize, u: [f64; 3]) -> f64 {
    let c = c(q);
    2.0 * W.get(q).copied().unwrap_or(0.0) * (c[0] * u[0] + c[1] * u[1] + c[2] * u[2]) / CS2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moments_of_the_velocity_set() {
        // Σ w = 1, Σ w c = 0, Σ w c_a c_b = c_s² δ_ab, and opposite directions pair up.
        let mut m0 = 0.0;
        let mut m1 = [0.0; 3];
        let mut m2 = [[0.0; 3]; 3];
        for q in 0..Q {
            let c = c(q);
            m0 += W[q];
            for a in 0..3 {
                m1[a] += W[q] * c[a];
                for b in 0..3 {
                    m2[a][b] += W[q] * c[a] * c[b];
                }
            }
            let o = opposite(q);
            assert_eq!(opposite(o), q);
            assert_eq!([C[o][0], C[o][1], C[o][2]], [-C[q][0], -C[q][1], -C[q][2]]);
            assert_eq!(W[o], W[q]);
        }
        assert!((m0 - 1.0f64).abs() < 1e-15);
        for a in 0..3 {
            assert!(m1[a].abs() < 1e-15);
            for b in 0..3 {
                let want = if a == b { CS2 } else { 0.0 };
                assert!((m2[a][b] - want).abs() < 1e-15);
            }
        }
        // The equilibrium carries the density and momentum it is built from.
        let (rho, u) = (1.03, [0.02, -0.01, 0.005]);
        let (mut r, mut j) = (0.0, [0.0; 3]);
        for q in 0..Q {
            let f = equilibrium(q, rho, u);
            r += f;
            for a in 0..3 {
                j[a] += f * c(q)[a];
            }
        }
        assert!((r - rho).abs() < 1e-15);
        for a in 0..3 {
            assert!((j[a] - rho * u[a]).abs() < 1e-15);
        }
    }
}
