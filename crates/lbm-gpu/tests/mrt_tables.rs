//! The constant matrices of `collide_mrt` in `lbm.wgsl` against the CPU's moment-space collision (`fieldcraft_lbm::mrt`): the
//! tables are read from the shader's text, the update `f* = f + F − A (f − f_eq + F/2)` of the shader is replayed in f64 with the
//! shader's even/odd layout, and every population must agree with the CPU's to the tables' f32 rounding. Headless.

#![allow(clippy::needless_range_loop)]

use fieldcraft_lbm::lattice::{C, Q, W};
use fieldcraft_lbm::mrt;

const SHADER: &str = include_str!("../src/lbm.wgsl");

/// The numbers of `const NAME: array<f32, N> = array<f32, N>(...);`.
fn table(name: &str, n: usize) -> Vec<f64> {
    let start = SHADER.find(&format!("const {name}: array<f32, {n}>")).expect("table in the shader");
    let open = SHADER[start..].find("array<f32, ").map(|i| start + i + 1).and_then(|s| SHADER[s..].find('(').map(|i| s + i + 1)).expect("open");
    let close = SHADER[open..].find(')').map(|i| open + i).expect("close");
    let v: Vec<f64> = SHADER[open..close].split(',').map(|x| x.trim().parse().expect("number")).collect();
    assert_eq!(v.len(), n, "{name}");
    v
}

/// Deterministic pseudo-random numbers in [-1, 1).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 52) as f64) - 1.0
    }
}

#[test]
fn the_shaders_constant_matrices_reproduce_the_cpu_mrt_collision() {
    let stress = table("MRT_EE_STRESS", 100);
    // the plain tables with the cell's rate omega, and the corner study's (every non-conserved mode at rate 1) against `collide_full`
    for corner in [false, true] {
        let fixed = table(if corner { "MRT_EE_FIXED_CORNER" } else { "MRT_EE_FIXED" }, 100);
        let odd = table(if corner { "MRT_OO_CORNER" } else { "MRT_OO" }, 81);
        // pairs of opposite directions are (1,2), (3,4), ... in the lattice's order
        for p in 0..9 {
            for a in 0..3 {
                assert_eq!(C[2 * p + 1][a], -C[2 * p + 2][a]);
            }
        }
        let mut rng = Rng(7);
        let mut worst = 0.0f64;
        for case in 0..200 {
            let f: [f64; Q] = std::array::from_fn(|_| 0.01 * rng.next());
            let drho: f64 = f.iter().sum();
            let rho = 1.0 + drho;
            let u = [0.1 * rng.next(), 0.1 * rng.next(), 0.1 * rng.next()];
            let forced = case % 2 == 0;
            let force = if forced { [1e-3 * rng.next(), 1e-3 * rng.next(), 1e-3 * rng.next()] } else { [0.0; 3] };
            let omega = if corner { 1.0 } else { 0.1 + 1.89 * (0.5 + 0.5 * rng.next()) };
            // the CPU's collision works on whole populations, the shader's on their deviations from the weights
            let full: [f64; Q] = std::array::from_fn(|q| W[q] + f[q]);
            let mut want = [0.0; Q];
            if corner {
                mrt::collide_full(&full, rho, u, force, forced, &mut want);
            } else {
                mrt::collide(&full, rho, u, force, omega, forced, &mut want);
            }
            for (q, w) in want.iter_mut().enumerate() {
                *w -= W[q];
            }
            // the shader's replay
            let uu: f64 = u.iter().map(|a| a * a).sum();
            let uf: f64 = (0..3).map(|a| u[a] * force[a]).sum();
            let (mut e, mut o, mut se, mut so) = ([0.0; 10], [0.0; 9], [0.0; 10], [0.0; 9]);
            let mut others = 0.0;
            for p in 0..9 {
                let q = 2 * p + 1;
                let c = C[q].map(f64::from);
                let cu: f64 = (0..3).map(|a| c[a] * u[a]).sum();
                let cf: f64 = (0..3).map(|a| c[a] * force[a]).sum();
                let w = W[q];
                let eq_plus = w * (drho + rho * (4.5 * cu * cu - 1.5 * uu));
                others += 2.0 * eq_plus;
                let eq_minus = w * rho * 3.0 * cu;
                if forced {
                    se[p + 1] = w * (9.0 * cu * cf - 3.0 * uf);
                    so[p] = w * 3.0 * cf;
                }
                e[p + 1] = 0.5 * (f[q] + f[q + 1]) - eq_plus + 0.5 * se[p + 1];
                o[p] = 0.5 * (f[q] - f[q + 1]) - eq_minus + 0.5 * so[p];
            }
            if forced {
                se[0] = -3.0 * W[0] * uf;
            }
            e[0] = f[0] - (drho - others) + 0.5 * se[0];
            let ae: Vec<f64> = (0..10).map(|r| (0..10).map(|c| (fixed[10 * r + c] + omega * stress[10 * r + c]) * e[c]).sum()).collect();
            let mut got = [0.0; Q];
            got[0] = f[0] + se[0] - ae[0];
            for p in 0..9 {
                let ao: f64 = (0..9).map(|c| odd[9 * p + c] * o[c]).sum();
                let (even, oddp) = (se[p + 1] - ae[p + 1], so[p] - ao);
                got[2 * p + 1] = f[2 * p + 1] + even + oddp;
                got[2 * p + 2] = f[2 * p + 2] + even - oddp;
            }
            for q in 0..Q {
                worst = worst.max((got[q] - want[q]).abs());
            }
        }
        // the tables are f32-rounded literals (9 digits): agreement to a few 1e-8 of the populations' deviations, 1e-2 at most
        assert!(worst < 5e-9, "largest population difference {worst:e}");
    }
}
