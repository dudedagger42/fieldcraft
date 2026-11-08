//! An absorbing layer at velocity inlets and pressure outlets: it drains the acoustic waves those
//! faces reflect, and it leaves a steady flow unchanged.
//!
//! **The term.** This is H. Xu and P. Sagaut's type II absorbing term (*Analysis of the absorbing
//! layers for the weakly-compressible lattice Boltzmann methods*, J. Comput. Phys. 245:14–42,
//! 2013, arXiv:1203.6350, eqs. 24–25 with the diagonal strength of eq. 16). Each layer cell's
//! post-collision populations get
//!
//! `f_q += w χ [f^eq_q(ρ̄, ū) − f^eq_q(ρ, u)]`
//!
//! - `w` is the profile below.
//! - `χ` is the strength per step.
//! - `(ρ̄, ū)` is the reference state.
//! - `(ρ, u)` are the cell's own moments.
//!
//! Type II leaves the relaxation rate alone, so the viscosity can't turn negative (type I's
//! failure, their eq. 23). Their analysis bounds the strength at `χ < 4/s` (§3); we refuse
//! `χ ≥ 2` per step, below it for every `s ∈ (0, 2)`.
//!
//! The term is applied after the collision to the cell's post-collision moments, as a separate
//! sub-step. This is the encapsulated (operator-split) form of B. E. Jordi, C. J. Cotter and
//! S. J. Sherwin (*An adaptive selective frequency damping method*, arXiv:1412.4372, eq. 4). The
//! moments are the plain sums: their eq. 26 with `n = m = 0`.
//!
//! Since it is a difference of equilibria, the term moves the cell's density and momentum by
//! exactly `w χ (ρ̄ − ρ)` and `w χ (ρ̄ū − ρu)`, and adds no non-equilibrium part.
//!
//! **The reference.** It is a low-pass filtered copy of the cell's own state, so a steady state
//! is a fixed point and the term vanishes there. This is selective frequency damping
//! (E. Åkervik, L. Brandt, D. S. Henningson, J. Hœpffner, O. Marxen, P. Schlatter, *Steady
//! solutions of the Navier–Stokes equations by selective frequency damping*, Phys. Fluids
//! 18:068102, 2006): `q̄ ← q̄ + (q − q̄)/N_f` per step. Xu and Sagaut note that a reference taken
//! from the far field is hard to choose with a background flow and can destabilise the layer
//! (§2.2); a time-filtered reference is the usual one.
//!
//! **The profile.** It is Israeli and Orszag's A = n = 4 polynomial as Xu and Sagaut normalise it
//! (eq. 34), scaled to a peak of 1 at ξ = 0.8: `σ̃(ξ) = 3125 (1 − ξ) ξ⁴ / 256`. Here ξ runs from 0
//! at the layer's inner edge to 1 at the open face's own cell layer. So it is zero at both ends,
//! and the open face's rule is untouched.

use crate::checkpoint::{Archive, fixed, vectors};
use crate::lattice::{C, Q, equilibrium};
use crate::{Error, Result};

/// The largest strength per step: below Xu and Sagaut's stability bound `4/s` for every `s`.
pub const MAX_STRENGTH: f64 = 2.0;

/// Israeli and Orszag's n = 4 profile, peak 1 at ξ = 0.8, zero at ξ = 0 and 1; 0 outside [0, 1].
pub fn profile(xi: f64) -> f64 {
    if (0.0..=1.0).contains(&xi) { 3125.0 * (1.0 - xi) * xi.powi(4) / 256.0 } else { 0.0 }
}

/// The weight of a cell `depth` cells in from an open face's own cell layer (depth 0), in a layer
/// `thickness` cells thick.
pub fn weight(depth: usize, thickness: usize) -> f64 {
    if thickness == 0 || depth >= thickness {
        return 0.0;
    }
    profile(1.0 - depth as f64 / thickness as f64)
}

/// One cell's density and momentum.
fn moments(f: &[f64; Q]) -> [f64; 4] {
    let mut m = [0.0; 4];
    for (q, v) in f.iter().enumerate() {
        let c = C.get(q).copied().unwrap_or([0; 3]);
        m[0] += v;
        m[1] += v * f64::from(c[0]);
        m[2] += v * f64::from(c[1]);
        m[3] += v * f64::from(c[2]);
    }
    m
}

fn velocity(m: [f64; 4]) -> [f64; 3] {
    if m[0] > 0.0 { [m[1] / m[0], m[2] / m[0], m[3] / m[0]] } else { [0.0; 3] }
}

/// The type II term on one cell's post-collision populations, `strength` being `w χ`. Then the
/// filter update of its reference `mean` (density and momentum) with `filter = 1/N_f`.
pub fn damp(f: &mut [f64; Q], strength: f64, mean: &mut [f64; 4], filter: f64) {
    let own = moments(f);
    if own[0] > 0.0 && mean[0] > 0.0 {
        let (u, ub) = (velocity(own), velocity(*mean));
        for (q, v) in f.iter_mut().enumerate() {
            *v += strength * (equilibrium(q, mean[0], ub) - equilibrium(q, own[0], u));
        }
    }
    let now = moments(f);
    for (m, n) in mean.iter_mut().zip(now) {
        *m += (n - *m) * filter;
    }
}

/// The layer cells of a flow, their strengths `w χ` and their filtered references.
#[derive(Debug, Clone, Default)]
pub(crate) struct Layer {
    cells: Vec<u32>,
    strength: Vec<f64>,
    mean: Vec<[f64; 4]>,
    filter: f64,
    /// Whether `mean` holds a reference yet: it is taken from the first state the layer sees.
    primed: bool,
    /// Each cell's damped populations, before they are stored (kept for its capacity).
    scratch: Vec<[f64; Q]>,
}

impl Layer {
    /// A layer over `cells` (in increasing order) with strengths `w χ`, and `filter` = 1/N_f.
    pub(crate) fn new(cells: Vec<u32>, strength: Vec<f64>, filter: f64) -> Result<Layer> {
        if cells.len() != strength.len() {
            return Err(Error::Invalid("the absorbing layer's cells and strengths differ in number".into()));
        }
        if !(filter > 0.0 && filter <= 1.0) {
            return Err(Error::Invalid(format!("the absorbing layer's filter rate {filter} is outside (0, 1]")));
        }
        if let Some(s) = strength.iter().find(|s| !(**s >= 0.0 && **s < MAX_STRENGTH)) {
            return Err(Error::Invalid(format!("an absorbing strength of {s} per step is outside [0, {MAX_STRENGTH})")));
        }
        let mut mean = Vec::new();
        mean.try_reserve_exact(cells.len()).map_err(|_| Error::Limit("not enough memory for the absorbing layer".into()))?;
        mean.resize(cells.len(), [0.0; 4]);
        Ok(Layer { cells, strength, mean, filter, primed: false, scratch: Vec::new() })
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    pub(crate) fn cells(&self) -> &[u32] {
        &self.cells
    }

    /// The term on every layer cell of the populations `f` (direction stride `stride`). Cells
    /// are worked out in parallel into scratch, then stored in cell order.
    pub(crate) fn apply(&mut self, f: &mut [f64], stride: usize, block: usize) {
        if self.cells.is_empty() {
            return;
        }
        let read = |cell: usize| -> [f64; Q] { std::array::from_fn(|q| f.get(q * stride + cell).copied().unwrap_or(0.0)) };
        if !self.primed {
            for (m, cell) in self.mean.iter_mut().zip(&self.cells) {
                *m = moments(&read(*cell as usize));
            }
            self.primed = true;
        }
        let mut done = std::mem::take(&mut self.scratch);
        done.clear();
        done.resize(self.cells.len(), [0.0; Q]);
        {
            let (cells, strength, filter) = (&self.cells, &self.strength, self.filter);
            let mut work: Vec<(&mut [f64; Q], &mut [f64; 4])> = done.iter_mut().zip(self.mean.iter_mut()).collect();
            fieldcraft_parallel::chunks_mut(&mut work, block.max(1), |b, chunk| {
                for (i, (p, mean)) in chunk.iter_mut().enumerate() {
                    let k = b * block.max(1) + i;
                    let (Some(&cell), Some(&s)) = (cells.get(k), strength.get(k)) else { continue };
                    **p = read(cell as usize);
                    damp(p, s, mean, filter);
                }
            });
        }
        for (cell, p) in self.cells.iter().zip(&done) {
            for (q, v) in p.iter().enumerate() {
                if let Some(x) = f.get_mut(q * stride + *cell as usize) {
                    *x = *v;
                }
            }
        }
        self.scratch = done;
    }

    /// The filtered references (the cells and strengths are rebuilt from the setup).
    pub(crate) fn checkpoint(&mut self, a: &mut dyn Archive) -> Result<()> {
        fixed(a, self.cells.len() as u64)?;
        let mut primed = self.primed;
        crate::checkpoint::boolean(a, &mut primed)?;
        self.primed = primed;
        let n = self.cells.len();
        vectors(a, &mut self.mean, n)?;
        if self.mean.iter().flatten().any(|v| !v.is_finite()) {
            return Err(Error::Invalid("checkpoint absorbing-layer reference is not finite".into()));
        }
        Ok(())
    }

    /// The bytes its arrays hold.
    pub(crate) fn heap_bytes(&self) -> u64 {
        (self.cells.capacity() * size_of::<u32>()
            + self.strength.capacity() * size_of::<f64>()
            + self.mean.capacity() * size_of::<[f64; 4]>()
            + self.scratch.capacity() * size_of::<[f64; Q]>()) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_profile_peaks_at_one_and_vanishes_at_both_edges() {
        assert_eq!(profile(0.0), 0.0);
        assert_eq!(profile(1.0), 0.0);
        assert!((profile(0.8) - 1.0).abs() < 1e-15);
        assert!((0..=100).all(|i| profile(i as f64 / 100.0) <= 1.0 + 1e-15));
        assert_eq!(profile(-0.1), 0.0);
        assert_eq!(profile(1.1), 0.0);
        assert_eq!(weight(0, 25), 0.0);
        assert_eq!(weight(25, 25), 0.0);
        assert!((weight(5, 25) - 1.0).abs() < 1e-15);
        assert_eq!(weight(3, 0), 0.0);
    }

    fn state(rho: f64, u: [f64; 3], wiggle: f64) -> [f64; Q] {
        std::array::from_fn(|q| equilibrium(q, rho, u) + wiggle * if q % 3 == 1 { 1e-4 } else { -0.5e-4 })
    }

    #[test]
    fn the_term_moves_density_and_momentum_exactly_toward_the_reference() {
        let mut f = state(1.01, [0.05, -0.02, 0.01], 1.0);
        let before = moments(&f);
        let mut mean = [1.0, 0.04, 0.0, 0.0];
        let reference = mean;
        let neq: [f64; Q] = std::array::from_fn(|q| f[q] - equilibrium(q, before[0], velocity(before)));
        damp(&mut f, 0.3, &mut mean, 0.0);
        let after = moments(&f);
        for i in 0..4 {
            let want = before[i] + 0.3 * (reference[i] - before[i]);
            assert!((after[i] - want).abs() < 1e-15, "moment {i}: {} vs {want}", after[i]);
        }
        // No non-equilibrium part is added: what changed is a difference of equilibria.
        for q in 0..Q {
            let added = f[q] - equilibrium(q, before[0], velocity(before)) - neq[q];
            let eq_diff = 0.3 * (equilibrium(q, reference[0], velocity(reference)) - equilibrium(q, before[0], velocity(before)));
            assert!((added - eq_diff).abs() < 1e-15);
        }
    }

    #[test]
    fn a_steady_state_is_a_fixed_point() {
        let f0 = state(1.0, [0.05, 0.0, 0.0], 1.0);
        let mut mean = moments(&f0);
        let mut f = f0;
        for _ in 0..100 {
            damp(&mut f, 0.5, &mut mean, 0.01);
        }
        assert_eq!(f, f0);
    }

    #[test]
    fn an_oscillation_decays_and_the_reference_follows_the_mean() {
        // One cell carrying an acoustic oscillation: its density and momentum excess rotate into
        // each other with a period of 400 steps (a stand-in for the flow between two layer
        // steps). Without the layer the amplitude stays; with it, it decays and the filtered
        // reference settles on the mean state.
        let omega = 2.0 * std::f64::consts::PI / 400.0;
        let run = |strength: f64| {
            let (mut a, mut b) = (1e-3, 0.0);
            let mut mean = [1.0, 0.0, 0.0, 0.0];
            for _ in 0..4000 {
                (a, b) = (a * omega.cos() - b * omega.sin(), a * omega.sin() + b * omega.cos());
                let mut f = state(1.0 + a, [b / (1.0 + a), 0.0, 0.0], 0.0);
                damp(&mut f, strength, &mut mean, 1.0 / 200.0);
                let m = moments(&f);
                (a, b) = (m[0] - 1.0, m[1]);
            }
            ((a * a + b * b).sqrt(), mean)
        };
        let (free, _) = run(0.0);
        let (damped, mean) = run(0.02);
        assert!((free / 1e-3 - 1.0).abs() < 1e-9, "{free}");
        assert!(damped < 1e-2 * free, "{damped} vs {free}");
        assert!((mean[0] - 1.0).abs() < 1e-5 && mean[1].abs() < 1e-5, "{mean:?}");
    }

    #[test]
    fn hostile_layers_are_refused() {
        assert!(Layer::new(vec![1, 2], vec![0.1], 0.1).is_err());
        assert!(Layer::new(vec![1], vec![2.0], 0.1).is_err());
        assert!(Layer::new(vec![1], vec![f64::NAN], 0.1).is_err());
        assert!(Layer::new(vec![1], vec![-0.1], 0.1).is_err());
        assert!(Layer::new(vec![1], vec![0.1], 0.0).is_err());
        assert!(Layer::new(vec![1], vec![0.1], 1.5).is_err());
        assert!(Layer::new(vec![1], vec![0.1], f64::NAN).is_err());
        assert!(Layer::new(vec![], vec![], 0.5).is_ok_and(|l| l.is_empty()));
    }

    #[test]
    fn apply_damps_only_the_layer_cells() {
        let stride = 4;
        let mut f = vec![0.0; Q * stride];
        for cell in 0..stride {
            let s = state(1.0 + 0.01 * cell as f64, [0.02, 0.0, 0.0], 0.0);
            for q in 0..Q {
                f[q * stride + cell] = s[q];
            }
        }
        let mut layer = Layer::new(vec![1, 3], vec![0.5, 0.5], 0.5).unwrap();
        let before = f.clone();
        layer.apply(&mut f, stride, 1);
        // Primed from this state: nothing moves yet.
        assert_eq!(f, before);
        // Disturb cell 1 and 2: only cell 1 (a layer cell) is pulled back.
        for q in 0..Q {
            f[q * stride + 1] *= 1.01;
            f[q * stride + 2] *= 1.01;
        }
        let disturbed = f.clone();
        layer.apply(&mut f, stride, 1);
        for q in 0..Q {
            assert_eq!(f[q * stride + 2], disturbed[q * stride + 2]);
            assert_eq!(f[q * stride], disturbed[q * stride]);
        }
        let rho1: f64 = (0..Q).map(|q| f[q * stride + 1]).sum();
        let rho1_disturbed: f64 = (0..Q).map(|q| disturbed[q * stride + 1]).sum();
        assert!((rho1 - (rho1_disturbed + 0.5 * (1.01 - rho1_disturbed))).abs() < 1e-14);
    }
}
