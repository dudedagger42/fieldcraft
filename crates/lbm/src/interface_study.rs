//! Study harness for a ratio-2 refinement interface (test code only; nothing here is a solver).
//!
//! Pseudo-2D D3Q19 boxes (one coarse cell thick, periodic in y and z) with one fine strip across the
//! box, between two interfaces normal to x, against a uniform fine box of the same physics. They
//! answer which interface treatment lets the hybrid recursive regularised collision cross a
//! refinement interface (the contract in the hub's `cfd-local-refinement-contract.md`):
//!
//! * **A**: M. Rohde, D. Kandhai, J. J. Derksen, H. E. A. van den Akker, *A generic, mass
//!   conservative local grid refinement technique for lattice-Boltzmann schemes*, Int. J. Numer.
//!   Meth. Fluids 51:439–468, 2006 (§2.1, steps 1–5): the populations of the coarse cell next to the
//!   interface that point into the fine strip are handed unchanged to the fine cells that cover it
//!   (a ghost layer one coarse cell thick) and travel with the fine streaming; the populations that
//!   come back are summed (÷ 8) into the coarse cell. No interpolation of populations, no rescaling.
//! * **A′**: A with σ = 0 (pure finite-difference strain) in the first `rows` cells on each side
//!   of the interface (Fable's addition to the contract).
//! * **B**: A with the non-equilibrium part of the crossing populations rescaled by the ratio of
//!   the relaxation times (O. Filippova, D. Hänel; B. Dupuis, B. Chopard; T. Astoul et al.,
//!   arXiv 2004.14887 eqs. 22–24: `f^neq,f = (τ_f / 2τ_c) f^neq,c`), with the mass the rescaling
//!   would change put back on the same crossing populations (a reflux).
//!
//! The strain rate of the hybrid collision at cells next to the interface uses centred
//! differences; the velocity of the ghost cells is interpolated bilinearly from the coarse field
//! (Astoul et al., arXiv 2004.11863 §4.3, 2004.14887 §3.3).

use crate::hrr;
use crate::lattice::{C, Q, W};
use crate::regularized::rebuild;

/// Cells per coarse cell along z (the coarse box is one cell thick).
const NZ_F: usize = 2;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Scheme {
    A,
    /// σ = 0 in the first `rows` cells on each side of the interface.
    APrime(usize),
    B,
}

pub struct Lev {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub f: Vec<[f64; Q]>,
}

impl Lev {
    fn new(nx: usize, ny: usize, nz: usize) -> Lev {
        Lev { nx, ny, nz, f: vec![[0.0; Q]; nx * ny * nz] }
    }
    pub fn at(&self, x: usize, y: usize, z: usize) -> usize {
        (z * self.ny + y) * self.nx + x
    }
}

pub fn macros(f: &[f64; Q]) -> (f64, [f64; 3]) {
    let mut r = 0.0;
    let mut j = [0.0; 3];
    for (q, fq) in f.iter().enumerate() {
        r += fq;
        for (a, ja) in j.iter_mut().enumerate() {
            *ja += fq * f64::from(C[q][a]);
        }
    }
    (r, [j[0] / r, j[1] / r, j[2] / r])
}

/// The equilibrium of the collision (with its third-order terms): the collision with the rate 1.
fn eq3(rho: f64, u: [f64; 3]) -> [f64; Q] {
    let mut out = [0.0; Q];
    rebuild(&[0.0; Q], rho, u, [0.0; 3], false, None, &mut out, |_| 1.0);
    out
}

fn strain_at(u: &[[f64; 3]], l: &Lev, x: usize, y: usize, z: usize, wrap_x: bool) -> ([f64; 3], [f64; 6]) {
    let ix = |x: isize| -> Option<usize> {
        if wrap_x {
            Some(x.rem_euclid(l.nx as isize) as usize)
        } else if x >= 0 && (x as usize) < l.nx {
            Some(x as usize)
        } else {
            None
        }
    };
    let wy = |y: isize| y.rem_euclid(l.ny as isize) as usize;
    let wz = |z: isize| z.rem_euclid(l.nz as isize) as usize;
    let (xi, yi, zi) = (x as isize, y as isize, z as isize);
    let get = |a: Option<usize>, b: usize, c: usize| a.map(|a| u[l.at(a, b, c)]);
    let nb: hrr::Neighbours = [
        [get(ix(xi - 1), y, z), get(ix(xi + 1), y, z)],
        [Some(u[l.at(x, wy(yi - 1), z)]), Some(u[l.at(x, wy(yi + 1), z)])],
        [Some(u[l.at(x, y, wz(zi - 1))]), Some(u[l.at(x, y, wz(zi + 1))])],
    ];
    let me = u[l.at(x, y, z)];
    (me, hrr::strain(&hrr::gradient(me, &nb)))
}

/// A uniform periodic box: the reference.
pub struct Uniform {
    pub l: Lev,
    pub omega: f64,
    pub sigma: f64,
    u: Vec<[f64; 3]>,
}

impl Uniform {
    pub fn new(nx: usize, ny: usize, nz: usize, tau_half: f64, sigma: f64, init: impl Fn(f64, f64) -> (f64, [f64; 3]), cell: f64) -> Uniform {
        let mut l = Lev::new(nx, ny, nz);
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let (r, u) = init((x as f64 + 0.5) * cell, (y as f64 + 0.5) * cell);
                    let i = l.at(x, y, z);
                    l.f[i] = eq3(r, u);
                }
            }
        }
        let n = l.f.len();
        Uniform { l, omega: 1.0 / (0.5 + tau_half), sigma, u: vec![[0.0; 3]; n] }
    }

    pub fn step(&mut self) {
        for (i, f) in self.l.f.iter().enumerate() {
            self.u[i] = macros(f).1;
        }
        let mut out = [0.0; Q];
        for z in 0..self.l.nz {
            for y in 0..self.l.ny {
                for x in 0..self.l.nx {
                    let i = self.l.at(x, y, z);
                    let (r, _) = macros(&self.l.f[i]);
                    let (u, s) = strain_at(&self.u, &self.l, x, y, z, true);
                    rebuild(&self.l.f[i], r, u, [0.0; 3], false, Some((self.sigma, &s)), &mut out, |_| self.omega);
                    self.l.f[i] = out;
                }
            }
        }
        let mut g = vec![[0.0; Q]; self.l.f.len()];
        let (nx, ny, nz) = (self.l.nx as isize, self.l.ny as isize, self.l.nz as isize);
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let i = self.l.at(x as usize, y as usize, z as usize);
                    for q in 0..Q {
                        let c = C[q];
                        let t = self.l.at(
                            (x + (c[0] as isize)).rem_euclid(nx) as usize,
                            (y + (c[1] as isize)).rem_euclid(ny) as usize,
                            (z + (c[2] as isize)).rem_euclid(nz) as usize,
                        );
                        g[t][q] = self.l.f[i][q];
                    }
                }
            }
        }
        self.l.f = g;
    }

    /// ρ and u averaged over blocks of `ratio`³-ish fine cells to coarse cells (`ratio` along x and y, all of z).
    pub fn coarsened(&self, ratio: usize) -> Vec<(f64, [f64; 3])> {
        let (cx, cy) = (self.l.nx / ratio, self.l.ny / ratio);
        let mut out = vec![(0.0, [0.0; 3]); cx * cy];
        for y in 0..self.l.ny {
            for x in 0..self.l.nx {
                for z in 0..self.l.nz {
                    let (r, u) = macros(&self.l.f[self.l.at(x, y, z)]);
                    let o = &mut out[(y / ratio) * cx + x / ratio];
                    o.0 += r;
                    for a in 0..3 {
                        o.1[a] += u[a];
                    }
                }
            }
        }
        let n = (ratio * ratio * self.l.nz) as f64;
        for o in &mut out {
            o.0 /= n;
            for a in 0..3 {
                o.1[a] /= n;
            }
        }
        out
    }
}

/// The coarse box with a fine strip over the coarse columns `ia..ib` (and a ghost layer of one
/// coarse column on each side).
pub struct Pair {
    pub c: Lev,
    pub f: Lev,
    pub ia: usize,
    pub ib: usize,
    pub scheme: Scheme,
    /// Whether the ghost cells collide too in the second fine step (the question the 1D tests settle).
    pub ghost_collision: bool,
    pub sigma: f64,
    omega_c: f64,
    omega_f: f64,
    uc: Vec<[f64; 3]>,
    uf: Vec<[f64; 3]>,
    /// ρ and u of the interface cells at the collision, `[side][y]`.
    edge: [Vec<(f64, [f64; 3])>; 2],
}

impl Pair {
    /// `init(x, y)` gives ρ and u at coarse-unit coordinates (cell centres at half-integers).
    pub fn new(
        nxc: usize,
        nyc: usize,
        ia: usize,
        ib: usize,
        tau_half: f64,
        sigma: f64,
        scheme: Scheme,
        init: impl Fn(f64, f64) -> (f64, [f64; 3]),
    ) -> Pair {
        let nxf = 2 * (ib - ia + 2);
        let mut c = Lev::new(nxc, nyc, 1);
        for y in 0..nyc {
            for x in 0..nxc {
                let (r, u) = init(x as f64 + 0.5, y as f64 + 0.5);
                let i = c.at(x, y, 0);
                c.f[i] = eq3(r, u);
            }
        }
        let mut f = Lev::new(nxf, 2 * nyc, NZ_F);
        for z in 0..NZ_F {
            for y in 0..f.ny {
                for x in 2..nxf - 2 {
                    let (r, u) = init((ia - 1) as f64 + (x as f64 + 0.5) * 0.5, (y as f64 + 0.5) * 0.5);
                    let i = f.at(x, y, z);
                    f.f[i] = eq3(r, u);
                }
            }
        }
        let omega_c = 1.0 / (0.5 + tau_half);
        let omega_f = 1.0 / (0.5 + 2.0 * tau_half);
        let (nc, nf) = (c.f.len(), f.f.len());
        Pair {
            c,
            f,
            ia,
            ib,
            scheme,
            ghost_collision: false,
            sigma,
            omega_c,
            omega_f,
            uc: vec![[0.0; 3]; nc],
            uf: vec![[0.0; 3]; nf],
            edge: [vec![(1.0, [0.0; 3]); nyc], vec![(1.0, [0.0; 3]); nyc]],
        }
    }

    fn covered(&self, x: usize) -> bool {
        x >= self.ia && x < self.ib
    }

    fn real_f(&self, x: usize) -> bool {
        x >= 2 && x < self.f.nx - 2
    }

    fn sigma_coarse(&self, x: usize) -> f64 {
        if let Scheme::APrime(rows) = self.scheme {
            let row = if x < self.ia { self.ia - 1 - x } else { x - self.ib };
            if row < rows {
                return 0.0;
            }
        }
        self.sigma
    }

    fn sigma_fine(&self, xf: usize) -> f64 {
        if let Scheme::APrime(rows) = self.scheme {
            let row = (xf as isize - 2).min(self.f.nx as isize - 3 - xf as isize);
            if row < rows as isize {
                return 0.0;
            }
        }
        self.sigma
    }

    /// Bilinear value of the coarse velocity at coarse index coordinates (cell centres at integers).
    fn interp(&self, xc: f64, yc: f64) -> [f64; 3] {
        let (x0, y0) = (xc.floor(), yc.floor());
        let (tx, ty) = (xc - x0, yc - y0);
        let ny = self.c.ny as isize;
        let mut out = [0.0; 3];
        for (dx, wx) in [(0isize, 1.0 - tx), (1, tx)] {
            for (dy, wy) in [(0isize, 1.0 - ty), (1, ty)] {
                let x = (x0 as isize + dx).rem_euclid(self.c.nx as isize) as usize;
                let y = (y0 as isize + dy).rem_euclid(ny) as usize;
                let u = self.uc[self.c.at(x, y, 0)];
                for a in 0..3 {
                    out[a] += wx * wy * u[a];
                }
            }
        }
        out
    }

    fn refresh_fine(&mut self) {
        for z in 0..self.f.nz {
            for y in 0..self.f.ny {
                for x in 0..self.f.nx {
                    let i = self.f.at(x, y, z);
                    self.uf[i] = if self.real_f(x) {
                        macros(&self.f.f[i]).1
                    } else {
                        let xc = (self.ia - 1) as f64 + (x as f64 + 0.5) * 0.5 - 0.5;
                        let yc = (y as f64 + 0.5) * 0.5 - 0.5;
                        self.interp(xc, yc)
                    };
                }
            }
        }
    }

    fn refresh_coarse(&mut self) {
        for y in 0..self.c.ny {
            for x in 0..self.c.nx {
                let i = self.c.at(x, y, 0);
                self.uc[i] = if self.covered(x) {
                    let mut s = [0.0; 3];
                    for dz in 0..NZ_F {
                        for dy in 0..2 {
                            for dx in 0..2 {
                                let u = macros(&self.f.f[self.f.at(2 * (x - (self.ia - 1)) + dx, 2 * y + dy, dz)]).1;
                                for a in 0..3 {
                                    s[a] += u[a] / 8.0;
                                }
                            }
                        }
                    }
                    s
                } else {
                    macros(&self.c.f[i]).1
                };
            }
        }
    }

    fn collide_coarse(&mut self) {
        self.refresh_coarse();
        let mut out = [0.0; Q];
        for y in 0..self.c.ny {
            let (ia, ib) = (self.ia, self.ib);
            for x in (0..self.c.nx).filter(|&x| x < ia || x >= ib) {
                let i = self.c.at(x, y, 0);
                let (r, _) = macros(&self.c.f[i]);
                let (u, s) = strain_at(&self.uc, &self.c, x, y, 0, true);
                rebuild(&self.c.f[i], r, u, [0.0; 3], false, Some((self.sigma_coarse(x), &s)), &mut out, |_| self.omega_c);
                self.c.f[i] = out;
            }
        }
    }

    fn collide_fine(&mut self) {
        self.refresh_fine();
        let mut out = [0.0; Q];
        for z in 0..self.f.nz {
            for y in 0..self.f.ny {
                let nxf = self.f.nx;
                let (lo, hi) = if self.ghost_collision { (1, nxf - 1) } else { (2, nxf - 2) };
                for x in lo..hi {
                    let i = self.f.at(x, y, z);
                    let (r, _) = macros(&self.f.f[i]);
                    if r <= 1e-9 {
                        continue;
                    }
                    let (u, s) = strain_at(&self.uf, &self.f, x, y, z, false);
                    rebuild(&self.f.f[i], r, u, [0.0; 3], false, Some((self.sigma_fine(x), &s)), &mut out, |_| self.omega_f);
                    self.f.f[i] = out;
                }
            }
        }
    }

    /// The factor on the non-equilibrium part of post-collision populations, fine from coarse.
    fn post_ratio(&self) -> f64 {
        let (tc, tf) = (1.0 / self.omega_c, 1.0 / self.omega_f);
        (1.0 - self.omega_f) / (1.0 - self.omega_c) * tf / (2.0 * tc)
    }

    /// Scheme B's transform of the crossing populations `crossing` of `a` by `ratio`, mass kept.
    fn transform(&self, a: &[f64; Q], rho: f64, u: [f64; 3], ratio: f64, crossing: impl Fn(usize) -> bool) -> [f64; Q] {
        let mut g = *a;
        if self.scheme != Scheme::B {
            return g;
        }
        let e = eq3(rho, u);
        let (mut delta, mut wsum) = (0.0, 0.0);
        for q in (0..Q).filter(|&q| crossing(q)) {
            g[q] = e[q] + ratio * (a[q] - e[q]);
            delta += g[q] - a[q];
            wsum += W[q];
        }
        for q in (0..Q).filter(|&q| crossing(q)) {
            g[q] -= delta * W[q] / wsum;
        }
        g
    }

    fn fill_ghosts(&mut self) {
        let nxf = self.f.nx;
        for side in 0..2 {
            let (xa, xf0) = if side == 0 { (self.ia - 1, 0) } else { (self.ib, nxf - 2) };
            let inbound = move |q: usize| if side == 0 { C[q][0] > 0 } else { C[q][0] < 0 };
            for y in 0..self.c.ny {
                let a = self.c.f[self.c.at(xa, y, 0)];
                let (r, u) = macros(&a);
                self.edge[side][y] = (r, u);
                let g = self.transform(&a, r, u, self.post_ratio(), inbound);
                for dz in 0..NZ_F {
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let i = self.f.at(xf0 + dx, 2 * y + dy, dz);
                            for q in (0..Q).filter(|&q| inbound(q)) {
                                self.f.f[i][q] = g[q];
                            }
                        }
                    }
                }
            }
        }
    }

    fn stream_fine(&mut self) {
        let mut g = vec![[0.0; Q]; self.f.f.len()];
        let (nx, ny, nz) = (self.f.nx as isize, self.f.ny as isize, self.f.nz as isize);
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let i = self.f.at(x as usize, y as usize, z as usize);
                    for q in 0..Q {
                        let c = C[q];
                        let tx = x + (c[0] as isize);
                        if tx < 0 || tx >= nx {
                            continue;
                        }
                        let t = self.f.at(tx as usize, (y + (c[1] as isize)).rem_euclid(ny) as usize, (z + (c[2] as isize)).rem_euclid(nz) as usize);
                        g[t][q] += self.f.f[i][q];
                    }
                }
            }
        }
        self.f.f = g;
    }

    fn stream_coarse(&mut self) {
        let mut g = vec![[0.0; Q]; self.c.f.len()];
        let (nx, ny) = (self.c.nx as isize, self.c.ny as isize);
        for y in 0..ny {
            for x in (0..nx).filter(|&x| !self.covered(x as usize)) {
                let i = self.c.at(x as usize, y as usize, 0);
                for q in 0..Q {
                    let c = C[q];
                    let tx = (x + (c[0] as isize)).rem_euclid(nx) as usize;
                    if self.covered(tx) {
                        continue;
                    }
                    let t = self.c.at(tx, (y + (c[1] as isize)).rem_euclid(ny) as usize, 0);
                    g[t][q] += self.c.f[i][q];
                }
            }
        }
        self.c.f = g;
    }

    fn restrict(&mut self) {
        let nxf = self.f.nx;
        let ratio = 1.0 / self.post_ratio();
        for side in 0..2 {
            let (xa, xf0) = if side == 0 { (self.ia - 1, 0) } else { (self.ib, nxf - 2) };
            let outbound = move |q: usize| if side == 0 { C[q][0] < 0 } else { C[q][0] > 0 };
            for y in 0..self.c.ny {
                let mut h = [0.0; Q];
                for dz in 0..NZ_F {
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let i = self.f.at(xf0 + dx, 2 * y + dy, dz);
                            for q in (0..Q).filter(|&q| outbound(q)) {
                                h[q] += self.f.f[i][q] / 8.0;
                                self.f.f[i][q] = 0.0;
                            }
                        }
                    }
                }
                let (r, u) = self.edge[side][y];
                let h = self.transform(&h, r, u, ratio, outbound);
                let t = self.c.at(xa, y, 0);
                for q in (0..Q).filter(|&q| outbound(q)) {
                    self.c.f[t][q] += h[q];
                }
            }
        }
    }

    /// One coarse step (two fine steps).
    pub fn step(&mut self) {
        self.collide_coarse();
        self.collide_fine();
        self.fill_ghosts();
        self.stream_coarse();
        self.stream_fine();
        self.collide_fine();
        self.stream_fine();
        self.restrict();
    }

    /// Mass in fine-cell units.
    pub fn mass(&self) -> f64 {
        let c: f64 = (0..self.c.ny)
            .flat_map(|y| (0..self.c.nx).filter(|&x| !self.covered(x)).map(move |x| (x, y)))
            .map(|(x, y)| self.c.f[self.c.at(x, y, 0)].iter().sum::<f64>() * 8.0)
            .sum();
        let f: f64 = self.f.f.iter().map(|f| f.iter().sum::<f64>()).sum();
        c + f
    }

    /// The composite field on coarse cells: coarse values outside the strip, fine averages inside.
    pub fn composite(&self) -> Vec<(f64, [f64; 3])> {
        let mut out = vec![(0.0, [0.0; 3]); self.c.nx * self.c.ny];
        for y in 0..self.c.ny {
            for x in 0..self.c.nx {
                let o = &mut out[y * self.c.nx + x];
                if self.covered(x) {
                    for dz in 0..NZ_F {
                        for dy in 0..2 {
                            for dx in 0..2 {
                                let (r, u) = macros(&self.f.f[self.f.at(2 * (x - (self.ia - 1)) + dx, 2 * y + dy, dz)]);
                                o.0 += r / 8.0;
                                for a in 0..3 {
                                    o.1[a] += u[a] / 8.0;
                                }
                            }
                        }
                    }
                } else {
                    *o = macros(&self.c.f[self.c.at(x, y, 0)]);
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The study's τ − ½ and σ, from `STUDY_TAU` and `STUDY_SIGMA`.
    fn tau_half() -> f64 {
        std::env::var("STUDY_TAU").ok().and_then(|v| v.parse().ok()).unwrap_or(2.4e-4)
    }
    fn sigma() -> f64 {
        std::env::var("STUDY_SIGMA").ok().and_then(|v| v.parse().ok()).unwrap_or(0.95)
    }
    const SCHEMES: [(&str, Scheme); 4] = [("A", Scheme::A), ("A'1", Scheme::APrime(1)), ("A'2", Scheme::APrime(2)), ("B", Scheme::B)];

    /// A Gaussian density pulse in a pseudo-1D box (coarse units).
    fn pulse(amp: f64) -> impl Fn(f64, f64) -> (f64, [f64; 3]) {
        move |x, _| (1.0 + amp * (-((x - 24.0) / 3.0).powi(2) / 2.0).exp(), [0.0; 3])
    }

    /// Mass is exact and the pulse crosses the strip: the report of gate 1.
    #[test]
    #[ignore = "study: cargo test -p fieldcraft-lbm --release interface_study -- --ignored --nocapture"]
    fn the_pulse_crosses_the_interface() {
        let amp = 1e-3;
        for (name, scheme) in SCHEMES {
            let mut p = Pair::new(128, 1, 48, 80, tau_half(), sigma(), scheme, pulse(amp));
            let mut r = Uniform::new(256, 2, NZ_F, 2.0 * tau_half(), sigma(), pulse(amp), 0.5);
            let m0 = p.mass();
            let mut line = format!("{name:5}");
            for step in 1..=130 {
                p.step();
                r.step();
                r.step();
                if step % 26 == 0 {
                    let rc = r.coarsened(2);
                    let comp = p.composite();
                    let err = comp.iter().zip(&rc).map(|(a, b)| (a.0 - b.0).abs()).fold(0.0f64, f64::max) / amp;
                    line += &format!("  t{step}: err {err:.1e}");
                }
                if step == 100 || step == 130 {
                    // Left of the strip only a reflection is left (the left-going half has wrapped round).
                    let refl = p.composite().iter().take(36).map(|(r, _)| (r - 1.0).abs()).fold(0.0f64, f64::max) / amp;
                    line += &format!(" refl {refl:.1e}");
                }
            }
            println!("{line}  mass drift {:.1e}", (p.mass() - m0) / m0);
        }
    }

    /// A uniform flow stays uniform: the equilibrium gate.
    #[test]
    #[ignore = "study: cargo test -p fieldcraft-lbm --release interface_study -- --ignored --nocapture"]
    fn a_uniform_flow_stays_uniform() {
        for (name, scheme) in SCHEMES {
            let init = |_: f64, _: f64| (1.0, [0.05, 0.02, 0.0]);
            let mut p = Pair::new(64, 4, 24, 40, tau_half(), sigma(), scheme, init);
            let m0 = p.mass();
            for _ in 0..400 {
                p.step();
            }
            let comp = p.composite();
            let du = comp.iter().map(|(_, u)| (u[0] - 0.05).abs().max((u[1] - 0.02).abs())).fold(0.0f64, f64::max);
            let dr = comp.iter().map(|(r, _)| (r - 1.0).abs()).fold(0.0f64, f64::max);
            println!("{name:5} after 400 steps: max |du| {du:.1e}, max |drho| {dr:.1e}, mass drift {:.1e}", (p.mass() - m0) / m0);
        }
    }

    /// Fourier amplitude and phase of `v(i)` for `n` samples at mode `m`.
    fn mode(v: &[f64], m: usize) -> (f64, f64) {
        let n = v.len() as f64;
        let (mut c, mut s) = (0.0, 0.0);
        for (i, x) in v.iter().enumerate() {
            let a = 2.0 * std::f64::consts::PI * (m * i) as f64 / n;
            c += x * a.cos();
            s += x * a.sin();
        }
        (2.0 * (c * c + s * s).sqrt() / n, s.atan2(c))
    }

    /// The convected shear wave crossing the strip normally (flow along x) and parallel (flow along y).
    #[test]
    #[ignore = "study: cargo test -p fieldcraft-lbm --release interface_study -- --ignored --nocapture"]
    fn the_shear_wave_crosses_the_interface() {
        let (a, u0) = (5e-3, 0.05);
        let tau = 2.0 * std::f64::consts::PI;
        for normal in [true, false] {
            // Normal: u = (U, a sin(kx)), wave along x, box 64 x 4. Parallel: u = (a sin(ky), U), box 64 x 16.
            let (nxc, nyc) = if normal { (64, 4) } else { (64, 16) };
            let init = move |x: f64, y: f64| -> (f64, [f64; 3]) {
                if normal { (1.0, [u0, a * (tau * x / 16.0).sin(), 0.0]) } else { (1.0, [a * (tau * y / 16.0).sin(), u0, 0.0]) }
            };
            let steps = 900;
            let mut r = Uniform::new(2 * nxc, 2 * nyc, NZ_F, 2.0 * tau_half(), sigma(), init, 0.5);
            for _ in 0..2 * steps {
                r.step();
            }
            let rc = r.coarsened(2);
            // The wave component and the profile it is read along.
            let read = |f: &[(f64, [f64; 3])]| -> (f64, f64, f64, f64) {
                let (mut v, comp, n, m) = if normal { (vec![0.0; nxc], 1, nxc, 4) } else { (vec![0.0; nyc], 0, nyc, 1) };
                for y in 0..nyc {
                    for x in 0..nxc {
                        let i = if normal { x } else { y };
                        v[i] += f[y * nxc + x].1[comp] / if normal { nyc as f64 } else { nxc as f64 };
                    }
                }
                let (amp, ph) = mode(&v, m);
                let other = f.iter().map(|(_, u)| (u[if normal { 0 } else { 1 }] - u0).abs()).fold(0.0f64, f64::max);
                let rho = f.iter().map(|(r, _)| (r - 1.0).abs()).fold(0.0f64, f64::max);
                let _ = n;
                (amp, ph, other, rho)
            };
            let (ra, rp, ro, rr) = read(&rc);
            println!(
                "{} flow: reference amp {:.3e} phase {rp:.3} (spurious |du_flow| {ro:.1e}, |drho| {rr:.1e})",
                if normal { "normal  " } else { "parallel" },
                ra
            );
            {
                let mut b = Uniform::new(nxc, nyc, 1, tau_half(), sigma(), init, 1.0);
                for _ in 0..steps {
                    b.step();
                }
                let bc = b.coarsened(1);
                let (ba, bp, bo, br) = read(&bc);
                let l2 = bc.iter().zip(&rc).map(|(x, y)| (x.1[0] - y.1[0]).powi(2) + (x.1[1] - y.1[1]).powi(2)).sum::<f64>().sqrt()
                    / (rc.len() as f64).sqrt()
                    / a;
                println!(
                    "  coarse-only (no strip) amp ratio {:.4} phase diff {:+.4} rad  spurious |du_flow| {bo:.1e} |drho| {br:.1e}  L2(du)/a {l2:.3}",
                    ba / ra,
                    bp - rp
                );
            }
            for (name, scheme) in SCHEMES {
                let mut p = Pair::new(nxc, nyc, 24, 40, tau_half(), sigma(), scheme, init);
                let m0 = p.mass();
                for _ in 0..steps {
                    p.step();
                }
                let (pa, pp, po, pr) = read(&p.composite());
                let l2 = p.composite().iter().zip(&rc).map(|(x, y)| (x.1[0] - y.1[0]).powi(2) + (x.1[1] - y.1[1]).powi(2)).sum::<f64>().sqrt()
                    / (rc.len() as f64).sqrt()
                    / a;
                println!(
                    "  {name:5} amp ratio {:.4} phase diff {:+.4} rad  spurious |du_flow| {po:.1e} |drho| {pr:.1e}  L2(du)/a {l2:.3}  mass drift {:.1e}",
                    pa / ra,
                    pp - rp,
                    (p.mass() - m0) / m0
                );
            }
        }
    }

    /// 1D question 1: do the ghost cells collide? A ghost cell holds only the populations of one
    /// direction half (plus the ones that have crossed), so a collision of it is not a collision of a
    /// distribution; the pulse and the mass say what it does.
    #[test]
    #[ignore = "study: cargo test -p fieldcraft-lbm --release interface_study -- --ignored --nocapture"]
    fn ghost_cells_colliding_breaks_mass() {
        let amp = 1e-3;
        for ghost in [false, true] {
            let mut p = Pair::new(128, 1, 48, 80, tau_half(), sigma(), Scheme::A, pulse(amp));
            p.ghost_collision = ghost;
            let m0 = p.mass();
            for _ in 0..130 {
                p.step();
            }
            let refl = p.composite().iter().take(36).map(|(r, _)| (r - 1.0).abs()).fold(0.0f64, f64::max) / amp;
            println!("ghost cells collide: {ghost:5}  mass drift {:.1e}  reflection {refl:.1e}", (p.mass() - m0) / m0);
        }
    }
}
