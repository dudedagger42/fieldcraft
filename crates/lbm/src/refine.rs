//! Local grid refinement: a fine strip across a coarse flow (ratio 2, acoustic scaling).
//!
//! The coarse flow is an ordinary [`Flow`] over the whole box. A strip of coarse columns
//! `first..last` (all of y and z) is refined by 2: the fine flow is a second [`Flow`] over those
//! columns plus one coarse column on each side (its ghost cells), periodic along x, with the same
//! faces along y and z, half the spacing and half the time step, so the lattice speed is the same
//! on both levels and the lattice viscosity doubles on the fine one (`τ_f − ½ = 2 (τ_c − ½)`,
//! M. Rohde et al., Int. J. Numer. Meth. Fluids 51:439, 2006, §1.1). The fine flow takes two steps
//! for each coarse step.
//!
//! # The interface
//! M. Rohde, D. Kandhai, J. J. Derksen, H. E. A. van den Akker, *A generic, mass conservative
//! local grid refinement technique for lattice-Boltzmann schemes*, Int. J. Numer. Meth. Fluids
//! 51:439–468, 2006, §2.1 (steps 1–5), with one change found by the study in `interface_study`:
//!
//! 1. Both levels collide. The two interface columns of the coarse flow (`first − 1` and `last`)
//!    hand the post-collision populations that point into the strip to the ghost cells of the fine
//!    flow that cover them (8 cells per coarse cell). Rohde puts the same value in all 8; here the
//!    populations are reconstructed linearly along the interface, `f ∓ slope / 4` in the two fine
//!    cells of a coarse cell along y and z (the slope a central difference of the neighbouring
//!    coarse cells, zero where a neighbour is missing): the offsets cancel, so mass is exact, and the
//!    equal split's smearing along the interface (an error that falls as the square of the
//!    wavelength, 4 % of a convected shear wave's amplitude at 16 cells) is gone.
//! 2. Both levels stream; the ghost cells' populations travel into the strip, and what the strip
//!    sends out collects in the ghost cells. The fine flow collides (the ghost cells do not: they
//!    keep what they hold) and streams once more.
//! 3. The populations that have collected in the ghost cells pointing out of the strip are summed
//!    (÷ 8) into the coarse interface cell; the ghost cells are cleared.
//!
//! The covered coarse columns are not used: their populations are set each step to the equilibrium
//! of the fine flow's average moments, which is what the coarse interface cells' finite-difference
//! strain reads. The ghost cells are filled with the equilibrium of the coarse flow's moments,
//! interpolated linearly, before each collision, so the fine strain stencil reads a velocity there.
//!
//! No interpolation or rescaling of populations is applied (Rohde's variant; the study found the
//! rescaling of the non-equilibrium part shifts a convected wave's phase and gains nothing).
//!
//! # Not supported yet
//! One strip with interfaces normal to x, spanning the box in y and z; the hybrid recursive
//! regularised or regularised collision without an eddy-viscosity model (a Smagorinsky constant
//! below 1e-3 is allowed: it is how a low τ is admitted); no heat, free surface, porous zone, fan,
//! absorbing layer, moving wall or obstacle surface condition; no wall-function bounce. Each is
//! refused with the reason.

use fieldcraft_model::{Collision, FaceCondition, Model, Procedure, Turbulence};

use crate::lattice::{C, Q};
use crate::regularized::rebuild;
use crate::setup::fluid_step;
use crate::{Error, Flow, Limits, Obstacle, Result, Setup};

/// Fine cells per coarse cell along each axis.
const RATIO: usize = 2;
/// The fewest coarse columns in a strip.
const MIN_COLUMNS: usize = 8;
/// A Smagorinsky constant at or below this is no eddy viscosity worth the name.
const NEGLIGIBLE_SMAGORINSKY: f64 = 1e-3;

/// The refined strip: coarse columns `first..last` (all of y and z), the interface columns being
/// `first − 1` and `last`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Strip {
    pub first: usize,
    pub last: usize,
    /// Reconstruct the crossing populations linearly along the interface (false: Rohde's equal split).
    pub linear: bool,
}

/// Two coupled flows: the coarse box and the fine strip.
pub struct Refined {
    coarse: Flow,
    fine: Flow,
    strip: Strip,
    /// The coarse cell counts and the fine ones.
    nc: [usize; 3],
    nf: [usize; 3],
    /// Whether the y and z faces are periodic (a population toward a wall face bounces back in its cell: it crosses nothing).
    periodic: [bool; 2],
    /// ρ and u of the coarse cells in the columns around the interface, by column offset (see `moment_columns`).
    moments: Vec<Vec<(f64, [f64; 3])>>,
}

/// ρ and u of a set of populations.
pub(crate) fn moments_of(f: &[f64; Q]) -> (f64, [f64; 3]) {
    let mut rho = 0.0;
    let mut j = [0.0; 3];
    for (q, fq) in f.iter().enumerate() {
        rho += fq;
        for (a, ja) in j.iter_mut().enumerate() {
            *ja += fq * f64::from(C[q][a]);
        }
    }
    if rho > 0.0 { (rho, [j[0] / rho, j[1] / rho, j[2] / rho]) } else { (rho, [0.0; 3]) }
}

/// The equilibrium of the collision with its third-order terms: the collision with rate 1.
fn equilibrium(rho: f64, u: [f64; 3]) -> [f64; Q] {
    let mut out = [0.0; Q];
    rebuild(&[0.0; Q], rho, u, [0.0; 3], false, None, &mut out, |_| 1.0);
    out
}

fn refuse<T>(what: &str) -> Result<T> {
    Err(Error::Invalid(format!("not supported yet in a refined flow: {what}")))
}

impl Strip {
    /// Whether the interface populations of the coarse column on `side` (0: `first − 1`, 1: `last`)
    /// point into the strip.
    fn into_strip(side: usize, q: usize) -> bool {
        if side == 0 { C[q][0] > 0 } else { C[q][0] < 0 }
    }
}

impl Refined {
    /// Build the coarse flow of `model`'s fluid step and the fine strip over it.
    pub fn new(model: &Model, step: Option<&str>, obstacles: &[Obstacle], limits: &Limits, strip: Strip) -> Result<Refined> {
        let (s, domain) = fluid_step(model, step)?;
        let Procedure::Fluid { domain: domain_name, steps, time_step, collision, turbulence, steady, .. } = &s.procedure else {
            return refuse("a step that is not a fluid step");
        };
        if !matches!(collision, Some(Collision::Hrr | Collision::Regularized)) {
            return refuse("a collision other than the hybrid recursive regularised or the regularised one");
        }
        match turbulence {
            None => {}
            Some(Turbulence::Smagorinsky { constant, van_driest: None, wall_model: None, .. }) if *constant <= NEGLIGIBLE_SMAGORINSKY => {}
            Some(_) => {
                return refuse("an eddy-viscosity or turbulence model (the fine level's relaxation time is scaled from the molecular viscosity)");
            }
        }
        if steady.is_some() {
            return refuse("a steady-state step");
        }
        let Some(time_step) = *time_step else { return refuse("a step without an explicit time step") };
        if domain.thermal.is_some() {
            return refuse("heat");
        }
        if domain.free_surface.is_some() {
            return refuse("a free surface");
        }
        if !domain.porous.is_empty() || !domain.fans.is_empty() {
            return refuse("porous zones and fans");
        }
        if domain.absorbing.is_some() {
            return refuse("an absorbing layer");
        }
        if !domain.surfaces.is_empty() {
            return refuse("conditions on obstacle surfaces");
        }
        if domain.moving_wall_ramp.is_some() {
            return refuse("a moving wall");
        }
        for face in [&domain.faces.y_min, &domain.faces.y_max, &domain.faces.z_min, &domain.faces.z_max] {
            if matches!(face, FaceCondition::MovingWall { .. }) {
                return refuse("a moving face along y or z");
            }
        }
        let (coarse_setup, coarse) = {
            let setup = Setup::new(model, step, obstacles, limits)?;
            let flow = Flow::new(&setup, obstacles)?;
            (setup, flow)
        };
        if coarse.on_node_bounce() {
            return refuse("the on-node wall-function bounce");
        }
        let nc = coarse_setup.grid.cells;
        let Strip { first, last, .. } = strip;
        if first < 3 || last.saturating_add(3) > nc[0] {
            return Err(Error::Invalid(format!(
                "the strip {first}..{last} must lie at least 3 coarse columns inside the box (0..{}): the interface columns and their neighbours stay clear of its x faces",
                nc[0]
            )));
        }
        if last < first + MIN_COLUMNS {
            return Err(Error::Invalid(format!("the strip {first}..{last} is narrower than {MIN_COLUMNS} coarse columns")));
        }
        // The fine model: the strip and a ghost column on each side, half the spacing and time step.
        let dx = domain.spacing;
        let mut fine_domain = domain.clone();
        fine_domain.min[0] = domain.min[0] + (first - 1) as f64 * dx;
        fine_domain.max[0] = domain.min[0] + (last + 1) as f64 * dx;
        fine_domain.spacing = dx / RATIO as f64;
        fine_domain.faces.x_min = FaceCondition::Periodic {};
        fine_domain.faces.x_max = FaceCondition::Periodic {};
        let mut fine_model = model.clone();
        fine_model.replace_fluid_domain(domain_name, fine_domain)?;
        let mut fine_step = s.clone();
        if let Procedure::Fluid { time_step: dt, steps: n, .. } = &mut fine_step.procedure {
            *dt = Some(time_step / RATIO as f64);
            *n = steps.map(|n| n.saturating_mul(RATIO as u64));
        }
        fine_model.replace_step(fine_step)?;
        let fine_setup = Setup::new(&fine_model, step, obstacles, limits)?;
        let fine = Flow::new(&fine_setup, obstacles)?;
        let nf = fine_setup.grid.cells;
        if nf != [RATIO * (last - first + 2), RATIO * nc[1], RATIO * nc[2]] {
            return Err(Error::Invalid(format!("the fine grid {nf:?} does not cover the strip {first}..{last} of the coarse grid {nc:?}")));
        }
        let periodic = [
            matches!(domain.faces.y_min, FaceCondition::Periodic {}) && matches!(domain.faces.y_max, FaceCondition::Periodic {}),
            matches!(domain.faces.z_min, FaceCondition::Periodic {}) && matches!(domain.faces.z_max, FaceCondition::Periodic {}),
        ];
        let mut refined = Refined { coarse, fine, strip, nc, nf, periodic, moments: Vec::new() };
        refined.check_interface_cells()?;
        // The ghost cells hold nothing between steps.
        for (cell, _) in refined.save_ghosts() {
            refined.fine.set_cell_populations(cell, &[0.0; Q]);
        }
        Ok(refined)
    }

    /// The columns next to the interface must be fluid on both levels (no obstacle, no wall node).
    fn check_interface_cells(&self) -> Result<()> {
        let Strip { first, last, .. } = self.strip;
        let [_, ny, nz] = self.nc;
        for x in [first - 2, first - 1, last, last + 1] {
            for z in 0..nz {
                for y in 0..ny {
                    if !self.coarse.is_fluid(self.coarse.grid().index([x, y, z])) {
                        return Err(Error::Invalid(format!(
                            "coarse cell ({x}, {y}, {z}) next to the interface is not a fluid cell: the interface must lie in open fluid, at least 2 coarse cells from a solid"
                        )));
                    }
                }
            }
        }
        let nxf = self.nf[0];
        for x in (0..4).chain(nxf - 4..nxf) {
            for z in 0..self.nf[2] {
                for y in 0..self.nf[1] {
                    if !self.fine.is_fluid(self.fine.grid().index([x, y, z])) {
                        return Err(Error::Invalid(format!(
                            "fine cell ({x}, {y}, {z}) next to the interface is not a fluid cell: the interface must lie in open fluid"
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    pub fn coarse(&self) -> &Flow {
        &self.coarse
    }

    pub fn fine(&self) -> &Flow {
        &self.fine
    }

    pub fn strip(&self) -> Strip {
        self.strip
    }

    /// The fine cell of the sub-cell `(dx, dy, dz)` of coarse cell `(x, y, z)` in the interface
    /// column on `side` (the ghost cells of that side).
    fn ghost(&self, side: usize, y: usize, z: usize, d: [usize; 3]) -> usize {
        let x0 = if side == 0 { 0 } else { self.nf[0] - RATIO };
        self.fine.grid().index([x0 + d[0], RATIO * y + d[1], RATIO * z + d[2]])
    }

    /// Whether the neighbour of coarse cell `(y, z)` along `(cy, cz)` exists: inside the box, or
    /// across a periodic face. A population toward a wall face bounces back within its cell.
    fn neighbour_exists(&self, y: usize, z: usize, cy: i32, cz: i32) -> bool {
        let ok = |i: usize, c: i32, n: usize, periodic: bool| periodic || (i as i64 + i64::from(c) >= 0 && ((i as i64 + i64::from(c)) as usize) < n);
        ok(y, cy, self.nc[1], self.periodic[0]) && ok(z, cz, self.nc[2], self.periodic[1])
    }

    fn interface_column(&self, side: usize) -> usize {
        if side == 0 { self.strip.first - 1 } else { self.strip.last }
    }

    /// The covered coarse cells take the equilibrium of the average moments of the fine cells in them.
    fn fill_covered(&mut self) {
        let Strip { first, last, .. } = self.strip;
        let [_, ny, nz] = self.nc;
        for z in 0..nz {
            for y in 0..ny {
                for x in first..last {
                    let cell = self.coarse.grid().index([x, y, z]);
                    if !self.coarse.is_fluid(cell) {
                        continue;
                    }
                    let (mut rho, mut j, mut n) = (0.0, [0.0; 3], 0.0);
                    for dz in 0..RATIO {
                        for dy in 0..RATIO {
                            for dx in 0..RATIO {
                                let fc = self.fine.grid().index([RATIO * (x - (first - 1)) + dx, RATIO * y + dy, RATIO * z + dz]);
                                if !self.fine.is_fluid(fc) {
                                    continue;
                                }
                                let (r, u) = moments_of(&self.fine.cell_populations(fc));
                                rho += r;
                                for a in 0..3 {
                                    j[a] += r * u[a];
                                }
                                n += 1.0;
                            }
                        }
                    }
                    if n > 0.0 && rho > 0.0 {
                        let mean = rho / n;
                        self.coarse.set_cell_populations(cell, &equilibrium(mean, [j[0] / rho, j[1] / rho, j[2] / rho]));
                    }
                }
            }
        }
    }

    /// ρ and u of the coarse cells in the columns that the ghost interpolation reads.
    fn snapshot_moments(&mut self) {
        let Strip { first, last, .. } = self.strip;
        let [_, ny, nz] = self.nc;
        let columns = [first - 2, first - 1, first, last - 1, last, last + 1];
        self.moments = columns
            .iter()
            .map(|&x| {
                let mut v = Vec::with_capacity(ny * nz);
                for z in 0..nz {
                    for y in 0..ny {
                        v.push(moments_of(&self.coarse.cell_populations(self.coarse.grid().index([x, y, z]))));
                    }
                }
                v
            })
            .collect();
    }

    /// The coarse moments at fine ghost cell `(xf, yf, zf)` of `side`, interpolated linearly in
    /// x, y and z between the coarse cell centres (clamped at the box faces).
    fn interpolate(&self, side: usize, dx: usize, yf: usize, zf: usize) -> (f64, [f64; 3]) {
        let [_, ny, nz] = self.nc;
        // Position in coarse cell units relative to the interface column's centre, along x.
        let along = (dx as f64 + 0.5) / RATIO as f64 - 0.5;
        let (col0, col1, tx) = if side == 0 {
            // ghost column is `first − 1`; neighbours `first − 2` (index 0) and `first` (index 2).
            if along < 0.0 { (0, 1, 1.0 + along) } else { (1, 2, along) }
        } else {
            // ghost column is `last` (index 4); neighbours `last − 1` (3) and `last + 1` (5).
            if along < 0.0 { (3, 4, 1.0 + along) } else { (4, 5, along) }
        };
        let fy = (yf as f64 + 0.5) / RATIO as f64 - 0.5;
        let fz = (zf as f64 + 0.5) / RATIO as f64 - 0.5;
        let axis = |f: f64, n: usize| -> (usize, usize, f64) {
            let f = f.clamp(0.0, (n.saturating_sub(1)) as f64);
            let i = f.floor();
            let j = (i as usize + 1).min(n - 1);
            (i as usize, j, f - i)
        };
        let (y0, y1, ty) = axis(fy, ny);
        let (z0, z1, tz) = axis(fz, nz);
        let mut rho = 0.0;
        let mut j = [0.0; 3];
        for (c, wx) in [(col0, 1.0 - tx), (col1, tx)] {
            for (z, wz) in [(z0, 1.0 - tz), (z1, tz)] {
                for (y, wy) in [(y0, 1.0 - ty), (y1, ty)] {
                    let w = wx * wy * wz;
                    let (r, u) = self.moments.get(c).and_then(|m| m.get(z * ny + y)).copied().unwrap_or((1.0, [0.0; 3]));
                    rho += w * r;
                    for a in 0..3 {
                        j[a] += w * u[a];
                    }
                }
            }
        }
        (rho, j)
    }

    /// Every ghost cell takes the equilibrium of the interpolated coarse moments.
    fn fill_ghost_equilibrium(&mut self) {
        let [_, ny, nz] = self.nc;
        for side in 0..2 {
            for zf in 0..RATIO * nz {
                for yf in 0..RATIO * ny {
                    for dx in 0..RATIO {
                        let xf = if side == 0 { dx } else { self.nf[0] - RATIO + dx };
                        let (rho, u) = self.interpolate(side, dx, yf, zf);
                        let cell = self.fine.grid().index([xf, yf, zf]);
                        self.fine.set_cell_populations(cell, &equilibrium(rho, u));
                    }
                }
            }
        }
    }

    /// The coarse interface columns' post-collision populations that point into the strip go to the ghost cells.
    fn transfer_in(&mut self) {
        let [_, ny, nz] = self.nc;
        let linear = self.strip.linear;
        for side in 0..2 {
            let xa = self.interface_column(side);
            let at = |y: usize, z: usize| self.coarse.grid().index([xa, y, z]);
            for z in 0..nz {
                for y in 0..ny {
                    let a = self.coarse.cell_populations(at(y, z));
                    // Central slopes along y and z, zero where a neighbour is missing or not fluid.
                    let slope = |n: Option<usize>, p: Option<usize>| -> [f64; Q] {
                        let (Some(n), Some(p)) = (n, p) else { return [0.0; Q] };
                        if !self.coarse.is_fluid(n) || !self.coarse.is_fluid(p) {
                            return [0.0; Q];
                        }
                        let (an, ap) = (self.coarse.cell_populations(n), self.coarse.cell_populations(p));
                        let mut s = [0.0; Q];
                        for q in 0..Q {
                            s[q] = 0.5 * (ap[q] - an[q]);
                        }
                        s
                    };
                    let sy = if linear { slope(y.checked_sub(1).map(|y| at(y, z)), (y + 1 < ny).then(|| at(y + 1, z))) } else { [0.0; Q] };
                    let sz = if linear { slope(z.checked_sub(1).map(|z| at(y, z)), (z + 1 < nz).then(|| at(y, z + 1))) } else { [0.0; Q] };
                    for dz in 0..RATIO {
                        for dy in 0..RATIO {
                            let (oy, oz) = (if dy == 0 { -0.25 } else { 0.25 }, if dz == 0 { -0.25 } else { 0.25 });
                            let mut g = [0.0; Q];
                            for q in (0..Q).filter(|&q| Strip::into_strip(side, q) && self.neighbour_exists(y, z, C[q][1], C[q][2])) {
                                g[q] = a[q] + oy * sy[q] + oz * sz[q];
                            }
                            for dx in 0..RATIO {
                                let cell = self.ghost(side, y, z, [dx, dy, dz]);
                                self.fine.set_cell_populations(cell, &g);
                            }
                        }
                    }
                }
            }
        }
    }

    /// What the ghost cells hold, to put back after a collision that must not touch them.
    fn save_ghosts(&self) -> Vec<(usize, [f64; Q])> {
        let mut saved = Vec::new();
        for side in 0..2 {
            for zf in 0..self.nf[2] {
                for yf in 0..self.nf[1] {
                    for dx in 0..RATIO {
                        let xf = if side == 0 { dx } else { self.nf[0] - RATIO + dx };
                        let cell = self.fine.grid().index([xf, yf, zf]);
                        saved.push((cell, self.fine.cell_populations(cell)));
                    }
                }
            }
        }
        saved
    }

    /// The populations that have collected in the ghost cells pointing out of the strip, summed
    /// (÷ 8) into the coarse interface cell; the ghost cells are cleared.
    fn restrict(&mut self) {
        let [_, ny, nz] = self.nc;
        let norm = (RATIO * RATIO * RATIO) as f64;
        for side in 0..2 {
            let xa = self.interface_column(side);
            for z in 0..nz {
                for y in 0..ny {
                    let cell = self.coarse.grid().index([xa, y, z]);
                    let mut a = self.coarse.cell_populations(cell);
                    let mut sum = [0.0; Q];
                    for dz in 0..RATIO {
                        for dy in 0..RATIO {
                            for dx in 0..RATIO {
                                let g = self.ghost(side, y, z, [dx, dy, dz]);
                                let f = self.fine.cell_populations(g);
                                for q in (0..Q).filter(|&q| C[q][0] != 0) {
                                    sum[q] += f[q] / norm;
                                }
                                self.fine.set_cell_populations(g, &[0.0; Q]);
                            }
                        }
                    }
                    // A slot whose source lies behind a wall face holds the wall's reflection of the
                    // coarse cell's own population, which the fine flow does not carry: what the fine
                    // flow sends is added to it. Any other slot held the covered cell's stand-in, which
                    // the fine flow's populations replace.
                    for q in (0..Q).filter(|&q| C[q][0] != 0) {
                        if Strip::into_strip(side, q) {
                            // Only a wall's reflection puts a population here, in a cell next to the wall.
                            if !self.neighbour_exists(y, z, -C[q][1], -C[q][2]) {
                                a[q] += sum[q];
                            }
                        } else if self.neighbour_exists(y, z, -C[q][1], -C[q][2]) {
                            a[q] = sum[q];
                        } else {
                            a[q] += sum[q];
                        }
                    }
                    self.coarse.set_cell_populations(cell, &a);
                }
            }
        }
    }

    /// One coarse step (two fine steps).
    pub fn step(&mut self) -> Result<()> {
        self.fill_covered();
        self.snapshot_moments();
        self.fill_ghost_equilibrium();
        self.coarse.step_collide()?;
        self.fine.step_collide()?;
        self.transfer_in();
        self.coarse.step_stream()?;
        self.fine.step_stream()?;
        // Second fine step: the ghost cells keep what they hold through the collision.
        let saved = self.save_ghosts();
        self.fill_ghost_equilibrium();
        self.fine.step_collide()?;
        for (cell, f) in &saved {
            self.fine.set_cell_populations(*cell, f);
        }
        self.fine.step_stream()?;
        self.restrict();
        Ok(())
    }

    /// The mass in fine-cell units: the coarse cells outside the strip (8 each) and the fine cells.
    pub fn mass(&self) -> f64 {
        let Strip { first, last, .. } = self.strip;
        let g = self.coarse.grid();
        let mut m = 0.0;
        for cell in 0..g.len() {
            let [x, _, _] = g.coords(cell);
            if (first..last).contains(&x) || !self.coarse.is_fluid(cell) {
                continue;
            }
            m += (RATIO * RATIO * RATIO) as f64 * self.coarse.cell_populations(cell).iter().sum::<f64>();
        }
        for cell in 0..self.fine.grid().len() {
            if self.fine.is_fluid(cell) {
                m += self.fine.cell_populations(cell).iter().sum::<f64>();
            }
        }
        m
    }

    /// ρ and u (lattice units) of every coarse cell: the coarse values outside the strip, the
    /// average of the fine cells inside (indexed as the coarse grid; solid cells give `(0, 0)`).
    pub fn composite_moments(&self) -> Vec<(f64, [f64; 3])> {
        let Strip { first, last, .. } = self.strip;
        let g = self.coarse.grid();
        (0..g.len())
            .map(|cell| {
                let [x, y, z] = g.coords(cell);
                if !(first..last).contains(&x) {
                    return if self.coarse.is_fluid(cell) { moments_of(&self.coarse.cell_populations(cell)) } else { (0.0, [0.0; 3]) };
                }
                let (mut rho, mut j, mut n) = (0.0, [0.0; 3], 0.0);
                for dz in 0..RATIO {
                    for dy in 0..RATIO {
                        for dx in 0..RATIO {
                            let fc = self.fine.grid().index([RATIO * (x - (first - 1)) + dx, RATIO * y + dy, RATIO * z + dz]);
                            if self.fine.is_fluid(fc) {
                                let (r, u) = moments_of(&self.fine.cell_populations(fc));
                                rho += r;
                                for a in 0..3 {
                                    j[a] += r * u[a];
                                }
                                n += 1.0;
                            }
                        }
                    }
                }
                if n > 0.0 && rho > 0.0 { (rho / n, [j[0] / rho, j[1] / rho, j[2] / rho]) } else { (0.0, [0.0; 3]) }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use fieldcraft_model::{BoxFaces, Fluid, FluidDomain, Material, Step};

    use super::*;

    fn flow() -> (Setup, Flow) {
        let mut m = Model::default();
        m.add_material("Fluid", Material { density: Some(1.0), fluid: Some(Fluid::Newtonian { viscosity: 1e-3 }), ..Material::default() }).unwrap();
        let d = FluidDomain {
            material: "Fluid".into(),
            min: [0.0; 3],
            max: [0.4, 0.2, 0.2],
            spacing: 0.05,
            faces: BoxFaces::default(),
            obstacles: vec![],
            surfaces: vec![],
            initial_velocity: [0.05, 0.01, 0.0],
            acceleration: [0.0, 0.0, -0.01],
            tolerance: None,
            thermal: None,
            porous: vec![],
            fans: vec![],
            free_surface: None,
            absorbing: None,
            moving_wall_ramp: None,
        };
        m.add_fluid_domain("Flow", d).unwrap();
        let p = Procedure::Fluid {
            domain: "Flow".into(),
            end_time: None,
            steps: Some(10),
            time_step: Some(0.01),
            frames: 2,
            magic: None,
            sigma: Some(fieldcraft_model::HrrSigma::Value(0.95)),
            collision: Some(Collision::Hrr),
            turbulence: None,
            steady: None,
        };
        m.add_step(Step::new("Run", p), None).unwrap();
        let s = Setup::new(&m, None, &[], &Limits::default()).unwrap();
        let f = Flow::new(&s, &[]).unwrap();
        (s, f)
    }

    /// The two halves of a step are the step: a refined flow's schedule changes nothing else.
    #[test]
    fn the_two_halves_of_a_step_are_the_step() {
        let ((_, mut whole), (_, mut halves)) = (flow(), flow());
        for _ in 0..25 {
            whole.step().unwrap();
            halves.step_collide().unwrap();
            halves.step_stream().unwrap();
        }
        for cell in 0..whole.grid().len() {
            assert_eq!(whole.cell_populations(cell), halves.cell_populations(cell), "cell {cell}");
        }
        assert_eq!(whole.steps_done(), halves.steps_done());
    }
}
