//! Local grid refinement: a fine strip across a coarse flow (ratio 2, acoustic scaling).
//!
//! (Written for a strip across x; a strip across y or z is the same code on permuted axes: `perm`.)
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
//!    cells of a coarse cell along y and z, the conservative piecewise-linear reconstruction of
//!    B. van Leer, J. Comput. Phys. 32:101–136, 1979 (the slope a central difference of the
//!    neighbouring coarse cells, one-sided next to a wall, zero where there is no neighbour): the offsets cancel, so mass is exact, and the
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
//! # Walls that meet the interface
//! A wall along y or z reaches the interface plane. A fine ghost cell at the wall cannot reflect a
//! population it never held, so the inner ghost cells next to a wall are handed, in the second
//! fine step, the population their reflection would have supplied (the coarse cell gives up the
//! same amount, so mass stays exact); what wall reflection leaves in the outer ghost layer
//! after the first fine step is collected before the fine box's periodic seam can carry it off.
//!
//! # The Spalart–Allmaras model
//! ν̃ is transported on both levels. The fine ghost cells take the coarse ν̃ interpolated like the
//! moments, in the fine level's lattice units (twice the coarse value: a lattice viscosity doubles
//! when the spacing halves at the same lattice speed), at the start of the coarse step for the
//! first fine step and half-way to its end for the second; the covered coarse cells take the mean
//! of the fine ν̃ in them. Each level has its own wall distance (its own grid). A wall function is
//! refused.
//!
//! # Not supported yet
//! One strip with interfaces normal to one box axis ([`Refined::new_along`]; x for [`Refined::new`]), spanning the box in the other two; the hybrid recursive
//! regularised or regularised collision with no eddy-viscosity model but Spalart–Allmaras (also
//! with DDES; a Smagorinsky constant below 1e-3 is allowed: it is how a low τ is admitted); no heat, free surface, porous zone, fan,
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
    /// Local axis `i` (0: across the strip) is box axis `perm[i]`; the cell counts and every index
    /// below are local, only the public accessors are in box order.
    perm: [usize; 3],
    /// The coarse cell counts and the fine ones.
    nc: [usize; 3],
    nf: [usize; 3],
    /// Whether the y and z faces are periodic (a population toward a wall face bounces back in its cell: it crosses nothing).
    periodic: [bool; 2],
    /// ρ and u of the coarse cells in the columns around the interface, by column offset (see `moment_columns`).
    moments: Vec<Vec<(f64, [f64; 3])>>,
    /// Outbound populations that wall reflection left in the outer ghost layer after the first fine step, per side and coarse (y, z) cell, in coarse units.
    pending: Vec<[f64; Q]>,
    /// Populations a wall-adjacent inner ghost cell must receive again after the first fine step
    /// (the reflection there has nothing to reflect): cell, direction, value, coarse slot.
    replenish: Vec<(usize, usize, f64, usize)>,
    /// What was put back, per side and coarse (y, z) cell, in coarse units: the coarse cell gives it up.
    replenished: Vec<[f64; Q]>,
    /// Whether the Spalart–Allmaras model runs on both levels.
    sa: bool,
    /// Scratch of [`Refined::step`]: the ghost cells' populations, kept for its capacity.
    saved: Vec<(usize, [f64; Q])>,
    /// ν̃ of the coarse cells in the same columns as `moments`.
    nu_tilde: Vec<Vec<f64>>,
    /// The same after the coarse step's transport (the end of the coarse step).
    nu_tilde_next: Vec<Vec<f64>>,
}

/// What the interface columns hold, see [`Refined::interface_report`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InterfaceReport {
    pub nu_t_over_nu_max: f64,
    pub nu_t_over_nu_mean: f64,
    /// The share of interface cells in DDES's LES mode (none without DDES).
    pub les_share: Option<f64>,
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

/// Component `i` (0: across the strip, 1 and 2: along its faces) of lattice direction `q`, with
/// `perm[i]` the box axis that local axis `i` is.
fn cdir(perm: [usize; 3], q: usize, i: usize) -> i32 {
    perm.get(i).and_then(|&a| C[q].get(a)).copied().unwrap_or(0)
}

/// Whether the interface populations of the coarse layer on `side` (0: `first − 1`, 1: `last`)
/// point into the strip.
fn into_strip(perm: [usize; 3], side: usize, q: usize) -> bool {
    if side == 0 { cdir(perm, q, 0) > 0 } else { cdir(perm, q, 0) < 0 }
}

/// The two faces of the box along `axis`.
fn faces_along(faces: &fieldcraft_model::BoxFaces, axis: usize) -> [&FaceCondition; 2] {
    match axis {
        0 => [&faces.x_min, &faces.x_max],
        1 => [&faces.y_min, &faces.y_max],
        _ => [&faces.z_min, &faces.z_max],
    }
}

impl Refined {
    /// Build the coarse flow of `model`'s fluid step and the fine strip over it.
    pub fn new(model: &Model, step: Option<&str>, obstacles: &[Obstacle], limits: &Limits, strip: Strip) -> Result<Refined> {
        Refined::new_along(model, step, obstacles, limits, strip, 0)
    }

    /// As [`Refined::new`], the strip's interfaces normal to box axis `axis` (0: x, 1: y, 2: z), the
    /// strip spanning the box in the other two.
    pub fn new_along(model: &Model, step: Option<&str>, obstacles: &[Obstacle], limits: &Limits, strip: Strip, axis: usize) -> Result<Refined> {
        let perm = match axis {
            0 => [0, 1, 2],
            1 => [1, 2, 0],
            2 => [2, 0, 1],
            _ => return Err(Error::Invalid(format!("the strip axis {axis} is not 0, 1 or 2"))),
        };
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
            Some(Turbulence::SpalartAllmaras { wall_model: None, .. }) => {}
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
        for axis in [perm[1], perm[2]] {
            for face in faces_along(&domain.faces, axis) {
                if matches!(face, FaceCondition::MovingWall { .. }) {
                    return refuse("a moving face along the strip");
                }
            }
        }
        let (coarse_setup, coarse) = {
            let setup = Setup::new(model, step, obstacles, limits)?;
            let flow = Flow::new(&setup, obstacles)?;
            (setup, flow)
        };
        let sa = coarse.spalart_allmaras().is_some();
        if sa && coarse.sa_wall_function().0.is_some() {
            return refuse("a wall function with the Spalart–Allmaras model");
        }
        if coarse.on_node_bounce() {
            return refuse("the on-node wall-function bounce");
        }
        let loc = |p: [usize; 3]| [p[perm[0]], p[perm[1]], p[perm[2]]];
        let nc = loc(coarse_setup.grid.cells);
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
        fine_domain.min[perm[0]] = domain.min[perm[0]] + (first - 1) as f64 * dx;
        fine_domain.max[perm[0]] = domain.min[perm[0]] + (last + 1) as f64 * dx;
        fine_domain.spacing = dx / RATIO as f64;
        match perm[0] {
            0 => (fine_domain.faces.x_min, fine_domain.faces.x_max) = (FaceCondition::Periodic {}, FaceCondition::Periodic {}),
            1 => (fine_domain.faces.y_min, fine_domain.faces.y_max) = (FaceCondition::Periodic {}, FaceCondition::Periodic {}),
            _ => (fine_domain.faces.z_min, fine_domain.faces.z_max) = (FaceCondition::Periodic {}, FaceCondition::Periodic {}),
        }
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
        let nf = loc(fine_setup.grid.cells);
        if nf != [RATIO * (last - first + 2), RATIO * nc[1], RATIO * nc[2]] {
            return Err(Error::Invalid(format!("the fine grid {nf:?} does not cover the strip {first}..{last} of the coarse grid {nc:?}")));
        }
        let periodic = [perm[1], perm[2]].map(|a| faces_along(&domain.faces, a).iter().all(|f| matches!(f, FaceCondition::Periodic {})));
        let mut refined = Refined {
            coarse,
            fine,
            strip,
            perm,
            nc,
            nf,
            periodic,
            moments: Vec::new(),
            pending: vec![[0.0; Q]; 2 * nc[1] * nc[2]],
            replenish: Vec::new(),
            replenished: vec![[0.0; Q]; 2 * nc[1] * nc[2]],
            sa,
            saved: Vec::new(),
            nu_tilde: Vec::new(),
            nu_tilde_next: Vec::new(),
        };
        refined.check_interface_cells()?;
        // The ghost cells hold nothing between steps.
        refined.clear_ghosts();
        Ok(refined)
    }

    /// Box coordinates of a local cell, and local ones of a box cell.
    fn nat(&self, l: [usize; 3]) -> [usize; 3] {
        let mut p = [0; 3];
        for (i, &a) in self.perm.iter().enumerate() {
            p[a] = l[i];
        }
        p
    }

    fn loc(&self, p: [usize; 3]) -> [usize; 3] {
        [p[self.perm[0]], p[self.perm[1]], p[self.perm[2]]]
    }

    /// The coarse and the fine cell at local coordinates.
    fn cidx(&self, l: [usize; 3]) -> usize {
        self.coarse.grid().index(self.nat(l))
    }

    fn fidx(&self, l: [usize; 3]) -> usize {
        self.fine.grid().index(self.nat(l))
    }

    /// The position in coarse cells, in box order, of a fine cell centre given by its box coordinates
    /// (the fine grid starts at the interface layer `first − 1`, in half cells).
    fn position(&self, fine: [usize; 3]) -> [f64; 3] {
        let l = self.loc(fine);
        let x = [
            (self.strip.first - 1) as f64 + (l[0] as f64 + 0.5) / RATIO as f64,
            (l[1] as f64 + 0.5) / RATIO as f64,
            (l[2] as f64 + 0.5) / RATIO as f64,
        ];
        let mut p = [0.0; 3];
        for (i, &a) in self.perm.iter().enumerate() {
            p[a] = x[i];
        }
        p
    }

    fn clear_ghosts(&mut self) {
        for (cell, _) in self.save_ghosts() {
            self.fine.set_cell_populations(cell, &[0.0; Q]);
        }
    }

    /// The columns next to the interface must be fluid on both levels (no obstacle, no wall node).
    fn check_interface_cells(&self) -> Result<()> {
        let Strip { first, last, .. } = self.strip;
        let [_, ny, nz] = self.nc;
        for x in [first - 2, first - 1, last, last + 1] {
            for z in 0..nz {
                for y in 0..ny {
                    if !self.coarse.is_fluid(self.cidx([x, y, z])) {
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
                    if !self.fine.is_fluid(self.fidx([x, y, z])) {
                        return Err(Error::Invalid(format!(
                            "fine cell ({x}, {y}, {z}) next to the interface is not a fluid cell: the interface must lie in open fluid"
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// The populations of a coarse cell, for studies.
    pub fn coarse_populations(&self, x: usize, y: usize, z: usize) -> [f64; Q] {
        self.coarse.cell_populations(self.coarse.grid().index([x, y, z]))
    }

    /// The populations of a fine cell, for studies.
    pub fn fine_populations(&self, x: usize, y: usize, z: usize) -> [f64; Q] {
        self.fine.cell_populations(self.fine.grid().index([x, y, z]))
    }

    /// ν̃ of a coarse and a fine cell in physical (coarse lattice) units, for studies.
    pub fn nu_tilde_coarse(&self, x: usize, y: usize, z: usize) -> Option<f64> {
        self.coarse.sa_value(self.coarse.grid().index([x, y, z]))
    }

    pub fn nu_tilde_fine(&self, x: usize, y: usize, z: usize) -> Option<f64> {
        self.fine.sa_value(self.fine.grid().index([x, y, z])).map(|v| v / RATIO as f64)
    }

    /// Both levels transport ν̃ without production or destruction, for verification.
    pub fn set_sa_transport_only(&mut self) {
        self.coarse.set_sa_model(false);
        self.fine.set_sa_model(false);
    }

    /// Start ν̃ on both levels from `value(x)` (coarse lattice units, `x` in coarse cells as in
    /// [`Refined::initialise`]).
    pub fn initialise_nu_tilde(&mut self, value: &dyn Fn([f64; 3]) -> f64) -> Result<()> {
        let mut c = Vec::new();
        for cell in 0..self.coarse.grid().len() {
            if self.coarse.is_fluid(cell) {
                let p = self.coarse.grid().coords(cell);
                c.push((cell, value([0, 1, 2].map(|a| p[a] as f64 + 0.5))));
            }
        }
        self.coarse.sa_impose(&c)?;
        let mut f = Vec::new();
        for cell in 0..self.fine.grid().len() {
            if self.fine.is_fluid(cell) {
                f.push((cell, RATIO as f64 * value(self.position(self.fine.grid().coords(cell)))));
            }
        }
        self.fine.sa_impose(&f)
    }

    /// ν̃ per coarse cell (coarse lattice units): the coarse level outside the strip, the mean of the fine cells in it.
    pub fn composite_nu_tilde(&self) -> Vec<f64> {
        let Strip { first, last, .. } = self.strip;
        let g = self.coarse.grid();
        (0..g.len())
            .map(|cell| {
                let [x, y, z] = self.loc(g.coords(cell));
                if !(first..last).contains(&x) {
                    return self.coarse.sa_value(cell).unwrap_or(0.0);
                }
                let mut sum = 0.0;
                for k in 0..8 {
                    let fc = self.fidx([RATIO * (x - (first - 1)) + k % 2, RATIO * y + (k / 2) % 2, RATIO * z + k / 4]);
                    sum += self.fine.sa_value(fc).unwrap_or(0.0);
                }
                sum / 8.0 / RATIO as f64
            })
            .collect()
    }

    /// ν̃ per coarse cell of the coarse level alone.
    pub fn coarse_nu_tilde(&self) -> Vec<f64> {
        (0..self.coarse.grid().len()).map(|c| self.coarse.sa_value(c).unwrap_or(0.0)).collect()
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
        self.fidx([x0 + d[0], RATIO * y + d[1], RATIO * z + d[2]])
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
    fn fill_covered(&mut self) -> Result<()> {
        let Strip { first, last, .. } = self.strip;
        let [_, ny, nz] = self.nc;
        for z in 0..nz {
            for y in 0..ny {
                for x in first..last {
                    let cell = self.cidx([x, y, z]);
                    if !self.coarse.is_fluid(cell) {
                        continue;
                    }
                    let (mut rho, mut j, mut n, mut nu_tilde) = (0.0, [0.0; 3], 0.0, 0.0);
                    for dz in 0..RATIO {
                        for dy in 0..RATIO {
                            for dx in 0..RATIO {
                                let fc = self.fidx([RATIO * (x - (first - 1)) + dx, RATIO * y + dy, RATIO * z + dz]);
                                if !self.fine.is_fluid(fc) {
                                    continue;
                                }
                                let (r, u) = moments_of(&self.fine.cell_populations(fc));
                                rho += r;
                                for a in 0..3 {
                                    j[a] += r * u[a];
                                }
                                n += 1.0;
                                nu_tilde += self.fine.sa_value(fc).unwrap_or(0.0);
                            }
                        }
                    }
                    if self.sa && n > 0.0 {
                        self.coarse.sa_impose(&[(cell, nu_tilde / n / RATIO as f64)])?;
                    }
                    if n > 0.0 && rho > 0.0 {
                        let mean = rho / n;
                        self.coarse.set_cell_populations(cell, &equilibrium(mean, [j[0] / rho, j[1] / rho, j[2] / rho]));
                    }
                }
            }
        }
        Ok(())
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
                        v.push(moments_of(&self.coarse.cell_populations(self.cidx([x, y, z]))));
                    }
                }
                v
            })
            .collect();
        if self.sa {
            self.nu_tilde = self.snapshot_nu_tilde();
        }
    }

    /// ν̃ of the coarse cells in the columns the ghost interpolation reads.
    fn snapshot_nu_tilde(&self) -> Vec<Vec<f64>> {
        let Strip { first, last, .. } = self.strip;
        let [_, ny, nz] = self.nc;
        [first - 2, first - 1, first, last - 1, last, last + 1]
            .iter()
            .map(|&x| {
                let mut v = Vec::with_capacity(ny * nz);
                for z in 0..nz {
                    for y in 0..ny {
                        v.push(self.coarse.sa_value(self.cidx([x, y, z])).unwrap_or(0.0));
                    }
                }
                v
            })
            .collect()
    }

    /// The coarse moments at fine ghost cell `(xf, yf, zf)` of `side`, interpolated linearly in
    /// x, y and z between the coarse cell centres (clamped at the box faces).
    fn stencil(&self, side: usize, dx: usize, yf: usize, zf: usize) -> [(usize, usize, f64); 8] {
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
        // Between the two cell centres around `f`; beyond the first or last centre the line through the
        // nearest two continues (next to a wall), across a periodic face the cells wrap round.
        let axis = |f: f64, n: usize, periodic: bool| -> (usize, usize, f64) {
            if periodic {
                let i = f.floor();
                let (i0, i1) = ((i as i64).rem_euclid(n as i64) as usize, ((i as i64) + 1).rem_euclid(n as i64) as usize);
                return (i0, i1, f - i);
            }
            if n < 2 {
                return (0, 0, 0.0);
            }
            let i = f.floor().clamp(0.0, (n - 2) as f64);
            (i as usize, i as usize + 1, f - i)
        };
        let (y0, y1, ty) = axis(fy, ny, self.periodic[0]);
        let (z0, z1, tz) = axis(fz, nz, self.periodic[1]);
        let mut out = [(0, 0, 0.0); 8];
        let mut k = 0;
        for (c, wx) in [(col0, 1.0 - tx), (col1, tx)] {
            for (z, wz) in [(z0, 1.0 - tz), (z1, tz)] {
                for (y, wy) in [(y0, 1.0 - ty), (y1, ty)] {
                    out[k] = (c, z * ny + y, wx * wy * wz);
                    k += 1;
                }
            }
        }
        out
    }

    /// The coarse moments at fine ghost cell `(dx, yf, zf)` of `side`.
    fn interpolate(&self, side: usize, dx: usize, yf: usize, zf: usize) -> (f64, [f64; 3]) {
        let mut rho = 0.0;
        let mut j = [0.0; 3];
        for (c, i, w) in self.stencil(side, dx, yf, zf) {
            let (r, u) = self.moments.get(c).and_then(|m| m.get(i)).copied().unwrap_or((1.0, [0.0; 3]));
            rho += w * r;
            for a in 0..3 {
                j[a] += w * u[a];
            }
        }
        (rho, j)
    }

    /// ν̃ of the coarse level interpolated to fine ghost cell `(dx, yf, zf)` of `side`, in the fine
    /// level's lattice units (a lattice viscosity doubles when the spacing halves at the same lattice speed).
    fn interpolate_nu_tilde(&self, side: usize, dx: usize, yf: usize, zf: usize, later: f64) -> f64 {
        let at = |v: &[Vec<f64>], c: usize, i: usize| v.get(c).and_then(|m| m.get(i)).copied().unwrap_or(0.0);
        let sum: f64 = self
            .stencil(side, dx, yf, zf)
            .into_iter()
            .map(|(c, i, w)| w * ((1.0 - later) * at(&self.nu_tilde, c, i) + later * at(&self.nu_tilde_next, c, i)))
            .sum();
        RATIO as f64 * sum
    }

    /// Every ghost cell takes the equilibrium of the interpolated coarse moments.
    /// `later` is the share of the coarse step's end in the ν̃ the ghost cells take (0: its start).
    fn fill_ghost_equilibrium(&mut self, later: f64) -> Result<()> {
        let [_, ny, nz] = self.nc;
        for side in 0..2 {
            for zf in 0..RATIO * nz {
                for yf in 0..RATIO * ny {
                    for dx in 0..RATIO {
                        let xf = if side == 0 { dx } else { self.nf[0] - RATIO + dx };
                        let (rho, u) = self.interpolate(side, dx, yf, zf);
                        let cell = self.fidx([xf, yf, zf]);
                        self.fine.set_cell_populations(cell, &equilibrium(rho, u));
                        if self.sa {
                            let nu = self.interpolate_nu_tilde(side, dx, yf, zf, later);
                            self.fine.sa_impose(&[(cell, nu)])?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// The coarse interface columns' post-collision populations that point into the strip go to the ghost cells.
    fn transfer_in(&mut self) {
        self.replenish.clear();
        let [_, ny, nz] = self.nc;
        let linear = self.strip.linear;
        for side in 0..2 {
            let xa = self.interface_column(side);
            let (g, perm) = (*self.coarse.grid(), self.perm);
            let at = move |y: usize, z: usize| {
                let mut p = [0; 3];
                for (i, &a) in perm.iter().enumerate() {
                    p[a] = [xa, y, z][i];
                }
                g.index(p)
            };
            for z in 0..nz {
                for y in 0..ny {
                    let a = self.coarse.cell_populations(at(y, z));
                    // Slopes along y and z: central, one-sided next to a wall (the periodic or missing neighbour
                    // does not count), zero where neither neighbour is a fluid cell.
                    let slope = |n: Option<usize>, p: Option<usize>| -> [f64; Q] {
                        let fluid = |c: Option<usize>| c.filter(|&c| self.coarse.is_fluid(c));
                        let (n, p) = (fluid(n), fluid(p));
                        let pop = |c: usize| self.coarse.cell_populations(c);
                        let mut s = [0.0; Q];
                        match (n, p) {
                            (Some(n), Some(p)) => {
                                let (an, ap) = (pop(n), pop(p));
                                for q in 0..Q {
                                    s[q] = 0.5 * (ap[q] - an[q]);
                                }
                            }
                            (None, Some(c)) | (Some(c), None) => {
                                let (ac, a0) = (pop(c), a);
                                let sign = if p.is_some() { 1.0 } else { -1.0 };
                                for q in 0..Q {
                                    s[q] = sign * (ac[q] - a0[q]);
                                }
                            }
                            (None, None) => {}
                        }
                        s
                    };
                    let sy = if linear { slope(y.checked_sub(1).map(|y| at(y, z)), (y + 1 < ny).then(|| at(y + 1, z))) } else { [0.0; Q] };
                    let sz = if linear { slope(z.checked_sub(1).map(|z| at(y, z)), (z + 1 < nz).then(|| at(y, z + 1))) } else { [0.0; Q] };
                    let sent: Vec<usize> = (0..Q).filter(|&q| into_strip(self.perm, side, q)).collect();
                    for dz in 0..RATIO {
                        for dy in 0..RATIO {
                            let (oy, oz) = (if dy == 0 { -0.25 } else { 0.25 }, if dz == 0 { -0.25 } else { 0.25 });
                            let mut g = [0.0; Q];
                            for &q in &sent {
                                g[q] = a[q] + oy * sy[q] + oz * sz[q];
                            }
                            // Next to a wall the reflection of the cell's own outgoing population, which a ghost
                            // cell does not have, would give the population that the inner ghost cell passes on in the second fine step.
                            let inner = if side == 0 { RATIO - 1 } else { 0 };
                            let fine_cell = self.ghost(side, y, z, [inner, dy, dz]);
                            let (yf, zf) = ((RATIO * y + dy) as i64, (RATIO * z + dz) as i64);
                            for &q in &sent {
                                let behind =
                                    |i: i64, c: i32, n: usize, periodic: bool| !periodic && (i - i64::from(c) < 0 || i - i64::from(c) >= n as i64);
                                if behind(yf, cdir(self.perm, q, 1), self.nf[1], self.periodic[0])
                                    || behind(zf, cdir(self.perm, q, 2), self.nf[2], self.periodic[1])
                                {
                                    self.replenish.push((fine_cell, q, g[q], (side * nz + z) * ny + y));
                                }
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

    /// After the first fine step: what wall reflection has left in the outer ghost layer, pointing
    /// out of the strip, would be pulled across the fine box's periodic seam by the next step. It is
    /// taken out now, into `pending`, and joins the sum at the end of the coarse step.
    fn harvest_outer_ghosts(&mut self) {
        let [_, ny, nz] = self.nc;
        for side in 0..2 {
            for z in 0..nz {
                for y in 0..ny {
                    for dz in 0..RATIO {
                        for dy in 0..RATIO {
                            let cell = self.ghost(side, y, z, [if side == 0 { 0 } else { RATIO - 1 }, dy, dz]);
                            let mut f = self.fine.cell_populations(cell);
                            let Some(slot) = self.pending.get_mut((side * nz + z) * ny + y) else { continue };
                            for q in (0..Q).filter(|&q| !into_strip(self.perm, side, q) && cdir(self.perm, q, 0) != 0) {
                                slot[q] += f[q] / (RATIO * RATIO * RATIO) as f64;
                                f[q] = 0.0;
                            }
                            self.fine.set_cell_populations(cell, &f);
                        }
                    }
                }
            }
        }
    }

    /// Put back the populations the inner ghost cells next to a wall lack (see `replenish`).
    fn replenish_inner_ghosts(&mut self) {
        let norm = (RATIO * RATIO * RATIO) as f64;
        for (cell, q, value, slot) in std::mem::take(&mut self.replenish) {
            let mut f = self.fine.cell_populations(cell);
            f[q] = value;
            self.fine.set_cell_populations(cell, &f);
            if let Some(r) = self.replenished.get_mut(slot) {
                r[q] += value / norm;
            }
        }
    }

    /// What the ghost cells hold, to put back after a collision that must not touch them.
    fn save_ghosts(&mut self) -> Vec<(usize, [f64; Q])> {
        let mut saved = std::mem::take(&mut self.saved);
        saved.clear();
        for side in 0..2 {
            for zf in 0..self.nf[2] {
                for yf in 0..self.nf[1] {
                    for dx in 0..RATIO {
                        let xf = if side == 0 { dx } else { self.nf[0] - RATIO + dx };
                        let cell = self.fidx([xf, yf, zf]);
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
                    let cell = self.cidx([xa, y, z]);
                    let mut a = self.coarse.cell_populations(cell);
                    let mut sum = [0.0; Q];
                    for dz in 0..RATIO {
                        for dy in 0..RATIO {
                            for dx in 0..RATIO {
                                let g = self.ghost(side, y, z, [dx, dy, dz]);
                                let f = self.fine.cell_populations(g);
                                for q in (0..Q).filter(|&q| cdir(self.perm, q, 0) != 0) {
                                    sum[q] += f[q] / norm;
                                }
                                self.fine.set_cell_populations(g, &[0.0; Q]);
                            }
                        }
                    }
                    // The slots that point out of the strip take what the fine flow sends. A population
                    // that wall reflection turned around inside a ghost cell is back in the inward
                    // direction: it belongs to the coarse cell's outgoing population of the opposite
                    // direction, which the coarse step reflects at the wall itself.
                    for q in (0..Q).filter(|&q| cdir(self.perm, q, 0) != 0 && !into_strip(self.perm, side, q)) {
                        a[q] = sum[q] + self.pending.get((side * nz + z) * ny + y).map_or(0.0, |p| p[q]);
                    }
                    for q in (0..Q).filter(|&q| cdir(self.perm, q, 0) != 0 && into_strip(self.perm, side, q)) {
                        if !self.neighbour_exists(y, z, -cdir(self.perm, q, 1), -cdir(self.perm, q, 2)) {
                            a[q] += sum[q] - self.replenished.get((side * nz + z) * ny + y).map_or(0.0, |r| r[q]);
                        }
                    }
                    self.coarse.set_cell_populations(cell, &a);
                }
            }
        }
        self.pending.iter_mut().for_each(|p| *p = [0.0; Q]);
        self.replenished.iter_mut().for_each(|p| *p = [0.0; Q]);
    }

    /// One coarse step (two fine steps).
    pub fn step(&mut self) -> Result<()> {
        self.fill_covered()?;
        self.snapshot_moments();
        self.fill_ghost_equilibrium(0.0)?;
        self.coarse.step_collide()?;
        if self.sa {
            self.nu_tilde_next = self.snapshot_nu_tilde();
        }
        self.fine.step_collide()?;
        self.transfer_in();
        self.coarse.step_stream()?;
        self.fine.step_stream()?;
        // Second fine step: the ghost cells keep what they hold through the collision.
        self.harvest_outer_ghosts();
        self.replenish_inner_ghosts();
        let saved = self.save_ghosts();
        self.fill_ghost_equilibrium(0.5)?;
        self.fine.step_collide()?;
        for (cell, f) in &saved {
            self.fine.set_cell_populations(*cell, f);
        }
        self.saved = saved;
        self.fine.step_stream()?;
        self.restrict();
        Ok(())
    }

    /// Start both levels from a given state: `state(x)` is the density and velocity (lattice
    /// units) at `x`, a position in coarse cells (the cell `i` has its centre at `i + 0.5`).
    pub fn initialise(&mut self, state: &dyn Fn([f64; 3]) -> (f64, [f64; 3])) {
        let at = |flow: &mut Flow, scale: f64| {
            for cell in 0..flow.grid().len() {
                let c = flow.grid().coords(cell);
                let x = [0, 1, 2].map(|a| (c[a] as f64 + 0.5) * scale);
                let (rho, u) = state(x);
                flow.set_cell_populations(cell, &equilibrium(rho, u));
            }
        };
        at(&mut self.coarse, 1.0);
        // The fine grid starts at the interface column `first − 1`, in half cells.
        for cell in 0..self.fine.grid().len() {
            let (rho, u) = state(self.position(self.fine.grid().coords(cell)));
            self.fine.set_cell_populations(cell, &equilibrium(rho, u));
        }
        self.clear_ghosts();
    }

    /// One coarse step of the coarse flow alone, as if there were no strip: the baseline a
    /// refined run is compared with.
    pub fn step_unrefined(&mut self) -> Result<()> {
        self.coarse.step()
    }

    /// The moments of every coarse cell, covered ones included (the coarse flow alone).
    pub fn coarse_moments(&self) -> Vec<(f64, [f64; 3])> {
        (0..self.coarse.grid().len())
            .map(|c| if self.coarse.is_fluid(c) { moments_of(&self.coarse.cell_populations(c)) } else { (0.0, [0.0; 3]) })
            .collect()
    }

    /// With the Spalart–Allmaras model, what the two interface columns of the coarse level hold:
    /// the largest and mean ν_t/ν there and, with DDES, the share of those cells in LES mode (the
    /// destruction length below the wall distance). An interface in a LES region has little eddy
    /// viscosity to damp what the interface produces.
    pub fn interface_report(&self) -> Option<InterfaceReport> {
        let sa = self.coarse.spalart_allmaras()?;
        let nu = (1.0 / self.coarse.omegas().0 - 0.5) / 3.0;
        let (distance, length) = (self.coarse.sa_wall_distance()?, self.coarse.ddes_length());
        let [_, ny, nz] = self.nc;
        let (mut max, mut sum, mut n, mut les) = (0.0f64, 0.0, 0usize, 0usize);
        for side in 0..2 {
            for z in 0..nz {
                for y in 0..ny {
                    let cell = self.cidx([self.interface_column(side), y, z]);
                    if !self.coarse.is_fluid(cell) {
                        continue;
                    }
                    let ratio = sa.nu_t(cell, nu) / nu;
                    max = max.max(ratio);
                    sum += ratio;
                    n += 1;
                    if let (Some(l), Some(d)) = (length.and_then(|l| l.get(cell)), distance.get(cell))
                        && *l > 0.0
                        && *l < 0.999 * d
                    {
                        les += 1;
                    }
                }
            }
        }
        (n > 0).then(|| InterfaceReport {
            nu_t_over_nu_max: max,
            nu_t_over_nu_mean: sum / n as f64,
            les_share: length.map(|_| les as f64 / n as f64),
        })
    }

    /// The mass in fine-cell units: the coarse cells outside the strip (8 each) and the fine cells.
    pub fn mass(&self) -> f64 {
        let Strip { first, last, .. } = self.strip;
        let g = self.coarse.grid();
        let mut m = 0.0;
        for cell in 0..g.len() {
            let [x, _, _] = self.loc(g.coords(cell));
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
                let [x, y, z] = self.loc(g.coords(cell));
                if !(first..last).contains(&x) {
                    return if self.coarse.is_fluid(cell) { moments_of(&self.coarse.cell_populations(cell)) } else { (0.0, [0.0; 3]) };
                }
                let (mut rho, mut j, mut n) = (0.0, [0.0; 3], 0.0);
                for dz in 0..RATIO {
                    for dy in 0..RATIO {
                        for dx in 0..RATIO {
                            let fc = self.fidx([RATIO * (x - (first - 1)) + dx, RATIO * y + dy, RATIO * z + dz]);
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
