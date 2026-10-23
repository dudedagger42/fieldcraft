//! Rotors ([`fieldcraft_model::Rotor`]): solid bodies that turn through the fixed lattice.
//!
//! The body is voxelised once, in its own frame, on a grid twice as fine as the lattice. At every
//! step a cell of the region the body sweeps is solid when its centre, turned back by the rotor's
//! angle, falls in a solid cell of that fine grid. Between the steps:
//!
//! - a fluid cell whose upstream neighbour along `c_q` is rotor-solid takes the bounced population
//!   with Ladd's moving-wall term, at the surface velocity `Ω × r` of the link's midpoint
//!   (A. J. C. Ladd, *Numerical simulations of particulate suspensions via a discretized Boltzmann
//!   equation*, J. Fluid Mech. 271:285–309, 1994), and the link gives the body the momentum
//!   `−c_q (f_out + f_back)` (the momentum exchange of the static walls);
//! - a cell the body covers gives its fluid momentum to the body and stops colliding;
//! - a cell the body uncovers is refilled with the equilibrium at the surface's velocity and the
//!   mean density of its fluid neighbours, and takes that momentum from the body (C. K. Aidun,
//!   Y. Lu, E.-J. Ding, *Direct analysis of particulate suspensions with inertia using the
//!   discrete Boltzmann equation*, J. Fluid Mech. 373:287–311, 1998: the covered and uncovered
//!   cells' momentum is part of the force on a moving body).
//!
//! The fluid a blade's leading face covers is denser than the fluid its trailing face uncovers is
//! refilled with, so covering and refilling alone lose mass every step (0.9 % in four turns of a
//! paddle), and Ladd's term adds a little where the staircase wall moves along its normal. The
//! body's interior is kept as a reservoir of that mass. What it holds beyond its covered cells'
//! share at the reference density goes back at once, spread over the fluid cells of the swept
//! region as density at rest (`w_q δρ` on each population: no momentum). The covered cells'
//! share stays with the body: their count changes a little with the angle (by a dozen cells for a
//! paddle), and giving that back would send a pressure pulse through the gap every step. The
//! fluid's mass and that share ([`RotorState::displaced`]) add up to the start's.
//!
//! The torque about the axis is the sum of those momenta's moments about the origin. The wall is
//! a staircase at the fine grid's resolution, so like the static obstacles it is first order in
//! the spacing.
//!
//! No code of any lattice Boltzmann package was read; this is written from the papers.

use crate::lattice::{C, Q, W, c, equilibrium, moving_wall, opposite};
use crate::voxel::{Grid, voxelise};
use crate::{Error, Result};
use fieldcraft_mesh::surface::BoundarySurface;

/// The fastest a rotor's surface may move through the lattice (lattice units): its blade tips
/// must stay well below the lattice's speed of sound (√⅓) and cross less than a cell per step.
pub const MAX_TIP_SPEED: f64 = 0.2;

/// One rotor as the flow steps it.
#[derive(Debug, Clone)]
pub struct RotorState {
    pub body: String,
    /// A point on the axis (model units) and the unit axis.
    origin: [f64; 3],
    axis: [f64; 3],
    /// The angle turned per time step, and so far.
    omega: f64,
    theta: f64,
    /// The body in its own frame, on a grid of half the lattice spacing.
    fine: Grid,
    solid_fine: Vec<bool>,
    /// The fluid cells the body may cover or neighbour, in order, and whether it covers each now.
    swept: Vec<u32>,
    covered: Vec<bool>,
    /// The largest distance of the body from the axis (model units).
    pub radius: f64,
    /// Whether the box is periodic along each axis: a neighbour across a periodic face is the
    /// cell the streaming pulls from there.
    periodic: [bool; 3],
    /// The momentum and its moment about the origin the fluid gave the body in the last step
    /// (lattice units: momentum per step is a force).
    pub force: [f64; 3],
    pub torque: [f64; 3],
    /// The covered cells at the start and now, and the mass the body holds beyond the start's.
    covered0: i64,
    count: i64,
    reservoir: f64,
    /// The axial torque of each step over the last turn (a ring of the steps of one turn, at
    /// most [`MAX_TURN_STEPS`]), and where the next goes.
    history: Vec<f64>,
    next: usize,
}

/// Most steps the torque over the last turn is averaged from.
pub const MAX_TURN_STEPS: usize = 1 << 16;

/// A rotor's state for the reply, lattice units.
#[derive(Debug, Clone, PartialEq)]
pub struct RotorSample {
    pub body: String,
    /// The torque the fluid exerts about the axis, in the last step and averaged over the last
    /// turn (or over the steps so far, before the first turn is complete).
    pub torque: f64,
    pub mean_torque: f64,
    /// The angle turned, and the angle turned per step.
    pub angle: f64,
    pub omega: f64,
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// `v` turned by `angle` about the unit axis `k` (Rodrigues).
fn turn(v: [f64; 3], k: [f64; 3], angle: f64) -> [f64; 3] {
    let (s, co) = angle.sin_cos();
    let kxv = cross(k, v);
    let kv = dot(k, v);
    [0, 1, 2].map(|a| v[a] * co + kxv[a] * s + k[a] * kv * (1.0 - co))
}

impl RotorState {
    /// The rotor of `surface` about `origin` along the unit `axis`, turning `omega` per step, in
    /// a flow on `grid` whose cells `fluid` marks before any rotor.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        body: &str,
        surface: &BoundarySurface,
        origin: [f64; 3],
        axis: [f64; 3],
        omega: f64,
        grid: &Grid,
        fluid: &[bool],
        periodic: [bool; 3],
    ) -> Result<RotorState> {
        let points = surface.points();
        if points.is_empty() {
            return Err(Error::Invalid(format!("rotor `{body}` has no surface")));
        }
        let h = grid.spacing;
        let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        let (mut radius, mut a_lo, mut a_hi) = (0.0f64, f64::INFINITY, f64::NEG_INFINITY);
        for p in points {
            let r = [0, 1, 2].map(|a| p[a] - origin[a]);
            let along = dot(r, axis);
            let off = [0, 1, 2].map(|a| r[a] - along * axis[a]);
            radius = radius.max(dot(off, off).sqrt());
            a_lo = a_lo.min(along);
            a_hi = a_hi.max(along);
            for a in 0..3 {
                lo[a] = lo[a].min(p[a]);
                hi[a] = hi[a].max(p[a]);
            }
        }
        // The fine grid: the body's box and a fine cell around it.
        let s = h / 2.0;
        let mut cells = [0usize; 3];
        let mut fine_origin = [0.0; 3];
        for a in 0..3 {
            fine_origin[a] = lo[a] - s;
            cells[a] = (((hi[a] - lo[a]) / s).ceil() as usize + 2).max(1);
        }
        let fine = Grid { origin: fine_origin, spacing: s, cells };
        // A body inside the box takes at most eight fine cells per lattice cell: one far larger
        // is not in this box (or not in its units), and is refused rather than allocated.
        let cap = 8 * grid.len() + 4096;
        if cells.iter().try_fold(1usize, |n, c| n.checked_mul(*c)).is_none_or(|n| n > cap) {
            return Err(Error::Invalid(format!(
                "rotor `{body}` spans {:.3} × {:.3} × {:.3}, far more than the fluid box holds: is it inside the box, in the same units?",
                hi[0] - lo[0],
                hi[1] - lo[1],
                hi[2] - lo[2]
            )));
        }
        let mut solid_fine = Vec::new();
        solid_fine.try_reserve_exact(fine.len()).map_err(|_| Error::Limit(format!("not enough memory for rotor `{body}`")))?;
        solid_fine.resize(fine.len(), false);
        voxelise(surface, &fine, &mut solid_fine)?;
        // Every fluid cell within two spacings of the cylinder the body sweeps: a covered cell's
        // diagonal neighbours lie up to √3 spacings out, and each must take its bounced populations.
        let mut swept = Vec::new();
        for (cell, &f) in fluid.iter().enumerate() {
            if !f {
                continue;
            }
            let p = grid.centre(cell);
            let r = [0, 1, 2].map(|a| p[a] - origin[a]);
            let along = dot(r, axis);
            if along < a_lo - 2.0 * h || along > a_hi + 2.0 * h {
                continue;
            }
            let off = [0, 1, 2].map(|a| r[a] - along * axis[a]);
            if dot(off, off).sqrt() <= radius + 2.0 * h {
                swept.push(u32::try_from(cell).map_err(|_| Error::Limit("too many cells for a rotor".into()))?);
            }
        }
        let covered = vec![false; swept.len()];
        Ok(RotorState {
            body: body.to_owned(),
            origin,
            axis,
            omega,
            theta: 0.0,
            fine,
            solid_fine,
            swept,
            covered,
            radius,
            periodic,
            force: [0.0; 3],
            torque: [0.0; 3],
            covered0: 0,
            count: 0,
            reservoir: 0.0,
            history: Vec::new(),
            next: 0,
        })
    }

    /// The cell at `ijk + d` (wrapped across periodic faces), if it is in the box.
    fn neighbour(&self, grid: &Grid, ijk: [i64; 3], d: [i32; 3]) -> Option<usize> {
        let mut out = [0usize; 3];
        for a in 0..3 {
            let n = grid.cells[a] as i64;
            let mut x = ijk[a] + i64::from(d[a]);
            if x < 0 || x >= n {
                if !self.periodic[a] {
                    return None;
                }
                x = x.rem_euclid(n);
            }
            out[a] = x as usize;
        }
        Some(grid.index(out))
    }

    /// The surface speed at the rim, lattice units.
    pub fn tip_speed(&self, spacing: f64) -> f64 {
        (self.omega * self.radius / spacing).abs()
    }

    /// Whether the body covers the point `p` (model units) at the current angle.
    fn covers(&self, p: [f64; 3]) -> bool {
        let r = [0, 1, 2].map(|a| p[a] - self.origin[a]);
        let back = turn(r, self.axis, -self.theta);
        let q = [0, 1, 2].map(|a| back[a] + self.origin[a]);
        self.fine.cell_at(q).is_some_and(|c| self.solid_fine.get(c).copied().unwrap_or(false))
    }

    /// The surface velocity at `p` (model units), lattice units: `Ω × r` with `Ω` per step and
    /// `r` in spacings.
    fn surface_velocity(&self, p: [f64; 3], spacing: f64) -> [f64; 3] {
        let r = [0, 1, 2].map(|a| (p[a] - self.origin[a]) / spacing);
        cross(self.axis.map(|k| k * self.omega), r)
    }

    /// Mark the cells the body covers at the current angle: `fluid` false there, `owner` set.
    /// Used once at the start (no momentum is counted).
    pub fn place(&mut self, grid: &Grid, fluid: &mut [bool], owner: &mut [u8], tag: u8) {
        for (i, &cell) in self.swept.iter().enumerate() {
            let cell = cell as usize;
            let on = self.covers(grid.centre(cell));
            if let Some(slot) = self.covered.get_mut(i) {
                *slot = on;
            }
            if let Some(f) = fluid.get_mut(cell) {
                *f = !on;
            }
            if let Some(o) = owner.get_mut(cell) {
                *o = if on { tag } else { 0 };
            }
        }
        self.count = self.covered.iter().filter(|c| **c).count() as i64;
        self.covered0 = self.count;
    }

    /// The covered cells beyond the start's at the reference density 1: the mass the body holds
    /// for the fluid (lattice units, cells of unit volume).
    pub fn displaced(&self) -> f64 {
        (self.count - self.covered0) as f64
    }

    /// After streaming (`g` streamed, `f` post-collision, both `stride` apart per direction): the
    /// bounced populations of every fluid cell next to the body, and the momentum they give it.
    pub fn bounce(&mut self, grid: &Grid, f: &[f64], g: &mut [f64], stride: usize, fluid: &[bool], owner: &[u8], tag: u8) {
        let h = grid.spacing;
        let (mut force, mut torque) = ([0.0; 3], [0.0; 3]);
        for &cell in &self.swept {
            let cell = cell as usize;
            if !fluid.get(cell).copied().unwrap_or(false) {
                continue;
            }
            let ijk = grid.coords(cell).map(|x| x as i64);
            let centre = grid.centre(cell);
            for q in 1..Q {
                let cq = C[q];
                let Some(up) = self.neighbour(grid, ijk, cq.map(|x| -x)) else { continue };
                if owner.get(up).copied() != Some(tag) {
                    continue;
                }
                let cv = c(q);
                let mid = [0, 1, 2].map(|a| centre[a] - 0.5 * cv[a] * h);
                let uw = self.surface_velocity(mid, h);
                let out = f.get(opposite(q) * stride + cell).copied().unwrap_or(0.0);
                let du = moving_wall(q, uw);
                let back = out + du;
                // The mass Ladd's term gives the fluid comes out of the body's reservoir.
                self.reservoir -= du;
                if let Some(slot) = g.get_mut(q * stride + cell) {
                    *slot = back;
                }
                let dp = cv.map(|x| -x * (out + back));
                let r = [0, 1, 2].map(|a| (mid[a] - self.origin[a]) / h);
                let m = cross(r, dp);
                for a in 0..3 {
                    force[a] += dp[a];
                    torque[a] += m[a];
                }
            }
        }
        self.force = force;
        self.torque = torque;
    }

    /// Turn by one step and update the cells the body covers, in `f` (the populations the next
    /// collision reads): covered cells stop being fluid and give their momentum and mass to the
    /// body; uncovered cells are refilled and take theirs from it; then what the body holds beyond
    /// its covered cells' reference share goes back to the fluid.
    pub fn advance(&mut self, grid: &Grid, f: &mut [f64], stride: usize, fluid: &mut [bool], owner: &mut [u8], tag: u8) {
        self.theta += self.omega;
        let h = grid.spacing;
        for i in 0..self.swept.len() {
            let cell = self.swept[i] as usize;
            let centre = grid.centre(cell);
            let now = self.covers(centre);
            let was = self.covered[i];
            if now == was {
                continue;
            }
            self.covered[i] = now;
            let r = [0, 1, 2].map(|a| (centre[a] - self.origin[a]) / h);
            let p = if now {
                // The fluid's momentum and mass go to the body.
                let mut j = [0.0; 3];
                for q in 0..Q {
                    let v = f.get(q * stride + cell).copied().unwrap_or(0.0);
                    self.reservoir += v;
                    let cv = c(q);
                    for a in 0..3 {
                        j[a] += v * cv[a];
                    }
                }
                fluid[cell] = false;
                owner[cell] = tag;
                self.count += 1;
                j
            } else {
                // Refilled at the surface's velocity and its fluid neighbours' mean density.
                let ijk = grid.coords(cell).map(|x| x as i64);
                let (mut sum, mut count) = (0.0, 0usize);
                for cq in C.iter().skip(1) {
                    let Some(nb) = self.neighbour(grid, ijk, *cq) else { continue };
                    if fluid.get(nb).copied().unwrap_or(false) {
                        sum += (0..Q).map(|q| f.get(q * stride + nb).copied().unwrap_or(0.0)).sum::<f64>();
                        count += 1;
                    }
                }
                let rho = if count > 0 { sum / count as f64 } else { 1.0 };
                let uw = self.surface_velocity(centre, h);
                for q in 0..Q {
                    if let Some(slot) = f.get_mut(q * stride + cell) {
                        *slot = equilibrium(q, rho, uw);
                    }
                }
                fluid[cell] = true;
                owner[cell] = 0;
                self.reservoir -= rho;
                self.count -= 1;
                uw.map(|u| -rho * u)
            };
            let m = cross(r, p);
            for a in 0..3 {
                self.force[a] += p[a];
                self.torque[a] += m[a];
            }
        }
        let excess = self.reservoir - self.displaced();
        let open = self.swept.iter().filter(|&&cell| fluid.get(cell as usize).copied().unwrap_or(false)).count();
        if excess != 0.0 && open > 0 {
            let share = excess / open as f64;
            for &cell in &self.swept {
                let cell = cell as usize;
                if !fluid.get(cell).copied().unwrap_or(false) {
                    continue;
                }
                for (q, w) in W.iter().enumerate() {
                    if let Some(slot) = f.get_mut(q * stride + cell) {
                        *slot += w * share;
                    }
                }
            }
            self.reservoir -= excess;
        }
    }

    /// The torque about the axis in the last step (lattice units).
    pub fn axial_torque(&self) -> f64 {
        dot(self.torque, self.axis)
    }

    /// Record the last step's axial torque in the ring of the last turn.
    pub fn record(&mut self) {
        let turn = if self.omega != 0.0 { (std::f64::consts::TAU / self.omega.abs()).ceil() as usize } else { 1 };
        let len = turn.clamp(1, MAX_TURN_STEPS);
        let t = self.axial_torque();
        if self.history.len() < len {
            self.history.push(t);
        } else if let Some(slot) = self.history.get_mut(self.next) {
            *slot = t;
        }
        self.next = (self.next + 1) % len;
    }

    /// The state for the reply.
    pub fn sample(&self) -> RotorSample {
        let mean = if self.history.is_empty() { 0.0 } else { self.history.iter().sum::<f64>() / self.history.len() as f64 };
        RotorSample { body: self.body.clone(), torque: self.axial_torque(), mean_torque: mean, angle: self.theta, omega: self.omega }
    }

    /// The angle turned so far.
    pub fn angle(&self) -> f64 {
        self.theta
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_quarter_turn_about_z_takes_x_to_y() {
        let v = turn([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], std::f64::consts::FRAC_PI_2);
        assert!((v[0]).abs() < 1e-15 && (v[1] - 1.0).abs() < 1e-15 && v[2].abs() < 1e-15, "{v:?}");
    }
}
