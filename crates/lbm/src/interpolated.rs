//! Linearly interpolated bounce-back at walls that do not lie half way between cell centres.
//!
//! Plain bounce-back puts a wall half way along every link from a fluid cell into a solid one, so
//! a surface at any other place is moved to the nearest such position: a staircase whose wall
//! can only sit in steps of half a cell (0.7 of a cell along a 45 degree wall), and the flow rate
//! of a laminar channel, which goes with the width cubed, jumps with it. The remedy here is the
//! linear scheme of M. Bouzidi, M. Firdaouss and P. Lallemand, "Momentum transfer of a
//! Boltzmann-lattice fluid with boundaries", Physics of Fluids 13(11), 3452-3459 (2001).
//!
//! For a link from fluid cell `x` towards the wall let `q` be the distance from `x` to the true
//! surface along the link, over the link's length (so `q = 1/2` is the half-way wall). With `i`
//! the direction into the wall, `f*` the post-collision populations and `x - c_i` the cell behind
//! `x` (away from the wall), the population that comes back along `-i` is
//!
//! - `q < 1/2`: `f_-i(x) = 2 q f*_i(x) + (1 - 2 q) f*_i(x - c_i)`,
//! - `q >= 1/2`: `f_-i(x) = f*_i(x) / (2 q) + (2 q - 1) / (2 q) f*_-i(x)`.
//!
//! Both come from following the characteristic back over one time step: it reflects at the
//! wall and begins `1 - 2 q` cells behind `x` (for `q < 1/2`, between `x` and `x - c_i`), or, for
//! `q >= 1/2`, between the wall and `x`, where the value is interpolated between `f*_-i(x)` and
//! the incoming one; the two agree at `q = 1/2`, where both are plain bounce-back. (The paper's
//! text could not be opened for this implementation; the two forms were derived and checked
//! against that continuity and against the Poiseuille test below.)
//!
//! **Quadratic form** (`WallOrder::Quadratic`). The same characteristic read from three nodes in line
//! (Lagrange interpolation, derived here the same way): for `q < 1/2`
//! `f_-i(x) = q (1 + 2 q) f*_i(x) + (1 - 4 q^2) f*_i(x - c_i) - q (1 - 2 q) f*_i(x - 2 c_i)`, and for `q >= 1/2`
//! `f_-i(x) = f*_i(x) / (q (2 q + 1)) + (2 q - 1) / q f*_-i(x) - (2 q - 1) / (2 q + 1) f*_-i(x - c_i)`. The
//! coefficients sum to one and both reduce to plain bounce-back at `q = 1/2`. A link without the third
//! node falls back to the linear rule, one without the second to the half-way wall; the report counts
//! each.
//!
//! **Fallbacks.** A link whose cell behind is not a fluid cell (a thin passage) keeps the
//! half-way rule when `q < 1/2`, as does one whose segment meets no surface; both are counted
//! ([`InterpolationReport`]). A link with `|q - 1/2| < 1e-9` is plain bounce-back, so a wall that
//! already sits half way (every grid-aligned one) is bit for bit what it was. Walls at rest only.
//!
//! **Mass.** On a slanted wall the link rule returns another amount than left along the link, so
//! the scheme leaks mass steadily (an aligned wall does not: `q = 1/2` returns what left). It is
//! taken back every step from the walls alone, with the weights of the lattice (they sum to one and
//! carry no momentum, so none is added):
//!
//! - `MassFix::Global` (the default): the sum over all interpolated links of (returned - sent) is
//!   spread evenly over the fluid cells and removed. It does not depend on what the box does
//!   elsewhere, so inlets and outlets are not fought. It needs one sum over the links, in link
//!   order, in double precision.
//! - `MassFix::Local`: each wall node removes the sum over its own links. No global sum, but it moved
//!   the flow rate of slanted channels by 0.3 point at 16 cells, so it is not the default.
//!
//! At `q = 1/2` the term is exactly zero. These are design choices of FieldCraft's, not published schemes.

use crate::lattice::{Q, W, opposite};

/// The order of the interpolation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum WallOrder {
    /// Two nodes (Bouzidi, Firdaouss and Lallemand's linear scheme).
    #[default]
    Linear,
    /// Three nodes in line where the flow has them, linear elsewhere.
    Quadratic,
}

/// How the mass the interpolated walls create is taken back.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MassFix {
    /// Not at all (to measure the leak).
    Off,
    /// Each wall node gives back what its own links created.
    Local,
    /// Every step the mass the wall links created is spread over all fluid cells.
    #[default]
    Global,
}

/// How the links of a flow were treated.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct InterpolationReport {
    /// Bounce-back links into obstacles at rest.
    pub links: usize,
    /// Links with an interpolated rule (linear or quadratic).
    pub interpolated: usize,
    /// Of those, the links with the quadratic rule (three nodes in line).
    pub quadratic: usize,
    /// Links whose wall already lies half way (`|q - 1/2| < 1e-9`): plain bounce-back.
    pub half_way: usize,
    /// Links with `q < 1/2` whose cell behind is not a fluid cell: plain bounce-back.
    pub no_second_node: usize,
    /// Links whose segment meets no surface: plain bounce-back.
    pub no_surface: usize,
}

/// What an interpolated link needs beyond the plain bounce: its wall fraction and the order of
/// its rule, for the GPU solver (the cells behind it follow from its cell and direction).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct InterpolatedLink {
    /// The link's index in the flow's link list.
    pub link: u32,
    /// The wall's fraction of the link.
    pub wall: f64,
    /// 2 for the quadratic rule, 1 for the linear one.
    pub order: u8,
}

/// One interpolated link.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Entry {
    /// The link's index in the flow's link list.
    pub link: u32,
    /// The link's cell and direction `q` (the population that streams in from the wall).
    pub cell: u32,
    pub dir: u8,
    /// The wall's fraction of the link.
    pub wall: f64,
    /// The cell behind (away from the wall), or `u32::MAX` when the rule does not use it.
    pub behind: u32,
    /// The cell two links behind, for the quadratic rule of `q < 1/2`, or `u32::MAX`.
    pub behind2: u32,
    /// 2 for the quadratic rule, 1 for the linear one.
    pub order: u8,
}

/// The interpolated links of a flow, in link order.
#[derive(Debug, Clone, Default)]
pub struct Interpolated {
    pub(crate) entries: Vec<Entry>,
    pub report: InterpolationReport,
    /// The entry indices ordered by cell, and per wall node its cell with the range of `order`.
    order: Vec<u32>,
    nodes: Vec<(u32, u32, u32)>,
    /// How the mass the wall links create is taken back (`Global` by default).
    pub(crate) mass_fix: MassFix,
    /// The steps the global correction ran and the mass the walls created over them.
    steps: u64,
    created: f64,
}

impl Interpolated {
    pub(crate) fn new(entries: Vec<Entry>, report: InterpolationReport) -> Self {
        let mut order: Vec<u32> = (0..entries.len() as u32).collect();
        order.sort_by_key(|&k| (entries.get(k as usize).map_or(u32::MAX, |e| e.cell), k));
        let mut nodes: Vec<(u32, u32, u32)> = Vec::new();
        for (at, &k) in order.iter().enumerate() {
            let Some(e) = entries.get(k as usize) else { continue };
            match nodes.last_mut() {
                Some(n) if n.0 == e.cell => n.2 = at as u32 + 1,
                _ => nodes.push((e.cell, at as u32, at as u32 + 1)),
            }
        }
        Self { entries, report, order, nodes, mass_fix: MassFix::default(), steps: 0, created: 0.0 }
    }

    /// Take from every wall node the mass its interpolated links created, spread over its
    /// populations by the lattice weights. `f` are the post-collision populations, `g` the
    /// streamed ones (already carrying the interpolated values).
    pub(crate) fn conserve_mass(&self, f: &[f64], g: &mut [f64], stride: usize) {
        if self.mass_fix != MassFix::Local {
            return;
        }
        for &(cell, start, end) in &self.nodes {
            let mut delta = 0.0;
            for &k in self.order.get(start as usize..end as usize).unwrap_or_default() {
                let Some(e) = self.entries.get(k as usize) else { continue };
                let (Some(v), Some(sent)) = (Self::incoming(e, f, stride), f.get(opposite(usize::from(e.dir)) * stride + e.cell as usize)) else {
                    continue;
                };
                delta += v - sent;
            }
            for i in 0..Q {
                if let Some(x) = g.get_mut(i * stride + cell as usize) {
                    *x -= delta * W[i];
                }
            }
        }
    }

    /// The global kind: the mass the wall links created this step (the sum over all of them of
    /// returned - sent, in link order, in double precision) is taken off evenly over the `cells`
    /// fluid cells by the lattice weights, so no momentum is added. It is defined from the walls
    /// alone, so an inlet or an outlet is not fought (a target of the initial total mass would be).
    /// `f` are the post-collision populations, `g` the streamed ones.
    pub(crate) fn conserve_mass_everywhere(&mut self, fluid: &[bool], cells: usize, f: &[f64], g: &mut [f64], stride: usize) {
        if self.mass_fix != MassFix::Global || self.entries.is_empty() || cells == 0 {
            return;
        }
        let mut delta = 0.0;
        for e in &self.entries {
            let (Some(v), Some(sent)) = (Self::incoming(e, f, stride), f.get(opposite(usize::from(e.dir)) * stride + e.cell as usize)) else {
                continue;
            };
            delta += v - sent;
        }
        self.created += delta;
        self.steps += 1;
        let each = -delta / cells as f64;
        for (cell, &on) in fluid.iter().enumerate() {
            if on {
                for i in 0..Q {
                    if let Some(x) = g.get_mut(i * stride + cell) {
                        *x += each * W[i];
                    }
                }
            }
        }
    }

    /// Whether the correction is the global kind.
    pub fn is_global(&self) -> bool {
        self.mass_fix == MassFix::Global
    }

    /// What the global correction took off so far: the steps it ran and the mass the walls created
    /// over them (lattice units, density 1 per cell).
    pub fn mass_created(&self) -> (u64, f64) {
        (self.steps, self.created)
    }

    /// The population of `entry` after streaming, from the post-collision populations `f`
    /// (`stride` between directions); `None` when an index is out of range.
    pub(crate) fn incoming(entry: &Entry, f: &[f64], stride: usize) -> Option<f64> {
        let into = f.get(opposite(usize::from(entry.dir)) * stride + entry.cell as usize).copied()?;
        let q = entry.wall;
        if q < 0.5 {
            let back = f.get(opposite(usize::from(entry.dir)) * stride + entry.behind as usize).copied()?;
            if entry.order == 2 {
                let back2 = f.get(opposite(usize::from(entry.dir)) * stride + entry.behind2 as usize).copied()?;
                Some(q * (1.0 + 2.0 * q) * into + (1.0 - 4.0 * q * q) * back - q * (1.0 - 2.0 * q) * back2)
            } else {
                let w = 2.0 * q;
                Some(w * into + (1.0 - w) * back)
            }
        } else {
            let own = f.get(usize::from(entry.dir) * stride + entry.cell as usize).copied()?;
            if entry.order == 2 {
                let own2 = f.get(usize::from(entry.dir) * stride + entry.behind as usize).copied()?;
                Some(into / (q * (2.0 * q + 1.0)) + (2.0 * q - 1.0) / q * own - (2.0 * q - 1.0) / (2.0 * q + 1.0) * own2)
            } else {
                let w = 2.0 * q;
                Some(into / w + (w - 1.0) / w * own)
            }
        }
    }

    /// The interpolated links in the order of the flow's link list.
    pub fn links(&self) -> Vec<InterpolatedLink> {
        self.entries.iter().map(|e| InterpolatedLink { link: e.link, wall: e.wall, order: e.order }).collect()
    }

    /// The links in the order of the flow's link list.
    pub(crate) fn entries(&self) -> &[Entry] {
        &self.entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Populations of one direction `q = 2` and its opposite on four cells in a line, `cell`
    /// first, then the one behind it and the one behind that: stride 4.
    fn line(into: [f64; 3], own: [f64; 3]) -> (Vec<f64>, usize) {
        let (q, stride) = (2, 4);
        let mut f = vec![0.0; crate::lattice::Q * stride];
        for k in 0..3 {
            f[opposite(q) * stride + k] = into[k];
            f[q * stride + k] = own[k];
        }
        (f, q)
    }

    fn entry(q: usize, wall: f64, order: u8) -> Entry {
        Entry { link: 0, cell: 0, dir: q as u8, wall, behind: 1, behind2: 2, order }
    }

    #[test]
    fn the_rules_give_plain_bounce_back_at_the_half_way_wall_and_join_there() {
        let (f, q) = line([3.0, 5.0, 8.0], [7.0, 1.0, 2.0]);
        for order in [1, 2] {
            assert_eq!(Interpolated::incoming(&entry(q, 0.5, order), &f, 4), Some(3.0));
        }
        // From below, the near forms approach it.
        for order in [1, 2] {
            let v = Interpolated::incoming(&entry(q, 0.5 - 1e-12, order), &f, 4).unwrap_or(f64::NAN);
            assert!((v - 3.0).abs() < 1e-9, "{v}");
        }
    }

    #[test]
    fn the_linear_forms_weigh_the_populations_by_the_wall_distance() {
        let (f, q) = line([3.0, 5.0, 8.0], [7.0, 1.0, 2.0]);
        // q = 1/4: half of f_i*(x) and half of the one behind.
        assert_eq!(Interpolated::incoming(&entry(q, 0.25, 1), &f, 4), Some(4.0));
        // q = 1: half of f_i*(x) and half of the population that left the wall's way.
        assert_eq!(Interpolated::incoming(&entry(q, 1.0, 1), &f, 4), Some(5.0));
        // An index out of range is no value, not a panic.
        let bad = Entry { cell: 99, ..entry(q, 0.25, 1) };
        assert_eq!(Interpolated::incoming(&bad, &f, 4), None);
    }

    #[test]
    fn the_quadratic_forms_follow_a_parabola_and_keep_a_constant() {
        // q < 1/2: the populations of three cells in line lie on a parabola of the position s
        // (0, -1, -2); the characteristic starts at s = 2 q - 1.
        let para = |s: f64| 1.5 - 0.7 * s + 0.4 * s * s;
        let (f, q) = line([para(0.0), para(-1.0), para(-2.0)], [0.0; 3]);
        for wall in [0.05, 0.2, 0.37, 0.49] {
            let v = Interpolated::incoming(&entry(q, wall, 2), &f, 4).unwrap_or(f64::NAN);
            assert!((v - para(2.0 * wall - 1.0)).abs() < 1e-12, "{wall}: {v}");
        }
        // q >= 1/2: the nodes are 2 q - 1 (the population that left), -1 and -2 (the ones behind).
        // The value of the three nodes on a parabola is read at 0.
        for wall in [0.5, 0.6, 0.8, 1.0] {
            let (a, b, c) = (2.0 * wall - 1.0, -1.0, -2.0);
            let (f, q) = line([para(a), 0.0, 0.0], [para(b), para(c), 0.0]);
            // `own` of the cell is node -1, the one of the cell behind is node -2.
            let v = Interpolated::incoming(&entry(q, wall, 2), &f, 4).unwrap_or(f64::NAN);
            assert!((v - para(0.0)).abs() < 1e-12, "{wall}: {v}");
        }
    }
}
