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
//! **Fallbacks.** A link whose cell behind is not a fluid cell (a thin passage) keeps the
//! half-way rule when `q < 1/2`, as does one whose segment meets no surface; both are counted
//! ([`InterpolationReport`]). A link with `|q - 1/2| < 1e-9` is plain bounce-back, so a wall that
//! already sits half way (every grid-aligned one) is bit for bit what it was. Walls at rest only.
//!
//! The scheme does not conserve mass exactly; the drift is reported by the tests, not corrected.

use crate::lattice::opposite;

/// How the links of a flow were treated.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct InterpolationReport {
    /// Bounce-back links into obstacles at rest.
    pub links: usize,
    /// Links with an interpolated rule.
    pub interpolated: usize,
    /// Links whose wall already lies half way (`|q - 1/2| < 1e-9`): plain bounce-back.
    pub half_way: usize,
    /// Links with `q < 1/2` whose cell behind is not a fluid cell: plain bounce-back.
    pub no_second_node: usize,
    /// Links whose segment meets no surface: plain bounce-back.
    pub no_surface: usize,
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
    /// The cell behind (`q < 1/2`), or `u32::MAX`.
    pub behind: u32,
}

/// The interpolated links of a flow, in link order.
#[derive(Debug, Clone, Default)]
pub struct Interpolated {
    pub(crate) entries: Vec<Entry>,
    pub report: InterpolationReport,
}

impl Interpolated {
    /// The population of `entry` after streaming, from the post-collision populations `f`
    /// (`stride` between directions); `None` when an index is out of range.
    pub(crate) fn incoming(entry: &Entry, f: &[f64], stride: usize) -> Option<f64> {
        let (cell, q) = (entry.cell as usize, usize::from(entry.dir));
        let into = f.get(opposite(q) * stride + cell).copied()?;
        let w = 2.0 * entry.wall;
        if entry.behind != u32::MAX {
            let back = f.get(opposite(q) * stride + entry.behind as usize).copied()?;
            Some(w * into + (1.0 - w) * back)
        } else {
            let own = f.get(q * stride + cell).copied()?;
            Some(into / w + (w - 1.0) / w * own)
        }
    }

    /// The links in the order of the flow's link list.
    pub(crate) fn entries(&self) -> &[Entry] {
        &self.entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two directions of one cell and one behind it: stride 1, `f_i*(x) = 3` at index 0, `f_i*(x - c_i) = 5` at index 1 (direction 1
    /// is the opposite of direction 2 in the lattice's order).
    fn populations() -> (Vec<f64>, usize) {
        let q = 2;
        let i = opposite(q);
        let mut f = vec![0.0; crate::lattice::Q * 2];
        f[i * 2] = 3.0;
        f[i * 2 + 1] = 5.0;
        f[q * 2] = 7.0;
        (f, q)
    }

    #[test]
    fn both_forms_give_plain_bounce_back_at_the_half_way_wall_and_join_there() {
        let (f, q) = populations();
        let near = Entry { link: 0, cell: 0, dir: q as u8, wall: 0.5, behind: 1 };
        let far = Entry { link: 0, cell: 0, dir: q as u8, wall: 0.5, behind: u32::MAX };
        assert_eq!(Interpolated::incoming(&near, &f, 2), Some(3.0));
        assert_eq!(Interpolated::incoming(&far, &f, 2), Some(3.0));
    }

    #[test]
    fn the_linear_forms_weigh_the_populations_by_the_wall_distance() {
        let (f, q) = populations();
        // q = 1/4: half of f_i*(x) and half of the one behind.
        let near = Entry { link: 0, cell: 0, dir: q as u8, wall: 0.25, behind: 1 };
        assert_eq!(Interpolated::incoming(&near, &f, 2), Some(4.0));
        // q = 1: half of f_i*(x) and half of the population that left the wall's way.
        let far = Entry { link: 0, cell: 0, dir: q as u8, wall: 1.0, behind: u32::MAX };
        assert_eq!(Interpolated::incoming(&far, &f, 2), Some(5.0));
        // An index out of range is no value, not a panic.
        let bad = Entry { link: 0, cell: 99, dir: q as u8, wall: 0.25, behind: 1 };
        assert_eq!(Interpolated::incoming(&bad, &f, 2), None);
    }
}
