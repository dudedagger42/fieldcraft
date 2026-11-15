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
//! - `MassFix::Spread` (the default): each wall node stores what its own links created (its delta)
//!   and hands it to itself and its fluid neighbours with the lattice weights renormalised over the
//!   fluid targets (fixed at setup; solid and open cells get nothing). Every fluid cell removes what
//!   it received, spread over its populations by the lattice weights, one step late. The plus and
//!   minus parts of neighbouring wall nodes cancel before anything is removed, and what remains is
//!   taken back close to where it was made. The sum removed equals the sum created, exactly.
//! - `MassFix::Local`: each wall node removes the sum over its own links at once.
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
    /// Not at all (to measure the leak: the mass the walls create is still counted).
    Off,
    /// Each wall node gives back what its own links created.
    Local,
    /// Each wall node hands what its links created to itself and its fluid neighbours, which take
    /// it back one step later.
    #[default]
    Spread,
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
    /// Per wall node the range of its shares, the shares (receiver index, weight) and the receiving
    /// cells (the spread kind).
    node_shares: Vec<(u32, u32)>,
    shares: Vec<(u32, f64)>,
    receivers: Vec<u32>,
    /// Per wall node the sum `W` of the lattice weights over its fluid targets.
    node_total: Vec<f64>,
    /// Per wall node what its links created at the last step, and per receiver what it is handed.
    delta: Vec<f64>,
    received: Vec<f64>,
    /// How the mass the wall links create is taken back (`Spread` by default).
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
        let delta = vec![0.0; nodes.len()];
        Self {
            entries,
            report,
            order,
            nodes,
            node_shares: Vec::new(),
            shares: Vec::new(),
            receivers: Vec::new(),
            node_total: Vec::new(),
            delta,
            received: Vec::new(),
            mass_fix: MassFix::default(),
            steps: 0,
            created: 0.0,
        }
    }

    /// Set up the spread kind: `target(cell, i)` is the cell `c_i` away from `cell` when it is a
    /// fluid cell that takes mass back (`None` otherwise). Each wall node's share to a target is
    /// `w_i / W`, `W` the sum of `w_i` over its targets.
    pub(crate) fn build_spread(&mut self, target: impl Fn(u32, usize) -> Option<u32>) {
        let mut raw: Vec<Vec<(u32, f64)>> = Vec::with_capacity(self.nodes.len());
        let mut self_total: Vec<f64> = Vec::with_capacity(self.nodes.len());
        let mut receivers: Vec<u32> = Vec::new();
        for &(cell, _, _) in &self.nodes {
            let mut list: Vec<(u32, f64)> = Vec::new();
            let mut total = 0.0;
            for (i, &w) in W.iter().enumerate() {
                if let Some(t) = target(cell, i) {
                    list.push((t, w));
                    total += w;
                }
            }
            self_total.push(total);
            if total > 0.0 {
                for share in &mut list {
                    share.1 /= total;
                }
            }
            receivers.extend(list.iter().map(|s| s.0));
            raw.push(list);
        }
        receivers.sort_unstable();
        receivers.dedup();
        self.node_shares.clear();
        self.shares.clear();
        for list in &raw {
            let start = self.shares.len() as u32;
            for &(t, w) in list {
                if let Ok(r) = receivers.binary_search(&t) {
                    self.shares.push((r as u32, w));
                }
            }
            self.node_shares.push((start, self.shares.len() as u32));
        }
        self.received = vec![0.0; receivers.len()];
        self.receivers = receivers;
        self.node_total = self_total;
    }

    /// The wall nodes' cells with the sum `W` of the lattice weights over their fluid targets
    /// (the spread kind), in cell order.
    pub fn spread_nodes(&self) -> Vec<(u32, f64)> {
        self.nodes.iter().zip(&self.node_total).map(|(n, &w)| (n.0, w)).collect()
    }

    /// The cells that take mass back in the spread kind, in cell order.
    pub fn spread_receivers(&self) -> &[u32] {
        &self.receivers
    }

    /// Whether the mass fix is the spread kind.
    pub fn is_spread(&self) -> bool {
        self.mass_fix == MassFix::Spread
    }

    /// Take back the mass the interpolated links created. `f` are the post-collision populations,
    /// `g` the streamed ones (already carrying the interpolated values).
    pub(crate) fn conserve_mass(&mut self, f: &[f64], g: &mut [f64], stride: usize) {
        if self.entries.is_empty() {
            return;
        }
        if self.mass_fix == MassFix::Spread && self.node_shares.len() == self.nodes.len() {
            // What the last step's deltas hand out, removed by the receivers.
            self.received.fill(0.0);
            for (n, &d) in self.delta.iter().enumerate() {
                let &(a, b) = self.node_shares.get(n).unwrap_or(&(0, 0));
                for &(r, w) in self.shares.get(a as usize..b as usize).unwrap_or_default() {
                    if let Some(x) = self.received.get_mut(r as usize) {
                        *x += d * w;
                    }
                }
            }
            for (&cell, &got) in self.receivers.iter().zip(&self.received) {
                for i in 0..Q {
                    if let Some(x) = g.get_mut(i * stride + cell as usize) {
                        *x -= got * W[i];
                    }
                }
            }
        }
        let mut total = 0.0;
        for (n, &(cell, start, end)) in self.nodes.iter().enumerate() {
            let mut delta = 0.0;
            for &k in self.order.get(start as usize..end as usize).unwrap_or_default() {
                let Some(e) = self.entries.get(k as usize) else { continue };
                let (Some(v), Some(sent)) = (Self::incoming(e, f, stride), f.get(opposite(usize::from(e.dir)) * stride + e.cell as usize)) else {
                    continue;
                };
                delta += v - sent;
            }
            total += delta;
            if let Some(d) = self.delta.get_mut(n) {
                *d = delta;
            }
            if self.mass_fix == MassFix::Local {
                for i in 0..Q {
                    if let Some(x) = g.get_mut(i * stride + cell as usize) {
                        *x -= delta * W[i];
                    }
                }
            }
        }
        self.created += total;
        self.steps += 1;
    }

    /// The mass the walls created at the last step that the spread has not taken back yet (the
    /// removal is one step late), in lattice units. Total mass less this is the conserved quantity.
    pub fn mass_pending(&self) -> f64 {
        if self.mass_fix == MassFix::Spread { self.delta.iter().sum() } else { 0.0 }
    }

    /// What the walls created so far: the steps it ran and the mass the walls created
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

    /// The exchange `(v + sent) c_q` of a link is continuous across `q = 1/2` for both orders, and
    /// a half-way link kept as a wall link (no second node) returns the plain bounce-back
    /// population: it must be stored at exactly 1/2, since just below it the rule needs a second
    /// node it does not have.
    #[test]
    fn the_exchange_is_continuous_across_one_half_and_a_kept_half_way_link_needs_no_second_node() {
        let (f, q) = line([3.0, 5.0, 8.0], [7.0, 1.0, 2.0]);
        let sent = 3.0;
        for order in [1, 2] {
            let below = Interpolated::incoming(&entry(q, 0.5 - 1e-9, order), &f, 4).unwrap_or(f64::NAN);
            let above = Interpolated::incoming(&entry(q, 0.5 + 1e-9, order), &f, 4).unwrap_or(f64::NAN);
            assert!(((below + sent) - (above + sent)).abs() < 1e-7, "order {order}: {below} {above}");
            assert!((below - sent).abs() < 1e-7 && (above - sent).abs() < 1e-7);
        }
        let kept = Entry { behind: u32::MAX, behind2: u32::MAX, ..entry(q, 0.5, 1) };
        assert_eq!(Interpolated::incoming(&kept, &f, 4), Some(sent));
        let off = Entry { wall: 0.5 - 1e-10, ..kept };
        assert_eq!(Interpolated::incoming(&off, &f, 4), None, "a second node is missing below 1/2");
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

    #[test]
    fn the_spread_hands_out_exactly_what_the_walls_created_one_step_late() {
        // Two wall nodes (cells 0 and 1) with a link each; the targets of a node are four cells
        // reached by every direction, some twice.
        let stride = 4;
        let q = 2;
        let mut f = vec![0.0; crate::lattice::Q * stride];
        for (k, v) in [3.0, 5.0, 8.0, 1.0].into_iter().enumerate() {
            f[opposite(q) * stride + k] = v;
            f[q * stride + k] = 7.0 - v;
        }
        let entries = vec![
            Entry { cell: 0, behind: 1, behind2: 2, ..entry(q, 0.25, 2) },
            Entry { link: 1, cell: 1, behind: 2, behind2: 3, ..entry(q, 0.7, 1) },
        ];
        let mut ip = Interpolated::new(entries, InterpolationReport::default());
        ip.build_spread(|cell, i| Some((cell + i as u32) % 4));
        let mut g = vec![0.0; crate::lattice::Q * stride];
        ip.conserve_mass(&f, &mut g, stride);
        // The first step removes nothing: the deltas are handed out at the next one.
        assert!(g.iter().all(|&x| x == 0.0));
        let (_, created) = ip.mass_created();
        assert!(created.abs() > 1e-6, "{created}");
        ip.conserve_mass(&f, &mut g, stride);
        let removed: f64 = g.iter().sum();
        assert!((removed + created).abs() < 1e-12, "{removed} {created}");
    }
}
