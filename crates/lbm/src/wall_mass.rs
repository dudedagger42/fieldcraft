//! The mass an interpolated wall creates, and how a flow takes it back.
//!
//! On a wall that does not lie half way along its links the interpolated rule returns another
//! amount than left along the link, so the walls make or lose mass every step (a wall that lies
//! half way returns what left). The correction here uses the walls alone, with the weights of the
//! lattice (they sum to one and carry no momentum, so none is added):
//!
//! - each wall node (a fluid cell with interpolated links) stores what its own links created at the
//!   step, its delta: the sum over its links of the population returned less the one that left;
//! - it hands the delta to itself and to its fluid neighbours with the lattice weights renormalised
//!   over those fluid targets (fixed when the correction is switched on; solid cells and open
//!   inlet and outlet cells get nothing, so the open boundaries are left to do their own work);
//! - every cell that received a share removes it, spread over its populations by the lattice
//!   weights, one step late: the next step's streaming takes it from the populations it just made.
//!
//! The shares of neighbouring wall nodes of opposite sign cancel before anything is removed, so
//! what is left is taken back close to where it was made, and what is removed in total equals what
//! was created, exactly (up to the one step in flight, [`WallMass::pending`]): total mass less the
//! pending amount is the conserved quantity. At `q = 1/2` the created term is exactly zero. This is
//! a design choice of FieldCraft's, not a published scheme; it follows the usual remedy for the
//! mass leak of interpolated bounce-back (the correction is local to the wall and carries no
//! momentum), with the weights of the lattice.
//!
//! A link on a moving wall adds Ladd's term (A. J. C. Ladd, *Numerical simulations of particulate
//! suspensions via a discretized Boltzmann equation. Part 1. Theoretical foundation*, J. Fluid Mech.
//! 271:285–309, 1994) to the population it returns. Over a closed surface of
//! a rigid body that term does not sum to zero once it is weighted by the link's fraction, so a
//! moving body in a periodic box leaks mass too (4.7e-3 of the mass over a sphere run at a wall speed of
//! 0.05); the created mass here includes it, and the weights carry no momentum, so the wall's
//! momentum exchange is left alone.
//!
//! # Known limits
//!
//! A flow with the correction on cannot be checkpointed (the mass in flight is state a checkpoint does
//! not carry yet).
//!
//! Stability. Measured on a one-dimensional laminar channel (ten cells across, impulsive start, no
//! turbulence model, 80 000 steps, a wall link cut at q = 0.75 of a cell): with the regularised and the
//! hybrid regularised collisions the correction makes the flow unstable at low viscosity, and not
//! cleanly. Hybrid regularised at σ 0.95: the linear rule blows up at τ − ½ of about 1.3e-2, the
//! quadratic rule at about 4e-2. At σ 1.0 and 0.99 the linear rule blows up below about 1.5e-2 and the
//! quadratic rule below 4e-2; between there and 5e-2 the run finishes but the answer is wrong, with no
//! error raised (linear rule, peak speed 0.0648 at σ 1.0 and 0.0757 at σ 0.99 at 5e-2, and 0.1367 and
//! 0.1342 at 3e-2, against 0.0485 with the correction off, up to 2.8 times). A wall link cut at
//! q = 0.25 stays stable in every arm, and with the correction off every arm is stable down to
//! τ − ½ = 5e-3. The MRT collision is stable at all ten viscosities with the correction on and reaches
//! the same steady state as with it off (the rows agree to every printed digit at 80 000 steps); the
//! switch does act under MRT (it created 3.5e-6 to 8e-6 of mass over 2 000 steps), so what is claimed for
//! MRT is "stable, same steady state". It has not been measured with the TRT collision. Handing back
//! only part of the pending mass each step (a factor of 0.5, 0.25 or 0.1) does not remove the distortion:
//! with 0.1 the linear rule is stable but the flow rate stays 1 to 6 % below the correction-off value,
//! the quadratic rule up to 22 % below it, and with the quadratic rule 7, 5 and 1 of the ten viscosities
//! blow up at 0.5, 0.25 and 0.1 (σ 1.0). On a sphere in a periodic box (linear rule, 80 000 steps)
//! the same collisions with the correction on ran to the end at τ − ½ = 2e-2 and 5e-3. Switching the
//! correction on with the regularised or the hybrid regularised collision below τ − ½ = 5e-2 is
//! therefore refused ([`crate::flow::WALL_MASS_MIN_RELAXATION_EXCESS`]).
//!
//! What the correction is for: with it off, the mass leaked by a sphere grows through a run (up to
//! 2.5e-2 of the mass at τ − ½ = 5e-3 for σ 0.95, 4.0e-2 for σ 1.0, 80 000 steps) where a straight
//! channel leaks nothing visible (a flat 9e-7 at most with MRT, rounding level with the hybrid
//! collision).
//!
//! Rounding. With the hybrid regularised collision the total mass drifts by about 1.1e-12 of itself
//! per 20 000 steps whatever the walls, the correction, τ or σ (the MRT collision stays at 1e-15); it
//! is a constant bias of about 5.5e-17 per step in that collision, not caused by the walls, and
//! negligible next to the wall leak. Its source has not been traced.

use crate::flow::{Interpolation, Link, Rule, interpolated_value};
use crate::lattice::{Q, W, opposite};

/// One wall node: its cell and the range of its links in [`WallMass::link_ids`].
#[derive(Debug, Clone, Copy)]
struct Node {
    cell: u32,
    start: u32,
    end: u32,
}

/// The mass correction of a flow's interpolated walls.
#[derive(Debug, Clone, Default)]
pub struct WallMass {
    nodes: Vec<Node>,
    /// Indices into the flow's link list, by wall node.
    link_ids: Vec<u32>,
    /// Per wall node the range of its shares in `shares`, and the shares (receiver, weight).
    node_shares: Vec<(u32, u32)>,
    shares: Vec<(u32, f64)>,
    /// The cells that take mass back, in cell order.
    receivers: Vec<u32>,
    /// Per wall node the sum of the lattice weights over its fluid targets.
    node_total: Vec<f64>,
    /// What each wall node created at the last step, and what each receiver is handed.
    delta: Vec<f64>,
    received: Vec<f64>,
    /// The steps run and the mass the walls created over them.
    steps: u64,
    created: f64,
}

impl WallMass {
    /// The correction for `links` (in cell order) with the spread over `target(cell, i)`: the cell
    /// `c_i` away from `cell` when it is a fluid cell that takes mass back, `None` otherwise.
    pub(crate) fn new(links: &[Link], target: impl Fn(u32, usize) -> Option<u32>) -> Self {
        let mut nodes: Vec<Node> = Vec::new();
        let mut link_ids: Vec<u32> = Vec::new();
        for (index, link) in links.iter().enumerate() {
            if !matches!(link.rule, Rule::Interpolated { .. }) {
                continue;
            }
            link_ids.push(index as u32);
            let at = link_ids.len() as u32;
            match nodes.last_mut() {
                Some(n) if n.cell == link.cell => n.end = at,
                _ => nodes.push(Node { cell: link.cell, start: at - 1, end: at }),
            }
        }
        let mut raw: Vec<Vec<(u32, f64)>> = Vec::with_capacity(nodes.len());
        let mut receivers: Vec<u32> = Vec::new();
        let mut node_total: Vec<f64> = Vec::with_capacity(nodes.len());
        for node in &nodes {
            let mut list: Vec<(u32, f64)> = Vec::new();
            let mut total = 0.0;
            for (i, &w) in W.iter().enumerate() {
                if let Some(t) = target(node.cell, i) {
                    list.push((t, w));
                    total += w;
                }
            }
            node_total.push(total);
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
        let mut node_shares = Vec::with_capacity(raw.len());
        let mut shares: Vec<(u32, f64)> = Vec::new();
        for list in &raw {
            let start = shares.len() as u32;
            for &(t, w) in list {
                if let Ok(r) = receivers.binary_search(&t) {
                    shares.push((r as u32, w));
                }
            }
            node_shares.push((start, shares.len() as u32));
        }
        Self {
            delta: vec![0.0; nodes.len()],
            received: vec![0.0; receivers.len()],
            nodes,
            link_ids,
            node_shares,
            node_total,
            shares,
            receivers,
            steps: 0,
            created: 0.0,
        }
    }

    /// Take back the mass the interpolated links created. `f` are the post-collision populations
    /// (stride `s`), `g` the streamed ones, already carrying the interpolated values.
    pub(crate) fn conserve(&mut self, f: &[f64], g: &mut [f64], s: usize, links: &[Link], interpolations: &[Interpolation]) {
        // What the last step's deltas handed out, removed by the receivers.
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
            for (i, w) in W.iter().enumerate().take(Q) {
                if let Some(x) = g.get_mut(i * s + cell as usize) {
                    *x -= got * w;
                }
            }
        }
        // What this step's links created: the population they return less the one that left.
        let mut total = 0.0;
        for (n, node) in self.nodes.iter().enumerate() {
            let mut delta = 0.0;
            for &id in self.link_ids.get(node.start as usize..node.end as usize).unwrap_or_default() {
                let Some(link) = links.get(id as usize) else { continue };
                let Rule::Interpolated { slot, .. } = link.rule else { continue };
                let (Some(i), q) = (interpolations.get(slot as usize), usize::from(link.q)) else { continue };
                let (Some(v), Some(sent)) = (interpolated_value(f, s, link.cell as usize, q, *i), f.get(opposite(q) * s + link.cell as usize)) else {
                    continue;
                };
                delta += v - sent;
            }
            total += delta;
            if let Some(d) = self.delta.get_mut(n) {
                *d = delta;
            }
        }
        self.created += total;
        self.steps += 1;
    }

    /// The mass the walls created at the last step that has not been taken back yet (the removal is
    /// one step late), in lattice units. Total mass less this is the conserved quantity.
    pub fn pending(&self) -> f64 {
        self.delta.iter().sum()
    }

    /// The steps run and the mass the walls created over them (lattice units, density 1 per cell).
    pub fn created(&self) -> (u64, f64) {
        (self.steps, self.created)
    }

    /// The wall nodes that hand mass out and the cells that take it back.
    pub fn counts(&self) -> (usize, usize) {
        (self.nodes.len(), self.receivers.len())
    }

    /// The wall nodes' cells, and per node the sum of the lattice weights over its fluid targets
    /// (for a device that runs the same correction).
    pub fn nodes(&self) -> Vec<(u32, f64)> {
        self.nodes.iter().zip(&self.node_total).map(|(n, &w)| (n.cell, w)).collect()
    }

    /// The cells that take mass back, in cell order.
    pub fn receivers(&self) -> &[u32] {
        &self.receivers
    }
}
