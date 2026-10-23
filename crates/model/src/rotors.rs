//! Rotors: solid bodies of a fluid domain that turn about an axis at a set speed, such as a
//! fan's or a blower's impeller, resolved blade by blade rather than lumped into a pressure–flow
//! curve ([`crate::Fan`]).
//!
//! The body turns rigidly about the axis through `origin` along `axis` (right-handed: a positive
//! `speed` turns counter-clockwise looking against `axis`), starting from where the geometry has
//! it. The flow solver moves the body through its fixed cells as it turns, the fluid at its
//! surface moving with it, and reports the torque the fluid exerts about the axis and the power
//! the rotor gives the fluid. The fan's operating point, flow and pressure rise, comes out of the
//! surroundings like any other flow.
//!
//! The lattice solver is weakly compressible: the blade tips must stay well below the speed of
//! sound (the solver refuses tip lattice speeds it can't hold); compressors and turbines faster
//! than about Mach 0.3 need a compressible solver.
//!
//! Units are the model's consistent system: `origin` a length, `speed` an angular velocity.

use serde::{Deserialize, Serialize};

use crate::{Error, Result, finite};

/// Most rotors in one domain.
pub const MAX_ROTORS: usize = 16;

/// A body that turns about an axis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rotor {
    /// The body of the geometry that turns (not also an obstacle of the domain).
    pub body: String,
    /// A point on the axis.
    pub origin: [f64; 3],
    /// The axis's direction (any length but zero).
    pub axis: [f64; 3],
    /// The angular speed about `axis` (right-handed).
    pub speed: f64,
}

impl Rotor {
    /// The axis as a unit vector.
    pub fn unit_axis(&self) -> [f64; 3] {
        let [x, y, z] = self.axis;
        let n = (x * x + y * y + z * z).sqrt();
        [x / n, y / n, z / n]
    }

    pub fn validate(&self) -> Result<()> {
        if self.body.trim().is_empty() {
            return Err(Error::Invalid("a rotor needs a body".into()));
        }
        for c in self.origin {
            finite("origin", c)?;
        }
        for c in self.axis {
            finite("axis", c)?;
        }
        let [x, y, z] = self.axis;
        if !(x * x + y * y + z * z > 0.0) {
            return Err(Error::Invalid(format!("rotor `{}`: the axis has no direction", self.body)));
        }
        finite("speed", self.speed)?;
        Ok(())
    }
}

/// Check the rotors of a domain: each valid, each body once and not an obstacle, within the cap.
pub(crate) fn validate_rotors(rotors: &[Rotor], obstacles: &[String]) -> Result<()> {
    if rotors.len() > MAX_ROTORS {
        return Err(Error::Limit(format!("a fluid domain holds at most {MAX_ROTORS} rotors")));
    }
    for (i, r) in rotors.iter().enumerate() {
        r.validate()?;
        if rotors.get(..i).is_some_and(|before| before.iter().any(|b| b.body == r.body)) {
            return Err(Error::Invalid(format!("body `{}` is two rotors", r.body)));
        }
        if obstacles.contains(&r.body) {
            return Err(Error::Invalid(format!("body `{}` is both an obstacle and a rotor: list it once, as a rotor", r.body)));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rotor() -> Rotor {
        Rotor { body: "Impeller".into(), origin: [0.0; 3], axis: [0.0, 0.0, 2.0], speed: 100.0 }
    }

    #[test]
    fn a_rotor_reads_back_and_its_axis_is_a_unit_vector() {
        let r = rotor();
        assert_eq!(r.unit_axis(), [0.0, 0.0, 1.0]);
        let json = serde_json::to_value(&r).unwrap();
        assert_eq!(serde_json::from_value::<Rotor>(json).unwrap(), r);
        assert!(validate_rotors(&[r], &[]).is_ok());
    }

    #[test]
    fn hostile_rotors_are_refused() {
        let bad = [
            Rotor { body: " ".into(), ..rotor() },
            Rotor { axis: [0.0; 3], ..rotor() },
            Rotor { origin: [f64::NAN, 0.0, 0.0], ..rotor() },
            Rotor { speed: f64::INFINITY, ..rotor() },
        ];
        for r in bad {
            assert!(validate_rotors(std::slice::from_ref(&r), &[]).is_err(), "{r:?}");
        }
        assert!(validate_rotors(&[rotor(), rotor()], &[]).is_err(), "a body once");
        assert!(validate_rotors(&[rotor()], &["Impeller".into()]).is_err(), "not also an obstacle");
        assert!(validate_rotors(&vec![rotor(); MAX_ROTORS + 1], &[]).is_err());
    }
}
