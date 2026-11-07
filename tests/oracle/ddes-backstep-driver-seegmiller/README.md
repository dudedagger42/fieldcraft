# Backward-facing step, Driver & Seegmiller (1985), zero upper-wall angle (pending)

A turbulent boundary layer leaves the edge of a step of height H, separates, and reattaches on
the floor downstream. The gated quantity is the reattachment length x_r/H, measured from the
step to the point where the time-mean floor shear changes sign. It is the acceptance case of
slice 15b (DDES, shielded hybrid RANS/LES) of the LES roadmap in #785.

## Reference

- D. M. Driver and H. L. Seegmiller, *Features of a reattaching turbulent shear layer in
  divergent channel flow*, AIAA Journal 23(2):163–171 (1985), doi:10.2514/3.8890. The case is the
  one with the upper wall parallel to the floor (0°).
- NASA Langley Turbulence Modeling Resource, *2D Backward Facing Step* validation case,
  <https://tmbwg.github.io/turbmodels/backstep_val.html> (read 2026-11-07), and its experimental
  data files `Backstep_validation/cf.exp.dat` and `cp.expnew.dat`. Facts used, in our words:
  Re_H ≈ 36 000 (TMR records a correction from 50 000, from the first author), Mach 0.128 upstream,
  momentum-thickness Reynolds number 5000 and boundary-layer thickness about 1.5 H upstream of the
  step, reference conditions (U_ref, the centre-channel speed; C_f and C_p) taken near x/H = −4,
  reattachment at x/H = 6.26 ± 0.10, profiles at x/H = −4, 1, 4, 6 and 10.
- `experiment.json` holds the floor C_f (with its uncertainty) and the floor and roof C_p, numbers
  only, from those files. Reading the C_f column: negative from x/H ≈ 1.8 to 5.9, positive from
  7.1, which brackets 6.26.

No TMR grid, solver file or result was used.

**Not yet read from the paper** (it is not open access): the channel heights upstream and
downstream of the step and the tunnel's span. The test takes them as parameters; its defaults (an
inlet channel 8 H high above the step, so 9 H downstream, and a span ≥ 2 H periodic) are the
commonly quoted expansion ratio of 9/8 and must be checked against the paper's §II before the case
becomes active.

## The lattice Boltzmann model

`crates/lbm/tests/ddes_backstep.rs` builds it (lengths in units of H, Re_H = U_ref H / ν = 36 000):

| Item | Setting |
|---|---|
| Domain | x from −L_up to L_down (30 H by default), z from −Δ to 9 H, span periodic: 1 cell for 2D RANS, S ≥ 2 H for the hybrid run (and 2 S for the span-sensitivity pair) |
| Step | a solid block x < 0, z < H; the step's top and the roof are no-slip walls along which the incoming boundary layers grow |
| Inlet, x_min | uniform velocity at −L_up; L_up is chosen so that Re_θ ≈ 5000 and δ ≈ 1.5 H at x/H = −4 (reported; flat-plate growth puts it near 70 H) |
| Floor, x > 0 | strips of solid cells H/8 long (one force group each, halfway wall at z = 0), so the local floor shear is the momentum exchange per strip |
| Roof | no-slip box face |
| Outlet, x_max | pressure, convective outflow |

**Reading x_r.** The time-mean carried floor shear per strip (its momentum exchange per area,
averaged through `fieldcraft_lbm::statistics::Blocks`); x_r is its one change of sign from
negative to non-negative, by linear interpolation between strip centres, divided by H. No such
change, or more than one (a second bubble, or a mean not yet converged), is refused and
reported rather than picked from. C_f and C_p are formed with ρ and U_ref at x/H = −4 (the centre-channel
speed there) for comparison with `experiment.json`.

**Statistics.** The hybrid run is unsteady: the reattachment's block means must pass the shared
acceptance of the LES studies (`fieldcraft_lbm::statistics`, #796); otherwise the result is
reported inconclusive.
