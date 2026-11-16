# Turbulent flat plate at zero pressure gradient: skin friction against Kármán–Schoenherr (pending)

A turbulent boundary layer grows along a flat plate in a uniform stream with no pressure gradient.
The quantity is the skin friction coefficient C_f as a function of the momentum-thickness Reynolds
number Re_θ, compared with the Kármán–Schoenherr correlation over 4000 ≤ Re_θ ≤ 13000. It is the
acceptance check for a turbulence model on a wall-bounded boundary layer: the friction a model
returns must follow the correlation, not only the velocity profile.

## Reference

- NASA Langley Turbulence Modeling Resource, *2D Zero Pressure Gradient Flat Plate Validation Case*,
  <https://tmbwg.github.io/turbmodels/flatplate_val.html> (read 2026-11-07). Facts used, in our
  words: Reynolds number 5 × 10⁶ per unit length, Mach number 0.2 or less, the plate from x = 0 to
  x = 2, the top boundary at y = 1, the largest boundary-layer thickness about 0.03, Re_θ just over
  14 000 at the end of the plate, and a theory comparison of C_f against Re_θ over
  4000 < Re_θ < 13 000.
- The correlation, as that page states it:
  c_f = 1 / (17.08 (log₁₀ Re_θ)² + 25.11 log₁₀ Re_θ + 6.012).
  It goes back to T. von Kármán, *Mechanical similitude and turbulence*, NACA TM 611 (1931), and
  K. E. Schoenherr, *Resistance of flat surfaces moving through a fluid*, Trans. SNAME 40:279–313
  (1932). The five values in `expected.json` are the formula evaluated at Re_θ = 4000, 6000, 8000,
  10 000 and 13 000, to five significant digits.
- Not a gate: the page's law-of-the-wall comparison at Re_θ = 10 000 states no constants, so no
  profile values are cited here.

## Tolerance

5 % relative on each value. This is our bar, chosen for a model that works; the page does not
state an accuracy for the correlation.

## Status

Pending: no result of ours is recorded yet, so there is no fixture. Each Re_θ is read from a run of
the case with the local edge speed (the stream above the layer speeds up by about δ*/H in a
bounded domain), C_f = τ_w / (½ ρ U_e²), Re_θ = U_e θ / ν, θ = ∫ (u/U_e)(1 − u/U_e) dz, and
interpolated linearly between the two stations that bracket it. No grid, solver file or result from
the reference page was used; the model is our own, built in code.
