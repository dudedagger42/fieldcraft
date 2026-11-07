# Turbulent flat plate at zero pressure gradient, NASA TMR 2DZP (pending)

A turbulent boundary layer grows along a flat plate in a uniform stream with no pressure gradient.
The quantity is the skin friction coefficient C_f as a function of the momentum-thickness Reynolds
number Re_θ, compared with Kármán–Schoenherr's correlation over 4000 ≤ Re_θ ≤ 13000. It is the
acceptance case of slice 15a (one RANS transport model, Spalart–Allmaras) of the LES roadmap in
#785.

## Reference

- NASA Langley Turbulence Modeling Resource (TMR), *2D Zero Pressure Gradient Flat Plate
  Validation Case*, <https://tmbwg.github.io/turbmodels/flatplate_val.html> (read 2026-11-07).
  Facts used, in our words: Reynolds number 5 × 10⁶ per unit length, Mach 0.2 or less, the plate
  from x = 0 to x = 2, the top boundary at y = 1, the largest boundary-layer thickness about 0.03,
  Re_θ just over 14 000 at the end of the plate, and the theory comparison of C_f against Re_θ
  over 4000 < Re_θ < 13 000.
- The correlation, as TMR states it:
  c_f = 1 / (17.08 (log₁₀ Re_θ)² + 25.11 log₁₀ Re_θ + 6.012).
  It goes back to T. von Kármán, *Mechanical similitude and turbulence*, NACA TM 611 (1931) and
  K. E. Schoenherr, *Resistance of flat surfaces moving through a fluid*, Trans. SNAME 40:279–313
  (1932). The five values in `expected.json` are the formula evaluated at Re_θ = 4000, 6000, 8000,
  10 000 and 13 000 (5 significant digits).
- Coles's law of the wall with wake (TMR's u⁺(y⁺) comparison at Re_θ = 10 000) is noted for later;
  TMR does not state κ, C and Π on that page, so no profile values are cited here yet.
- Experimental companion for a capability release (not part of the gate): Wieghardt & Tillmann,
  *On the turbulent friction layer for rising pressure*, NACA TM 1314 (1951),
  <https://ntrs.nasa.gov/citations/19930093916>; values to be transcribed when that check is built.

No TMR grid, solver file or result was used. The model is our own, built in code.

## The lattice Boltzmann model

`crates/lbm/tests/sa_flat_plate.rs` builds it (lengths in units of the plate's reference length,
so Re per unit length = U / ν = 5 × 10⁶):

| Item | Setting |
|---|---|
| Domain | x from −x_up (0.33 by default) to 2, z from −Δ to H (0.25 by default, ≥ 7 δ_max), one cell periodic in y (2D) |
| Inlet, x_min | uniform velocity U; with SA, ν̃ = `inflow_ratio` ν from the model |
| Floor upstream, z_min for x < 0 | slip (the symmetry plane TMR places ahead of the plate) |
| Plate, 0 ≤ x ≤ 2 | strips of solid cells (cuboids from z = −2Δ to z = 0 covering the bottom cell row, one force group each) whose halfway bounce-back wall lies at z = 0; the slip floor upstream is the box face at z = −Δ, so the leading edge is a one-cell step |
| Top, z_max | slip; the stream above the layer speeds up by about δ*/H (≈ 1.4 % at x = 2 with H = 0.25), so C_f and Re_θ are formed with the local edge speed U_e(x) read at the top cell row |
| Outlet, x_max | pressure, convective outflow |
| Grid | uniform spacing Δ; the default puts the first cell centre at y⁺ ≈ 40–60 over the gated range (wall function), see the sizing note |

**Reading C_f(Re_θ).** For each strip, the time-mean x momentum its wall receives per step
(`Flow::forces`, the momentum exchange, which includes the wall model's reconstruction share)
divided by its wetted area is the **carried** wall stress τ_w; C_f = τ_w / (½ ρ U_e²) with U_e at
the strip's centre. Re_θ = U_e θ / ν with θ = ∫ (u/U_e)(1 − u/U_e) dz integrated from the wall to
the top through the column at the strip's centre. Each gated Re_θ is interpolated linearly between
the two strips that bracket it. Where a wall model supplies the wall stress, its requested stress is
reported beside the carried one.

**Convergence.** The run stops when every gated strip's C_f changes by less than 0.1 % over the
last 10 % of the steps (checked every sample), or at the step cap, which is reported as
unconverged.

**Measurement check.** Before the turbulent case is trusted, the same pipeline runs a laminar
plate and is compared with Blasius's C_f Re_θ = 0.664² ≈ 0.441 (test
`a_laminar_plate_reads_blasius_friction`).
