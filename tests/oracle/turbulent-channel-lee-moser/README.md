# Turbulent channel flow against direct numerical simulation (Lee and Moser)

Fully developed turbulent flow between two parallel walls, driven by a uniform body force, at friction
Reynolds numbers Re_τ = 182, 543 and 1000. The measured quantity is the skin friction coefficient on the bulk
velocity, C_f = 2 (u_τ / U_b)², where u_τ is the friction velocity and U_b the mean velocity over the height.
The reference is a direct numerical simulation, so there is no model in the answer.

**Source.** M. Lee and R. D. Moser, *Direct numerical simulation of turbulent channel flow up to Re_τ ≈ 5200*,
J. Fluid Mech. 774:395–415 (2015), doi:10.1017/jfm.2015.268. Their published mean-profile files give, per
Reynolds number, the half-height δ = 1, the viscosity ν, the friction velocity u_τ and the bulk velocity U_b
in the header. This case uses those three scalars and nothing else of the data (no profile, no table): u_τ =
0.0637309 (Re_τ 182.088), 0.0543496 (Re_τ 543.496) and 0.0500256 (Re_τ 1000.512) at U_b = 1. The files carry no licence
statement; only these three published numbers are used and nothing is copied. (Licence to be confirmed before
the first upstream push.)

**Setup.** Half-height δ = 1, u_τ = 1 and ν = 1 / Re_τ in physical units; body force a = u_τ² / δ. The grid has
The channel is `cells` cells per δ high (the table), periodic along the flow and
across, a wall at each side on the half-way wall with the Spalart–Allmaras wall law (no turbulence closure
tuned to this case), regularised collision with σ = 0.95. Time step Δt = 0.1 Δx / U_b⁺ with U_b⁺ the DNS bulk
velocity, so the lattice bulk speed starts at 0.1; ν_lat = ν Δt / Δx², τ = 3 ν_lat + ½; the lattice body force is
Δt² / Δx.

| Case | ν_lat | τ | lattice force | steps to settle | y⁺ of the first cell | C_f of the DNS | C_f here | error |
|---|---|---|---|---|---|---|---|---|
| Re_τ 182.088, 10 cells per δ | 3.50001e-04 | 0.50105 | 4.0616e-06 | 41100 | 27.3 | 8.123255e-03 | 8.039611e-03 | -1.03 % |
| Re_τ 182.088, 20 cells per δ | 7.00001e-04 | 0.50210 | 2.0308e-06 | 106400 | 13.7 | 8.123255e-03 | 7.723343e-03 | -4.92 % |
| Re_τ 543.496, 20 cells per δ | 2.00000e-04 | 0.50060 | 1.4769e-06 | 96900 | 40.8 | 5.907758e-03 | 5.826601e-03 | -1.37 % |
| Re_τ 1000, 50 cells per δ | 2.50128e-04 | 0.50075 | 5.0051e-07 | 67200 | 30.0 | 5.005121e-03 | 4.946613e-03 | -1.17 % |
| Re_τ 1000, 80 cells per δ | 4.00205e-04 | 0.50120 | 3.1282e-07 | 108200 | 18.8 | 5.005121e-03 | 4.919915e-03 | -1.70 % |

The Re_τ 1000 rows run at Re_τ = 1000 exactly; the DNS values are those of Re_τ = 1000.512 (0.05 % apart,
inside the reference's own spread).

**Settling.** A run is read only when it is settled: at least two flow-through times, the bulk velocity's
drift below 1e-3 per flow-through, and the applied drag within 1e-4 of the body force. Every row above settled
with the requested force 0.99995 to 0.99999 of the carried drag.

**Result.** Every row is below the DNS, and every row misses the 1 % bar: by 0.03 point at Re_τ 182 with 10
cells, by 3.9 points at Re_τ 182 with 20 cells (the first cell is then at y⁺ 13.7, in the buffer layer, where
the wall law is least reliable), and by 0.4 to 0.7 point elsewhere. The case records this as a known miss;
the bar is not widened. Each quantity carries a second, tight band (`regression`, 0.05 % of our own value)
so that any later change that moves the result has to explain itself. Each row takes 40,000 to 110,000
steps, so the case is `nightly`, and pending the feature it exercises.

**Not a gate on the wall law's constants.** The law and its constants were fixed on the Re_τ 182 and 1000
channels before this case was written, and the Re_τ 543 row was not used to choose them.
