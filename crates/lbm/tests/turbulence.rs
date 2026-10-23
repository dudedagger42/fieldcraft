//! Smagorinsky's eddy viscosity (`turbulence: {"smagorinsky": C}`) against the closed form: in
//! plane Couette flow the strain rate is the same everywhere, so the eddy viscosity is the uniform
//! `(C Δx)² U / h`, the profile stays linear and the wall shear is `ρ (ν + (C Δx)² U / h) U / h`.
//!
//! Each test prints its numbers (`cargo test -p fieldcraft-lbm --test turbulence -- --nocapture`).

mod common;

use common::*;
use fieldcraft_lbm::{Error, Flow, Limits, Setup, run};
use fieldcraft_model::{Collision, FaceCondition, FluidThermal, Procedure, ThermalCondition, Turbulence};

const PERIODIC: FaceCondition = FaceCondition::Periodic {};

/// [`collide`] with Smagorinsky's constant `c`.
fn turbulent(end_time: f64, time_step: f64, collision: fieldcraft_model::Collision, c: f64) -> Procedure {
    let mut p = collide(end_time, time_step, None, collision);
    if let Procedure::Fluid { turbulence, .. } = &mut p {
        *turbulence = Some(Turbulence::Smagorinsky { constant: c, prandtl: 0.9 });
    }
    p
}

#[test]
fn couette_flow_takes_the_smagorinsky_viscosity_of_its_shear() {
    // A thin fluid and a fast wall, so the eddy viscosity is a few per cent of the fluid's.
    let (h, nu, rho, u_wall, n, c) = (1.0, 1e-3, 2.0, 0.1, 16, 0.4);
    let dx = h / n as f64;
    for collision in COLLISIONS {
        let mut d = domain([0.0; 3], [dx, h, dx], dx);
        for f in [&mut d.faces.x_min, &mut d.faces.x_max, &mut d.faces.z_min, &mut d.faces.z_max] {
            *f = PERIODIC;
        }
        d.faces.y_max = FaceCondition::MovingWall { velocity: [u_wall, 0.0, 0.0] };
        let m = model(rho, nu, d, turbulent(5.0 * h * h / nu, dx, collision, c));
        let (setup, mut flow) = prepare(&m, &[]);
        assert_eq!(setup.smagorinsky, c);
        run_all(&setup, &mut flow);
        let mut err: f64 = 0.0;
        for j in 0..n {
            let y = (j as f64 + 0.5) * dx;
            let (_, u) = flow.moments(j).unwrap();
            err = err.max((u[0] * setup.velocity_scale() - u_wall * y / h).abs() / u_wall);
        }
        let gamma = u_wall / h;
        let nu_t = (c * dx) * (c * dx) * gamma;
        let tau = rho * (nu + nu_t) * gamma * dx * dx;
        let laminar = rho * nu * gamma * dx * dx;
        let g = |name: &str| flow.groups().iter().position(|g| g.name == name).unwrap();
        let fs = setup.force_scale();
        let (bottom, top) = (flow.forces()[g("y_min")][0] * fs, flow.forces()[g("y_max")][0] * fs);
        println!(
            "{} Couette with C = {c}, {n} cells: max error {err:.2e}; wall shear {bottom:.6e} and {top:.6e}, expected ±{tau:.6e} (laminar {laminar:.6e}, ν_t/ν = {:.4})",
            collision.name(),
            nu_t / nu
        );
        // Lattice speed 0.1 at the wall: the profile and stress carry its small compressibility
        // error, well inside the 6% the eddy viscosity adds.
        assert!(err < 2e-3, "{}: {err}", collision.name());
        assert!((bottom - tau).abs() < 2e-3 * tau, "{}: {bottom} vs {tau}", collision.name());
        assert!((top + tau).abs() < 2e-3 * tau, "{}: {top} vs {tau}", collision.name());
        assert!(nu_t / nu > 0.05 && bottom > laminar * (1.0 + 0.5 * nu_t / nu), "the eddy viscosity shows");
        // The reply's largest eddy viscosity ratio: the uniform closed form. It is a maximum
        // over the cells, and the two-relaxation-time cells next to the walls read a little more
        // flux than the shear's (their wall shear above is right).
        let ratio = flow.eddy_viscosity_ratio();
        let within = if collision == Collision::Regularized { 2e-3 } else { 0.2 };
        assert!((ratio - nu_t / nu).abs() < within * nu_t / nu, "{}: ν_t/ν {ratio} vs {}", collision.name(), nu_t / nu);
    }
}

#[test]
fn a_smagorinsky_constant_out_of_range_is_refused() {
    for c in [0.0, -0.1, 0.6, f64::NAN, f64::INFINITY] {
        let m = model(1.0, 0.1, domain([0.0; 3], [4.0, 2.0, 2.0], 1.0), turbulent(10.0, 1.0, fieldcraft_model::Collision::Trt, c));
        let r = Setup::new(&m, None, &[], &Limits::default());
        assert!(matches!(r, Err(Error::Invalid(ref e)) if e.contains("Smagorinsky")), "C = {c}: {r:?}");
    }
    // Without a turbulence model the step is unchanged: no eddy viscosity.
    let m = model(1.0, 0.1, domain([0.0; 3], [4.0, 2.0, 2.0], 1.0), step(10.0, 1.0, None));
    let (setup, flow) = prepare(&m, &[]);
    assert_eq!((setup.smagorinsky, Flow::smagorinsky(&flow)), (0.0, 0.0));
}

/// Heated Couette flow: a flux q into the still wall and the moving wall held at a temperature.
/// The shear and so the eddy viscosity are the same everywhere, so the turbulent diffusivity
/// α_t = ν_t / Pr_t is too, and the steady drop over the gap is q h / (ρ c (α + α_t)).
#[test]
fn heated_couette_flow_conducts_with_the_turbulent_diffusivity() {
    let (h, nu, alpha, rho, u_wall, n, c, prandtl, q) = (1.0, 1e-3, 1e-3, 2.0, 0.1, 16, 0.4, 0.9, 0.3);
    let dx = h / n as f64;
    for collision in COLLISIONS {
        let mut d = domain([0.0; 3], [dx, h, dx], dx);
        for f in [&mut d.faces.x_min, &mut d.faces.x_max, &mut d.faces.z_min, &mut d.faces.z_max] {
            *f = PERIODIC;
        }
        d.faces.y_max = FaceCondition::MovingWall { velocity: [u_wall, 0.0, 0.0] };
        let mut t = FluidThermal::default();
        t.faces.y_min = ThermalCondition::Flux { value: q };
        t.faces.y_max = ThermalCondition::Temperature { value: kelvin(1.0) };
        d.thermal = Some(t);
        let mut p = collide(6.0 * h * h / alpha, dx, None, collision);
        if let Procedure::Fluid { turbulence, .. } = &mut p {
            *turbulence = Some(Turbulence::Smagorinsky { constant: c, prandtl });
        }
        let m = hot_model(rho, nu, alpha, d, p);
        let setup = Setup::new(&m, None, &[], &Limits::default()).unwrap();
        let mut flow = Flow::new(&setup, &[]).unwrap();
        let r = run(&setup, &mut flow, &Limits::default()).unwrap();
        let alpha_t = (c * dx) * (c * dx) * (u_wall / h) / prandtl;
        let k = rho * (alpha + alpha_t);
        let t0 = f64::from(r.frames.last().unwrap().temperature[0]) + 0.5 * dx * q / k;
        let drop = t0 - 1.0;
        let expected = q * h / k;
        let laminar = q * h / (rho * alpha);
        println!(
            "{} heated Couette with C = {c}, Pr_t = {prandtl}: drop {drop:.6} vs q h / (ρ (α + α_t)) = {expected:.6} (laminar {laminar:.6}, α_t/α = {:.4})",
            collision.name(),
            alpha_t / alpha
        );
        assert!(alpha_t / alpha > 0.05, "the turbulent diffusivity shows");
        assert!((drop - expected).abs() < 2e-3 * expected, "{}: {drop} vs {expected}", collision.name());
        assert_eq!(r.stats.eddy_viscosity_ratio.map(|x| x > 0.0), Some(true));
    }
}

/// A lid-driven cavity, 48² cells (quasi-2D), lid at lattice speed 0.1 and Reynolds number `re`,
/// run `steps_run` steps: the step it went unstable at, if it did; an error when the step is
/// refused before it runs.
fn cavity_failure(re: f64, steps_run: u64, c: Option<f64>) -> Result<Option<u64>, Error> {
    let l = 48.0;
    let mut d = domain([0.0; 3], [l, l, 1.0], 1.0);
    d.faces.y_max = FaceCondition::MovingWall { velocity: [0.1, 0.0, 0.0] };
    d.faces.z_min = PERIODIC;
    d.faces.z_max = PERIODIC;
    let mut p = collide(steps_run as f64, 1.0, None, Collision::Regularized);
    if let (Procedure::Fluid { turbulence, .. }, Some(c)) = (&mut p, c) {
        *turbulence = Some(Turbulence::Smagorinsky { constant: c, prandtl: 0.9 });
    }
    let m = model(1.0, 0.1 * l / re, d, p);
    let setup = Setup::new(&m, None, &[], &Limits::default())?;
    let mut flow = Flow::new(&setup, &[])?;
    Ok((0..steps_run).find(|_| match flow.step() {
        Ok(()) => false,
        Err(Error::Unstable(_)) => true,
        Err(e) => panic!("{e}"),
    }))
}

/// At a cell Reynolds number past what the lattice resolves on its own (u Δx / ν at the lid:
/// 104, τ⁺ = 0.5029) a step without a turbulence model is refused before it runs, pointing to
/// one, and with the closure the cavity runs stably.
#[test]
fn the_closure_holds_a_cavity_too_fast_for_the_lattice_alone() {
    let (re, steps_run) = (5000.0, 4000);
    let plain = cavity_failure(re, steps_run, None);
    assert!(matches!(&plain, Err(Error::Invalid(e)) if e.contains("turbulence")), "{plain:?}");
    let closed = cavity_failure(re, steps_run, Some(0.17));
    println!("cavity 48², Re {re} (cell Re {:.0}): without a model {plain:?}; with C = 0.17, unstable at {closed:?} of {steps_run} steps", re / 48.0);
    assert!(matches!(closed, Ok(None)), "{closed:?}");
}

/// Plane Poiseuille flow (walls at y = 0 and 1, driven by a body force, peak 0.1) with
/// Smagorinsky's model on, `n` cells across and diffusive time steps (τ fixed): the largest
/// error of the velocity against the laminar parabola, over its peak. With Λ = 3/16 the lattice
/// alone is exact for the parabola, so the error left is the eddy viscosity's, (C Δ)² |S|.
fn smagorinsky_poiseuille(n: usize, c: f64, collision: Collision) -> f64 {
    let (h, nu, umax) = (1.0, 1.0 / 6.0, 0.1);
    let dx = h / n as f64;
    let a = 8.0 * nu * umax / (h * h);
    let mut d = domain([0.0; 3], [dx, h, dx], dx);
    for f in [&mut d.faces.x_min, &mut d.faces.x_max, &mut d.faces.z_min, &mut d.faces.z_max] {
        *f = PERIODIC;
    }
    d.acceleration = [a, 0.0, 0.0];
    let mut p = collide(6.0 * h * h / nu, dx * dx, Some(3.0 / 16.0), collision);
    if let Procedure::Fluid { turbulence, .. } = &mut p {
        *turbulence = Some(Turbulence::Smagorinsky { constant: c, prandtl: 0.9 });
    }
    let (setup, mut flow) = prepare(&model(1.0, nu, d, p), &[]);
    run_all(&setup, &mut flow);
    let mut err: f64 = 0.0;
    for j in 0..n {
        let y = (j as f64 + 0.5) * dx;
        let exact = a / (2.0 * nu) * y * (h - y);
        let (_, u) = flow.moments(j).unwrap();
        err = err.max((u[0] * setup.velocity_scale() - exact).abs() / umax);
    }
    err
}

/// #664: Smagorinsky's model adds viscosity wherever there is shear, laminar flow included, so
/// with it on the laminar answer is reached only as the grid is refined: the eddy viscosity
/// (C Δ)² |S| falls like Δ², and so does Poiseuille's error against the parabola.
#[test]
fn with_the_model_on_poiseuille_converges_to_the_laminar_profile_like_dx_squared() {
    let c = 0.4;
    for collision in COLLISIONS {
        let errors: Vec<f64> = [8, 16, 32, 64].iter().map(|&n| smagorinsky_poiseuille(n, c, collision)).collect();
        let p = orders(&errors, 2.0);
        println!("{} Poiseuille with C = {c}, 8 → 64 cells: errors {errors:?}, observed orders {p:.3?}", collision.name());
        assert!(errors[0] > 1e-5, "the eddy viscosity shows: {errors:?}");
        // Observed (C = 0.4): TRT 1.970, 1.993, 1.998; regularised 1.839, 1.906, 1.950, which
        // approaches 2 from below, as its own O(Δ²) wall error joins the eddy viscosity's.
        assert!(p.iter().all(|o| *o > 1.8), "{}: second order, observed {p:.3?}", collision.name());
        assert!(p.windows(2).all(|w| w[1] >= w[0] - 1e-3) && (p[p.len() - 1] - 2.0).abs() < 0.06, "{}: {p:.3?}", collision.name());
    }
}

/// The decaying Taylor–Green vortex on [0, 2π]² (as in `verification.rs`) with Smagorinsky's
/// model on: the amplitude's relative error after its half-life against the laminar decay.
fn smagorinsky_taylor_green(n: usize, c: f64) -> f64 {
    let (nu, u0, l) = (0.1, 0.1, 2.0 * std::f64::consts::PI);
    let dx = l / n as f64;
    let mut d = domain([0.0; 3], [l, l, dx], dx);
    for f in fieldcraft_model::BoxFace::ALL {
        *d.faces.get_mut(f) = PERIODIC;
    }
    let dt = 0.3 * dx * dx / (3.0 * nu);
    let half_life = std::f64::consts::LN_2 / (2.0 * nu);
    let mut p = collide(half_life, dt, None, Collision::Regularized);
    if let Procedure::Fluid { turbulence, .. } = &mut p {
        *turbulence = Some(Turbulence::Smagorinsky { constant: c, prandtl: 0.9 });
    }
    let (setup, mut flow) = prepare(&model(1.0, nu, d, p), &[]);
    let grid = *flow.grid();
    let mode = |cell: usize| {
        let [x, y, _] = grid.centre(cell);
        [-x.cos() * y.sin(), x.sin() * y.cos()]
    };
    let to_lattice = 1.0 / setup.velocity_scale();
    flow.set_equilibrium(|cell| {
        let [x, y, _] = grid.centre(cell);
        let pressure = -u0 * u0 / 4.0 * ((2.0 * x).cos() + (2.0 * y).cos());
        let m = mode(cell);
        (setup.lattice_density(pressure), [u0 * m[0] * to_lattice, u0 * m[1] * to_lattice, 0.0])
    });
    run_all(&setup, &mut flow);
    let (mut num, mut den) = (0.0, 0.0);
    for cell in 0..grid.len() {
        let (_, u) = flow.moments(cell).unwrap();
        let m = mode(cell);
        num += u[0] * m[0] + u[1] * m[1];
        den += m[0] * m[0] + m[1] * m[1];
    }
    let amplitude = num / den * setup.velocity_scale();
    let exact = u0 * (-2.0 * nu * setup.steps as f64 * setup.time_step).exp();
    (amplitude / exact - 1.0).abs()
}

/// The vortex decays faster with the model on, and approaches the laminar decay to second order.
#[test]
fn with_the_model_on_taylor_green_approaches_the_laminar_decay_to_second_order() {
    let c = 0.4;
    let errors: Vec<f64> = [16, 32, 64].iter().map(|&n| smagorinsky_taylor_green(n, c)).collect();
    let p = orders(&errors, 2.0);
    println!("Taylor–Green with C = {c}, 16 → 64 cells: amplitude errors {errors:?}, observed orders {p:.3?}");
    for o in p {
        assert!(o > 1.8, "second order: {o:.3}");
    }
}
