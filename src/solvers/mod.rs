//! Solver families.
//!
//! - `ddd`     — Dynamic Discretization Discovery solvers (all 6 main
//!               solvers including the thesis MaxSAT-Default).
//! - `milp`    — MILP baselines (Big-M, TI, and experimental variants).
//! - `legacy`  — older pre-ladder DDD variants + non-DDD experiments (TI MaxSAT, IDL).
//! - `util`    — shared utilities (heuristics, counting solver, value trace).

pub mod ddd;
pub mod legacy;
pub mod milp;
pub mod util;

#[derive(Debug)]
pub enum SolverError {
    NoSolution,
    GurobiError(grb::Error),
    Timeout,
    OutOfMemory,
}
