//! Solver families.
//!
//! - `ladder`  — MaxSAT-DDD ladder solvers (thesis main).
//! - `milp`    — MILP baselines (Big-M, TI, and experimental variants).
//! - `ddd`     — DDD framework solvers (PureSAT, IncSAT, MaxSAT-RC2).
//! - `legacy`  — pre-ladder MaxSAT solvers and experimental approaches.
//! - `util`    — shared utilities (heuristics, counting solver, value trace).

pub mod ddd;
pub mod ladder;
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
