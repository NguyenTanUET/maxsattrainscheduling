//! Shared utilities used by multiple solver families.
//!
//! - `greedy`            — greedy initial schedule heuristic.
//! - `heuristic`         — improved heuristic, used as MaxSAT warm-start UB.
//! - `minimize`          — UNSAT core minimisation.
//! - `counting_solver`   — `SatSolver` wrapper recording vars/clauses.
//! - `value_trace`       — per-iteration trace of cost values during a solve.
pub mod counting_solver;
pub mod greedy;
pub mod heuristic;
pub mod minimize;
pub mod value_trace;
