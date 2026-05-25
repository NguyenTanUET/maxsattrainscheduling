//! Shared utilities used by multiple DDD solver families.
//!
//! - `common` — shared types (Occ, VisitId, SolveStats helpers) used by `maxsat_rc2`.
//! - `costtree` — cost-tree representation used across ladder solvers + maxsat_rc2.
//! - `precedence` — within-train chain propagation (`chain_earliest`).
//!                  Thesis Contribution 2.
//! - `greedy` — greedy feasible schedule for warm-start UB; used by
//!              MaxSAT solvers to seed `best_heur`.

pub mod common;
pub mod costtree;
pub mod greedy;
pub mod precedence;
