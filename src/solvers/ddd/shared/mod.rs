//! Shared utilities used by multiple DDD solver families.
//!
//! - `common` — shared types (Occ, VisitId, SolveStats helpers) used by `maxsat_rc2`.
//! - `costtree` — cost-tree representation used across ladder solvers + maxsat_rc2.
//! - `precedence` — within-train chain propagation (`chain_earliest`)
//!                  and greedy warm-start schedule. Thesis Contribution 2.

pub mod common;
pub mod costtree;
pub mod precedence;
