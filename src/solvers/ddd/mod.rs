//! Dynamic Discretization Discovery (DDD) solvers for TRP.
//!
//! All solvers in this folder implement the iterative DDD framework
//! (Boland 2017, Croella 2024): start with a small CNF, solve, detect new
//! conflicts on the candidate schedule, add clauses + time points, re-solve.
//! They differ in the SAT/MaxSAT backend and in the cost encoding.
//!
//! - `maxsat_ladder` — Croella 2024 baseline (MaxSAT-Base in the thesis).
//! - `maxsat_ladder_sc` — thesis main solver (MaxSAT-Default): SC AMO +
//!   precedence preprocessing on top of the ladder baseline.
//! - `maxsat_ladder_abstract` — experimental abstract-MaxSAT / IPAMIR
//!   backend over the ladder encoding.
//! - `maxsat_rc2` — MaxSAT-RC2 (core-guided) using the shared DDD types.
//! - `incremental_sat` — incremental SAT-DDD (used by IncSAT-Default).
//! - `puresat` — non-incremental SAT-DDD with per-iteration rebuild
//!   (used by PureSAT-Default).
//! - `shared` — utilities shared across the family (common types,
//!   cost-tree, chain-earliest precedence).

pub mod incremental_sat;
pub mod maxsat_ladder;
pub mod maxsat_ladder_abstract;
pub mod maxsat_ladder_sc;
pub mod maxsat_rc2;
pub mod puresat;
pub mod shared;

pub use incremental_sat::solve as solve_sat;
pub use maxsat_rc2::solve as solve_maxsat_rc2;

#[derive(Clone, Copy, Debug)]
pub enum SolveMode { MaxSatRc2, Sat }
