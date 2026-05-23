//! MaxSAT-DDD with ladder encoding + SC AMO + precedence preprocessing —
//! the thesis main solver (Chapter 3 of the thesis).
//!
//! Module layout:
//! - [`settings`] — `MaxSatDddLadderScSettings` + tunable constants.
//! - [`sc_amo`] — Contribution 1: SC AMO encoding helpers
//!                (`add_sc_amo`, `add_hybrid_amo`, Tseitin
//!                helpers `get_delay_lit_at` and `build_active_lit`).
//! - [`precedence`] — Contribution 2: precedence-graph propagation
//!                    (`add_fixed_precedence_row`, `propagate_precedence`).
//! - [`experimental_approach`] — Alternatives that are **not** part of the
//!                               thesis result (TRUE SCAMO encoding, etc.).
//! - [`solve`] — main DDD loop, common types, cost-tree integration.
//!
//! The public API of this module is the `solve*` entry points and
//! `MaxSatDddLadderScSettings`, both re-exported here for compatibility
//! with the old flat-file layout.

mod experimental_approach;
mod precedence;
mod sc_amo;
mod settings;
mod solve;

pub use settings::MaxSatDddLadderScSettings;
pub use solve::{solve, solve_debug, solve_debug_with_settings, solve_with_settings, IterationType, SolveStats};
