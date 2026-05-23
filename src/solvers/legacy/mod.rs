//! Older MaxSAT-DDD variants and experimental approaches — kept for
//! reference and historical comparison; not part of the thesis main
//! solver set.
//!
//! - `maxsat_ddd`  — pre-ladder MaxSAT-DDD (External / IPAMIR / Incremental / PairwiseCustomRC2 variants).
//! - `maxsat_ti`   — Time-Indexed MaxSAT.
//! - `idl`         — Integer Difference Logic experimental solver. The
//!                   dispatch in `main.rs` short-circuits to `NoSolution`;
//!                   the implementation is preserved here for reference.
pub mod idl;
pub mod maxsat_ddd;
pub mod maxsat_ti;
