//! Mixed-Integer Linear Programming baselines via Gurobi.
//!
//! - `bigm`           — Big-M formulation (in-thesis baseline).
//! - `milp_ti`        — Time-Indexed MILP (in-thesis baseline).
//! - `binarizedbigm`  — experimental binarized Big-M variants.
//! - `mipdddpack`     — experimental MIP-DDD-Pack solver.
pub mod bigm;
pub mod binarizedbigm;
pub mod milp_ti;
pub mod mipdddpack;
