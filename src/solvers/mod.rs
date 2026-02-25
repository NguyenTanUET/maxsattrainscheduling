pub mod bigm;
pub mod binarizedbigm;
pub mod costtree;
pub mod ddd;
pub mod greedy;
pub mod heuristic;
pub mod idl;
pub mod maxsat_ddd;
pub mod maxsat_ti;
pub mod maxsatddd_ladder;
pub mod maxsatddd_ladder_abstract;
pub mod maxsatddd_ladder_scl;
pub mod milp_ti;
mod minimize;
pub mod mipdddpack;
// pub mod cutting;

#[derive(Debug)]
pub enum SolverError {
    NoSolution,
    GurobiError(grb::Error),
    Timeout,
}
