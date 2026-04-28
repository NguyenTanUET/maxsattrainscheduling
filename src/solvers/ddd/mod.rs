pub mod common;
pub mod costtree;
pub mod incremental_sat;
pub mod maxsat_rc2;
pub mod puresat;

pub use incremental_sat::solve as solve_sat;
pub use maxsat_rc2::solve as solve_maxsat_rc2;

#[derive(Clone, Copy, Debug)]
pub enum SolveMode { MaxSatRc2, Sat }
