pub mod common;
pub mod costtree;
pub mod maxsat_rc2;
pub mod sat;

pub use maxsat_rc2::solve as solve_maxsat_rc2;
pub use sat::solve as solve_sat;

#[derive(Clone, Copy, Debug)]
pub enum SolveMode { MaxSatRc2, Sat }
