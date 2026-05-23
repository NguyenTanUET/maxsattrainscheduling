//! MaxSAT-DDD with ladder encoding — the main solver family of the thesis.
//!
//! - `maxsatddd_ladder`        — baseline (Croella 2024): pairwise AMO + raw cận dưới.
//! - `maxsatddd_ladder_sc`     — thesis default: SC AMO + precedence-graph preprocessing.
//! - `maxsatddd_ladder_abstract` — experimental Abstract-MaxSAT / IPAMIR backend.
pub mod maxsatddd_ladder;
pub mod maxsatddd_ladder_abstract;
pub mod maxsatddd_ladder_sc;
