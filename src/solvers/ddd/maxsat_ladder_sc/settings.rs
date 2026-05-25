//! Configuration for `maxsat_ladder_sc`.
#[derive(Clone, Copy, Debug)]
pub struct MaxSatDddLadderScSettings {
    pub use_precedence_graph: bool,
    pub use_sc_amo: bool,
    pub use_touched_clique_amo: bool,

    // ── Alternative / experimental knobs (default OFF) ───────────────────
    /// Eagerly expand long travel-time precedence chains into per-step
    /// 3-literal clauses.
    pub use_eager_chain_expansion: bool,
    /// Seed fixed precedence rows from earliest time points.
    pub seed_sc_from_earliest: bool,
    /// Pre-allocate SAT variables and monotonicity clauses for the
    /// per-visit cost-threshold time points at INIT.
    pub prealloc_cost_thresholds: bool,
}

impl Default for MaxSatDddLadderScSettings {
    fn default() -> Self {
        Self {
            use_precedence_graph: true,
            use_eager_chain_expansion: false,
            use_sc_amo: true,
            use_touched_clique_amo: true,
            seed_sc_from_earliest: false,
            prealloc_cost_thresholds: false,
        }
    }
}

// ─── Tunable for SC AMO encoding ─────────────────────────────────────────
pub(super) const PAIRWISE_AMO_MAX_SIZE: usize = 8;
