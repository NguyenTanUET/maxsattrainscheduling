//! Configuration for `maxsat_ladder_sc`.
//!
//! Field groups (see thesis Chapter 3):
//! - **Contribution 1 (SC AMO encoding)**: `use_sc_amo`, `use_touched_clique_amo`
//! - **Contribution 2 (precedence preprocessing)**: `use_precedence_graph`
//! - **Alternative / experimental knobs** (default OFF, not part of the
//!   thesis main result): `use_eager_chain_expansion`,
//!   `use_interval_graph_conflicts`, `seed_sc_from_earliest`,
//!   `use_scamo_encoding`, `prealloc_cost_thresholds`. The supporting code for
//!   `use_scamo_encoding` lives under `experimental_approach::scamo`.

#[derive(Clone, Copy, Debug)]
pub struct MaxSatDddLadderScSettings {
    // ── Contribution 2 (precedence preprocessing) ────────────────────────
    /// Toggle precedence-graph preprocessing/queue seeding.
    pub use_precedence_graph: bool,

    // ── Contribution 1 (SC AMO encoding) ────────────────────────────────
    /// Use SC (Sequential Counter) AMO encoding from Truong/Kieu/To
    /// (ICAART 2025) for resource-conflict cliques of size > [`PAIRWISE_AMO_MAX_SIZE`].
    /// If false, the encoding stays pairwise regardless of clique size.
    pub use_sc_amo: bool,
    /// Lite clique-AMO mode for the pair-based conflict path: during the
    /// pair-by-pair conflict scan, aggregate visits sharing a (resource,
    /// tau) into mini-cliques. For each clique of size ≥ 3, emit an SC
    /// AMO over the active literals — in addition to the per-pair "delay
    /// one of them" clauses. Effective only when
    /// `use_interval_graph_conflicts = false` (otherwise the full
    /// interval-graph clique-cover already handles AMOs).
    pub use_touched_clique_amo: bool,

    // ── Alternative / experimental knobs (default OFF) ───────────────────
    /// Eagerly expand long travel-time precedence chains into per-step
    /// 3-literal clauses (instead of one 2-literal implication that relies
    /// on ladder monotonicity propagation).
    pub use_eager_chain_expansion: bool,
    /// Use interval-graph clique-cover conflict encoding (AMO over cliques)
    /// instead of the lite `use_touched_clique_amo` path. Has no effect on
    /// soundness — both feed cliques into the AMO encoder.
    pub use_interval_graph_conflicts: bool,
    /// Seed fixed precedence rows from earliest time points (only used when
    /// SC is enabled).
    pub seed_sc_from_earliest: bool,
    /// **Experimental**: TRUE SCAMO (Staircase AMO) encoding per
    /// Truong/Kieu/To (ICAART 2025) §3.1. Currently a placeholder — the
    /// primitives are implemented under
    /// `experimental_approach::scamo` but integration into the main
    /// DDD loop requires further work.
    pub use_scamo_encoding: bool,
    /// Pre-allocate SAT variables and monotonicity clauses for the
    /// per-visit cost-threshold time points at INIT. When OFF (default):
    /// matches `maxsat_ladder` lazy behaviour — cost timepoints are
    /// created only when an actual conflict needs them.
    pub prealloc_cost_thresholds: bool,
}

impl Default for MaxSatDddLadderScSettings {
    fn default() -> Self {
        // Option B baseline (precedence + touched-clique AMO + SC encoding):
        // empirically best on infsteps180 + finsteps123, lighter than the
        // full interval-graph clique cover.
        Self {
            use_precedence_graph: true,
            use_eager_chain_expansion: false,
            use_interval_graph_conflicts: false,
            use_sc_amo: true,
            use_touched_clique_amo: true,
            seed_sc_from_earliest: false,
            use_scamo_encoding: false,
            prealloc_cost_thresholds: false,
        }
    }
}

// ─── Tunables for SC AMO encoding ────────────────────────────────────────
//
// Two compile-time knobs control how AMO over conflict cliques is encoded.

/// Maximum clique size encoded by pairwise AMO. Cliques strictly larger
/// than this use SC (Sequential Counter) AMO from Truong/Kieu/To, ICAART
/// 2025 §3.1 (see [`super::sc_amo::add_sc_amo`]). Larger value keeps
/// pairwise for medium cliques whose simplicity may beat SC's tighter
/// propagation in the DDD setting.
///
/// Empirically tuned to 5 via threshold sweep on the Croella2024 TRP
/// benchmark (3 objectives × 72 instances). Tested values {3, 5, 10}:
///   - n=3  → too aggressive: SC on clique 4-5 adds 2n-1 aux vars that
///            don't pay back in clause savings; stationB1 (cont) blew
///            up 22s → 99s.
///   - n=5  → SWEET SPOT: pairwise for trivial cliques, SC for medium+
///            cliques where register chain helps CDCL learn quality
///            clauses. Best total sol_time on all 3 objectives.
///   - n=10 → too conservative: clique 6-10 falls back to pairwise,
///            losing SC's propagation-chain advantage; net +15–46%
///            slowdown vs n=5 on infsteps180/cont.
///
/// Theoretical crossover on raw clause count is at n ≈ 8
/// (4n-5 < C(n,2) when n ≥ 8), but CDCL learnt-clause quality from
/// the SC register chain shifts the practical optimum to n=5.
pub(super) const PAIRWISE_AMO_MAX_SIZE: usize = 5;

/// Lazy AMO threshold. A clique with > 2 members must be detected this
/// many times across iterations (counted by member visit-set) before its
/// full AMO is encoded. Until then each detection emits only a single
/// pair clause for the clique's first two members — same shape as the
/// 2-member fast path. 0 = eager (encode AMO on first detection); ≥ 2
/// = lazy with that many pair-clause "warmups" first.
pub(super) const LAZY_AMO_THRESHOLD: usize = 0;
