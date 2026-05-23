//! True SCAMO (Staircase At-Most-One) encoding primitives — Truong/Kieu/To
//! ICAART 2025 §3.1.
//!
//! **Status: experimental, NOT wired into the main DDD loop.** The three
//! primitives below build the staircase block-by-block but the integration
//! step (deciding how to group consecutive (resource, τ) cliques into a
//! sliding-window staircase, and where to put block boundaries) is
//! non-trivial and intentionally left out for now. Until that design lands,
//! the main loop keeps using [`super::super::sc_amo::add_hybrid_amo`] which
//! is sound for any single clique.

use satcoder::{Bool, SatInstance};

/// Encode an **AMO block** of an SCAMO chain (paper §3.1, all four formulas).
///
/// Allocates `vars.len() - 1` fresh register bits and emits clauses
/// equivalent to At-Most-One over `vars`, in the staircase-friendly form
/// where the register bits are exposed for use by neighbouring blocks.
///
/// Returns the register bits `R[1..vars.len()]` (so `out[i]` represents
/// `R_{i+1}` in the paper's 1-indexed notation; `R_1 ≡ vars[0]` implicitly).
///
/// Use this for the *first* block of each subset and for any subset whose
/// neighbour relationship requires the AMO-enforcement clauses.
#[allow(dead_code)]
pub fn encode_scamo_amo_block<L: satcoder::Lit>(
    solver: &mut impl SatInstance<L>,
    vars: &[Bool<L>],
) -> Vec<Bool<L>> {
    if vars.len() < 2 {
        return Vec::new();
    }
    let mut prefix = Vec::with_capacity(vars.len() - 1);
    for _ in 0..(vars.len() - 1) {
        prefix.push(solver.new_var());
    }
    // R_1 layer: x_1 ↔ R_1 (formulas 1 + 3 for j=1 give a biconditional).
    solver.add_clause(vec![!vars[0], prefix[0]]);
    solver.add_clause(vec![vars[0], !prefix[0]]);
    for i in 1..(vars.len() - 1) {
        solver.add_clause(vec![!vars[i], prefix[i]]);                        // (1)
        solver.add_clause(vec![!prefix[i - 1], prefix[i]]);                  // (2)
        solver.add_clause(vec![vars[i], prefix[i - 1], !prefix[i]]);         // (3)
        solver.add_clause(vec![!vars[i], !prefix[i - 1]]);                   // (4)
    }
    // (4) for the last variable, against the topmost prefix.
    solver.add_clause(vec![!vars[vars.len() - 1], !prefix[prefix.len() - 1]]);
    prefix
}

/// Encode an **AMZ block** of an SCAMO chain (paper §3.1, formulas 1, 2, 3
/// only — formula 4 is intentionally skipped).
///
/// AMZ blocks **do NOT enforce at-most-one** on their own. They piggy-back
/// on the AMO block of an adjacent subset: the connection clauses
/// (Proposition 1) ensure that if both halves were "active", we'd have a
/// joint AMO-zero, otherwise the AMO block fires.
///
/// Returns the register bits `R[1..vars.len()]` for connecting to neighbouring
/// blocks via [`connect_scamo_blocks`].
#[allow(dead_code)]
pub fn encode_scamo_amz_block<L: satcoder::Lit>(
    solver: &mut impl SatInstance<L>,
    vars: &[Bool<L>],
) -> Vec<Bool<L>> {
    if vars.len() < 2 {
        return Vec::new();
    }
    let mut prefix = Vec::with_capacity(vars.len() - 1);
    for _ in 0..(vars.len() - 1) {
        prefix.push(solver.new_var());
    }
    // R_1 layer: x_1 ↔ R_1 (formulas 1 + 3 for j=1).
    solver.add_clause(vec![!vars[0], prefix[0]]);
    solver.add_clause(vec![vars[0], !prefix[0]]);
    for i in 1..(vars.len() - 1) {
        solver.add_clause(vec![!vars[i], prefix[i]]);                        // (1)
        solver.add_clause(vec![!prefix[i - 1], prefix[i]]);                  // (2)
        solver.add_clause(vec![vars[i], prefix[i - 1], !prefix[i]]);         // (3)
        // Formula (4) intentionally skipped — this is what makes the block
        // an AMZ rather than an AMO. The neighbouring AMO block plus the
        // connection clauses together still enforce the global SCAMO
        // property (paper §3.1 Proposition 1 + §3.1 derivation).
    }
    prefix
}

/// Connect two SCAMO blocks that represent two consecutive sliding windows.
///
/// `bits_a`: register bits of the *trailing* block (built with the right
/// variable ordering — last variable first in the underlying SC).
/// `bits_b`: register bits of the *leading* block (left ordering).
/// `overlap`: number of overlapping positions between the two windows
/// (= w - 1 in paper's fixed-width staircase).
///
/// Emits `overlap` disjunction clauses of the form
///   `(¬R_a[k] ∨ ¬R_b[overlap - k])`
/// for `k = 1..overlap`, which is the application of paper's Proposition 1
/// (pairs of partial sums where at least one must be zero).
#[allow(dead_code)]
pub fn connect_scamo_blocks<L: satcoder::Lit>(
    solver: &mut impl SatInstance<L>,
    bits_a: &[Bool<L>],
    bits_b: &[Bool<L>],
    overlap: usize,
) {
    // For each k in 1..overlap:
    //   bits_a[k-1] is R_a,k  (sum of first k vars in block a, in its ordering)
    //   bits_b[overlap-k-1] is R_b,(overlap-k)
    // Disjunction: at least one of the two partial sums is zero.
    for k in 1..overlap {
        let r_a = bits_a[k - 1];
        let r_b_idx = overlap - k;
        if r_b_idx == 0 || r_b_idx > bits_b.len() {
            continue;
        }
        let r_b = bits_b[r_b_idx - 1];
        solver.add_clause(vec![!r_a, !r_b]);
    }
}
