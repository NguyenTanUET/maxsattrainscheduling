//! Sequential Counter (SC) AMO encoding for resource-conflict cliques —
//! Contribution 1 of the thesis (Chapter 3 §3.2.1).
//!
//! `add_sc_amo` emits the SC AMO encoding (Sinz 2005 + Truong/Kieu/To
//! ICAART 2025 four-formula form). `add_pairwise_amo` is the legacy
//! O(n²) pairwise encoding for comparison. `add_hybrid_amo` dispatches
//! between the two based on clique size (`PAIRWISE_AMO_MAX_SIZE`).
//!
//! `get_delay_lit_at` and `build_active_lit` are the Tseitin helpers that
//! materialise the per-visit indicator literals `x_v` over which the AMO
//! is enforced (thesis Eq. eq:xv-cnf).

use std::collections::{HashMap, HashSet};

use satcoder::{Bool, SatInstance};
use typed_index_collections::TiVec;

use crate::problem::Problem;

use super::precedence::add_fixed_precedence_row;
use super::settings::PAIRWISE_AMO_MAX_SIZE;
use super::solve::{Occ, VisitId};

/// Sequential Counter (SC) encoding for At-Most-One — Truong, Kieu, To
/// (ICAART 2025), §3.1, four-formula form.
///
/// Auxiliary register bits `prefix[0..n-1]` where `prefix[i]` plays the role
/// of `R_{i+1}` in the paper (R_1 ≡ x_1 implicitly via the `(¬lits[0] ∨
/// prefix[0])` clause we emit before the loop).
///
/// All four paper formulas are emitted:
///   (1) x_j → R_j                           — `(¬lits[i] ∨ prefix[i])`
///   (2) R_{j-1} → R_j                       — `(¬prefix[i-1] ∨ prefix[i])`
///   (3) ¬x_j ∧ ¬R_{j-1} → ¬R_j              — `(lits[i] ∨ prefix[i-1] ∨ ¬prefix[i])`
///   (4) x_j → ¬R_{j-1}                      — `(¬lits[i] ∨ ¬prefix[i-1])`
///
/// Formula (3) is the *downward* propagation (register stays false when no
/// var has fired yet). Earlier versions of this file omitted it, which kept
/// soundness for AMO but weakened unit propagation. The added clauses are
/// O(n) and let CDCL conclude `prefix[i]` cannot be forced true unless some
/// `lits[k≤i]` is.
pub(super) fn add_sc_amo<L: satcoder::Lit>(solver: &mut impl SatInstance<L>, lits: &[Bool<L>]) {
    match lits.len() {
        0 | 1 => return,
        2 => {
            solver.add_clause(vec![!lits[0], !lits[1]]);
            return;
        }
        _ => {}
    }

    let mut prefix = Vec::with_capacity(lits.len() - 1);
    for _ in 0..(lits.len() - 1) {
        prefix.push(solver.new_var());
    }

    // R_1 layer: x_1 ↔ R_1 via (1) one direction + (3) the other.
    // Together they make prefix[0] equivalent to lits[0].
    solver.add_clause(vec![!lits[0], prefix[0]]);                // (1) for j=1
    solver.add_clause(vec![lits[0], !prefix[0]]);                // (3) for j=1

    for i in 1..(lits.len() - 1) {
        solver.add_clause(vec![!lits[i], prefix[i]]);            // (1)
        solver.add_clause(vec![!prefix[i - 1], prefix[i]]);      // (2)
        solver.add_clause(vec![lits[i], prefix[i - 1], !prefix[i]]); // (3)
        solver.add_clause(vec![!lits[i], !prefix[i - 1]]);       // (4)
    }
    solver.add_clause(vec![
        !lits[lits.len() - 1],
        !prefix[prefix.len() - 1],
    ]); // (4) for j=w
}

pub(super) fn add_pairwise_amo<L: satcoder::Lit>(solver: &mut impl SatInstance<L>, lits: &[Bool<L>]) {
    for i in 0..lits.len() {
        for j in (i + 1)..lits.len() {
            solver.add_clause(vec![!lits[i], !lits[j]]);
        }
    }
}

pub(super) fn add_hybrid_amo<L: satcoder::Lit>(
    solver: &mut impl SatInstance<L>,
    lits: &[Bool<L>],
    use_sc_amo: bool,
) {
    if !use_sc_amo || lits.len() <= PAIRWISE_AMO_MAX_SIZE {
        add_pairwise_amo(solver, lits);
    } else {
        add_sc_amo(solver, lits);
    }
}

/// Monotone delay literal `visit_id.start ≥ t`.
/// Returns `true.into()` if t ≤ earliest (always satisfied),
/// `false.into()` if t ≥ infinity sentinel (never satisfied),
/// otherwise creates the timepoint (if new) and returns its literal.
pub(super) fn get_delay_lit_at<L: satcoder::Lit>(
    solver: &mut impl SatInstance<L>,
    problem: &Problem,
    visits: &TiVec<VisitId, (usize, usize)>,
    occupations: &mut TiVec<VisitId, Occ<L>>,
    new_time_points: &mut Vec<(VisitId, Bool<L>, i32)>,
    fixed_prec_rows: &mut HashSet<(VisitId, i32)>,
    use_eager_chain_expansion: bool,
    visit_id: VisitId,
    t: i32,
) -> Bool<L> {
    let (earliest_t, last_t) = {
        let occ = &occupations[visit_id];
        (occ.delays[0].1, occ.delays[occ.delays.len() - 1].1)
    };
    if t <= earliest_t {
        return true.into();
    }
    if t >= last_t {
        return false.into();
    }
    let (lit, is_new) = occupations[visit_id].time_point(solver, t);
    if is_new {
        new_time_points.push((visit_id, lit, t));
        let _ = add_fixed_precedence_row(
            solver,
            problem,
            visits,
            occupations,
            new_time_points,
            fixed_prec_rows,
            visit_id,
            lit,
            t,
            use_eager_chain_expansion,
        );
    }
    lit
}

/// Build a sound "active at tau" aux variable via Tseitin:
///   active_i = (start_i ≤ tau) ∧ (end_i > tau)
///           = !delay_i(tau+1) ∧ delay_next_i(tau+1)
/// For the last visit of a train (no next visit), end_i = start_i + travel,
/// so `end_i > tau` ⟺ `start_i ≥ tau - travel + 1`, encoded as `delay_i(tau+1-travel)`.
///
/// Cached per (visit_id, tau+1) to avoid duplicate aux vars.
pub(super) fn build_active_lit<L: satcoder::Lit>(
    solver: &mut impl SatInstance<L>,
    problem: &Problem,
    visits: &TiVec<VisitId, (usize, usize)>,
    occupations: &mut TiVec<VisitId, Occ<L>>,
    new_time_points: &mut Vec<(VisitId, Bool<L>, i32)>,
    fixed_prec_rows: &mut HashSet<(VisitId, i32)>,
    active_cache: &mut HashMap<(VisitId, i32), Bool<L>>,
    use_eager_chain_expansion: bool,
    visit_id: VisitId,
    tau_plus_1: i32,
) -> Bool<L> {
    if let Some(&lit) = active_cache.get(&(visit_id, tau_plus_1)) {
        return lit;
    }

    let (train_idx, visit_idx) = visits[visit_id];

    let delay_start = get_delay_lit_at(
        solver,
        problem,
        visits,
        occupations,
        new_time_points,
        fixed_prec_rows,
        use_eager_chain_expansion,
        visit_id,
        tau_plus_1,
    );

    let delay_end = if visit_idx + 1 < problem.trains[train_idx].visits.len() {
        let next_id: VisitId = (usize::from(visit_id) + 1).into();
        get_delay_lit_at(
            solver,
            problem,
            visits,
            occupations,
            new_time_points,
            fixed_prec_rows,
            use_eager_chain_expansion,
            next_id,
            tau_plus_1,
        )
    } else {
        let travel = problem.trains[train_idx].visits[visit_idx].travel_time;
        get_delay_lit_at(
            solver,
            problem,
            visits,
            occupations,
            new_time_points,
            fixed_prec_rows,
            use_eager_chain_expansion,
            visit_id,
            tau_plus_1 - travel,
        )
    };

    let active = solver.new_var();
    // For AMO clique soundness we only need the CONVERSE direction:
    //   (!delay_start ∧ delay_end) → active
    // i.e. when a visit is genuinely occupying the resource at τ, its
    // `active` literal must be true; the AMO over `active`s then forces
    // all other actives in the clique to false, which (by their own
    // converse clauses) constrains their delay literals correctly.
    //
    // The two forward clauses
    //   active → !delay_start
    //   active → delay_end
    // make the encoding a full biconditional `active ↔ (!delay_start ∧ delay_end)`,
    // giving the solver the ability to propagate from `active` back to
    // delay literals. They are NOT required for soundness — a `true`
    // assignment to `active` without the corresponding delay literals
    // does not break AMO. They only help propagation strength.
    //
    // Toggle this const off to A/B-test whether the forward clauses are
    // actually buying us search speedup or are over-tightening (extra
    // propagation that pulls the solver down unhelpful branches).
    const ENCODE_ACTIVE_FORWARD_DIRECTION: bool = false;
    if ENCODE_ACTIVE_FORWARD_DIRECTION {
        // active → !delay_start
        solver.add_clause(vec![!active, !delay_start]);
        // active → delay_end
        solver.add_clause(vec![!active, delay_end]);
    }
    // (!delay_start ∧ delay_end) → active   — required for soundness.
    solver.add_clause(vec![active, delay_start, !delay_end]);

    active_cache.insert((visit_id, tau_plus_1), active);
    active
}
