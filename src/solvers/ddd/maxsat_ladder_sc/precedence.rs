//! Precedence-graph preprocessing — Contribution 2 of the thesis (Chapter 3 §3.2.2).
//!
//! `add_fixed_precedence_row` encodes one within-train travel-time
//! implication `d^{i,r}(t) → d^{i,q}(t + l^{r}_i)` between two consecutive
//! visits of the same train.
//!
//! `propagate_precedence` walks the chain forward from a starting visit,
//! eagerly emitting all reachable fixed-precedence rows in one pass.

use std::collections::{HashSet, VecDeque};

use satcoder::{Bool, SatInstance};
use typed_index_collections::TiVec;

use crate::problem::Problem;

use super::solve::{Occ, VisitId};

/// Add a fixed precedence row for one chosen time point and return the
/// propagated successor time point for further queue-based propagation.
pub(super) fn add_fixed_precedence_row<L: satcoder::Lit>(
    solver: &mut impl SatInstance<L>,
    problem: &Problem,
    visits: &TiVec<VisitId, (usize, usize)>,
    occupations: &mut TiVec<VisitId, Occ<L>>,
    new_time_points: &mut Vec<(VisitId, Bool<L>, i32)>,
    added: &mut HashSet<(VisitId, i32)>,
    visit_id: VisitId,
    in_var: Bool<L>,
    in_t: i32,
    use_eager_chain_expansion: bool,
) -> Option<(VisitId, Bool<L>, i32)> {
    if !added.insert((visit_id, in_t)) {
        return None;
    }

    let (train_idx, visit_idx) = visits[visit_id];
    if visit_idx + 1 >= problem.trains[train_idx].visits.len() {
        return None;
    }

    let travel = problem.trains[train_idx].visits[visit_idx].travel_time;
    let next_visit: VisitId = (usize::from(visit_id) + 1).into();
    let req_t = in_t + travel;

    let earliest_next = occupations[next_visit].delays[0].1;
    if req_t <= earliest_next {
        return Some((next_visit, true.into(), earliest_next));
    }

    let (req_var, is_new) = occupations[next_visit].time_point(solver, req_t);
    if use_eager_chain_expansion {
        // Eager chain expansion for long travel-time precedence chains.
        // For chains ≤ threshold: a single 2-literal implication is
        // sufficient — ladder monotonicity propagates the rest cheaply.
        // For chains > threshold: expand into per-step 3-literal clauses
        // so unit propagation reaches the far end in one shot rather than
        // multi-hop through the monotonicity ladder.
        const LADDER_SHORTCUT_THRESHOLD: usize = 8;
        let idx = occupations[next_visit]
            .delays
            .partition_point(|(_, t0)| *t0 < req_t);
        if idx > LADDER_SHORTCUT_THRESHOLD {
            for i in 0..idx {
                let lit_i = occupations[next_visit].delays[i].0;
                let lit_next = occupations[next_visit].delays[i + 1].0;
                solver.add_clause(vec![!in_var, !lit_i, lit_next]);
            }
        } else {
            solver.add_clause(vec![!in_var, req_var]);
        }
    } else {
        // Plain encoding: a single 2-literal implication. Relies on the
        // monotonicity chain among delay literals (already established in
        // `Occ::time_point`) for correct forward propagation.
        solver.add_clause(vec![!in_var, req_var]);
    }
    if is_new {
        new_time_points.push((next_visit, req_var, req_t));
    }

    Some((next_visit, req_var, req_t))
}

pub(super) fn propagate_precedence<L: satcoder::Lit>(
    solver: &mut impl SatInstance<L>,
    problem: &Problem,
    visits: &TiVec<VisitId, (usize, usize)>,
    occupations: &mut TiVec<VisitId, Occ<L>>,
    new_time_points: &mut Vec<(VisitId, Bool<L>, i32)>,
    added: &mut HashSet<(VisitId, i32)>,
    start_visit: VisitId,
    start_var: Bool<L>,
    start_t: i32,
    use_eager_chain_expansion: bool,
) {
    let mut queue = VecDeque::from([(start_visit, start_var, start_t)]);

    while let Some((visit_id, in_var, in_t)) = queue.pop_front() {
        if let Some(next) = add_fixed_precedence_row(
            solver,
            problem,
            visits,
            occupations,
            new_time_points,
            added,
            visit_id,
            in_var,
            in_t,
            use_eager_chain_expansion,
        ) {
            queue.push_back(next);
        }
    }
}
