//! Precedence preprocessing for TRP — Contribution 2 of the thesis
//! (Chapter 3 §3.2.2).
//!
//! [`chain_earliest`] — within-train chain propagation
//! `est[v+1] = max(visit.earliest, est[v] + travel[v])`. Sound for all
//! objectives; used by `incremental_sat`, `puresat`, and
//! `maxsat_ladder_sc`. **This is the thesis' Contribution 2.**

use crate::problem::Problem;

/// Simple within-train chain propagation of earliest start times.
///
/// For each train, iterate visits in order and set
/// `earliest[v] = max(visit.earliest, earliest[v-1] + travel[v-1])`.
pub fn chain_earliest(problem: &Problem) -> Vec<Vec<i32>> {
    let mut effective = Vec::with_capacity(problem.trains.len());
    for train in &problem.trains {
        let mut train_bounds = Vec::with_capacity(train.visits.len());
        let mut propagated_lb: Option<i32> = None;
        for visit in &train.visits {
            let lb = propagated_lb
                .map_or(visit.earliest, |prev_lb: i32| prev_lb.max(visit.earliest));
            train_bounds.push(lb);
            propagated_lb = Some(lb.saturating_add(visit.travel_time));
        }
        effective.push(train_bounds);
    }
    effective
}
