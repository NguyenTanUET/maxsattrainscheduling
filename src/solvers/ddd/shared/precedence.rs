//! Precedence preprocessing for TRP — Contribution 2 of the thesis
//! (Chapter 3 §3.2.2).
//!
//! - [`chain_earliest`] — within-train chain propagation
//!   `est[v+1] = max(visit.earliest, est[v] + travel[v])`. Sound for all
//!   objectives; used by `incremental_sat`, `puresat`, and
//!   `maxsat_ladder_sc`. **This is the thesis' Contribution 2.**
//! - [`greedy_schedule`] — greedy initial schedule used as warm-start UB
//!   for solvers without a Gurobi-based heuristic.

use crate::problem::{Problem, Train};
use std::collections::HashMap;

/// Simple within-train chain propagation of earliest start times.
///
/// For each train, iterate visits in order and set
/// `earliest[v] = max(visit.earliest, earliest[v-1] + travel[v-1])`.
///
/// Sound for ALL objectives — no energetic reasoning, never declares
/// infeasibility.
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

fn propagate_est_forward(
    est: &mut [i32],
    travel: &[i32],
    trains: &[Train],
    train_offset: &[usize],
) {
    for (t_idx, train) in trains.iter().enumerate() {
        for v_idx in 1..train.visits.len() {
            let prev = train_offset[t_idx] + v_idx - 1;
            let cur = train_offset[t_idx] + v_idx;
            let bound = est[prev].saturating_add(travel[prev]);
            if est[cur] < bound {
                est[cur] = bound;
            }
        }
    }
}

/// Greedy feasible schedule for warm-start UB.
///
/// Returns per-train `[t_v0, t_v1, ..., t_last, t_last + travel_last]` as
/// expected by [`crate::problem::Problem::verify_solution`] /
/// [`crate::problem::Problem::cost`]: each train has `n_visits + 1`
/// entries (one per visit start + one final completion time).
///
/// Algorithm: forward-chain `est`, sort visits by `est` ascending (ties
/// by linear index = stable within-train order), then sweep — each visit
/// starts at `max(est, conflict-resource latest finish, predecessor
/// finish)`. The resulting schedule respects:
///   1. `visit.earliest` lower bounds,
///   2. within-train chain `t[v+1] ≥ t[v] + travel[v]`,
///   3. resource non-overlap on any pair `(r, r')` in `problem.conflicts`.
/// → feasible by construction; cost is a valid upper bound on optimal.
///
/// Used by solvers that lack a Gurobi-based heuristic UB to seed
/// `best_heur` so timeout cases can still report a finite `ub` / GAP.
pub fn greedy_schedule(problem: &Problem) -> Vec<Vec<i32>> {
    let n_trains = problem.trains.len();
    let mut train_offset: Vec<usize> = Vec::with_capacity(n_trains + 1);
    train_offset.push(0);
    for t in &problem.trains {
        train_offset.push(train_offset.last().unwrap() + t.visits.len());
    }
    let n = train_offset[n_trains];

    let mut resource: Vec<usize> = Vec::with_capacity(n);
    let mut travel: Vec<i32> = Vec::with_capacity(n);
    let mut est: Vec<i32> = Vec::with_capacity(n);
    for train in &problem.trains {
        for visit in &train.visits {
            resource.push(visit.resource_id);
            travel.push(visit.travel_time);
            est.push(visit.earliest);
        }
    }
    propagate_est_forward(&mut est, &travel, &problem.trains, &train_offset);

    let mut train_visit: Vec<(usize, usize)> = Vec::with_capacity(n);
    for (t_idx, train) in problem.trains.iter().enumerate() {
        for v_idx in 0..train.visits.len() {
            train_visit.push((t_idx, v_idx));
        }
    }

    let mut conflicts_of: HashMap<usize, Vec<usize>> = HashMap::new();
    for &(a, b) in &problem.conflicts {
        conflicts_of.entry(a).or_default().push(b);
        if a != b {
            conflicts_of.entry(b).or_default().push(a);
        }
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&v| (est[v], v));

    let mut t_greedy: Vec<i32> = vec![0; n];
    let mut last_finish: HashMap<usize, i32> = HashMap::new();

    for &v in &order {
        let r = resource[v];
        let (t_idx, v_idx) = train_visit[v];

        let chain_lb = if v_idx > 0 {
            let prev = train_offset[t_idx] + v_idx - 1;
            t_greedy[prev].saturating_add(travel[prev])
        } else {
            0
        };

        let resource_lb = conflicts_of
            .get(&r)
            .map(|rs| {
                rs.iter()
                    .map(|r2| last_finish.get(r2).copied().unwrap_or(0))
                    .max()
                    .unwrap_or(0)
            })
            .unwrap_or(0);

        let t_v = est[v].max(resource_lb).max(chain_lb);
        t_greedy[v] = t_v;
        let finish = t_v.saturating_add(travel[v]);

        if let Some(rs) = conflicts_of.get(&r) {
            for &r2 in rs {
                let cur = last_finish.entry(r2).or_insert(0);
                if finish > *cur {
                    *cur = finish;
                }
            }
        } else {
            let cur = last_finish.entry(r).or_insert(0);
            if finish > *cur {
                *cur = finish;
            }
        }
    }

    // Scatter linear `t_greedy` back into per-train `[t_v0, t_v1, ...,
    // t_last, t_last + travel_last]` shape.
    problem
        .trains
        .iter()
        .enumerate()
        .map(|(t_idx, train)| {
            let n_visits = train.visits.len();
            let mut times: Vec<i32> = Vec::with_capacity(n_visits + 1);
            for v_idx in 0..n_visits {
                times.push(t_greedy[train_offset[t_idx] + v_idx]);
            }
            if n_visits > 0 {
                let last = train_offset[t_idx] + n_visits - 1;
                times.push(t_greedy[last].saturating_add(travel[last]));
            }
            times
        })
        .collect()
}
