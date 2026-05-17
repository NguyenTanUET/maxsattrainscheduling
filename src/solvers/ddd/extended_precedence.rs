//! Extended precedence graph + unary energetic reasoning preprocessing.
//!
//! Adapts Bofill, Coll, Suy, Villaret (2020) RCPSP preprocessing and the
//! unary energetic reasoning of Baptiste, Le Pape, Nuijten (2001, ch. 4)
//! to the TRP data model in this codebase.
//!
//! Output is a per-train, per-visit lower bound `est[t][v]` on the start
//! time of each visit, derived from:
//!   - within-train chain `est[v+1] ≥ est[v] + travel[v]`
//!   - unary capacity on each conflict clique of resources (visits that
//!     mutually exclude each other in time cannot all pack into a window
//!     smaller than the sum of their mandatory work)
//! plus an `infeasible` flag set when the unary energetic feasibility test
//! detects no schedule can exist.
//!
//! Horizon: per-visit `lst[v] = est[v] + BIG_M` where `BIG_M = 900` follows
//! the convention from [`crate::solvers::maxsat_ti`] for the same TRP
//! benchmark. Initialization uses chain-propagated `est` (not raw
//! `visit.earliest`) so `lst` stays consistent with within-train precedence
//! on long chains. Sound under the (empirically verified) assumption that
//! cost-optimal schedules of this benchmark have no visit delayed beyond
//! `BIG_M` past its chain-earliest.
//!
//! Greedy-makespan-based `lst` was tried earlier but is unsound for
//! saturating cost functions (`finsteps123`, `infsteps180`) where
//! cost-optimal schedules can have a visit delayed past greedy makespan
//! without extra cost penalty.

use crate::problem::{Problem, Train};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct ExtendedPrecedence {
    pub est: Vec<Vec<i32>>,
    pub infeasible: bool,
}

const MAX_FIXEDPOINT_ITERS: usize = 20;

/// Per-visit horizon offset: `lst[v] = est[v] + BIG_M`. Matches the
/// `big_m` parameter used by [`crate::solvers::maxsat_ti`] for the same
/// benchmark (see main.rs site that passes `900`).
const BIG_M: i32 = 900;

/// Enable the Baptiste–Le Pape–Nuijten left-shift adjustment rule.
///
/// Sound when `lst` is a true upper bound on each visit's start time. With
/// `lst = est + BIG_M`, the rule is sound under the assumption that
/// cost-optimal schedules have no visit delayed beyond BIG_M past its
/// chain-earliest.
const ENABLE_EST_ADJUSTMENT: bool = true;

/// Simple within-train chain propagation of earliest start times.
///
/// For each train, iterate visits in order and set
/// `earliest[v] = max(visit.earliest, earliest[v-1] + travel[v-1])`.
///
/// Sound for ALL objectives — no energetic reasoning, never declares
/// infeasibility. Use this when you want chain-tightened earliest times
/// without the saturating-cost soundness risk of [`compute`].
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

pub fn compute(problem: &Problem) -> ExtendedPrecedence {
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

    // Forward-propagate est first so it reflects within-train chain.
    propagate_est_forward(&mut est, &travel, &problem.trains, &train_offset);

    // Per-visit lst = est + BIG_M (matching the convention used by
    // `maxsat_ti` for the same benchmark: each visit may start at most
    // BIG_M time units after its chain-propagated earliest). Initializing
    // from `est` (post-chain) instead of raw `visit.earliest` keeps lst
    // consistent with within-train precedence (avoids false infeasibility
    // on long chains).
    //
    // Sound assumption: cost-optimal schedule has no visit delayed beyond
    // BIG_M past its chain-earliest. Empirically true on Croella benchmark
    // (`maxsat_ti` uses big_m = 900 successfully).
    let mut lst: Vec<i32> = est.iter().map(|&e| e.saturating_add(BIG_M)).collect();
    propagate_lst_backward(&mut lst, &travel, &problem.trains, &train_offset);

    if has_infeasible_window(&est, &lst) {
        return ExtendedPrecedence {
            est: scatter_back(&est, &train_offset, &problem.trains),
            infeasible: true,
        };
    }

    let clique_visits = build_clique_visits(problem, &resource, n);

    for _ in 0..MAX_FIXEDPOINT_ITERS {
        let est_at_start = est.clone();

        for clique in clique_visits.values() {
            if run_energetic_reasoning(clique, &mut est, &lst, &travel) {
                return ExtendedPrecedence {
                    est: scatter_back(&est, &train_offset, &problem.trains),
                    infeasible: true,
                };
            }
        }
        propagate_est_forward(&mut est, &travel, &problem.trains, &train_offset);
        if has_infeasible_window(&est, &lst) {
            return ExtendedPrecedence {
                est: scatter_back(&est, &train_offset, &problem.trains),
                infeasible: true,
            };
        }

        if est == est_at_start {
            break;
        }
    }

    ExtendedPrecedence {
        est: scatter_back(&est, &train_offset, &problem.trains),
        infeasible: false,
    }
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

fn propagate_lst_backward(
    lst: &mut [i32],
    travel: &[i32],
    trains: &[Train],
    train_offset: &[usize],
) {
    for (t_idx, train) in trains.iter().enumerate() {
        let n_visits = train.visits.len();
        if n_visits < 2 {
            continue;
        }
        for v_idx in (0..n_visits - 1).rev() {
            let cur = train_offset[t_idx] + v_idx;
            let next = train_offset[t_idx] + v_idx + 1;
            // visit v+1 must start by lst[v+1] → visit v must start by lst[v+1] - travel[v]
            let bound = lst[next].saturating_sub(travel[cur]);
            if lst[cur] > bound {
                lst[cur] = bound;
            }
        }
    }
}

fn has_infeasible_window(est: &[i32], lst: &[i32]) -> bool {
    est.iter().zip(lst.iter()).any(|(&e, &l)| e > l)
}

/// Quick greedy list-schedule producing a FEASIBLE solution for the
/// problem. Returns the per-train schedule in the `Vec<Vec<i32>>` shape
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

/// Greedy list schedule: process visits in `est` ascending order; each
/// visit starts at the max of `est[v]`, the latest finish on conflicting
/// resources, and its within-train predecessor's actual greedy finish.
/// Returns a feasible-schedule makespan upper bound.
///
/// Currently unused: greedy makespan was tried as the source for `lst` but
/// is unsound for saturating cost functions (cost-optimal schedule can
/// have makespan > greedy makespan). Kept for potential future use with
/// `DelayCostType::Continuous` where the bound is sound.
#[allow(dead_code)]
fn greedy_makespan_ub(
    problem: &Problem,
    est: &[i32],
    resource: &[usize],
    travel: &[i32],
    train_visit: &[(usize, usize)],
    train_offset: &[usize],
    n: usize,
) -> i32 {
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
    let mut makespan: i32 = 0;
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
        if finish > makespan {
            makespan = finish;
        }
    }
    makespan
}

/// `tail[v]` = travel time from v to end of v's train (inclusive of v).
/// Used to derive `lst[v] = makespan_UB - tail[v]` in the greedy approach;
/// not used in the current BIG_M-based pipeline.
#[allow(dead_code)]
fn compute_tail(
    travel: &[i32],
    trains: &[Train],
    train_offset: &[usize],
    n: usize,
) -> Vec<i32> {
    let mut tail = vec![0; n];
    for (t_idx, train) in trains.iter().enumerate() {
        let n_visits = train.visits.len();
        if n_visits == 0 {
            continue;
        }
        let last = train_offset[t_idx] + n_visits - 1;
        tail[last] = travel[last];
        for v_idx in (0..n_visits - 1).rev() {
            let cur = train_offset[t_idx] + v_idx;
            let next = cur + 1;
            tail[cur] = travel[cur].saturating_add(tail[next]);
        }
    }
    tail
}

/// Run unary energetic reasoning on one conflict clique.
/// Returns `true` if infeasibility is detected; otherwise updates `est`
/// in place via the Baptiste–Le Pape–Nuijten "left-shift adjustment" rule.
///
/// Mandatory work of visit i in window `[t1, t2]` is `min(W_left, W_right)`:
///   W_left  = overlap of [est_i, est_i+p_i] with [t1, t2]
///   W_right = overlap of [lst_i, lst_i+p_i] with [t1, t2]
/// If W_right = 0 the visit can be right-shifted entirely past the window
/// and contributes zero mandatory work (sound).
///
/// Adjustment: if `Σ_{i≠j} w_i + W_left_j > t2 - t1`, then j cannot start
/// as left as est_j; its earliest start is pushed to `t1 + Σ_{i≠j} w_i`.
fn run_energetic_reasoning(
    clique: &[usize],
    est: &mut [i32],
    lst: &[i32],
    travel: &[i32],
) -> bool {
    if clique.len() < 2 {
        return false;
    }

    let mut o_pts: Vec<i32> = Vec::with_capacity(4 * clique.len());
    for &i in clique {
        o_pts.push(est[i]);
        o_pts.push(est[i] + travel[i]);
        o_pts.push(lst[i]);
        o_pts.push(lst[i] + travel[i]);
    }
    o_pts.sort_unstable();
    o_pts.dedup();

    for i1 in 0..o_pts.len() {
        for i2 in (i1 + 1)..o_pts.len() {
            let t1 = o_pts[i1];
            let t2 = o_pts[i2];
            let window = t2 - t1;

            let mut w_sum: i32 = 0;
            for &i in clique {
                let p = travel[i];
                let w_left = ((est[i] + p).min(t2) - est[i].max(t1)).max(0);
                let w_right = ((lst[i] + p).min(t2) - lst[i].max(t1)).max(0);
                w_sum = w_sum.saturating_add(w_left.min(w_right));
            }
            if w_sum > window {
                return true;
            }

            if ENABLE_EST_ADJUSTMENT {
                // For each j, recompute Σ_{i≠j} w_i FRESHLY from current
                // est/lst — pushing one est[j] during this loop changes that
                // visit's mandatory work (typically decreases it as it moves
                // into the valid region). Using a cached `w_sum` from before
                // the j-loop would over-estimate Σ_{i≠j} w_i for later j's,
                // causing the push `t1 + Σ_{i≠j} w_i` to land further than
                // sound. The extra O(|clique|) per j is cheap.
                for &j in clique {
                    let p_j = travel[j];
                    let w_left_j = ((est[j] + p_j).min(t2) - est[j].max(t1)).max(0);

                    let mut w_minus_j: i32 = 0;
                    for &i in clique {
                        if i == j {
                            continue;
                        }
                        let p_i = travel[i];
                        let w_left_i = ((est[i] + p_i).min(t2) - est[i].max(t1)).max(0);
                        let w_right_i = ((lst[i] + p_i).min(t2) - lst[i].max(t1)).max(0);
                        w_minus_j = w_minus_j.saturating_add(w_left_i.min(w_right_i));
                    }

                    if w_minus_j + w_left_j > window {
                        let new_est = t1.saturating_add(w_minus_j);
                        if new_est > est[j] {
                            if new_est > lst[j] {
                                return true;
                            }
                            est[j] = new_est;
                        }
                    }
                }
            }
        }
    }
    false
}

/// Build conflict cliques from `problem.conflicts` via union-find.
/// A clique is included only if every resource in it has a self-conflict
/// `(r, r)` — this is the unary-capacity declaration. Mixed cliques
/// (some unary, some not) are skipped because the energetic feasibility
/// rule assumes every pair of visits in the clique mutually excludes in
/// time.
fn build_clique_visits(
    problem: &Problem,
    resource: &[usize],
    n: usize,
) -> HashMap<usize, Vec<usize>> {
    let max_res = resource.iter().copied().max().map_or(0, |m| m + 1);
    if max_res == 0 {
        return HashMap::new();
    }

    let mut uf: Vec<usize> = (0..max_res).collect();
    let self_unary: HashSet<usize> = problem
        .conflicts
        .iter()
        .filter(|(a, b)| a == b)
        .map(|(a, _)| *a)
        .collect();

    for &(a, b) in &problem.conflicts {
        if a != b && a < max_res && b < max_res {
            let ra = find_root(&mut uf, a);
            let rb = find_root(&mut uf, b);
            if ra != rb {
                uf[ra] = rb;
            }
        }
    }

    let mut resources_per_root: HashMap<usize, Vec<usize>> = HashMap::new();
    for r in 0..max_res {
        let root = find_root(&mut uf, r);
        resources_per_root.entry(root).or_default().push(r);
    }

    let sound_roots: HashSet<usize> = resources_per_root
        .iter()
        .filter(|(_, rs)| rs.iter().all(|r| self_unary.contains(r)))
        .map(|(root, _)| *root)
        .collect();

    let mut clique_visits: HashMap<usize, Vec<usize>> = HashMap::new();
    for v_idx in 0..n {
        let r = resource[v_idx];
        if r >= max_res {
            continue;
        }
        let root = find_root(&mut uf, r);
        if !sound_roots.contains(&root) {
            continue;
        }
        clique_visits.entry(root).or_default().push(v_idx);
    }
    clique_visits
}

fn find_root(uf: &mut [usize], mut x: usize) -> usize {
    while uf[x] != x {
        uf[x] = uf[uf[x]];
        x = uf[x];
    }
    x
}

fn scatter_back(flat: &[i32], train_offset: &[usize], trains: &[Train]) -> Vec<Vec<i32>> {
    trains
        .iter()
        .enumerate()
        .map(|(t_idx, train)| {
            let start = train_offset[t_idx];
            (0..train.visits.len())
                .map(|v_idx| flat[start + v_idx])
                .collect()
        })
        .collect()
}
