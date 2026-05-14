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
//! No horizon `M` is available in this codebase, so `lst` is not computed
//! and the mandatory-work formula drops the `lst_i + travel_i - t1` term.
//! Deductions are sound but weaker than the full Baptiste rule.

use crate::problem::{Problem, Train};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct ExtendedPrecedence {
    pub est: Vec<Vec<i32>>,
    pub infeasible: bool,
}

const MAX_FIXEDPOINT_ITERS: usize = 20;

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

    propagate_within_train(&mut est, &travel, &problem.trains, &train_offset);

    let clique_visits = build_clique_visits(problem, &resource, n);

    for _ in 0..MAX_FIXEDPOINT_ITERS {
        let est_at_start = est.clone();

        for clique in clique_visits.values() {
            if run_energetic_reasoning(clique, &mut est, &travel) {
                return ExtendedPrecedence {
                    est: scatter_back(&est, &train_offset, &problem.trains),
                    infeasible: true,
                };
            }
        }
        propagate_within_train(&mut est, &travel, &problem.trains, &train_offset);

        if est == est_at_start {
            break;
        }
    }

    ExtendedPrecedence {
        est: scatter_back(&est, &train_offset, &problem.trains),
        infeasible: false,
    }
}

fn propagate_within_train(
    est: &mut [i32],
    travel: &[i32],
    trains: &[Train],
    train_offset: &[usize],
) {
    for (t_idx, train) in trains.iter().enumerate() {
        for v_idx in 1..train.visits.len() {
            let prev = train_offset[t_idx] + v_idx - 1;
            let cur = train_offset[t_idx] + v_idx;
            let bound = est[prev] + travel[prev];
            if est[cur] < bound {
                est[cur] = bound;
            }
        }
    }
}

/// Run unary energetic reasoning on one conflict clique.
/// Returns `true` if infeasibility is detected; otherwise updates `est`
/// in place via the "left-shift adjustment" rule.
fn run_energetic_reasoning(clique: &[usize], est: &mut [i32], travel: &[i32]) -> bool {
    if clique.len() < 2 {
        return false;
    }

    let mut o_pts: Vec<i32> = Vec::with_capacity(2 * clique.len());
    for &i in clique {
        o_pts.push(est[i]);
        o_pts.push(est[i] + travel[i]);
    }
    o_pts.sort_unstable();
    o_pts.dedup();

    let work = |i: usize, t1: i32, t2: i32, est: &[i32]| -> i32 {
        let a = travel[i];
        let b = t2 - t1;
        let c = t2 - est[i];
        a.min(b).min(c).max(0)
    };

    for i1 in 0..o_pts.len() {
        for i2 in (i1 + 1)..o_pts.len() {
            let t1 = o_pts[i1];
            let t2 = o_pts[i2];
            let window = t2 - t1;

            let mut w_sum: i32 = 0;
            for &i in clique {
                w_sum = w_sum.saturating_add(work(i, t1, t2, est));
            }
            if w_sum > window {
                return true;
            }

            // Left-shift adjustment: if j's mandatory work plus the work of
            // all other visits exceeds the window, j cannot complete inside
            // [t1, t2] given est_j ≥ t1, so its earliest start must be pushed
            // past t2 - travel_j.
            for &j in clique {
                let wj = work(j, t1, t2, est);
                let w_other = w_sum - wj;
                if w_other + travel[j] > window
                    && est[j] >= t1
                    && est[j] + travel[j] <= t2
                {
                    let new_est = t2 - travel[j] + 1;
                    if new_est > est[j] {
                        est[j] = new_est;
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
