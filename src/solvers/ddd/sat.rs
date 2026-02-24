use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap, HashSet},
    time::Instant,
};

use crate::{
    debug::{DebugInfo, ResourceInterval, SolverAction},
    problem::{DelayCostType, Problem},
    solvers::heuristic,
};
use satcoder::{
    constraints::Totalizer, prelude::SymbolicModel, Bool, SatInstance, SatSolverWithCore,
};
use typed_index_collections::TiVec;

use super::{
    common::{do_output_stats, extract_solution, IterationType, Occ, SolveStats, VisitId},
    costtree::CostTree,
    SolverError,
};

#[derive(Clone, Copy, Debug)]
pub enum SatBoundMode {
    /// Ràng buộc UB được thêm vĩnh viễn bằng clause (phù hợp siết UB đơn điệu: UB = UB-1).
    AddClauses,
    /// Ràng buộc UB được áp bằng assumptions (incremental “sạch”, tiện cho thử nhiều UB mà không làm bẩn CNF).
    Assumptions,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SatPrecEncoding {
    Plain,
    Scl,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SatSearchMode {
    UbSearch,
    Invalid,
}

/// NSC (Nested Sorting or Number Series) Pseudo-Boolean Encoder
/// Implements incremental PB encoding sums up to a required threshold (k).
/// It creates bits R_{i, j} where R_{i, j} = 1 means the sum of the first i weights is at least j.
struct NscEncoder<L: satcoder::Lit> {
    // Each element is a vector representing R_{i, j} where j goes from 1 to pos_i.
    // Index i in `r` corresponds to the i-th added weight.
    r: Vec<Vec<Bool<L>>>,
}

impl<L: satcoder::Lit + Copy> NscEncoder<L> {
    fn new() -> Self {
        Self { r: Vec::new() }
    }

    /// Incrementally adds a new variable `x` with positive integer weight `w` to the sum.
    /// `max_k` is the maximum threshold we care about tracking exactly.
    /// If the running sum can exceed `max_k`, the variables stop at `max_k`.
    fn add_variable(
        &mut self,
        solver: &mut impl SatInstance<L>,
        x: Bool<L>,
        w: usize,
        max_k: usize,
    ) {
        if w == 0 {
            return;
        }

        // This is the (i)-th item. In our vectors, it's index `i`.
        let i = self.r.len();

        let prev_pos = if i == 0 { 0 } else { self.r[i - 1].len() };
        let pos_i = std::cmp::min(max_k, prev_pos + w);

        let mut r_i = Vec::with_capacity(pos_i);
        for _ in 0..pos_i {
            r_i.push(SatInstance::new_var(solver));
        }

        // C1: If x is chosen, sum is at least 1..min(w_i, max_k)
        for j in 1..=std::cmp::min(w, max_k) {
            SatInstance::add_clause(solver, vec![!x, r_i[j - 1]]);
        }

        if i > 0 {
            // C2: Copy 1s from previous layer
            for j in 1..=prev_pos {
                SatInstance::add_clause(solver, vec![!self.r[i - 1][j - 1], r_i[j - 1]]);
            }

            // C3: If x is chosen and prev layer reached j, this layer reaches j+w
            for j in 1..=prev_pos {
                if j + w <= pos_i {
                    SatInstance::add_clause(
                        solver,
                        vec![!x, !self.r[i - 1][j - 1], r_i[(j + w) - 1]],
                    );
                }
            }
        }

        self.r.push(r_i);
    }

    /// Returns the exact bound variable indicating whether the sum is >= target.
    /// To enforce sum <= k, we will assert NOT(has_sum_reached(k + 1)).
    fn has_sum_reached(&self, target: usize) -> Option<Bool<L>> {
        if target == 0 {
            return None; /* Always true implicitly, but hard to return a literal */
        }
        let last_layer = self.r.last()?;
        if target <= last_layer.len() {
            Some(last_layer[target - 1])
        } else {
            None // Sum structurally cannot reach `target` yet
        }
    }
}

/// SAT-only version of the DDD Ladder solver.
///
/// Idea:
/// - keep the exact same DDD refinement (time-point generation + conflict clauses)
/// - replace MaxSAT objective by a SAT cardinality constraint on unit-cost ladder vars
/// - solve repeatedly by tightening an upper bound (UB) on total cost
pub fn solve<L: satcoder::Lit + Copy + std::fmt::Debug>(
    mk_env: impl Fn() -> grb::Env + Send + 'static,
    solver: impl SatInstance<L> + SatSolverWithCore<Lit = L> + std::fmt::Debug,
    problem: &Problem,
    timeout: f64,
    delay_cost_type: DelayCostType,
    output_stats: impl FnMut(String, serde_json::Value),
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError> {
    solve_with_mode(
        mk_env,
        solver,
        problem,
        timeout,
        delay_cost_type,
        SatBoundMode::AddClauses,
        output_stats,
    )
}

pub fn solve_incremental<L: satcoder::Lit + Copy + std::fmt::Debug>(
    mk_env: impl Fn() -> grb::Env + Send + 'static,
    solver: impl SatInstance<L> + SatSolverWithCore<Lit = L> + std::fmt::Debug,
    problem: &Problem,
    timeout: f64,
    delay_cost_type: DelayCostType,
    output_stats: impl FnMut(String, serde_json::Value),
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError> {
    solve_debug_with_mode(
        mk_env,
        solver,
        problem,
        timeout,
        delay_cost_type,
        SatBoundMode::Assumptions,
        SatPrecEncoding::Plain,
        SatSearchMode::Invalid,
        |_| {},
        output_stats,
    )
}

pub fn solve_scl<L: satcoder::Lit + Copy + std::fmt::Debug>(
    mk_env: impl Fn() -> grb::Env + Send + 'static,
    solver: impl SatInstance<L> + SatSolverWithCore<Lit = L> + std::fmt::Debug,
    problem: &Problem,
    timeout: f64,
    delay_cost_type: DelayCostType,
    output_stats: impl FnMut(String, serde_json::Value),
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError> {
    solve_with_mode_scl(
        mk_env,
        solver,
        problem,
        timeout,
        delay_cost_type,
        SatBoundMode::Assumptions,
        SatSearchMode::UbSearch,
        output_stats,
    )
}

pub fn solve_incremental_scl<L: satcoder::Lit + Copy + std::fmt::Debug>(
    mk_env: impl Fn() -> grb::Env + Send + 'static,
    solver: impl SatInstance<L> + SatSolverWithCore<Lit = L> + std::fmt::Debug,
    problem: &Problem,
    timeout: f64,
    delay_cost_type: DelayCostType,
    output_stats: impl FnMut(String, serde_json::Value),
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError> {
    solve_with_mode_scl(
        mk_env,
        solver,
        problem,
        timeout,
        delay_cost_type,
        SatBoundMode::Assumptions,
        SatSearchMode::UbSearch,
        output_stats,
    )
}

pub fn solve_with_mode<L: satcoder::Lit + Copy + std::fmt::Debug>(
    mk_env: impl Fn() -> grb::Env + Send + 'static,
    solver: impl SatInstance<L> + SatSolverWithCore<Lit = L> + std::fmt::Debug,
    problem: &Problem,
    timeout: f64,
    delay_cost_type: DelayCostType,
    mode: SatBoundMode,
    output_stats: impl FnMut(String, serde_json::Value),
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError> {
    solve_debug_with_mode(
        mk_env,
        solver,
        problem,
        timeout,
        delay_cost_type,
        mode,
        SatPrecEncoding::Plain,
        SatSearchMode::UbSearch,
        |_| {},
        output_stats,
    )
}

pub fn solve_with_mode_scl<L: satcoder::Lit + Copy + std::fmt::Debug>(
    mk_env: impl Fn() -> grb::Env + Send + 'static,
    solver: impl SatInstance<L> + SatSolverWithCore<Lit = L> + std::fmt::Debug,
    problem: &Problem,
    timeout: f64,
    delay_cost_type: DelayCostType,
    mode: SatBoundMode,
    search: SatSearchMode,
    output_stats: impl FnMut(String, serde_json::Value),
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError> {
    solve_debug_with_mode(
        mk_env,
        solver,
        problem,
        timeout,
        delay_cost_type,
        mode,
        SatPrecEncoding::Scl,
        search,
        |_| {},
        output_stats,
    )
}

thread_local! { pub static WATCH : RefCell<Option<(usize,usize)>> = RefCell::new(None); }

fn inject_solution_timepoints_sat<L: satcoder::Lit>(
    solver: &mut impl SatInstance<L>,
    problem: &Problem,
    train_visit_ids: &[Vec<VisitId>],
    occupations: &mut TiVec<VisitId, Occ<L>>,
    new_time_points: &mut Vec<(VisitId, Bool<L>, i32)>,
    sol: &[Vec<i32>],
) {
    for (train_idx, train) in problem.trains.iter().enumerate() {
        for visit_idx in 0..train.visits.len() {
            let t = sol[train_idx][visit_idx];
            let vid = train_visit_ids[train_idx][visit_idx];
            let (v, is_new) = occupations[vid].time_point(solver, t);
            if is_new {
                new_time_points.push((vid, v, t));
            }
        }
    }
}

pub fn solve_debug<L: satcoder::Lit + Copy + std::fmt::Debug>(
    mk_env: impl Fn() -> grb::Env + Send + 'static,
    solver: impl SatInstance<L> + SatSolverWithCore<Lit = L> + std::fmt::Debug,
    problem: &Problem,
    timeout: f64,
    delay_cost_type: DelayCostType,
    debug_out: impl Fn(DebugInfo),
    output_stats: impl FnMut(String, serde_json::Value),
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError> {
    solve_debug_with_mode(
        mk_env,
        solver,
        problem,
        timeout,
        delay_cost_type,
        SatBoundMode::AddClauses,
        SatPrecEncoding::Plain,
        SatSearchMode::UbSearch,
        debug_out,
        output_stats,
    )
}

pub fn solve_debug_with_mode<L: satcoder::Lit + Copy + std::fmt::Debug>(
    mk_env: impl Fn() -> grb::Env + Send + 'static,
    mut solver: impl SatInstance<L> + SatSolverWithCore<Lit = L> + std::fmt::Debug,
    problem: &Problem,
    timeout: f64,
    delay_cost_type: DelayCostType,
    mode: SatBoundMode,
    prec: SatPrecEncoding,
    search: SatSearchMode,
    debug_out: impl Fn(DebugInfo),
    mut output_stats: impl FnMut(String, serde_json::Value),
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError> {
    let _p = hprof::enter("sat_solver");
    let search_label = match search {
        SatSearchMode::UbSearch => "ub_search",
        SatSearchMode::Invalid => "invalid",
    };
    println!("SAT search mode: {}", search_label);

    let start_time: Instant = Instant::now();
    let mut solver_time = std::time::Duration::ZERO;
    let mut stats = SolveStats::default();

    let mut visits: TiVec<VisitId, (usize, usize)> = TiVec::new();
    let mut train_visit_ids: Vec<Vec<VisitId>> = vec![Vec::new(); problem.trains.len()];
    let mut resource_visits: Vec<Vec<VisitId>> = Vec::new();
    let mut occupations: TiVec<VisitId, Occ<_>> = TiVec::new();
    let mut touched_intervals = Vec::new();
    let mut conflicts: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut new_time_points: Vec<(VisitId, Bool<L>, i32)> = Vec::new();

    let mut iteration_types: BTreeMap<IterationType, usize> = BTreeMap::new();

    let mut n_timepoints = 0usize;
    let mut n_conflict_constraints = 0usize;

    for (a, b) in problem.conflicts.iter() {
        conflicts.entry(*a).or_default().push(*b);
        if *a != *b {
            conflicts.entry(*b).or_default().push(*a);
        }
    }

    for (train_idx, train) in problem.trains.iter().enumerate() {
        for (visit_idx, visit) in train.visits.iter().enumerate() {
            let visit_id: VisitId = visits.push_and_get_key((train_idx, visit_idx));
            train_visit_ids[train_idx].push(visit_id);

            occupations.push(Occ {
                cost: vec![true.into()],
                cost_tree: CostTree::new(),
                delays: vec![(true.into(), visit.earliest), (false.into(), i32::MAX)],
                incumbent_idx: 0,
            });
            n_timepoints += 1;

            while resource_visits.len() <= visit.resource_id {
                resource_visits.push(Vec::new());
            }

            resource_visits[visit.resource_id].push(visit_id);
            touched_intervals.push(visit_id);
            new_time_points.push((visit_id, true.into(), visit.earliest));
        }
    }

    // Search state.
    let mut best_sol: Option<(i32, Vec<Vec<i32>>)> = None;
    let mut lower_bound: i32 = 0;
    // Upper bound on optimal cost (exclusive). When None, no bound yet.
    let mut upper_bound: Option<i32> = None;

    // NSC Encoder state (replaces Totalizer & unary ladder `budget_units`)
    let mut nsc_enc: NscEncoder<L> = NscEncoder::new();
    let mut nsc_current_max_k = 0; // We keep track of the maximum Ub we've built the encoder up to

    // Conflict-choice vars (optional; currently unused because USE_CHOICE_VAR=false below).
    let mut conflict_vars: HashMap<(VisitId, VisitId), Bool<L>> = Default::default();
    // Rows already added for SCL fixed-precedence encoding: (visit_id, time).
    let mut scl_fixed_prec_rows: HashSet<(VisitId, i32)> = HashSet::new();

    // Optional: seed fixed-precedence (travel-time) constraints from the earliest time points.
    const SEED_SCL_FROM_EARLIEST: bool = true;
    if prec == SatPrecEncoding::Scl && SEED_SCL_FROM_EARLIEST {
        for visit_id in visits.keys() {
            let (train_idx, visit_idx) = visits[visit_id];
            if visit_idx + 1 >= problem.trains[train_idx].visits.len() {
                continue;
            }
            let (in_var, in_t) = occupations[visit_id].delays[0];
            add_fixed_precedence_scl(
                &mut solver,
                problem,
                &visits,
                &mut occupations,
                &mut new_time_points,
                &mut scl_fixed_prec_rows,
                visit_id,
                in_var,
                in_t,
            );
        }
    }

    // Heuristic thread: produces feasible UB solutions (cost, solution).
    const USE_HEURISTIC: bool = true;
    let heur_thread = USE_HEURISTIC.then(|| {
        let (sol_in_tx, sol_in_rx) = std::sync::mpsc::channel();
        let (sol_out_tx, sol_out_rx) = std::sync::mpsc::channel();
        let problem = problem.clone();
        heuristic::spawn_heuristic_thread(mk_env, sol_in_rx, problem, delay_cost_type, sol_out_tx);
        (sol_in_tx, sol_out_rx)
    });
    let mut heur_active = USE_HEURISTIC;

    let mut iteration: usize = 1;
    let mut is_sat: bool = true;

    // Preallocate to avoid reallocation inside the loop
    let mut invalid_clause: Vec<Bool<L>> = Vec::new();

    loop {
        let mut bound_assumption: Option<Bool<L>> = None;
        let mut bound_used: Option<i32> = None;
        if start_time.elapsed().as_secs_f64() > timeout {
            let ub = best_sol.as_ref().map(|(c, _)| *c).unwrap_or(i32::MAX);
            let lb = lower_bound;
            println!("TIMEOUT LB={} UB={}", lb, ub);

            do_output_stats(
                &mut output_stats,
                iteration,
                &iteration_types,
                &stats,
                &occupations,
                start_time,
                solver_time,
                lb,
                ub,
            );
            return Err(SolverError::Timeout);
        }

        if is_sat {
            // Send current incumbent to heuristic and read improved UBs.
            if heur_active {
                if let Some((sol_tx, sol_rx)) = heur_thread.as_ref() {
                    let sol = extract_solution(problem, &occupations);
                    let _ = sol_tx.send(sol);

                    while let Ok((ub_cost, ub_sol)) = sol_rx.try_recv() {
                        if best_sol.as_ref().map(|(c, _)| ub_cost < *c).unwrap_or(true) {
                            best_sol = Some((ub_cost, ub_sol.clone()));
                        }

                        if search == SatSearchMode::UbSearch {
                            // Use heuristic solution as a starting UB (search for strictly better).
                            let candidate_ub = ub_cost - 1;
                            upper_bound = Some(
                                upper_bound
                                    .map(|b| b.min(candidate_ub))
                                    .unwrap_or(candidate_ub),
                            );
                        }

                        // Make sure these time points exist in the encoding.
                        inject_solution_timepoints_sat(
                            &mut solver,
                            problem,
                            &train_visit_ids,
                            &mut occupations,
                            &mut new_time_points,
                            &ub_sol,
                        );
                    }
                }
            }

            let mut found_travel_time_conflict = false;
            let mut found_resource_conflict = false;

            // ----- Travel time conflicts -----
            for visit_id in touched_intervals.iter().copied() {
                let (train_idx, visit_idx) = visits[visit_id];
                let next_visit: Option<VisitId> =
                    if visit_idx + 1 < problem.trains[train_idx].visits.len() {
                        Some((usize::from(visit_id) + 1).into())
                    } else {
                        None
                    };

                let t1_in = occupations[visit_id].incumbent_time();
                let visit = problem.trains[train_idx].visits[visit_idx];

                if let Some(next_visit) = next_visit {
                    let v1 = &occupations[visit_id];
                    let v2 = &occupations[next_visit];
                    let t1_out = v2.incumbent_time();

                    if t1_in + visit.travel_time > t1_out {
                        found_travel_time_conflict = true;

                        // Debug info
                        let mut debug_actions = Vec::new();
                        debug_actions.push(SolverAction::TravelTimeConflict(ResourceInterval {
                            train_idx,
                            visit_idx,
                            resource_idx: visit.resource_id,
                            time_in: t1_in,
                            time_out: t1_out,
                        }));
                        debug_out(DebugInfo {
                            iteration,
                            actions: debug_actions,
                            solution: extract_solution(problem, &occupations),
                        });

                        if prec == SatPrecEncoding::Scl {
                            let in_var = v1.delays[v1.incumbent_idx].0;
                            let in_t = v1.incumbent_time();
                            add_fixed_precedence_scl(
                                &mut solver,
                                problem,
                                &visits,
                                &mut occupations,
                                &mut new_time_points,
                                &mut scl_fixed_prec_rows,
                                visit_id,
                                in_var,
                                in_t,
                            );
                            stats.n_travel += 1;
                        } else {
                            let t1_in_var = v1.delays[v1.incumbent_idx].0;
                            let new_t = v1.incumbent_time() + visit.travel_time;
                            let (t1_earliest_out_var, t1_is_new) =
                                occupations[next_visit].time_point(&mut solver, new_t);

                            SatInstance::add_clause(
                                &mut solver,
                                vec![!t1_in_var, t1_earliest_out_var],
                            );
                            stats.n_travel += 1;

                            if t1_is_new {
                                new_time_points.push((next_visit, t1_earliest_out_var, new_t));
                            }
                        }
                    }
                }
            }

            // ----- Resource conflicts via Sweep Line -----
            let mut deconflicted_train_pairs: HashSet<(usize, usize)> = HashSet::new();

            // To run a sweep line, we first collect all intervals grouped by resource.
            // interval: [time_in, time_out)
            #[derive(Debug, Clone, Copy, PartialEq, Eq)]
            struct SweepEvent {
                time: i32,
                is_start: bool,
                visit_id: VisitId,
                train_idx: usize,
                visit_idx: usize,
            }

            // We need a predictable ordering for SweepEvent. We sort primarily by time.
            // If times are equal, END events (is_start = false) must process BEFORE START events
            // so we don't flag [A, B) and [B, C) as overlapping.
            impl Ord for SweepEvent {
                fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                    self.time
                        .cmp(&other.time)
                        .then_with(|| match (self.is_start, other.is_start) {
                            (true, true) => std::cmp::Ordering::Equal,
                            (false, false) => std::cmp::Ordering::Equal,
                            (false, true) => std::cmp::Ordering::Less, // END before START
                            (true, false) => std::cmp::Ordering::Greater,
                        })
                        .then_with(|| self.visit_id.cmp(&other.visit_id))
                }
            }
            impl PartialOrd for SweepEvent {
                fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                    Some(self.cmp(other))
                }
            }

            let mut retain_touched = vec![false; touched_intervals.len()];

            // Only process resources that actually have potential conflicts.
            for (&conflicting_resource, conflicting_set) in conflicts.iter() {
                // Determine if any touched interval requires us to check this resource set
                let mut needs_check = false;
                for visit_id in touched_intervals.iter() {
                    let (train_idx, visit_idx) = visits[*visit_id];
                    let r = problem.trains[train_idx].visits[visit_idx].resource_id;
                    if r == conflicting_resource || conflicting_set.contains(&r) {
                        needs_check = true;
                        break;
                    }
                }
                if !needs_check {
                    continue;
                }

                // Collect all current intervals for the conflicting_resource and its conflicting_set
                let mut events = Vec::new();
                let mut relevant_resources = conflicting_set.clone();
                relevant_resources.push(conflicting_resource);

                for &res in &relevant_resources {
                    if res >= resource_visits.len() {
                        continue;
                    }
                    for &visit_id in &resource_visits[res] {
                        let (train_idx, visit_idx) = visits[visit_id];
                        let t_in = occupations[visit_id].incumbent_time();

                        let next_visit: Option<VisitId> =
                            if visit_idx + 1 < problem.trains[train_idx].visits.len() {
                                Some((usize::from(visit_id) + 1).into())
                            } else {
                                None
                            };
                        let t_out = next_visit
                            .map(|nx| occupations[nx].incumbent_time())
                            .unwrap_or(
                                t_in + problem.trains[train_idx].visits[visit_idx].travel_time,
                            );

                        events.push(SweepEvent {
                            time: t_in,
                            is_start: true,
                            visit_id,
                            train_idx,
                            visit_idx,
                        });
                        events.push(SweepEvent {
                            time: t_out,
                            is_start: false,
                            visit_id,
                            train_idx,
                            visit_idx,
                        });
                    }
                }

                events.sort();

                let mut active_intervals: Vec<SweepEvent> = Vec::new();

                for ev in events {
                    if ev.is_start {
                        // Check against all currently active intervals
                        for active in &active_intervals {
                            if active.train_idx == ev.train_idx {
                                continue;
                            }

                            // Conflict found between `ev` and `active`
                            let v1_id = ev.visit_id;
                            let v2_id = active.visit_id;

                            // We only process it if it's a new pair we haven't deconflicted
                            if !deconflicted_train_pairs.insert((ev.train_idx, active.train_idx))
                                || !deconflicted_train_pairs
                                    .insert((active.train_idx, ev.train_idx))
                            {
                                // Mark touched to retain
                                for (i, t_vid) in touched_intervals.iter().enumerate() {
                                    if *t_vid == v1_id || *t_vid == v2_id {
                                        retain_touched[i] = true;
                                    }
                                }
                                continue;
                            }

                            found_resource_conflict = true;
                            stats.n_conflict += 1;

                            // Re-calculate bounds for clause generation
                            let (train_idx1, visit_idx1) = (ev.train_idx, ev.visit_idx);
                            let t1_in = occupations[v1_id].incumbent_time();
                            let t1_out = if visit_idx1 + 1 < problem.trains[train_idx1].visits.len()
                            {
                                occupations[VisitId::from(usize::from(v1_id) + 1)].incumbent_time()
                            } else {
                                t1_in + problem.trains[train_idx1].visits[visit_idx1].travel_time
                            };

                            let (train_idx2, visit_idx2) = (active.train_idx, active.visit_idx);
                            let t2_in = occupations[v2_id].incumbent_time();
                            let t2_out = if visit_idx2 + 1 < problem.trains[train_idx2].visits.len()
                            {
                                occupations[VisitId::from(usize::from(v2_id) + 1)].incumbent_time()
                            } else {
                                t2_in + problem.trains[train_idx2].visits[visit_idx2].travel_time
                            };

                            let (delay_t2, t2_is_new) =
                                occupations[v2_id].time_point(&mut solver, t1_out);
                            let (delay_t1, t1_is_new) =
                                occupations[v1_id].time_point(&mut solver, t2_out);

                            if t1_is_new {
                                new_time_points.push((v1_id, delay_t1, t2_out));
                            }
                            if t2_is_new {
                                new_time_points.push((v2_id, delay_t2, t1_out));
                            }

                            if prec == SatPrecEncoding::Scl {
                                add_fixed_precedence_scl(
                                    &mut solver,
                                    problem,
                                    &visits,
                                    &mut occupations,
                                    &mut new_time_points,
                                    &mut scl_fixed_prec_rows,
                                    v1_id,
                                    delay_t1,
                                    t2_out,
                                );
                                add_fixed_precedence_scl(
                                    &mut solver,
                                    problem,
                                    &visits,
                                    &mut occupations,
                                    &mut new_time_points,
                                    &mut scl_fixed_prec_rows,
                                    v2_id,
                                    delay_t2,
                                    t1_out,
                                );
                            }

                            let t1_out_lit =
                                if visit_idx1 + 1 < problem.trains[train_idx1].visits.len() {
                                    let nx = VisitId::from(usize::from(v1_id) + 1);
                                    occupations[nx].delays[occupations[nx].incumbent_idx].0
                                } else {
                                    true.into()
                                };

                            let t2_out_lit =
                                if visit_idx2 + 1 < problem.trains[train_idx2].visits.len() {
                                    let nx = VisitId::from(usize::from(v2_id) + 1);
                                    occupations[nx].delays[occupations[nx].incumbent_idx].0
                                } else {
                                    true.into()
                                };

                            const USE_CHOICE_VAR: bool = true;
                            n_conflict_constraints += 1;

                            if USE_CHOICE_VAR {
                                let (pa, pb) = (v1_id, v2_id);
                                let choose =
                                    conflict_vars.get(&(pa, pb)).copied().unwrap_or_else(|| {
                                        let new_var = SatInstance::new_var(&mut solver);
                                        conflict_vars.insert((pa, pb), new_var);
                                        conflict_vars.insert((pb, pa), !new_var);
                                        new_var
                                    });
                                SatInstance::add_clause(
                                    &mut solver,
                                    vec![!choose, !t1_out_lit, delay_t2],
                                );
                                SatInstance::add_clause(
                                    &mut solver,
                                    vec![choose, !t2_out_lit, delay_t1],
                                );
                            } else {
                                SatInstance::add_clause(
                                    &mut solver,
                                    vec![!t1_out_lit, !t2_out_lit, delay_t1, delay_t2],
                                );
                            }
                        }
                        active_intervals.push(ev);
                    } else {
                        // Remove from active intervals
                        if let Some(pos) = active_intervals
                            .iter()
                            .position(|x| x.visit_id == ev.visit_id)
                        {
                            active_intervals.remove(pos);
                        }
                    }
                }
            }

            let mut new_touched = Vec::new();
            for (i, vid) in touched_intervals.into_iter().enumerate() {
                if retain_touched[i] {
                    new_touched.push(vid);
                }
            }
            touched_intervals = new_touched;

            let iterationtype = if found_travel_time_conflict && found_resource_conflict {
                IterationType::TravelAndResourceConflict
            } else if found_travel_time_conflict {
                IterationType::TravelTimeConflict
            } else if found_resource_conflict {
                IterationType::ResourceConflict
            } else {
                IterationType::Solution
            };
            *iteration_types.entry(iterationtype).or_default() += 1;

            // If there are no conflicts, current incumbent is a feasible schedule for the current discretization.
            if !(found_resource_conflict || found_travel_time_conflict) {
                let sol = extract_solution(problem, &occupations);
                let cost = problem.cost(&sol, delay_cost_type);

                if best_sol.as_ref().map(|(c, _)| cost < *c).unwrap_or(true) {
                    best_sol = Some((cost, sol.clone()));
                }

                if search == SatSearchMode::UbSearch {
                    // Tighten UB to search for a strictly better solution.
                    let candidate_ub = cost - 1;
                    upper_bound = Some(
                        upper_bound
                            .map(|b| b.min(candidate_ub))
                            .unwrap_or(candidate_ub),
                    );
                }

                debug_out(DebugInfo {
                    iteration,
                    actions: Vec::new(),
                    solution: sol,
                });

                // If we cannot improve further, we can stop.
                if search == SatSearchMode::UbSearch {
                    if let Some(ub) = upper_bound {
                        if ub < lower_bound {
                            let (c, s) = best_sol.clone().unwrap();
                            stats.satsolver = format!("{:?}", solver);
                            do_output_stats(
                                &mut output_stats,
                                iteration,
                                &iteration_types,
                                &stats,
                                &occupations,
                                start_time,
                                solver_time,
                                c,
                                c,
                            );
                            println!("SAT OPTIMAL (cost={})", c);
                            return Ok((s, stats));
                        }
                    }
                } else if search == SatSearchMode::Invalid {
                    // Block the current schedule (timepoint boundaries only).
                    invalid_clause.clear();
                    for occ in occupations.iter() {
                        let idx = occ.incumbent_idx;
                        let (lit_at, _) = occ.delays[idx];
                        invalid_clause.push(!lit_at);
                        if idx + 1 < occ.delays.len() {
                            let (lit_next, _) = occ.delays[idx + 1];
                            invalid_clause.push(lit_next);
                        }
                    }
                    if !invalid_clause.is_empty() {
                        SatInstance::add_clause(&mut solver, invalid_clause.clone());
                    }
                }
            }
        }

        // ----- Encode costs for newly-created time points -----
        // SAT budget now uses the NSC Pseudo-Boolean encoding
        const USE_COST_TREE: bool = false;

        // If we know a strict UB, we only need to build NSC up to UB.
        // If not, we build up to an arbitrary reasonable max initially (e.g. 500)
        let nsc_target_k = upper_bound.map(|ub| ub as usize).unwrap_or(2000);
        nsc_current_max_k = std::cmp::max(nsc_current_max_k, nsc_target_k);

        for (visit, new_timepoint_var, new_t) in new_time_points.drain(..) {
            n_timepoints += 1;
            let (train_idx, visit_idx) = visits[visit];

            let new_timepoint_cost =
                problem.trains[train_idx].visit_delay_cost(delay_cost_type, visit_idx, new_t);

            if new_timepoint_cost > 0 {
                if !USE_COST_TREE {
                    // We only need ONE boolean variable to represent this decision reaching this delay.
                    // Actually, `new_timepoint_var` ALREADY represents the decision "time >= new_t".
                    // But our cost model incurs `new_timepoint_cost` if time >= new_t AND time < new_t_next.
                    // For the pb formulation, we just add `new_timepoint_var` as having weight `new_timepoint_cost`?
                    // Actually, the DDD ladder encoding says:
                    // cost occurs if var is true. But costs stack!
                    // If t=1 has cost 1, t=2 has cost 2, then var(t>=1) contributes 1, var(t>=2) contributes 1.
                    // So we must figure out the *marginal* cost of this timepoint (cost at new_t - cost at previous t).

                    let prev_cost = if occupations[visit].cost.len() > 1 {
                        occupations[visit].cost.len() - 1 // the cost at the state we were previously tracking
                    } else {
                        0
                    };

                    // The marginal increase in cost from the previous timepoint to this one
                    let marginal_cost = if new_timepoint_cost as usize > prev_cost {
                        new_timepoint_cost as usize - prev_cost
                    } else {
                        0
                    };

                    for cost in occupations[visit].cost.len()..=new_timepoint_cost as usize {
                        let prev_cost_var = occupations[visit].cost[cost - 1];
                        let next_cost_var = SatInstance::new_var(&mut solver);
                        SatInstance::add_clause(&mut solver, vec![!next_cost_var, prev_cost_var]);
                        occupations[visit].cost.push(next_cost_var);

                        // Feed each unit weight into NSC (or we could feed the single `next_cost_var` with weight 1)
                        // This exactly mirrors `budget_units.push` linearly but pipes into the PB encoder directly!
                        nsc_enc.add_variable(&mut solver, next_cost_var, 1, nsc_current_max_k);
                    }

                    SatInstance::add_clause(
                        &mut solver,
                        vec![
                            !new_timepoint_var,
                            occupations[visit].cost[new_timepoint_cost as usize],
                        ],
                    );
                } else {
                    let _ = new_timepoint_var;
                    let _ = new_t;
                    let _ = new_timepoint_cost;
                }
            }
        }

        // ----- Enforce budget UB (if known) -----
        if search == SatSearchMode::UbSearch {
            if let Some(ub) = upper_bound {
                if ub < lower_bound {
                    // Search space exhausted.
                    if let Some((c, s)) = best_sol.clone() {
                        stats.n_unsat += 1;
                        stats.satsolver = format!("{:?}", solver);
                        do_output_stats(
                            &mut output_stats,
                            iteration,
                            &iteration_types,
                            &stats,
                            &occupations,
                            start_time,
                            solver_time,
                            c,
                            c,
                        );
                        return Ok((s, stats));
                    }
                    return Err(SolverError::NoSolution);
                }

                let target_ub = match mode {
                    SatBoundMode::AddClauses => ub,
                    SatBoundMode::Assumptions => (lower_bound + ub) / 2,
                };
                let ub_usize = target_ub as usize;

                // Enforce sum(budget_units) <= target_ub
                // This means the sum cannot reach target_ub + 1
                if let Some(reached_var) = nsc_enc.has_sum_reached(ub_usize + 1) {
                    let bound_lit = !reached_var;
                    bound_used = Some(target_ub);
                    match mode {
                        SatBoundMode::AddClauses => {
                            SatInstance::add_clause(&mut solver, vec![bound_lit]);
                        }
                        SatBoundMode::Assumptions => {
                            bound_assumption = Some(bound_lit);
                        }
                    }
                } else {
                    // The encoder structurally hasn't even grown to `ub_usize + 1` yet,
                    // which means the sum is trivially <= target_ub based purely on the number of variables added so far.
                    bound_used = Some(target_ub);
                }
            }
        }

        // ----- Solve SAT -----
        *iteration_types.entry(IterationType::Objective).or_default() += 1;

        let solver_debug = format!("{:?}", solver);
        let solve_start = Instant::now();
        let result =
            SatSolverWithCore::solve_with_assumptions(&mut solver, bound_assumption.into_iter());
        solver_time += solve_start.elapsed();

        match result {
            satcoder::SatResultWithCore::Sat(model) => {
                is_sat = true;
                stats.n_sat += 1;

                // Update incumbents from the model and mark touched intervals.
                for (visit, this_occ) in occupations.iter_mut_enumerated() {
                    let mut touched = false;

                    while model.value(&this_occ.delays[this_occ.incumbent_idx + 1].0) {
                        this_occ.incumbent_idx += 1;
                        touched = true;
                    }
                    while !model.value(&this_occ.delays[this_occ.incumbent_idx].0) {
                        this_occ.incumbent_idx -= 1;
                        touched = true;
                    }

                    let (_, visit_idx) = visits[visit];
                    if touched {
                        if visit_idx > 0 {
                            let prev_visit = (Into::<usize>::into(visit) - 1).into();
                            if touched_intervals.last() != Some(&prev_visit) {
                                touched_intervals.push(prev_visit);
                            }
                        }
                        touched_intervals.push(visit);
                    }
                }
            }
            satcoder::SatResultWithCore::Unsat(_core) => {
                is_sat = false;
                stats.n_unsat += 1;
                if search == SatSearchMode::Invalid {
                    if let Some((c, s)) = best_sol.clone() {
                        stats.satsolver = solver_debug;
                        do_output_stats(
                            &mut output_stats,
                            iteration,
                            &iteration_types,
                            &stats,
                            &occupations,
                            start_time,
                            solver_time,
                            c,
                            c,
                        );
                        return Ok((s, stats));
                    }
                    return Err(SolverError::NoSolution);
                }

                if let Some(bound) = bound_used {
                    lower_bound = bound + 1;
                    if let (Some((c, s)), Some(ub)) = (best_sol.clone(), upper_bound) {
                        if ub < lower_bound {
                            stats.satsolver = solver_debug;
                            do_output_stats(
                                &mut output_stats,
                                iteration,
                                &iteration_types,
                                &stats,
                                &occupations,
                                start_time,
                                solver_time,
                                c,
                                c,
                            );
                            println!("SAT OPTIMAL (cost={})", c);
                            return Ok((s, stats));
                        }
                    }
                    iteration += 1;
                    continue;
                }

                return Err(SolverError::NoSolution);
            }
        }

        iteration += 1;
    }
}

/// Add the SCL-compressed fixed-precedence constraint for a single time point.
///
/// Given an in-visit `visit_id` on train i at time t with ladder literal `in_var` (= "time >= t"),
/// we enforce that the next visit on the same train must satisfy time >= t + travel_time.
fn add_fixed_precedence_scl<L: satcoder::Lit>(
    solver: &mut impl SatInstance<L>,
    problem: &Problem,
    visits: &TiVec<VisitId, (usize, usize)>,
    occupations: &mut TiVec<VisitId, Occ<L>>,
    new_time_points: &mut Vec<(VisitId, Bool<L>, i32)>,
    added: &mut HashSet<(VisitId, i32)>,
    visit_id: VisitId,
    in_var: Bool<L>,
    in_t: i32,
) {
    if !added.insert((visit_id, in_t)) {
        return;
    }

    let (train_idx, visit_idx) = visits[visit_id];
    if visit_idx + 1 >= problem.trains[train_idx].visits.len() {
        return;
    }

    let travel = problem.trains[train_idx].visits[visit_idx].travel_time;
    let next_visit: VisitId = (usize::from(visit_id) + 1).into();
    let req_t = in_t + travel;

    let earliest_next = occupations[next_visit].delays[0].1;
    if req_t <= earliest_next {
        return;
    }

    let (req_var, is_new) = occupations[next_visit].time_point(solver, req_t);
    // For small cliques, use pairwise clauses; otherwise use SCL (single implication).
    const SCL_PAIRWISE_THRESHOLD: usize = 5;
    let idx = occupations[next_visit]
        .delays
        .partition_point(|(_, t0)| *t0 < req_t);
    if idx <= SCL_PAIRWISE_THRESHOLD {
        for i in 0..idx {
            let lit_i = occupations[next_visit].delays[i].0;
            let lit_next = occupations[next_visit].delays[i + 1].0;
            solver.add_clause(vec![!in_var, !lit_i, lit_next]);
        }
    } else {
        solver.add_clause(vec![!in_var, req_var]);
    }
    if is_new {
        new_time_points.push((next_visit, req_var, req_t));
    }
}
