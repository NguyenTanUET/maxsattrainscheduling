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
use satcoder::{constraints::Totalizer, prelude::SymbolicModel, Bool, SatInstance, SatSolverWithCore};
use typed_index_collections::TiVec;

use super::{common::{do_output_stats, extract_solution, IterationType, Occ, SolveStats, VisitId}, costtree::CostTree, SolverError};

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
    solve_debug(
        mk_env,
        solver,
        problem,
        timeout,
        delay_cost_type,
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
    mut solver: impl SatInstance<L> + SatSolverWithCore<Lit = L> + std::fmt::Debug,
    problem: &Problem,
    timeout: f64,
    delay_cost_type: DelayCostType,
    debug_out: impl Fn(DebugInfo),
    mut output_stats: impl FnMut(String, serde_json::Value),
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError> {
    let _p = hprof::enter("sat_solver");

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

    // Search state (SAT with decreasing UB).
    let mut best_sol: Option<(i32, Vec<Vec<i32>>)> = None;
    // Current budget (<=). When None, we haven't got any UB yet.
    let mut budget_ub: Option<i32> = None;

    // Unit-cost ladder vars used for the budget constraint.
    // Each pushed var corresponds to +1 cost.
    let mut budget_units: Vec<Bool<L>> = Vec::new();

    // Cache a totalizer built over `budget_units` up to `budget_tot_max_bound`.
    let mut budget_tot: Option<Totalizer<L>> = None;
    let mut budget_tot_len: usize = 0;
    let mut budget_tot_max_bound: usize = 0;

    // Conflict-choice vars (optional; currently unused because USE_CHOICE_VAR=false below).
    let mut conflict_vars: HashMap<(VisitId, VisitId), Bool<L>> = Default::default();

    // Heuristic thread: produces feasible UB solutions (cost, solution).
    const USE_HEURISTIC: bool = true;
    let heur_thread = USE_HEURISTIC.then(|| {
        let (sol_in_tx, sol_in_rx) = std::sync::mpsc::channel();
        let (sol_out_tx, sol_out_rx) = std::sync::mpsc::channel();
        let problem = problem.clone();
        heuristic::spawn_heuristic_thread(mk_env, sol_in_rx, problem, delay_cost_type, sol_out_tx);
        (sol_in_tx, sol_out_rx)
    });

    let mut iteration: usize = 1;
    let mut is_sat: bool = true;

    loop {
        if start_time.elapsed().as_secs_f64() > timeout {
            let ub = best_sol.as_ref().map(|(c, _)| *c).unwrap_or(i32::MAX);
            let lb = budget_ub.map(|b| b + 1).unwrap_or(0);
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
            if let Some((sol_tx, sol_rx)) = heur_thread.as_ref() {
                let sol = extract_solution(problem, &occupations);
                let _ = sol_tx.send(sol);

                while let Ok((ub_cost, ub_sol)) = sol_rx.try_recv() {
                    if best_sol.as_ref().map(|(c, _)| ub_cost < *c).unwrap_or(true) {
                        best_sol = Some((ub_cost, ub_sol.clone()));
                    }

                    // Use heuristic solution as a starting UB.
                    budget_ub = Some(budget_ub.map(|b| b.min(ub_cost)).unwrap_or(ub_cost));

                    // Make sure these time points exist in the encoding.
                    inject_solution_timepoints_sat(&mut solver, problem, &train_visit_ids, &mut occupations, &mut new_time_points, &ub_sol);
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

                        let t1_in_var = v1.delays[v1.incumbent_idx].0;
                        let new_t = v1.incumbent_time() + visit.travel_time;
                        let (t1_earliest_out_var, t1_is_new) =
                            occupations[next_visit].time_point(&mut solver, new_t);

                        SatInstance::add_clause(&mut solver, vec![!t1_in_var, t1_earliest_out_var]);
                        stats.n_travel += 1;

                        if t1_is_new {
                            new_time_points.push((next_visit, t1_earliest_out_var, new_t));
                        }
                    }
                }
            }

            // ----- Resource conflicts -----
            let mut deconflicted_train_pairs: HashSet<(usize, usize)> = HashSet::new();
            touched_intervals.retain(|visit_id| {
                let visit_id = *visit_id;
                let (train_idx, visit_idx) = visits[visit_id];
                let next_visit: Option<VisitId> =
                    if visit_idx + 1 < problem.trains[train_idx].visits.len() {
                        Some((usize::from(visit_id) + 1).into())
                    } else {
                        None
                    };

                let t1_in = occupations[visit_id].incumbent_time();
                let visit = problem.trains[train_idx].visits[visit_idx];

                let mut retain = false;

                if let Some(conflicting_resources) = conflicts.get(&visit.resource_id) {
                    for other_resource in conflicting_resources.iter().copied() {
                        let t1_out = next_visit
                            .map(|nx| occupations[nx].incumbent_time())
                            .unwrap_or(t1_in + visit.travel_time);

                        for other_visit in resource_visits[other_resource].iter().copied() {
                            if usize::from(visit_id) == usize::from(other_visit) {
                                continue;
                            }
                            let v2 = &occupations[other_visit];
                            let t2_in = v2.incumbent_time();
                            let (other_train_idx, other_visit_idx) = visits[other_visit];

                            if other_train_idx == train_idx {
                                continue;
                            }

                            let other_next_visit: Option<VisitId> = if other_visit_idx + 1
                                < problem.trains[other_train_idx].visits.len()
                            {
                                Some((usize::from(other_visit) + 1).into())
                            } else {
                                None
                            };

                            let t2_out = other_next_visit
                                .map(|v| occupations[v].incumbent_time())
                                .unwrap_or_else(|| {
                                    let other_visit =
                                        problem.trains[other_train_idx].visits[other_visit_idx];
                                    t2_in + other_visit.travel_time
                                });

                            if t1_out <= t2_in || t2_out <= t1_in {
                                continue;
                            }

                            if !deconflicted_train_pairs.insert((train_idx, other_train_idx))
                                || !deconflicted_train_pairs.insert((other_train_idx, train_idx))
                            {
                                retain = true;
                                continue;
                            }

                            found_resource_conflict = true;
                            stats.n_conflict += 1;

                            let (delay_t2, t2_is_new) =
                                occupations[other_visit].time_point(&mut solver, t1_out);
                            let (delay_t1, t1_is_new) =
                                occupations[visit_id].time_point(&mut solver, t2_out);

                            if t1_is_new {
                                new_time_points.push((visit_id, delay_t1, t2_out));
                            }
                            if t2_is_new {
                                new_time_points.push((other_visit, delay_t2, t1_out));
                            }

                            let v1 = &occupations[visit_id];
                            let v2 = &occupations[other_visit];

                            let t1_out_lit = next_visit
                                .map(|v| occupations[v].delays[occupations[v].incumbent_idx].0)
                                .unwrap_or_else(|| true.into());
                            let t2_out_lit = other_next_visit
                                .map(|v| occupations[v].delays[occupations[v].incumbent_idx].0)
                                .unwrap_or_else(|| true.into());

                            const USE_CHOICE_VAR: bool = false;
                            n_conflict_constraints += 1;

                            if USE_CHOICE_VAR {
                                let (pa, pb) = (visit_id, other_visit);
                                let choose = conflict_vars.get(&(pa, pb)).copied().unwrap_or_else(|| {
                                    let new_var = SatInstance::new_var(&mut solver);
                                    conflict_vars.insert((pa, pb), new_var);
                                    conflict_vars.insert((pb, pa), !new_var);
                                    new_var
                                });

                                SatInstance::add_clause(&mut solver, vec![!choose, !t1_out_lit, delay_t2]);
                                SatInstance::add_clause(&mut solver, vec![choose, !t2_out_lit, delay_t1]);
                            } else {
                                SatInstance::add_clause(
                                    &mut solver,
                                    vec![!t1_out_lit, !t2_out_lit, delay_t1, delay_t2],
                                );
                            }
                        }
                    }
                }

                retain
            });

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

                // Tighten UB to search for a strictly better solution.
                budget_ub = Some(cost - 1);

                debug_out(DebugInfo {
                    iteration,
                    actions: Vec::new(),
                    solution: sol,
                });

                // If we cannot improve further, we can stop.
                if budget_ub.unwrap() < 0 {
                    let (c, s) = best_sol.unwrap();
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
        }

        // ----- Encode costs for newly-created time points -----
        // SAT budget uses *unit-cost ladder vars* only (no weighted CostTree).
        const USE_COST_TREE: bool = false;

        for (visit, new_timepoint_var, new_t) in new_time_points.drain(..) {
            n_timepoints += 1;
            let (train_idx, visit_idx) = visits[visit];

            let new_timepoint_cost =
                problem.trains[train_idx].visit_delay_cost(delay_cost_type, visit_idx, new_t);

            if new_timepoint_cost > 0 {
                if !USE_COST_TREE {
                    for cost in occupations[visit].cost.len()..=new_timepoint_cost {
                        let prev_cost_var = occupations[visit].cost[cost - 1];
                        let next_cost_var = SatInstance::new_var(&mut solver);

                        SatInstance::add_clause(&mut solver, vec![!next_cost_var, prev_cost_var]);

                        occupations[visit].cost.push(next_cost_var);
                        assert!(cost + 1 == occupations[visit].cost.len());

                        // Each such var is one unit of cost.
                        budget_units.push(next_cost_var);
                    }

                    SatInstance::add_clause(
                        &mut solver,
                        vec![!new_timepoint_var, occupations[visit].cost[new_timepoint_cost]],
                    );
                } else {
                    // Weighted PB encoding is not implemented in the SAT-only backend.
                    // Keep this branch to make the intent explicit.
                    // If you really need weighted costs in SAT, implement a PB encoding here.
                    let _ = new_timepoint_var;
                    let _ = new_t;
                    let _ = new_timepoint_cost;
                }
            }
        }

        // ----- Enforce budget UB (if known) -----
        if let Some(ub) = budget_ub {
            if ub < 0 {
                // Impossible to satisfy cost <= -1; return best.
                stats.n_unsat += 1;
                let (c, s) = best_sol.or_else(|| {
                    heur_thread
                        .as_ref()
                        .and_then(|(_, rx)| rx.try_recv().ok())
                        .map(|(c, s)| (c, s))
                }).unwrap_or((i32::MAX, Vec::new()));

                if s.is_empty() {
                    return Err(SolverError::NoSolution);
                }
                stats.satsolver = format!("{:?}", solver);
                return Ok((s, stats));
            }

            let ub_usize = ub as usize;

            if ub_usize < budget_units.len() {
                let need_rebuild = budget_tot.is_none()
                    || budget_tot_len != budget_units.len()
                    || budget_tot_max_bound < ub_usize;

                if need_rebuild {
                    let tot = Totalizer::count(&mut solver, budget_units.iter().copied(), ub_usize as u32);
                    budget_tot_len = budget_units.len();
                    budget_tot_max_bound = ub_usize;
                    budget_tot = Some(tot);
                }

                // Enforce sum(budget_units) <= ub
                if let Some(tot) = budget_tot.as_ref() {
                    debug_assert!(ub_usize < tot.rhs().len());
                    SatInstance::add_clause(&mut solver, vec![!tot.rhs()[ub_usize]]);
                }
            }
        }

        // ----- Solve SAT -----
        *iteration_types.entry(IterationType::Objective).or_default() += 1;

        let solver_debug = format!("{:?}", solver);
        let solve_start = Instant::now();
        let result =
            SatSolverWithCore::solve_with_assumptions(&mut solver, std::iter::empty::<Bool<L>>());
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

                // If we have a best solution, UNSAT under the tightened UB means it is optimal (for this encoding).
                if let Some((c, s)) = best_sol {
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

                return Err(SolverError::NoSolution);
            }
        }

        iteration += 1;
    }
}
