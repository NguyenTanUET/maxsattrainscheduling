use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque},
    time::Instant,
};

#[allow(unused)]
use crate::{
    debug::{ResourceInterval, SolverAction},
    minimize_core,
    problem::Problem,
    trim_core,
};
use satcoder::{
    constraints::{BooleanFormulas, Totalizer},
    prelude::{Binary, SymbolicModel},
    Bool, SatInstance, SatSolverWithCore,
};
use typed_index_collections::TiVec;

// -----------------------------------------------------------------------------
// SC-style encoding for fixed-precedence cliques (rail-path precedence).
//
// In the IAP/DDD formulation, a fixed precedence clique for two consecutive
// visits r \prec q on the SAME train has the staircase form:
//   x_{ir}^p + \sum_{t=1}^{K(p)} x_{iq}^t \le 1.
//
// This solver represents each visit by a monotone ladder of time-point literals
// (Occ::delays): for increasing times t_1 < t_2 < ... we maintain literals
//   d(t)  ==  ("arrival time at least t").
// The ladder clauses ensure d(t_j) => d(t_{j-1}) and d(t_{j+1}) => d(t_j).
//
// Under this representation, the clique above compresses to ONE implication per
// time point (the SC idea):
//   d_r(t) -> d_q(t + travel_r).
// which forbids all too early choices at q without enumerating them.
// -----------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(super) struct VisitId(pub(super) u32);

impl From<VisitId> for usize {
    fn from(v: VisitId) -> Self {
        v.0 as usize
    }
}

impl From<usize> for VisitId {
    fn from(x: usize) -> Self {
        VisitId(x as u32)
    }
}

#[derive(Clone, Copy, Debug)]
struct ResourceId(u32);

impl From<ResourceId> for usize {
    fn from(v: ResourceId) -> Self {
        v.0 as usize
    }
}

impl From<usize> for ResourceId {
    fn from(x: usize) -> Self {
        ResourceId(x as u32)
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum IterationType {
    Objective,
    TravelTimeConflict,
    ResourceConflict,
    TravelAndResourceConflict,
    Solution,
}

#[derive(Default)]
pub struct SolveStats {
    pub n_sat: usize,
    pub n_unsat: usize,
    pub n_travel: usize,
    pub n_conflict: usize,
    pub satsolver: String,
}

// Settings + tunable constants moved to `super::settings`.
// SC AMO helpers moved to `super::sc_amo`.
// Precedence helpers moved to `super::precedence`.
use super::precedence::{add_fixed_precedence_row, propagate_precedence};
use super::sc_amo::{add_hybrid_amo, build_active_lit, get_delay_lit_at};
use super::settings::MaxSatDddLadderScSettings;

enum Soft<L: satcoder::Lit> {
    Primary,
    Totalizer(Totalizer<L>, usize),
}

fn bits_needed(max_value: usize) -> usize {
    if max_value == 0 {
        1
    } else {
        (usize::BITS as usize) - (max_value.leading_zeros() as usize)
    }
}

fn build_binary_register<L: satcoder::Lit + Copy + 'static>(reg_bits: &[Bool<L>]) -> Binary<L> {
    Binary::from_list(reg_bits.iter().copied())
}

fn build_weighted_binary_term<L: satcoder::Lit + Copy + 'static>(
    lit: Bool<L>,
    weight: usize,
) -> Binary<L> {
    if weight == 0 || lit == false.into() {
        return Binary::constant(0);
    }

    let mut bits = Vec::new();
    let mut remaining = weight;
    while remaining > 0 {
        bits.push(if (remaining & 1usize) == 1usize {
            lit
        } else {
            false.into()
        });
        remaining >>= 1;
    }

    Binary::from_list(bits)
}

fn subtract_binary<L: satcoder::Lit + Copy + 'static>(
    solver: &mut impl SatInstance<L>,
    a: &Binary<L>,
    b: &Binary<L>,
) -> (Binary<L>, Bool<L>) {
    use std::iter::repeat;

    let len = a.clone().into_list().len().max(b.clone().into_list().len());
    let a_bits = a
        .clone()
        .into_list()
        .into_iter()
        .chain(repeat(false.into()))
        .take(len)
        .collect::<Vec<_>>();
    let b_bits = b
        .clone()
        .into_list()
        .into_iter()
        .chain(repeat(false.into()))
        .take(len)
        .collect::<Vec<_>>();

    let mut diff_bits = Vec::with_capacity(len);
    let mut borrow = false.into();

    for idx in 0..len {
        let ai = a_bits[idx];
        let bi = b_bits[idx];

        let diff = solver.xor_literal([ai, bi, borrow]);
        let bi_or_borrow = solver.or_literal([bi, borrow]);
        let borrow_from_subtrahend = solver.and_literal([!ai, bi_or_borrow]);
        let borrow_from_borrow = solver.and_literal([bi, borrow]);
        let next_borrow = solver.or_literal([borrow_from_subtrahend, borrow_from_borrow]);

        diff_bits.push(diff);
        borrow = next_borrow;
    }

    (Binary::from_list(diff_bits), borrow)
}

fn binary_le_literal_bits<L: satcoder::Lit + Copy>(
    solver: &mut impl SatInstance<L>,
    a: &[Bool<L>], // MSB -> LSB
    b: &[Bool<L>], // MSB -> LSB
) -> Bool<L> {
    assert_eq!(a.len(), b.len());

    if a.is_empty() {
        return true.into();
    }

    if a.len() == 1 {
        let le_lit = solver.new_var();
        solver.add_clause(vec![!le_lit, !a[0], b[0]]);
        return le_lit;
    }

    let rest = binary_le_literal_bits(solver, &a[1..], &b[1..]);
    let le_lit = solver.new_var();

    solver.add_clause(vec![!le_lit, !a[0], b[0]]);
    solver.add_clause(vec![!le_lit, a[0], b[0], rest]);
    solver.add_clause(vec![!le_lit, !a[0], !b[0], rest]);

    le_lit
}

fn assert_binary_le<L: satcoder::Lit + Copy + 'static>(
    solver: &mut impl SatInstance<L>,
    a: &Binary<L>,
    b: &Binary<L>,
) {
    use std::iter::repeat;

    let len = a.clone().into_list().len().max(b.clone().into_list().len());
    if len == 0 {
        return;
    }

    let mut a_bits = a
        .clone()
        .into_list()
        .into_iter()
        .chain(repeat(false.into()))
        .take(len)
        .collect::<Vec<_>>();
    a_bits.reverse();

    let mut b_bits = b
        .clone()
        .into_list()
        .into_iter()
        .chain(repeat(false.into()))
        .take(len)
        .collect::<Vec<_>>();
    b_bits.reverse();

    let le = binary_le_literal_bits(solver, &a_bits, &b_bits);
    solver.add_clause(vec![le]);
}

struct BinaryObjective<L: satcoder::Lit> {
    reg_bits: Vec<Bool<L>>,
    remaining: Option<Binary<L>>,
}

impl<L: satcoder::Lit + Copy + 'static> BinaryObjective<L> {
    fn new() -> Self {
        Self {
            reg_bits: Vec::new(),
            remaining: None,
        }
    }

    fn add_term(&mut self, solver: &mut impl SatInstance<L>, lit: Bool<L>, weight: usize) {
        if weight == 0 || lit == false.into() {
            return;
        }

        let term = build_weighted_binary_term(lit, weight);
        let current_remaining = self
            .remaining
            .clone()
            .expect("binary objective requires an upper bound before adding terms");
        let (next_remaining, underflow) = subtract_binary(solver, &current_remaining, &term);
        solver.add_clause(vec![!underflow]);
        self.remaining = Some(next_remaining);
    }

    fn ensure_capacity(
        &mut self,
        solver: &mut impl SatInstance<L>,
        soft_constraints: &mut HashMap<Bool<L>, (Soft<L>, usize, usize)>,
        capacity: usize,
    ) {
        let need_bits = bits_needed(capacity);
        let old_len = self.reg_bits.len();

        if need_bits > self.reg_bits.len() {
            while self.reg_bits.len() < need_bits {
                let bit = self.reg_bits.len();
                let reg_bit = solver.new_var();
                let weight = 1usize << bit;
                self.reg_bits.push(reg_bit);
                soft_constraints.insert(!reg_bit, (Soft::Primary, weight, weight));
            }
        }

        if old_len == self.reg_bits.len() {
            if self.remaining.is_none() {
                self.remaining = Some(build_binary_register(&self.reg_bits));
            }
        } else if let Some(remaining) = self.remaining.take() {
            let mut bits = remaining.into_list();
            bits.extend(self.reg_bits[old_len..].iter().copied());
            self.remaining = Some(Binary::from_list(bits));
        } else {
            self.remaining = Some(build_binary_register(&self.reg_bits));
        }
    }
}

#[allow(dead_code)]
fn compute_initial_heuristic_upper_bound<L: satcoder::Lit>(
    mk_env: &impl Fn() -> grb::Env,
    problem: &Problem,
    delay_cost_type: DelayCostType,
    occupations: &TiVec<VisitId, Occ<L>>,
) -> Result<Option<(i32, Vec<Vec<i32>>)>, SolverError> {
    let initial_solution = extract_solution(problem, occupations);
    let env = mk_env();

    for use_strong_branching in [false, true] {
        if let Some(ub_sol) = heuristic::solve_heuristic_better(
            &env,
            problem,
            delay_cost_type,
            use_strong_branching,
            Some(&initial_solution),
        )? {
            let ub_cost = problem.verify_solution(&ub_sol, delay_cost_type).unwrap();
            return Ok(Some((ub_cost, ub_sol)));
        }
    }

    Ok(None)
}

// `compute_effective_earliest` has moved to
// `super::ddd::shared::precedence::chain_earliest` so that
// `maxsat_ladder_sc` and `incremental_sat` use byte-identical
// chain-propagation logic for `delays[0]` setup.

pub fn solve<L: satcoder::Lit + Copy + std::fmt::Debug + 'static>(
    mk_env: impl Fn() -> grb::Env + Send + 'static,
    solver: impl SatInstance<L> + SatSolverWithCore<Lit = L> + std::fmt::Debug,
    problem: &Problem,
    timeout: f64,
    delay_cost_type: DelayCostType,
    output_stats: impl FnMut(String, serde_json::Value),
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError> {
    solve_with_settings(
        mk_env,
        solver,
        problem,
        timeout,
        delay_cost_type,
        MaxSatDddLadderScSettings::default(),
        output_stats,
    )
}

pub fn solve_with_settings<L: satcoder::Lit + Copy + std::fmt::Debug + 'static>(
    mk_env: impl Fn() -> grb::Env + Send + 'static,
    solver: impl SatInstance<L> + SatSolverWithCore<Lit = L> + std::fmt::Debug,
    problem: &Problem,
    timeout: f64,
    delay_cost_type: DelayCostType,
    settings: MaxSatDddLadderScSettings,
    output_stats: impl FnMut(String, serde_json::Value),
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError> {
    solve_debug_with_settings(
        mk_env,
        solver,
        problem,
        timeout,
        delay_cost_type,
        settings,
        |_| {},
        output_stats,
    )
}

thread_local! { pub static  WATCH : std::cell::RefCell<Option<(usize,usize)>>  = RefCell::new(None);}

use crate::{debug::DebugInfo, problem::DelayCostType, solvers::util::heuristic};

use crate::solvers::{ddd::shared::costtree::CostTree, SolverError};
pub fn solve_debug<L: satcoder::Lit + Copy + std::fmt::Debug>(
    mk_env: impl Fn() -> grb::Env + Send + 'static,
    solver: impl SatInstance<L> + SatSolverWithCore<Lit = L> + std::fmt::Debug,
    problem: &Problem,
    timeout: f64,
    delay_cost_type: DelayCostType,
    debug_out: impl Fn(DebugInfo),
    output_stats: impl FnMut(String, serde_json::Value),
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError>
where
    L: 'static,
{
    solve_debug_with_settings(
        mk_env,
        solver,
        problem,
        timeout,
        delay_cost_type,
        MaxSatDddLadderScSettings::default(),
        debug_out,
        output_stats,
    )
}

pub fn solve_debug_with_settings<L: satcoder::Lit + Copy + std::fmt::Debug + 'static>(
    mk_env: impl Fn() -> grb::Env + Send + 'static,
    mut solver: impl SatInstance<L> + SatSolverWithCore<Lit = L> + std::fmt::Debug,
    problem: &Problem,
    timeout: f64,
    delay_cost_type: DelayCostType,
    settings: MaxSatDddLadderScSettings,
    debug_out: impl Fn(DebugInfo),
    mut output_stats: impl FnMut(String, serde_json::Value),
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError> {
    // TODO
    //  - more eager constraint generation
    //    - propagate simple presedences?
    //    - update all conflicts and presedences when adding new time points?
    //    - smt refinement of the simple presedences?
    //  - get rid of the multiple adding of constraints
    //  - cadical doesn't use false polarity, so it can generate unlimited conflicts when cost is maxed. Two trains pushing each other forward.

    let _p = hprof::enter("solver");

    let start_time: Instant = Instant::now();
    let mut solver_time = std::time::Duration::ZERO;
    let mut stats = SolveStats::default();

    let mut visits: TiVec<VisitId, (usize, usize)> = TiVec::new();
    let mut resource_visits: Vec<Vec<VisitId>> = Vec::new();
    let mut occupations: TiVec<VisitId, Occ<_>> = TiVec::new();
    let mut touched_intervals = Vec::new();
    let mut conflicts: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut new_time_points = Vec::new();
    let effective_earliest = settings
        .use_precedence_graph
        .then(|| crate::solvers::ddd::shared::precedence::chain_earliest(problem));

    #[allow(unused)]
    let mut core_sizes: BTreeMap<usize, usize> = BTreeMap::new();
    #[allow(unused)]
    let mut processed_core_sizes: BTreeMap<usize, usize> = BTreeMap::new();
    let mut iteration_types: BTreeMap<IterationType, usize> = BTreeMap::new();

    let mut n_timepoints = 0;
    let mut n_conflict_constraints = 0;

    //Build conflict adjacency 
    for (a, b) in problem.conflicts.iter() {
        conflicts.entry(*a).or_default().push(*b);
        if *a != *b {
            conflicts.entry(*b).or_default().push(*a);
        }
    }

    //Build Occ 
    for (train_idx, train) in problem.trains.iter().enumerate() {
        for (visit_idx, visit) in train.visits.iter().enumerate() {
            let visit_id: VisitId = visits.push_and_get_key((train_idx, visit_idx));
            let earliest = effective_earliest
                .as_ref()
                .map(|bounds| bounds[train_idx][visit_idx])
                .unwrap_or(visit.earliest);

            occupations.push(Occ {
                cost: vec![true.into()],
                cost_tree: CostTree::new(),
                delays: vec![(true.into(), earliest), (false.into(), i32::MAX)],
                incumbent_idx: 0,
            });
            n_timepoints += 1;

            while resource_visits.len() <= visit.resource_id {
                resource_visits.push(Vec::new());
            }

            // (1.1)
            resource_visits[visit.resource_id].push(visit_id);
            //
            
            touched_intervals.push(visit_id);
            new_time_points.push((visit_id, true.into(), earliest));

            // Pre-allocate SAT vars + monotonicity clauses for each cost-step
            // threshold time. 
            if settings.prealloc_cost_thresholds {
                for t in problem.trains[train_idx]
                    .visit_cost_threshold_times(delay_cost_type, visit_idx, earliest)
                {
                    let (var, is_new) = occupations[visit_id].time_point(&mut solver, t);
                    if is_new {
                        n_timepoints += 1;
                        new_time_points.push((visit_id, var, t));
                    }
                }
            }
        }
    }

    // The first iteration (0) does not need a solve call; we
    // know it's SAT because there are no constraints yet.
    let mut iteration = 1;
    let mut is_sat = true;

    let mut total_cost = 0;
    let mut soft_constraints = HashMap::new();
    let mut debug_actions = Vec::new();

    // Remember which cliques have already had their full AMO encoded
    // (keyed by sorted visit-set) to dedup AMO emission across iterations.
    //SEED PRECEDENCE ROWS
    let mut clique_amo_encoded: HashSet<Vec<VisitId>> = HashSet::new();
    // Rows already added for fixed-precedence encoding: (visit_id, time).
    let mut fixed_prec_rows: HashSet<(VisitId, i32)> = HashSet::new();

    // Optional: seed fixed-precedence (travel-time) constraints from the earliest
    // time points to reduce the number of "travel-time conflict" iterations.
    if settings.seed_sc_from_earliest {
        for visit_id in visits.keys() {
            let (train_idx, visit_idx) = visits[visit_id];
            if visit_idx + 1 >= problem.trains[train_idx].visits.len() {
                continue;
            }
            let (in_var, in_t) = occupations[visit_id].delays[0];
            if settings.use_precedence_graph {
                propagate_precedence(
                    &mut solver,
                    problem,
                    &visits,
                    &mut occupations,
                    &mut new_time_points,
                    &mut fixed_prec_rows,
                    visit_id,
                    in_var,
                    in_t,
                    settings.use_eager_chain_expansion,
                );
            } else if settings.use_eager_chain_expansion {
                let _ = add_fixed_precedence_row(
                    &mut solver,
                    problem,
                    &visits,
                    &mut occupations,
                    &mut new_time_points,
                    &mut fixed_prec_rows,
                    visit_id,
                    in_var,
                    in_t,
                    settings.use_eager_chain_expansion,
                );
            }
        }
    }

    // Async heuristic: instead of blocking on initial heuristic computation
    // at init (5-30s of Gurobi call time), spawn the heuristic thread
    // immediately and let SAT solver start working in parallel. The heuristic
    // result flows in via `heur_thread`'s channel during the main loop, where
    // we update `best_heur` and inject solution timepoints when it arrives.
    const USE_HEURISTIC: bool = true;
    let mut best_heur: Option<(i32, Vec<Vec<i32>>)> = None;
    let mut injected_heuristic_cost: Option<i32> = None;

    // Seed `best_heur` with a quick greedy schedule. Gives a sound UB even
    // when the Gurobi-based heuristic thread isn't running (e.g. license
    // expired), so timeout cases can still report a finite `ub` and
    // downstream tooling can compute a real GAP.
    {
        let greedy_sol = crate::solvers::ddd::shared::greedy::greedy_schedule(problem);
        if let Some(cost) = problem.verify_solution(&greedy_sol, delay_cost_type) {
            best_heur = Some((cost, greedy_sol));
        }
    }

    //UB from Gurobi
    let heur_thread = USE_HEURISTIC.then(|| {
        let (sol_in_tx, sol_in_rx) = std::sync::mpsc::channel();
        let (sol_out_tx, sol_out_rx) = std::sync::mpsc::channel();
        let problem = problem.clone();
        heuristic::spawn_heuristic_thread(mk_env, sol_in_rx, problem, delay_cost_type, sol_out_tx);
        (sol_in_tx, sol_out_rx)
    });

    /// Main DDD loop
    loop {
        // Check timeout at the start of each iteration.
        if start_time.elapsed().as_secs_f64() > timeout {
            let ub = best_heur.map(|(c, _)| c).unwrap_or(i32::MAX);
            println!("TIMEOUT LB={} UB={}", total_cost, ub);

            do_output_stats(
                &mut output_stats,
                iteration,
                &iteration_types,
                &stats,
                &occupations,
                start_time,
                solver_time,
                total_cost,
                ub,
            );
            return Err(SolverError::Timeout);
        }

        // Check SAT/UNSAT of the current iteration's formula.
        let _p = hprof::enter("iteration");
        if is_sat {
            
            // If SAT, extract the solution and check for optimality via the async.
            if let Some((sol_tx, sol_rx)) = heur_thread.as_ref() {
                let sol = extract_solution(problem, &occupations);
                let _ = sol_tx.send(sol);

                while let Ok((ub_cost, ub_sol)) = sol_rx.try_recv() {
                    if ub_cost < total_cost as i32 {
                        println!(
                            "HEURISTIC UB={} is below current LB={}; keeping UB but skipping LB==UB termination this iteration",
                            ub_cost, total_cost
                        );
                        if ub_cost < best_heur.as_ref().map(|(c, _)| *c).unwrap_or(i32::MAX) {
                            best_heur = Some((ub_cost, ub_sol));
                        }
                        continue;
                    }
                    if ub_cost == total_cost as i32 {
                        println!("HEURISTIC UB=LB");
                        println!("TERMINATE HEURISTIC");
                        println!(
                            "MAXSAT ITERATIONS {}  {}",
                            n_conflict_constraints, iteration
                        );
                        do_output_stats(
                            &mut output_stats,
                            iteration,
                            &iteration_types,
                            &stats,
                            &occupations,
                            start_time,
                            solver_time,
                            total_cost,
                            ub_cost,
                        );

                        return Ok((ub_sol, stats));
                    }

                    if ub_cost < best_heur.as_ref().map(|(c, _)| *c).unwrap_or(i32::MAX) {
                        best_heur = Some((ub_cost, ub_sol));
                    }
                }

            }

            // Extract the current solution and check for travel-time and resource conflicts.
               let mut found_travel_time_conflict = false;
            let mut found_resource_conflict = false;

            // travel time conflict check.
            for visit_id in touched_intervals.iter().copied() {
                let _p = hprof::enter("travel time check");
                let (train_idx, visit_idx) = visits[visit_id];
                let next_visit: Option<VisitId> =
                    if visit_idx + 1 < problem.trains[train_idx].visits.len() {
                        Some((usize::from(visit_id) + 1).into())
                    } else {
                        None
                    };

                // GATHER INFORMATION ABOUT TWO CONSECUTIVE TIME POINTS
                let t1_in = occupations[visit_id].incumbent_time();
                let visit = problem.trains[train_idx].visits[visit_idx];

                if let Some(next_visit) = next_visit {
                    let v1 = &occupations[visit_id];
                    let v2 = &occupations[next_visit];
                    let t1_out = v2.incumbent_time();

                    // TRAVEL TIME CONFLICT
                    if t1_in + visit.travel_time > t1_out {
                        found_travel_time_conflict = true;

                        debug_actions.push(SolverAction::TravelTimeConflict(ResourceInterval {
                            train_idx,
                            visit_idx,
                            resource_idx: visit.resource_id,
                            time_in: t1_in,
                            time_out: t1_out,
                        }));

                        // When `use_eager_chain_expansion` is OFF, emit the
                        // travel-time clause inline — matches ladder semantics
                        // exactly (no dedup, no short-circuit). The dedup +
                        // earliest-next short-circuit inside
                        // `add_fixed_precedence_row` change the emitted CNF in
                        // subtle ways that make `ScNothing != Ldr` even when
                        // every other flag is off.
                        if settings.use_eager_chain_expansion {
                            let in_var = v1.delays[v1.incumbent_idx].0;
                            let in_t = v1.incumbent_time();
                            let _ = add_fixed_precedence_row(
                                &mut solver,
                                problem,
                                &visits,
                                &mut occupations,
                                &mut new_time_points,
                                &mut fixed_prec_rows,
                                visit_id,
                                in_var,
                                in_t,
                                settings.use_eager_chain_expansion,
                            );
                        } else {
                            let t1_in_var = v1.delays[v1.incumbent_idx].0;
                            let new_t = v1.incumbent_time() + visit.travel_time;
                            let (t1_earliest_out_var, t1_is_new) =
                                occupations[next_visit].time_point(&mut solver, new_t);
                            SatInstance::add_clause(
                                &mut solver,
                                vec![!t1_in_var, t1_earliest_out_var],
                            );
                            if t1_is_new {
                                new_time_points.push((next_visit, t1_earliest_out_var, new_t));
                            }
                        }
                        stats.n_travel += 1;
                    }




                }
            }


            //Resource conflict check: find all pairs of overlapping visits on conflicting resources, 
            //and add a clause to forbid the current overlap.
            let _p = hprof::enter("conflict check");
            let mut deconflicted_train_pairs: HashSet<(usize, usize)> = HashSet::new();

            // Lite clique aggregation for the pair-based scan: each
            // (resource, tau_plus_1) accumulates visits that ended up in a
            // detected conflict. After the scan, cliques with ≥3 members
            // are encoded as a single AMO over `active(v, tau_plus_1)`
            // literals — opt-in via `use_touched_clique_amo`. This lets
            // SC AMO encoding fire WITHOUT the heavier interval-
            // graph clique-cover pass.
            let mut touched_pair_cliques: HashMap<(usize, i32), HashSet<VisitId>> =
                HashMap::new();

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
                                    let other_v =
                                        problem.trains[other_train_idx].visits[other_visit_idx];
                                    t2_in + other_v.travel_time
                                });

                            if t1_out <= t2_in || t2_out <= t1_in {
                                continue;
                            }

                            if !deconflicted_train_pairs.insert((train_idx, other_train_idx))
                                || !deconflicted_train_pairs
                                    .insert((other_train_idx, train_idx))
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

                            // Eager SC-style precedence propagation: pre-emptively add a
                            // fixed-precedence row at the new time point (delay_t1, delay_t2)
                            // for the train's NEXT visit, propagating travel time forward.
                            //
                            // Gated on `use_eager_chain_expansion` so that this entire block
                            // is a no-op when the flag is OFF — that yields semantics
                            // equivalent to `maxsat_ladder.rs` (which only enforces travel
                            // time lazily in the dedicated travel-time-conflict branch above).
                            //
                            // With the flag ON we keep the previous eager behaviour: better
                            // unit propagation, but +50–70% more time points per benchmark
                            // (see sc-vs-ladder analysis).
                            if settings.use_eager_chain_expansion {
                                let _ = add_fixed_precedence_row(
                                    &mut solver,
                                    problem,
                                    &visits,
                                    &mut occupations,
                                    &mut new_time_points,
                                    &mut fixed_prec_rows,
                                    visit_id,
                                    delay_t1,
                                    t2_out,
                                    settings.use_eager_chain_expansion,
                                );

                                let _ = add_fixed_precedence_row(
                                    &mut solver,
                                    problem,
                                    &visits,
                                    &mut occupations,
                                    &mut new_time_points,
                                    &mut fixed_prec_rows,
                                    other_visit,
                                    delay_t2,
                                    t1_out,
                                    settings.use_eager_chain_expansion,
                                );
                            }

                            let t1_out_lit = next_visit
                                .map(|v| occupations[v].delays[occupations[v].incumbent_idx].0)
                                .unwrap_or_else(|| true.into());
                            let t2_out_lit = other_next_visit
                                .map(|v| occupations[v].delays[occupations[v].incumbent_idx].0)
                                .unwrap_or_else(|| true.into());

                            n_conflict_constraints += 1;
                            SatInstance::add_clause(
                                &mut solver,
                                vec![!t1_out_lit, !t2_out_lit, delay_t1, delay_t2],
                            );

                            // Touched-clique-AMO aggregation: record
                            // both visits at the (resource, tau_plus_1)
                            // of their overlap. Only meaningful for
                            // self-conflicts (both on same resource);
                            // cross-resource conflicts skipped.
                            if settings.use_touched_clique_amo
                                && visit.resource_id == other_resource
                            {
                                let tau_plus_1 = t1_out.min(t2_out);
                                let entry = touched_pair_cliques
                                    .entry((other_resource, tau_plus_1))
                                    .or_default();
                                entry.insert(visit_id);
                                entry.insert(other_visit);
                            }

                            retain = true;
                        }
                    }
                }
                retain
            });

            // ───────── Touched-clique AMO (post-retain) ─────────
            // For each (resource, tau_plus_1) that accumulated ≥ 3
            // visits during the pair-based scan, emit a single AMO
            // over their `active(v, tau_plus_1)` literals. Choice of
            // pairwise vs SC encoding follows `use_sc_amo`.
            if settings.use_touched_clique_amo {
                let mut active_lit_cache_tcamo: HashMap<(VisitId, i32), Bool<L>> =
                    HashMap::new();
                let entries: Vec<((usize, i32), HashSet<VisitId>)> =
                    touched_pair_cliques.into_iter().collect();
                for ((_, tau_plus_1), visit_set) in entries {
                    if visit_set.len() < 3 {
                        continue;
                    }
                    let mut visits_sorted: Vec<VisitId> =
                        visit_set.into_iter().collect();
                    visits_sorted.sort_by_key(|v| v.0);
                    if !clique_amo_encoded.insert(visits_sorted.clone()) {
                        continue;
                    }
                    let mut active_lits: Vec<Bool<L>> =
                        Vec::with_capacity(visits_sorted.len());
                    for &v in &visits_sorted {
                        let lit = build_active_lit(
                            &mut solver,
                            problem,
                            &visits,
                            &mut occupations,
                            &mut new_time_points,
                            &mut fixed_prec_rows,
                            &mut active_lit_cache_tcamo,
                            settings.use_eager_chain_expansion,
                            v,
                            tau_plus_1,
                        );
                        active_lits.push(lit);
                    }
                    add_hybrid_amo(&mut solver, &active_lits, settings.use_sc_amo);
                    n_conflict_constraints += 1;
                }
            }


            // If UNSAT, add conflict-graph-based constraints, optimal check
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

            if !(found_resource_conflict || found_travel_time_conflict) {
                // Incumbent times are feasible and optimal.

                const USE_LP_MINIMIZE: bool = false;

                let trains = if !USE_LP_MINIMIZE {
                    extract_solution(problem, &occupations)
                } else {
                    panic!()
                };

                println!(
                    "Finished with cost {} iterations {} solver {:?}",
                    total_cost, iteration, solver
                );
                println!("Core size bins {:?}", core_sizes);
                println!("Iteration types {:?}", iteration_types);
                debug_out(DebugInfo {
                    iteration,
                    actions: std::mem::take(&mut debug_actions),
                    solution: extract_solution(problem, &occupations),
                });

                stats.satsolver = format!("{:?}", solver);

                println!(
                    "STATS {} {} {} {} {} {} {} {}",
                    /* iter */ iteration,
                    /* objective iters */
                    iteration_types.get(&IterationType::Objective).unwrap_or(&0),
                    /* travel iters */
                    iteration_types
                        .get(&IterationType::TravelTimeConflict)
                        .unwrap_or(&0),
                    /* resource iters */
                    iteration_types
                        .get(&IterationType::ResourceConflict)
                        .unwrap_or(&0),
                    /* both iters */
                    iteration_types
                        .get(&IterationType::TravelAndResourceConflict)
                        .unwrap_or(&0),
                    /* solution iters */
                    iteration_types.get(&IterationType::Solution).unwrap_or(&0),
                    /* num traveltime */ stats.n_travel,
                    /* num conflicts */ stats.n_conflict,
                );

                do_output_stats(
                    &mut output_stats,
                    iteration,
                    &iteration_types,
                    &stats,
                    &occupations,
                    start_time,
                    solver_time,
                    total_cost,
                    total_cost,
                );

                println!("VARSCLAUSES {:?}", solver);

                println!(
                    "MAXSAT ITERATIONS {}  {}",
                    n_conflict_constraints, iteration
                );
                return Ok((trains, stats));
            }
        }

        // Add new time points and their costs to the solver.
        for (visit, new_timepoint_var, new_t) in new_time_points.drain(..) {
            n_timepoints += 1;
            let (train_idx, visit_idx) = visits[visit];

            let new_timepoint_cost =
                problem.trains[train_idx].visit_delay_cost(delay_cost_type, visit_idx, new_t);

            if new_timepoint_cost > 0 {


                const USE_COST_TREE: bool = true;
                if !USE_COST_TREE {
                    for cost in occupations[visit].cost.len()..=new_timepoint_cost {
                        let prev_cost_var = occupations[visit].cost[cost - 1];
                        let next_cost_var = SatInstance::new_var(&mut solver);

                        SatInstance::add_clause(&mut solver, vec![!next_cost_var, prev_cost_var]);

                        occupations[visit].cost.push(next_cost_var);
                        assert!(cost + 1 == occupations[visit].cost.len());

                        soft_constraints.insert(!next_cost_var, (Soft::Primary, 1, 1));
                    }

                    SatInstance::add_clause(
                        &mut solver,
                        vec![
                            !new_timepoint_var,
                            occupations[visit].cost[new_timepoint_cost],
                        ],
                    );

                } else {

                    // Direct insertion in callback — no Vec buffering overhead.
                    // Matches the ladder (non-SC) pattern for lower allocation cost.
                    occupations[visit].cost_tree.add_cost(
                        &mut solver,
                        new_timepoint_var,
                        new_timepoint_cost,
                        &mut |weight, cost_var| {
                            soft_constraints
                                .insert(!cost_var, (Soft::Primary, weight, weight));
                        },
                    );
                }
            }

            // set the cost for this new time point.

        }

        //Solve
        //Build asssumption
        let mut n_assumps = 20;
        let mut assumptions = soft_constraints
            .iter()
            .map(|(k, (_, w, _))| (*k, *w))
            .collect::<Vec<_>>();
        assumptions.sort_by(|a, b| b.1.cmp(&a.1));

        log::info!(
            "solving it{} with {} timepoints {} conflicts",
            iteration,
            n_timepoints,
            n_conflict_constraints
        );

        //grow assumptions
        let core = loop {
            let solve_start = Instant::now();
            let result = {
                let _p = hprof::enter("sat check");
                SatSolverWithCore::solve_with_assumptions(
                    &mut solver,
                    assumptions.iter().map(|(k, _)| *k).take(n_assumps),
                )
            };
            solver_time += solve_start.elapsed();

            //Update incumbent
            match result {
                satcoder::SatResultWithCore::Sat(_) if n_assumps < soft_constraints.len() => {
                    n_assumps += 20;
                }
                satcoder::SatResultWithCore::Sat(model) => {
                    is_sat = true;
                    stats.n_sat += 1;
                    let _p = hprof::enter("update times");

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
                        let (_train_idx, visit_idx) = visits[visit];



                        if touched {

                            // We are really interested not in the visits, but the resource occupation
                            // intervals. Therefore, also the previous visit has been touched by this visit.
                            if visit_idx > 0 {
                                let prev_visit = (Into::<usize>::into(visit) - 1).into();
                                if touched_intervals.last() != Some(&prev_visit) {
                                    touched_intervals.push(prev_visit);
                                }
                            }
                            touched_intervals.push(visit);
                        }

                    }

                    //Local Minimization: Optimize incumbent solution by trying to move each visit earlier as much as possible
                    const USE_LOCAL_MINIMIZE: bool = true;
                    if USE_LOCAL_MINIMIZE {
                        let mut last_mod = 0;
                        let mut i = 0;
                        let occs_len = occupations.len();
                        assert!(visits.len() == occupations.len());
                        while last_mod < occs_len {
                            let mut touched = false;

                            let visit_id = VisitId(i % occs_len as u32);
                            while occupations[visit_id].incumbent_idx > 0 {
                                // We can always leave earlier, so the critical interval is
                                // from this event to the next.

                                let t1_in = occupations[visit_id].delays
                                    [occupations[visit_id].incumbent_idx]
                                    .1;
                                let t1_in_new = occupations[visit_id].delays
                                    [occupations[visit_id].incumbent_idx - 1]
                                    .1;

                                let (train_idx, visit_idx) = visits[visit_id];
                                let visit = problem.trains[train_idx].visits[visit_idx];

                                let next_visit: Option<VisitId> =
                                    if visit_idx + 1 < problem.trains[train_idx].visits.len() {
                                        Some((usize::from(visit_id) + 1).into())
                                    } else {
                                        None
                                    };

                                let prev_visit: Option<VisitId> = if visit_idx > 0 {
                                    Some((usize::from(visit_id) - 1).into())
                                } else {
                                    None
                                };

                                let t1_prev_earliest_out = prev_visit
                                    .map(|v| {
                                        let (tidx, vidx) = visits[v];
                                        let travel_time =
                                            problem.trains[tidx].visits[vidx].travel_time;
                                        occupations[v].incumbent_time() + travel_time
                                    })
                                    .unwrap_or(i32::MIN);

                                let travel_ok = t1_prev_earliest_out <= t1_in_new;

                                let t1_out = next_visit
                                    .map(|nx| occupations[nx].incumbent_time())
                                    .unwrap_or(t1_in + visit.travel_time);

                                let can_reduce = travel_ok
                                    && conflicts
                                        .get(&visit.resource_id)
                                        .iter()
                                        .flat_map(|rs| rs.iter())
                                        .copied()
                                        .all(|other_resource| {
                                            resource_visits[other_resource]
                                                .iter()
                                                .copied()
                                                .filter(|other_visit| {
                                                    usize::from(visit_id)
                                                        != usize::from(*other_visit)
                                                })
                                                .filter(|other_visit| {
                                                    visits[*other_visit].0 != train_idx
                                                })
                                                .all(|other_visit| {
                                                    let v2 = &occupations[other_visit];
                                                    let t2_in = v2.incumbent_time();
                                                    let (other_train_idx, other_visit_idx) =
                                                        visits[other_visit];
                                                    let other_next_visit: Option<VisitId> =
                                                        if other_visit_idx + 1
                                                            < problem.trains[other_train_idx]
                                                                .visits
                                                                .len()
                                                        {
                                                            Some(
                                                                (usize::from(other_visit) + 1)
                                                                    .into(),
                                                            )
                                                        } else {
                                                            None
                                                        };

                                                    let t2_out = other_next_visit
                                                        .map(|v| occupations[v].incumbent_time())
                                                        .unwrap_or_else(|| {
                                                            let other_visit = problem.trains
                                                                [other_train_idx]
                                                                .visits[other_visit_idx];
                                                            t2_in + other_visit.travel_time
                                                        });
                                                    t1_out <= t2_in || t2_out <= t1_in_new
                                                })
                                        });

                                if can_reduce {
                                    occupations[visit_id].incumbent_idx -= 1;
                                    touched = true;
                                    last_mod = 0;
                                } else {
                                    break;
                                }
                            }

                            i += 1;

                            if touched {
                                let visit_idx = visits[visit_id].1;
                                if visit_idx > 0 {
                                    let prev_visit = (Into::<usize>::into(visit_id) - 1).into();
                                    if touched_intervals.last() != Some(&prev_visit) {
                                        touched_intervals.push(prev_visit);
                                    }
                                }
                                touched_intervals.push(visit_id);
                            } else {
                                last_mod += 1;
                            }
                        }
                    }



                    debug_out(DebugInfo {
                        iteration,
                        actions: std::mem::take(&mut debug_actions),
                        solution: extract_solution(problem, &occupations),
                    });

                    break None;
                }

                satcoder::SatResultWithCore::Unsat(core) => {
                    is_sat = false;
                    stats.n_unsat += 1;
                    break Some(core);
                }
            }
        };

        //Core handling RC2
        if let Some(core) = core {
            let _p = hprof::enter("treat core");

            if core.len() == 0 {
                // SC-specific fallback: when the precedence graph is OFF but
                // some other SC feature is ON, an empty core may indicate
                // missing precedence info rather than true infeasibility — try
                // injecting the best heuristic as a recovery step.
                //
                // Pure-ladder mode (all SC features off) skips this branch
                // entirely so behaviour matches `maxsat_ladder.rs` exactly.
                let any_other_sc_feature = settings.use_eager_chain_expansion
                    || settings.use_touched_clique_amo;
                if !settings.use_precedence_graph && any_other_sc_feature {
                    if let Some((ub_cost, ub_sol)) = best_heur.as_ref() {
                        if injected_heuristic_cost != Some(*ub_cost) {
                            inject_solution_timepoints_maxsat(
                                &mut solver,
                                problem,
                                &visits,
                                &mut occupations,
                                &mut new_time_points,
                                &mut fixed_prec_rows,
                                settings.use_eager_chain_expansion,
                                ub_sol,
                            );
                            injected_heuristic_cost = Some(*ub_cost);
                            iteration += 1;
                            continue;
                        }
                    }
                }
                return Err(SolverError::NoSolution); // UNSAT
            }

            let core = core.iter().map(|c| Bool::Lit(*c)).collect::<Vec<_>>();


            *iteration_types.entry(IterationType::Objective).or_default() += 1;
            debug_actions.push(SolverAction::Core(core.len()));

            let min_weight = core.iter().map(|c| soft_constraints[c].1).min().unwrap();
            assert!(min_weight >= 1);


            for c in core.iter() {
                let (soft, cost, original_cost) = soft_constraints.remove(c).unwrap();



                assert!(cost >= min_weight);
                let new_cost = cost - min_weight;
                match soft {
                    Soft::Primary => {
                        if new_cost > 0 {
                            soft_constraints.insert(*c, (Soft::Primary, new_cost, original_cost));
                        } else {
                        }
                        /* primary soft constraint, when we relax to new_cost=0 we are done */
                    }
                    Soft::Totalizer(mut tot, bound) => {
                        if new_cost > 0 {

                            soft_constraints
                                .insert(*c, (Soft::Totalizer(tot, bound), new_cost, original_cost));
                        } else {
                            let new_bound = bound + 1;
                            tot.increase_bound(&mut solver, new_bound as u32);
                            if new_bound < tot.rhs().len() {


                                soft_constraints.insert(
                                    !tot.rhs()[new_bound], // tot <= 2, 3, 4...
                                    (
                                        Soft::Totalizer(tot, new_bound),
                                        original_cost,
                                        original_cost,
                                    ),
                                );
                            } else {
                            }
                        }
                    }
                }
            }

            total_cost += min_weight as i32;
            println!("    LB={}", total_cost);

            if total_cost as i32 == best_heur.as_ref().map(|(c, _)| *c).unwrap_or(i32::MAX) {
                println!("TERMINATE HEURISTIC");
                println!(
                    "MAXSAT ITERATIONS {}  {}",
                    n_conflict_constraints, iteration
                );
                do_output_stats(
                    &mut output_stats,
                    iteration,
                    &iteration_types,
                    &stats,
                    &occupations,
                    start_time,
                    solver_time,
                    total_cost,
                    total_cost,
                );

                return Ok((best_heur.unwrap().1, stats));
            }

            if core.len() > 1 {
                let bound = 1;
                let tot = Totalizer::count(&mut solver, core.iter().map(|c| !*c), bound as u32);
                assert!(bound < tot.rhs().len());

                soft_constraints.insert(
                    !tot.rhs()[bound], // tot <= 1
                    (Soft::Totalizer(tot, bound), min_weight, min_weight),
                );
            } else {
                SatInstance::add_clause(&mut solver, vec![!core[0]]);
            }
        }

        iteration += 1;
    }
}

fn do_output_stats<L: satcoder::Lit>(
    output_stats: &mut impl FnMut(String, serde_json::Value),
    iteration: usize,
    iteration_types: &BTreeMap<IterationType, usize>,
    stats: &SolveStats,
    occupations: &TiVec<VisitId, Occ<L>>,
    start_time: Instant,
    solver_time: std::time::Duration,
    lb: i32,
    ub: i32,
) {
    output_stats("iterations".to_string(), iteration.into());
    output_stats(
        "objective_iters".to_string(),
        (*iteration_types.get(&IterationType::Objective).unwrap_or(&0)).into(),
    );
    output_stats(
        "travel_iters".to_string(),
        (*iteration_types
            .get(&IterationType::TravelTimeConflict)
            .unwrap_or(&0))
        .into(),
    );
    output_stats(
        "resource_iters".to_string(),
        (*iteration_types
            .get(&IterationType::ResourceConflict)
            .unwrap_or(&0))
        .into(),
    );
    output_stats(
        "travel_and_resource_iters".to_string(),
        (*iteration_types
            .get(&IterationType::TravelAndResourceConflict)
            .unwrap_or(&0))
        .into(),
    );
    output_stats("num_traveltime".to_string(), stats.n_travel.into());
    output_stats("num_conflicts".to_string(), stats.n_conflict.into());
    // Total SAT variables allocated and CNF clauses added throughout the
    // entire solve (initial encoding + every DDD refinement iteration).
    // Counts come from the per-thread `CountingSolver` wrapper around the
    // underlying solver; caller must reset before invoking `solve()`.
    let (n_vars_total, n_clauses_total) = crate::solvers::util::counting_solver::get_counts();
    output_stats("num_vars_total".to_string(), n_vars_total.into());
    output_stats("num_clauses_total".to_string(), n_clauses_total.into());
    output_stats(
        "num_time_points".to_string(),
        occupations
            .iter()
            .map(|o| o.delays.len() - 1)
            .sum::<usize>()
            .into(),
    );
    output_stats(
        "max_time_points".to_string(),
        occupations
            .iter()
            .map(|o| o.delays.len() - 1)
            .max()
            .unwrap()
            .into(),
    );
    output_stats(
        "avg_time_points".to_string(),
        ((occupations
            .iter()
            .map(|o| o.delays.len() - 1)
            .sum::<usize>() as f64)
            / (occupations.len() as f64))
            .into(),
    );

    output_stats(
        "total_time".to_string(),
        start_time.elapsed().as_secs_f64().into(),
    );
    output_stats("solver_time".to_string(), solver_time.as_secs_f64().into());
    output_stats(
        "algorithm_time".to_string(),
        (start_time.elapsed().as_secs_f64() - solver_time.as_secs_f64()).into(),
    );
    output_stats("lb".to_string(), lb.into());
    output_stats("ub".to_string(), ub.into());
}

fn extract_solution<L: satcoder::Lit>(
    problem: &Problem,
    occupations: &TiVec<VisitId, Occ<L>>,
) -> Vec<Vec<i32>> {
    let _p = hprof::enter("extract solution");
    let mut trains = Vec::new();
    let mut i = 0;
    for (train_idx, train) in problem.trains.iter().enumerate() {
        let mut train_times = Vec::new();
        for _ in train.visits.iter().enumerate() {
            train_times.push(occupations[VisitId(i)].incumbent_time());
            i += 1;
        }

        let visit = problem.trains[train_idx].visits[train_times.len() - 1];
        let last_t = train_times[train_times.len() - 1] + visit.travel_time;
        train_times.push(last_t);

        trains.push(train_times);
    }
    trains
}

fn inject_solution_timepoints_maxsat<L: satcoder::Lit>(
    solver: &mut impl SatInstance<L>,
    problem: &Problem,
    visits: &TiVec<VisitId, (usize, usize)>,
    occupations: &mut TiVec<VisitId, Occ<L>>,
    new_time_points: &mut Vec<(VisitId, Bool<L>, i32)>,
    fixed_prec_rows: &mut HashSet<(VisitId, i32)>,
    use_eager_chain_expansion: bool,
    sol: &[Vec<i32>],
) {
    let mut flat_visit = 0usize;
    for (train_idx, train) in problem.trains.iter().enumerate() {
        for visit_idx in 0..train.visits.len() {
            let visit_id = VisitId(flat_visit as u32);
            flat_visit += 1;
            let time = sol[train_idx][visit_idx];
            let (lit, is_new) = occupations[visit_id].time_point(solver, time);
            if is_new {
                new_time_points.push((visit_id, lit, time));
            }
            if visit_idx + 1 < train.visits.len() {
                let _ = add_fixed_precedence_row(
                    solver,
                    problem,
                    visits,
                    occupations,
                    new_time_points,
                    fixed_prec_rows,
                    visit_id,
                    lit,
                    time,
                    use_eager_chain_expansion,
                );
            }
        }
    }
}

#[derive(Debug)]
pub(super) struct Occ<L: satcoder::Lit> {
    pub(super) cost: Vec<Bool<L>>,
    pub(super) cost_tree: CostTree<L>,
    pub(super) delays: Vec<(Bool<L>, i32)>,
    pub(super) incumbent_idx: usize,
}

impl<L: satcoder::Lit> Occ<L> {
    pub fn incumbent_time(&self) -> i32 {
        self.delays[self.incumbent_idx].1
    }

    pub fn time_point(&mut self, solver: &mut impl SatInstance<L>, t: i32) -> (Bool<L>, bool) {

        let idx = self.delays.partition_point(|(_, t0)| *t0 < t);


        assert!(idx > 0 || t == self.delays[0].1); // cannot insert before the earliest time.
        assert!(idx < self.delays.len()); // cannot insert after infinity.

        assert!(idx == 0 || self.delays[idx - 1].1 < t);
        assert!(self.delays[idx].1 >= t);

        if self.delays[idx].1 == t || (idx > 0 && self.delays[idx - 1].1 == t) {
            return (self.delays[idx].0, false);
        }

        let var = solver.new_var();
        self.delays.insert(idx, (var, t));

        // Keep `incumbent_idx` pointing to the same logical timepoint after
        // insertion. If we inserted at or before the incumbent's array slot,
        // the old incumbent shifted forward by one — compensate so callers
        // reading `incumbent_time()` or `delays[incumbent_idx]` still see
        // the same data they had before this call. Without this, an insertion
        // before the incumbent leaves `delays[incumbent_idx]` pointing to the
        // newly-inserted (wrong) timepoint, causing travel-time violations.
        if idx <= self.incumbent_idx {
            self.incumbent_idx += 1;
        }

        if idx > 0 {
            solver.add_clause(vec![!var, self.delays[idx - 1].0]);
        }

        if idx < self.delays.len() {
            solver.add_clause(vec![!self.delays[idx + 1].0, var]);
        }

        (var, true)
    }
}
