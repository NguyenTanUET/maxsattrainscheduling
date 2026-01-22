use std::{collections::BTreeMap, time::Instant};

use satcoder::{Bool, SatInstance};
use typed_index_collections::TiVec;

use crate::problem::Problem;

use super::costtree::CostTree;

/// Internal identifier for a (train, visit) pair.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct VisitId(pub u32);

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

/// Internal identifier for a resource.
#[derive(Clone, Copy, Debug)]
pub struct ResourceId(pub u32);

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

pub fn do_output_stats<L: satcoder::Lit>(
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
    output_stats("num_conflicts".to_string(), stats.n_travel.into());
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

pub fn extract_solution<L: satcoder::Lit>(
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

#[derive(Debug)]
pub struct Occ<L: satcoder::Lit> {
    pub cost: Vec<Bool<L>>,
    pub cost_tree: CostTree<L>,
    pub delays: Vec<(Bool<L>, i32)>,
    pub incumbent_idx: usize,
}

impl<L: satcoder::Lit> Occ<L> {
    pub fn incumbent_time(&self) -> i32 {
        self.delays[self.incumbent_idx].1
    }

    /// Insert (or reuse) a time-point in the monotone chain of delay variables.
    ///
    /// Returns (var, is_new): `var` is the literal representing reaching time `t`,
    /// and `is_new` indicates whether a new variable was allocated.
    pub fn time_point(&mut self, solver: &mut impl SatInstance<L>, t: i32) -> (Bool<L>, bool) {
        let idx = self.delays.partition_point(|(_, t0)| *t0 < t);

        assert!(idx > 0 || t == self.delays[0].1); // cannot insert before the earliest time
        assert!(idx < self.delays.len()); // cannot insert after infinity

        assert!(idx == 0 || self.delays[idx - 1].1 < t);
        assert!(self.delays[idx].1 >= t);

        if self.delays[idx].1 == t || (idx > 0 && self.delays[idx - 1].1 == t) {
            return (self.delays[idx].0, false);
        }

        let var = solver.new_var();
        self.delays.insert(idx, (var, t));

        if idx > 0 {
            solver.add_clause(vec![!var, self.delays[idx - 1].0]);
        }

        if idx < self.delays.len() {
            solver.add_clause(vec![!self.delays[idx + 1].0, var]);
        }

        (var, true)
    }
}
