//! `CountingSolver<S>` — a transparent SAT-instance wrapper that tracks the
//! total number of variables allocated via `new_var()` and the total number
//! of clauses added via `add_clause()`. Counts are stored in thread-local
//! statics so the solve function can read them without changing the
//! generic signatures it accepts.
//!
//! Intended workflow:
//!   1. Caller resets counts with [`reset_counts`] before invoking the
//!      solver pipeline.
//!   2. Caller wraps the underlying SAT solver via [`CountingSolver::new`]
//!      and passes the wrapper into a `solve(...)` function expecting a
//!      `SatInstance<L> + SatSolverWithCore<Lit = L> + Debug` implementor.
//!   3. After `solve(...)` returns, caller reads counts via [`get_counts`]
//!      and reports them through `output_stats` for inclusion in the JSON
//!      result.
//!
//! Note: counts are global within the current thread. Resetting between
//! per-instance solves is important when running a batch of instances in
//! the same process.

use std::cell::Cell;

use satcoder::{Bool, Lit, SatInstance, SatResult, SatResultWithCore, SatSolver, SatSolverWithCore};

thread_local! {
    /// Total `new_var()` calls observed by [`CountingSolver`] this thread.
    pub static VAR_COUNT: Cell<usize> = const { Cell::new(0) };
    /// Total `add_clause()` calls observed by [`CountingSolver`] this thread.
    pub static CLAUSE_COUNT: Cell<usize> = const { Cell::new(0) };
}

/// Reset both per-thread counters to zero.
pub fn reset_counts() {
    VAR_COUNT.with(|c| c.set(0));
    CLAUSE_COUNT.with(|c| c.set(0));
}

/// Read the current per-thread `(n_vars, n_clauses)` counts.
pub fn get_counts() -> (usize, usize) {
    (
        VAR_COUNT.with(|c| c.get()),
        CLAUSE_COUNT.with(|c| c.get()),
    )
}

/// Transparent wrapper that increments thread-local counters on every
/// `new_var()` and `add_clause()` call before forwarding to the inner
/// solver. Implements `SatInstance`, `SatSolver`, `SatSolverWithCore`, and
/// `Debug` so it can drop in anywhere the underlying solver type works.
pub struct CountingSolver<S> {
    inner: S,
}

impl<S> CountingSolver<S> {
    pub fn new(inner: S) -> Self {
        Self { inner }
    }

    pub fn into_inner(self) -> S {
        self.inner
    }
}

impl<L: Lit, S: SatInstance<L>> SatInstance<L> for CountingSolver<S> {
    fn new_var(&mut self) -> Bool<L> {
        VAR_COUNT.with(|c| c.set(c.get() + 1));
        self.inner.new_var()
    }

    fn add_clause<IL: Into<Bool<L>>, I: IntoIterator<Item = IL>>(&mut self, clause: I) {
        CLAUSE_COUNT.with(|c| c.set(c.get() + 1));
        self.inner.add_clause(clause)
    }
}

impl<S: SatSolver> SatSolver for CountingSolver<S> {
    type Lit = S::Lit;

    fn solve<'a>(&'a mut self) -> SatResult<'a, Self::Lit> {
        self.inner.solve()
    }
}

impl<S: SatSolverWithCore> SatSolverWithCore for CountingSolver<S> {
    type Lit = S::Lit;

    fn solve_with_assumptions<'a>(
        &'a mut self,
        assumptions: impl IntoIterator<Item = Bool<Self::Lit>>,
    ) -> SatResultWithCore<'a, Self::Lit> {
        self.inner.solve_with_assumptions(assumptions)
    }
}

impl<S: std::fmt::Debug> std::fmt::Debug for CountingSolver<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let (v, c) = get_counts();
        write!(f, "CountingSolver(vars={}, clauses={}, inner={:?})", v, c, self.inner)
    }
}
