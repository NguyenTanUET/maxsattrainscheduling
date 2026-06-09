//! Standalone binary: build TRP MILP model trong Rust, export .lp file.
//!
//! Mục đích: thay thế cplex_milp/build.py — không cần Python, không cần
//! Gurobi license. Chỉ cần Rust toolchain.
//!
//! Workflow:
//! ```text
//! instance.txt -- export_lp --> .lp file -- CPLEX CLI --> .sol file
//! ```
//!
//! Hai formulation hỗ trợ:
//! - bigm: Big-M continuous time formulation (port từ src/solvers/milp/bigm.rs)
//! - ti:   Time-indexed binary formulation (port từ src/solvers/milp/milp_ti.rs)
//!
//! Usage:
//!   cargo build --release --bin export_lp
//!   target/release/export_lp <instance.txt> <bigm|ti> <cost_type> [out_dir]
//!
//! Examples:
//!   target/release/export_lp instances/original/InstanceA1.txt bigm finsteps123 results/
//!   target/release/export_lp instances/original/InstanceA1.txt ti cont results/

use ddd::parser;
use ddd::problem::{DelayCostThresholds, DelayCostType, DelayMeasurementType, Problem};

use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

const BIG_M: i32 = 2 * 6 * 3600; // = 43200, giống bigm.rs

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "Usage: {} <instance.txt> <bigm|ti> <cost_type> [out_dir=.]",
            args[0]
        );
        eprintln!("\nCost types: finsteps123, finsteps12345, finsteps139,");
        eprintln!("            finsteps1_3min, finsteps1_5min,");
        eprintln!("            infsteps60, infsteps180, infsteps360, cont");
        std::process::exit(1);
    }

    let instance_path = &args[1];
    let solver = args[2].to_lowercase();
    let cost_str = &args[3];
    let out_dir = if args.len() > 4 {
        args[4].clone()
    } else {
        ".".to_string()
    };

    let cost_type = parse_cost_type(cost_str).unwrap_or_else(|| {
        eprintln!("Unknown cost type: {}", cost_str);
        std::process::exit(1);
    });

    // TI parameters từ env vars (giống Python build)
    let interval: i32 = env::var("TI_INTERVAL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let big_m_ti: i32 = env::var("TI_BIG_M")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(900);

    println!("Loading {}", instance_path);
    let (named, _) = parser::read_txt_file(
        instance_path,
        DelayMeasurementType::FinalStationArrival,
        false,
        None,
        |_| {},
    );
    let problem = &named.problem;
    let n_visits: usize = problem.trains.iter().map(|t| t.visits.len()).sum();
    println!(
        "  Trains: {}, Resources: {}, Visits: {}",
        problem.trains.len(),
        named.resource_names.len(),
        n_visits,
    );

    std::fs::create_dir_all(&out_dir).expect("mkdir out_dir");
    let stem = PathBuf::from(instance_path)
        .file_stem()
        .unwrap()
        .to_string_lossy()
        .to_string();

    let (lp_path, n_vars, n_constraints) = match solver.as_str() {
        "bigm" => {
            let path = format!(
                "{}/{}_bigm_{}.lp",
                out_dir,
                stem,
                cost_str
            );
            let start = Instant::now();
            let (v, c) = write_bigm_lp(&path, problem, cost_type);
            println!("  Build time: {:.2}s", start.elapsed().as_secs_f64());
            (path, v, c)
        }
        "ti" => {
            let path = format!(
                "{}/{}_ti_{}_i{}_m{}.lp",
                out_dir, stem, cost_str, interval, big_m_ti
            );
            let start = Instant::now();
            let (v, c) = write_ti_lp(&path, problem, cost_type, interval, big_m_ti);
            println!("  Build time: {:.2}s", start.elapsed().as_secs_f64());
            (path, v, c)
        }
        _ => {
            eprintln!("Unknown solver: {}. Use 'bigm' or 'ti'.", solver);
            std::process::exit(1);
        }
    };

    println!("  Vars: {}", n_vars);
    println!("  Constraints: {}", n_constraints);
    println!("  -> {}", lp_path);
}

fn parse_cost_type(s: &str) -> Option<DelayCostType> {
    match s.to_lowercase().as_str() {
        "finsteps1_3min" => Some(DelayCostType::FiniteSteps1_3Min),
        "finsteps1_5min" => Some(DelayCostType::FiniteSteps1_5Min),
        "finsteps123" => Some(DelayCostType::FiniteSteps123),
        "finsteps12345" => Some(DelayCostType::FiniteSteps12345),
        "finsteps139" => Some(DelayCostType::FiniteSteps139),
        "infsteps60" => Some(DelayCostType::InfiniteSteps60),
        "infsteps180" => Some(DelayCostType::InfiniteSteps180),
        "infsteps360" => Some(DelayCostType::InfiniteSteps360),
        "infsteps123" => Some(DelayCostType::InfiniteSteps180),
        "cont" => Some(DelayCostType::Continuous),
        _ => None,
    }
}

// ────────────────────────────────────────────────────────────
// BigM-Eager LP Writer (port 1:1 từ src/solvers/milp/bigm.rs)
// ────────────────────────────────────────────────────────────
fn write_bigm_lp(
    path: &str,
    problem: &Problem,
    cost_type: DelayCostType,
) -> (usize, usize) {
    let f = File::create(path).expect("create .lp");
    let mut w = BufWriter::new(f);

    let mut n_vars = 0usize;
    let mut n_constraints = 0usize;

    writeln!(w, "\\ TRP BigM MILP, exported by export_lp.rs").unwrap();
    writeln!(w, "Minimize").unwrap();
    writeln!(w, " obj:").unwrap();

    // ───── Objective ─────
    let mut obj_terms: Vec<String> = Vec::new();
    let mut cost_var_count = 0usize;

    match cost_type {
        DelayCostType::FiniteSteps1_3Min
        | DelayCostType::FiniteSteps1_5Min
        | DelayCostType::FiniteSteps123
        | DelayCostType::FiniteSteps12345
        | DelayCostType::FiniteSteps139 => {
            let thr = match cost_type {
                DelayCostType::FiniteSteps1_3Min => DelayCostThresholds::f1_3min(),
                DelayCostType::FiniteSteps1_5Min => DelayCostThresholds::f1_5min(),
                DelayCostType::FiniteSteps123 => DelayCostThresholds::f123(),
                DelayCostType::FiniteSteps12345 => DelayCostThresholds::f12345(),
                DelayCostType::FiniteSteps139 => DelayCostThresholds::f139(),
                _ => unreachable!(),
            };
            for (ti, train) in problem.trains.iter().enumerate() {
                for (vi, visit) in train.visits.iter().enumerate() {
                    if visit.aimed.is_some() {
                        for thr_idx in (0..thr.thresholds.len()).rev() {
                            let prev_cost = thr
                                .thresholds
                                .get(thr_idx + 1)
                                .map(|x| x.1)
                                .unwrap_or(0);
                            let (threshold, cost) = thr.thresholds[thr_idx];
                            let diff = cost as i32 - prev_cost as i32;
                            assert!(diff > 0);
                            let name = format!("thr_{}_{}_{}", ti, vi, threshold);
                            obj_terms.push(format!("{} {}", diff, name));
                            cost_var_count += 1;
                        }
                    }
                }
            }
        }
        DelayCostType::InfiniteSteps60
        | DelayCostType::InfiniteSteps180
        | DelayCostType::InfiniteSteps360 => {
            for (ti, train) in problem.trains.iter().enumerate() {
                for (vi, visit) in train.visits.iter().enumerate() {
                    if visit.aimed.is_some() {
                        let name = format!("steps_{}_{}", ti, vi);
                        obj_terms.push(format!("1 {}", name));
                        cost_var_count += 1;
                    }
                }
            }
        }
        DelayCostType::Continuous => {
            for (ti, train) in problem.trains.iter().enumerate() {
                for (vi, visit) in train.visits.iter().enumerate() {
                    if visit.aimed.is_some() {
                        let name = format!("delay_{}_{}", ti, vi);
                        obj_terms.push(format!("1 {}", name));
                        cost_var_count += 1;
                    }
                }
            }
        }
    }

    // Write objective: split into multiple lines if too long
    write_terms(&mut w, &obj_terms);

    writeln!(w, "Subject To").unwrap();

    // ───── Travel constraints ─────
    let mut c_idx = 0usize;
    for (ti, train) in problem.trains.iter().enumerate() {
        for vi in 0..(train.visits.len().saturating_sub(1)) {
            let travel = train.visits[vi].travel_time;
            // t[ti][vi+1] - t[ti][vi] >= travel
            writeln!(
                w,
                " c{}: t_{}_{} - t_{}_{} >= {}",
                c_idx,
                ti,
                vi + 1,
                ti,
                vi,
                travel,
            )
            .unwrap();
            c_idx += 1;
            n_constraints += 1;
        }
    }

    // ───── Conflict constraints (BigM) ─────
    let pairs = visit_conflicts(problem);
    for ((t1, v1), (t2, v2)) in &pairs {
        if v1 + 1 >= problem.trains[*t1].visits.len() {
            continue;
        }
        if v2 + 1 >= problem.trains[*t2].visits.len() {
            continue;
        }
        // y=1: t1 first → t[t1][v1+1] - t[t2][v2] - M*(1-y) <= 0
        //                → t[t1][v1+1] - t[t2][v2] + M*y <= M
        writeln!(
            w,
            " c{}: t_{}_{} - t_{}_{} + {} y_{}_{}_{}_{} <= {}",
            c_idx,
            t1, v1 + 1,
            t2, v2,
            BIG_M,
            t1, v1, t2, v2,
            BIG_M,
        )
        .unwrap();
        c_idx += 1;
        n_constraints += 1;

        // y=0: t2 first → t[t2][v2+1] - t[t1][v1] - M*y <= 0
        writeln!(
            w,
            " c{}: t_{}_{} - t_{}_{} - {} y_{}_{}_{}_{} <= 0",
            c_idx,
            t2, v2 + 1,
            t1, v1,
            BIG_M,
            t1, v1, t2, v2,
        )
        .unwrap();
        c_idx += 1;
        n_constraints += 1;
    }

    // ───── Cost variable constraints ─────
    match cost_type {
        DelayCostType::FiniteSteps1_3Min
        | DelayCostType::FiniteSteps1_5Min
        | DelayCostType::FiniteSteps123
        | DelayCostType::FiniteSteps12345
        | DelayCostType::FiniteSteps139 => {
            let thr = match cost_type {
                DelayCostType::FiniteSteps1_3Min => DelayCostThresholds::f1_3min(),
                DelayCostType::FiniteSteps1_5Min => DelayCostThresholds::f1_5min(),
                DelayCostType::FiniteSteps123 => DelayCostThresholds::f123(),
                DelayCostType::FiniteSteps12345 => DelayCostThresholds::f12345(),
                DelayCostType::FiniteSteps139 => DelayCostThresholds::f139(),
                _ => unreachable!(),
            };
            for (ti, train) in problem.trains.iter().enumerate() {
                for (vi, visit) in train.visits.iter().enumerate() {
                    if let Some(aimed) = visit.aimed {
                        for thr_idx in (0..thr.thresholds.len()).rev() {
                            let (threshold, _cost) = thr.thresholds[thr_idx];
                            // t - aimed <= threshold + M * thr_var
                            // → t - M * thr_var <= threshold + aimed
                            writeln!(
                                w,
                                " c{}: t_{}_{} - {} thr_{}_{}_{} <= {}",
                                c_idx,
                                ti, vi,
                                BIG_M,
                                ti, vi, threshold,
                                threshold + aimed,
                            )
                            .unwrap();
                            c_idx += 1;
                            n_constraints += 1;
                        }
                    }
                }
            }
        }
        DelayCostType::InfiniteSteps60
        | DelayCostType::InfiniteSteps180
        | DelayCostType::InfiniteSteps360 => {
            let interval = match cost_type {
                DelayCostType::InfiniteSteps60 => 60,
                DelayCostType::InfiniteSteps180 => 180,
                DelayCostType::InfiniteSteps360 => 360,
                _ => unreachable!(),
            };
            for (ti, train) in problem.trains.iter().enumerate() {
                for (vi, visit) in train.visits.iter().enumerate() {
                    if let Some(aimed) = visit.aimed {
                        // interval * steps >= t - aimed
                        //                  → t - interval * steps <= aimed
                        writeln!(
                            w,
                            " c{}: t_{}_{} - {} steps_{}_{} <= {}",
                            c_idx,
                            ti, vi,
                            interval,
                            ti, vi,
                            aimed,
                        )
                        .unwrap();
                        c_idx += 1;
                        n_constraints += 1;
                    }
                }
            }
        }
        DelayCostType::Continuous => {
            for (ti, train) in problem.trains.iter().enumerate() {
                for (vi, visit) in train.visits.iter().enumerate() {
                    if let Some(aimed) = visit.aimed {
                        // delay >= t - aimed → t - delay <= aimed
                        writeln!(
                            w,
                            " c{}: t_{}_{} - delay_{}_{} <= {}",
                            c_idx, ti, vi, ti, vi, aimed,
                        )
                        .unwrap();
                        c_idx += 1;
                        n_constraints += 1;
                    }
                }
            }
        }
    }

    // ───── Bounds ─────
    writeln!(w, "Bounds").unwrap();
    for (ti, train) in problem.trains.iter().enumerate() {
        for (vi, visit) in train.visits.iter().enumerate() {
            writeln!(
                w,
                " {} <= t_{}_{} <= {}",
                visit.earliest, ti, vi, BIG_M
            )
            .unwrap();
            n_vars += 1;
        }
    }

    // delay vars (continuous) bounds
    if let DelayCostType::Continuous = cost_type {
        for (ti, train) in problem.trains.iter().enumerate() {
            for (vi, visit) in train.visits.iter().enumerate() {
                if visit.aimed.is_some() {
                    writeln!(w, " 0 <= delay_{}_{} <= {}", ti, vi, BIG_M).unwrap();
                    n_vars += 1;
                }
            }
        }
    }

    // ───── Binary vars ─────
    writeln!(w, "Binary").unwrap();
    // y vars
    for ((t1, v1), (t2, v2)) in &pairs {
        if v1 + 1 >= problem.trains[*t1].visits.len() {
            continue;
        }
        if v2 + 1 >= problem.trains[*t2].visits.len() {
            continue;
        }
        writeln!(w, " y_{}_{}_{}_{}", t1, v1, t2, v2).unwrap();
        n_vars += 1;
    }

    // threshold vars (binary)
    if matches!(
        cost_type,
        DelayCostType::FiniteSteps1_3Min
            | DelayCostType::FiniteSteps1_5Min
            | DelayCostType::FiniteSteps123
            | DelayCostType::FiniteSteps12345
            | DelayCostType::FiniteSteps139
    ) {
        let thr = match cost_type {
            DelayCostType::FiniteSteps1_3Min => DelayCostThresholds::f1_3min(),
            DelayCostType::FiniteSteps1_5Min => DelayCostThresholds::f1_5min(),
            DelayCostType::FiniteSteps123 => DelayCostThresholds::f123(),
            DelayCostType::FiniteSteps12345 => DelayCostThresholds::f12345(),
            DelayCostType::FiniteSteps139 => DelayCostThresholds::f139(),
            _ => unreachable!(),
        };
        for (ti, train) in problem.trains.iter().enumerate() {
            for (vi, visit) in train.visits.iter().enumerate() {
                if visit.aimed.is_some() {
                    for (threshold, _cost) in &thr.thresholds {
                        writeln!(w, " thr_{}_{}_{}", ti, vi, threshold).unwrap();
                        n_vars += 1;
                    }
                }
            }
        }
    }

    // ───── Integer vars (for InfiniteSteps) ─────
    if matches!(
        cost_type,
        DelayCostType::InfiniteSteps60
            | DelayCostType::InfiniteSteps180
            | DelayCostType::InfiniteSteps360
    ) {
        writeln!(w, "General").unwrap();
        for (ti, train) in problem.trains.iter().enumerate() {
            for (vi, visit) in train.visits.iter().enumerate() {
                if visit.aimed.is_some() {
                    writeln!(w, " steps_{}_{}", ti, vi).unwrap();
                    n_vars += 1;
                }
            }
        }
    }

    writeln!(w, "End").unwrap();
    let _ = cost_var_count;
    (n_vars, n_constraints)
}

// ────────────────────────────────────────────────────────────
// MILP-TI LP Writer (port 1:1 từ src/solvers/milp/milp_ti.rs)
// ────────────────────────────────────────────────────────────
fn write_ti_lp(
    path: &str,
    problem: &Problem,
    cost_type: DelayCostType,
    interval: i32,
    big_m: i32,
) -> (usize, usize) {
    let f = File::create(path).expect("create .lp");
    let mut w = BufWriter::new(f);

    let mut n_vars = 0usize;
    let mut n_constraints = 0usize;

    writeln!(w, "\\ TRP MILP-TI, exported by export_lp.rs").unwrap();
    writeln!(
        w,
        "\\ interval={}s, big_m_range={}s",
        interval, big_m
    )
    .unwrap();

    let round = |t: i32| ((t + interval / 2) / interval) * interval;

    // Time discretization (1:1 Rust lines 226-244)
    let time_disc: Vec<Vec<Vec<i32>>> = problem
        .trains
        .iter()
        .map(|train| {
            train
                .visits
                .iter()
                .map(|visit| {
                    let mut slots = Vec::new();
                    let mut t = round(visit.earliest);
                    while t < visit.earliest + big_m {
                        slots.push(t);
                        t += interval;
                    }
                    slots
                })
                .collect()
        })
        .collect();

    // Conflicting resources map (1:1 Rust lines 33-39)
    let mut conflicting: HashMap<usize, Vec<usize>> = HashMap::new();
    for (a, b) in problem.conflicts.iter() {
        conflicting.entry(*a).or_default().push(*b);
        if *a != *b {
            conflicting.entry(*b).or_default().push(*a);
        }
    }

    let mut resource_visits: Vec<Vec<(usize, usize)>> = Vec::new();
    for (ti, train) in problem.trains.iter().enumerate() {
        for (vi, visit) in train.visits.iter().enumerate() {
            while resource_visits.len() <= visit.resource_id {
                resource_visits.push(Vec::new());
            }
            resource_visits[visit.resource_id].push((ti, vi));
        }
    }

    // ───── Objective (1:1 Rust lines 55-69) ─────
    writeln!(w, "Minimize").unwrap();
    writeln!(w, " obj:").unwrap();

    let mut obj_terms: Vec<String> = Vec::new();
    for (ti, train) in problem.trains.iter().enumerate() {
        for (vi, _visit) in train.visits.iter().enumerate() {
            for &slot_t in &time_disc[ti][vi] {
                let cost = train.visit_delay_cost(cost_type, vi, slot_t) as i32;
                if cost > 0 {
                    obj_terms.push(format!("{} x_{}_{}_{}", cost, ti, vi, slot_t));
                }
                n_vars += 1;
            }
        }
    }
    write_terms(&mut w, &obj_terms);

    writeln!(w, "Subject To").unwrap();

    // ───── CONSTRAINT 1: Selection (1:1 Rust lines 72-85) ─────
    let mut c_idx = 0usize;
    for (ti, train) in problem.trains.iter().enumerate() {
        for (vi, _visit) in train.visits.iter().enumerate() {
            let slots = &time_disc[ti][vi];
            let mut sum_terms = String::new();
            for (i, &slot_t) in slots.iter().enumerate() {
                if i == 0 {
                    sum_terms.push_str(&format!("x_{}_{}_{}", ti, vi, slot_t));
                } else {
                    sum_terms.push_str(&format!(" + x_{}_{}_{}", ti, vi, slot_t));
                }
            }
            writeln!(w, " c{}: {} = 1", c_idx, sum_terms).unwrap();
            c_idx += 1;
            n_constraints += 1;
        }
    }

    // ───── CONSTRAINT 2: Travel (1:1 Rust lines 88-115) ─────
    for (ti, train) in problem.trains.iter().enumerate() {
        for (v1_idx, visit) in train.visits.iter().enumerate() {
            if v1_idx + 1 >= train.visits.len() {
                continue;
            }
            let v2_idx = v1_idx + 1;
            for &t1 in &time_disc[ti][v1_idx] {
                let mut constraint: Vec<i32> = Vec::new();
                for &t2 in &time_disc[ti][v2_idx] {
                    let can_reach = t1 + visit.travel_time <= t2;
                    if !can_reach {
                        constraint.push(t2);
                    }
                }
                if constraint.len() > 1 {
                    // Rust: push v1, sum <= 1
                    let mut s = String::new();
                    s.push_str(&format!("x_{}_{}_{}", ti, v1_idx, t1));
                    for t2 in &constraint {
                        s.push_str(&format!(" + x_{}_{}_{}", ti, v2_idx, t2));
                    }
                    writeln!(w, " c{}: {} <= 1", c_idx, s).unwrap();
                    c_idx += 1;
                    n_constraints += 1;
                }
            }
        }
    }

    // ───── CONSTRAINT 3: Resource conflict (1:1 Rust lines 117-149) ─────
    for (t1_idx, train) in problem.trains.iter().enumerate() {
        eprintln!("   c3 t{}", t1_idx);
        for (v1_idx, visit) in train.visits.iter().enumerate() {
            if let Some(conf_res) = conflicting.get(&visit.resource_id) {
                for &other_res in conf_res.iter() {
                    if other_res >= resource_visits.len() {
                        continue;
                    }
                    for &(t2_idx, v2_idx) in resource_visits[other_res].iter() {
                        if t2_idx == t1_idx {
                            continue;
                        }
                        for &t1_in in &time_disc[t1_idx][v1_idx] {
                            for &t2_in in &time_disc[t2_idx][v2_idx] {
                                let t1_out = t1_in + visit.travel_time;
                                let t2_out = t2_in
                                    + problem.trains[t2_idx].visits[v2_idx]
                                        .travel_time;
                                let separation = (t2_in - t1_out).max(t1_in - t2_out);
                                let has_separation = separation >= 0;
                                if !has_separation {
                                    writeln!(
                                        w,
                                        " c{}: x_{}_{}_{} + x_{}_{}_{} <= 1",
                                        c_idx,
                                        t1_idx, v1_idx, t1_in,
                                        t2_idx, v2_idx, t2_in,
                                    )
                                    .unwrap();
                                    c_idx += 1;
                                    n_constraints += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ───── Bounds: binary vars 0/1 (implicit) ─────
    writeln!(w, "Binary").unwrap();
    for (ti, train) in problem.trains.iter().enumerate() {
        for (vi, _visit) in train.visits.iter().enumerate() {
            for &slot_t in &time_disc[ti][vi] {
                writeln!(w, " x_{}_{}_{}", ti, vi, slot_t).unwrap();
            }
        }
    }

    writeln!(w, "End").unwrap();
    (n_vars, n_constraints)
}

// ────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────

/// Visit pairs giữa 2 tàu khác nhau dùng conflicting resources.
fn visit_conflicts(problem: &Problem) -> Vec<((usize, usize), (usize, usize))> {
    use std::collections::HashSet;
    let mut conflicts = Vec::new();
    let resource_conflicts: HashSet<(usize, usize)> =
        problem.conflicts.iter().copied().collect();
    for train_idx1 in 0..problem.trains.len() {
        for train_idx2 in (train_idx1 + 1)..problem.trains.len() {
            for visit_idx1 in 0..problem.trains[train_idx1].visits.len() {
                for visit_idx2 in 0..problem.trains[train_idx2].visits.len() {
                    let r1 = problem.trains[train_idx1].visits[visit_idx1].resource_id;
                    let r2 = problem.trains[train_idx2].visits[visit_idx2].resource_id;
                    let is_conflict1 = resource_conflicts.contains(&(r1, r2));
                    let is_conflict2 = resource_conflicts.contains(&(r2, r1));
                    if is_conflict1 || is_conflict2 {
                        conflicts.push(((train_idx1, visit_idx1), (train_idx2, visit_idx2)));
                    }
                }
            }
        }
    }
    conflicts
}

/// Write một dãy obj_terms vào LP file, chia thành nhiều dòng (LP format
/// có giới hạn ~560 ký tự/dòng theo CPLEX).
fn write_terms<W: Write>(w: &mut W, terms: &[String]) {
    const MAX_LEN: usize = 400;
    let mut line = String::from("    ");
    let mut first = true;
    for term in terms {
        let prefix = if first { "" } else { " + " };
        if line.len() + prefix.len() + term.len() > MAX_LEN {
            writeln!(w, "{}", line).unwrap();
            line = String::from("    + ");
            line.push_str(term);
        } else {
            line.push_str(prefix);
            line.push_str(term);
        }
        first = false;
    }
    if !line.trim().is_empty() {
        writeln!(w, "{}", line).unwrap();
    }
}
