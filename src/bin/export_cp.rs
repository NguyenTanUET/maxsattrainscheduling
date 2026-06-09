//! Standalone binary: build TRP CP model trong Rust, export .cpo file.
//!
//! Mục đích: tương tự `export_lp` cho MILP, nhưng cho CP Optimizer.
//!
//! Workflow:
//! ```text
//! instance.txt -- export_cp --> .cpo file -- cpoptimizer CLI --> output (text)
//! ```
//!
//! CP formulation:
//! - Mỗi visit là một `intervalVar` với size = travel_time
//! - Precedence trong tàu: endBeforeStart(v[i], v[i+1])
//! - Disjunctive resource: noOverlap([visits on resource r])
//! - Objective: minimize sum of delay costs
//!
//! Usage:
//!   cargo build --release --bin export_cp
//!   target/release/export_cp <instance.txt> <cost_type> [out_dir]

use ddd::parser;
use ddd::problem::{DelayCostThresholds, DelayCostType, DelayMeasurementType, Problem};

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

const HORIZON: i32 = 2 * 6 * 3600; // 43200s = 12 giờ

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: {} <instance.txt> <cost_type> [out_dir=.]",
            args[0]
        );
        eprintln!("\nCost types: finsteps123, finsteps12345, finsteps139,");
        eprintln!("            finsteps1_3min, finsteps1_5min,");
        eprintln!("            infsteps60, infsteps180, infsteps360, cont");
        std::process::exit(1);
    }

    let instance_path = &args[1];
    let cost_str = &args[2];
    let out_dir = if args.len() > 3 {
        args[3].clone()
    } else {
        ".".to_string()
    };

    let cost_type = parse_cost_type(cost_str).unwrap_or_else(|| {
        eprintln!("Unknown cost type: {}", cost_str);
        std::process::exit(1);
    });

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

    let cpo_path = format!("{}/{}_cp_{}.cpo", out_dir, stem, cost_str);

    let start = Instant::now();
    let (n_vars, n_constraints) = write_cp_cpo(&cpo_path, problem, cost_type);
    println!("  Build time: {:.2}s", start.elapsed().as_secs_f64());
    println!("  Interval vars: {}", n_vars);
    println!("  Constraints: {}", n_constraints);
    println!("  -> {}", cpo_path);
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

/// Write CP Optimizer .cpo file format.
///
/// Format mẫu:
/// ```
/// // comment
/// intervalVar v_0_0 = intervalVar(start=0..43200, size=30);
/// constraints {
///     endBeforeStart(v_0_0, v_0_1);
///     noOverlap([v_0_1, v_1_3, v_2_5]);
/// }
/// dexpr int delay_0_0 = max(0, startOf(v_0_0) - 1000);
/// dexpr int cost_0_0 = ... ;
/// objective minimize sum of cost terms;
/// ```
fn write_cp_cpo(
    path: &str,
    problem: &Problem,
    cost_type: DelayCostType,
) -> (usize, usize) {
    let f = File::create(path).expect("create .cpo");
    let mut w = BufWriter::new(f);

    let mut n_vars = 0usize;
    let mut n_constraints = 0usize;

    writeln!(w, "// TRP CP Model, exported by export_cp.rs").unwrap();
    writeln!(w, "// Cost type: {:?}", cost_type).unwrap();
    writeln!(w).unwrap();

    // ───── Interval variables ─────
    writeln!(w, "// Interval variables (one per visit)").unwrap();
    for (ti, train) in problem.trains.iter().enumerate() {
        for (vi, visit) in train.visits.iter().enumerate() {
            // intervalVar v_i_j = intervalVar(start=earliest..HORIZON, size=travel_time);
            writeln!(
                w,
                "intervalVar v_{}_{} = intervalVar(start={}..{}, size={});",
                ti, vi, visit.earliest, HORIZON, visit.travel_time,
            )
            .unwrap();
            n_vars += 1;
        }
    }
    writeln!(w).unwrap();

    // ───── Build resource → list of visits map ─────
    let mut resource_intervals: std::collections::HashMap<usize, Vec<(usize, usize)>> =
        std::collections::HashMap::new();
    for (ti, train) in problem.trains.iter().enumerate() {
        for (vi, visit) in train.visits.iter().enumerate() {
            resource_intervals
                .entry(visit.resource_id)
                .or_default()
                .push((ti, vi));
        }
    }

    // ───── Constraints block ─────
    writeln!(w, "constraints {{").unwrap();

    // Travel time / precedence within train
    writeln!(w, "  // Precedence within train").unwrap();
    for (ti, train) in problem.trains.iter().enumerate() {
        for vi in 0..train.visits.len().saturating_sub(1) {
            writeln!(
                w,
                "  endBeforeStart(v_{}_{}, v_{}_{});",
                ti, vi, ti, vi + 1
            )
            .unwrap();
            n_constraints += 1;
        }
    }

    // Disjunctive resource constraints (noOverlap)
    writeln!(w, "  // Disjunctive resource constraints").unwrap();
    let conflict_set: std::collections::HashSet<(usize, usize)> =
        problem.conflicts.iter().copied().collect();

    let mut resource_ids: Vec<&usize> = resource_intervals.keys().collect();
    resource_ids.sort();

    for &res_id in &resource_ids {
        if !conflict_set.contains(&(*res_id, *res_id)) {
            continue; // skip non-exclusive resource
        }
        let visits = &resource_intervals[res_id];
        if visits.len() < 2 {
            continue; // not a conflict if only 1 visit
        }

        // Filter visits with conflicting train pairs
        // (In TRP, no_overlap on resource is sufficient since all visits compete)
        let visit_names: Vec<String> = visits
            .iter()
            .map(|(ti, vi)| format!("v_{}_{}", ti, vi))
            .collect();
        let array = visit_names.join(", ");
        writeln!(w, "  noOverlap([{}]);", array).unwrap();
        n_constraints += 1;
    }

    writeln!(w, "}}").unwrap();
    writeln!(w).unwrap();

    // ───── Objective: minimize total delay cost ─────
    writeln!(w, "// Delay cost expressions").unwrap();

    let mut cost_exprs: Vec<String> = Vec::new();

    match cost_type {
        DelayCostType::Continuous => {
            // Linear: cost = max(0, startOf(v) - aimed)
            for (ti, train) in problem.trains.iter().enumerate() {
                for (vi, visit) in train.visits.iter().enumerate() {
                    if let Some(aimed) = visit.aimed {
                        writeln!(
                            w,
                            "dexpr int delay_{}_{} = max(0, startOf(v_{}_{}) - {});",
                            ti, vi, ti, vi, aimed
                        )
                        .unwrap();
                        cost_exprs.push(format!("delay_{}_{}", ti, vi));
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
            // ceil(delay / interval) = (delay + interval - 1) / interval
            for (ti, train) in problem.trains.iter().enumerate() {
                for (vi, visit) in train.visits.iter().enumerate() {
                    if let Some(aimed) = visit.aimed {
                        writeln!(
                            w,
                            "dexpr int delay_{}_{} = max(0, startOf(v_{}_{}) - {});",
                            ti, vi, ti, vi, aimed
                        )
                        .unwrap();
                        // cost = ceil(delay / interval) = (delay + interval - 1) / interval
                        writeln!(
                            w,
                            "dexpr int cost_{}_{} = (delay_{}_{} + {}) / {};",
                            ti, vi, ti, vi, interval - 1, interval
                        )
                        .unwrap();
                        cost_exprs.push(format!("cost_{}_{}", ti, vi));
                    }
                }
            }
        }
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
            // Bậc thang: thresholds [(360,3), (180,2), (0,1)] cho finsteps123
            // cost = max(c_k * (delay > t_k)) cho mỗi threshold
            // Tương đương: sum cho mỗi (threshold, cost_diff) → 1 nếu delay > threshold
            //
            // Iterate từ thresholds (đã sort theo cost giảm dần)
            for (ti, train) in problem.trains.iter().enumerate() {
                for (vi, visit) in train.visits.iter().enumerate() {
                    if let Some(aimed) = visit.aimed {
                        writeln!(
                            w,
                            "dexpr int delay_{}_{} = max(0, startOf(v_{}_{}) - {});",
                            ti, vi, ti, vi, aimed
                        )
                        .unwrap();

                        // Build cost expression theo bậc thang
                        // cost = sum_{k} cost_diff_k * (delay > threshold_k)
                        let mut parts = Vec::new();
                        for thr_idx in (0..thr.thresholds.len()).rev() {
                            let prev_cost = thr
                                .thresholds
                                .get(thr_idx + 1)
                                .map(|x| x.1)
                                .unwrap_or(0);
                            let (threshold, cost_val) = thr.thresholds[thr_idx];
                            let diff = cost_val as i32 - prev_cost as i32;
                            assert!(diff > 0);
                            // CPO ternary: (delay > threshold ? diff : 0)
                            // OR: diff * (delay > threshold)
                            // CPO syntax: dùng if-then-else hoặc boolean->int
                            parts.push(format!(
                                "({} * (delay_{}_{} > {}))",
                                diff, ti, vi, threshold
                            ));
                        }
                        let expr = parts.join(" + ");
                        writeln!(
                            w,
                            "dexpr int cost_{}_{} = {};",
                            ti, vi, expr
                        )
                        .unwrap();
                        cost_exprs.push(format!("cost_{}_{}", ti, vi));
                    }
                }
            }
        }
    }

    writeln!(w).unwrap();

    // Write objective
    writeln!(w, "// Objective: minimize total delay cost").unwrap();
    if cost_exprs.is_empty() {
        writeln!(w, "minimize 0;").unwrap();
    } else {
        writeln!(w, "minimize").unwrap();
        // Multi-line sum if too many
        let chunks: Vec<String> = cost_exprs.chunks(10).map(|c| c.join(" + ")).collect();
        let sum_expr = chunks.join("\n    + ");
        writeln!(w, "    {};", sum_expr).unwrap();
    }

    (n_vars, n_constraints)
}
