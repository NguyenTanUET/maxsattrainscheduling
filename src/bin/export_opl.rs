//! Standalone binary: build TRP CP model dạng OPL, export .mod + .dat files.
//!
//! OPL (Optimization Programming Language) là IBM modeling language hỗ trợ
//! cả CPLEX MILP và CP Optimizer. `oplrun` CLI đọc .mod + .dat → solve →
//! print output.
//!
//! Workflow:
//! ```text
//! instance.txt -- export_opl --> .mod + .dat -- oplrun CLI --> output.txt
//! ```
//!
//! Usage:
//!   cargo build --release --bin export_opl
//!   target/release/export_opl <instance.txt> <cost_type> [out_dir]

use ddd::parser;
use ddd::problem::{DelayCostThresholds, DelayCostType, DelayMeasurementType, Problem};

use std::collections::HashMap;
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

    let mod_path = format!("{}/{}_cp_{}.mod", out_dir, stem, cost_str);
    let dat_path = format!("{}/{}_cp_{}.dat", out_dir, stem, cost_str);

    let start = Instant::now();
    write_opl_mod(&mod_path, cost_type);
    write_opl_dat(&dat_path, problem);
    println!("  Build time: {:.2}s", start.elapsed().as_secs_f64());
    println!("  -> {}", mod_path);
    println!("  -> {}", dat_path);
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
// Write OPL .mod file (template + cost expression)
// ────────────────────────────────────────────────────────────
fn write_opl_mod(path: &str, cost_type: DelayCostType) {
    let f = File::create(path).expect("create .mod");
    let mut w = BufWriter::new(f);

    // Read solver parameters from env vars
    let time_limit = env::var("CPO_TIME_LIMIT")
        .ok()
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(120);
    let workers = env::var("CPO_WORKERS")
        .ok()
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(4);

    writeln!(w, "// TRP CP Model, exported by export_opl.rs").unwrap();
    writeln!(w, "// Cost type: {:?}", cost_type).unwrap();
    writeln!(w).unwrap();
    writeln!(w, "using CP;").unwrap();
    writeln!(w).unwrap();

    // ───── CP Optimizer parameters block ─────
    writeln!(w, "// CP Optimizer parameters (set before solve)").unwrap();
    writeln!(w, "execute SETTINGS {{").unwrap();
    writeln!(w, "  cp.param.Workers = {};", workers).unwrap();
    writeln!(w, "  cp.param.TimeLimit = {};", time_limit).unwrap();
    writeln!(w, "  cp.param.OptimalityTolerance = 0.001;").unwrap();
    writeln!(w, "  // Stronger inference for scheduling propagation").unwrap();
    writeln!(w, "  cp.param.NoOverlapInferenceLevel = \"Extended\";").unwrap();
    writeln!(w, "  cp.param.PrecedenceInferenceLevel = \"Extended\";").unwrap();
    writeln!(w, "  cp.param.IntervalSequenceInferenceLevel = \"Extended\";").unwrap();
    writeln!(w, "}}").unwrap();
    writeln!(w).unwrap();

    // ───── Data declarations ─────
    writeln!(w, "// Data (from .dat file)").unwrap();
    writeln!(w, "int nVisits = ...;").unwrap();
    writeln!(w, "range Visits = 1..nVisits;").unwrap();
    writeln!(w, "int travelTime[Visits] = ...;").unwrap();
    writeln!(w, "int earliest[Visits] = ...;").unwrap();
    writeln!(w, "int aimed[Visits] = ...;  // aimed time (có thể âm), 0 nếu không tính cost").unwrap();
    writeln!(w, "{{int}} VisitsWithAimed = ...;  // set visits có aimed (cost contribution)").unwrap();
    writeln!(w).unwrap();

    writeln!(w, "// Precedence within train").unwrap();
    writeln!(w, "tuple Edge {{ int u; int v; }}").unwrap();
    writeln!(w, "{{Edge}} PrecArcs = ...;").unwrap();
    writeln!(w).unwrap();

    writeln!(w, "// Disjunctive resource grouping").unwrap();
    writeln!(w, "int nResourceGroups = ...;").unwrap();
    writeln!(w, "range ResGroups = 1..nResourceGroups;").unwrap();
    writeln!(w, "{{int}} VisitsOnResource[ResGroups] = ...;").unwrap();
    writeln!(w).unwrap();

    // ───── Decision variables ─────
    writeln!(w, "// Decision variables: interval per visit").unwrap();
    writeln!(
        w,
        "dvar interval visit[i in Visits] in earliest[i]..{} size travelTime[i];",
        HORIZON
    )
    .unwrap();
    writeln!(w).unwrap();
    writeln!(w, "// Auxiliary delay vars (CHỈ cho visits có aimed)").unwrap();
    writeln!(
        w,
        "dvar int delay[i in VisitsWithAimed] in 0..{};",
        HORIZON
    )
    .unwrap();
    writeln!(w).unwrap();

    // Cost variable per visit (Option C - propagation tốt hơn)
    let cost_upper = match cost_type {
        DelayCostType::Continuous => HORIZON,
        DelayCostType::InfiniteSteps60 => HORIZON / 60 + 1,
        DelayCostType::InfiniteSteps180 => HORIZON / 180 + 1,
        DelayCostType::InfiniteSteps360 => HORIZON / 360 + 1,
        DelayCostType::FiniteSteps1_3Min => 1,
        DelayCostType::FiniteSteps1_5Min => 1,
        DelayCostType::FiniteSteps123 => 3,
        DelayCostType::FiniteSteps12345 => 5,
        DelayCostType::FiniteSteps139 => 9,
    };
    writeln!(w, "// Cost per visit (max cost = {})", cost_upper).unwrap();
    writeln!(
        w,
        "dvar int cost[i in VisitsWithAimed] in 0..{};",
        cost_upper
    )
    .unwrap();
    writeln!(w).unwrap();

    // ───── Objective (Option C: sum of cost vars - đơn giản, propagate tốt) ─────
    writeln!(w, "// Objective: minimize sum of cost vars").unwrap();
    writeln!(w, "minimize sum(i in VisitsWithAimed) cost[i];").unwrap();
    writeln!(w).unwrap();

    // ───── Constraints ─────
    writeln!(w, "subject to {{").unwrap();
    writeln!(w, "  // Delay bound: delay[i] >= startOf(visit[i]) - aimed[i]").unwrap();
    writeln!(w, "  // (delay's lower bound 0 implicitly enforces max(0, ...))").unwrap();
    writeln!(w, "  forall(i in VisitsWithAimed)").unwrap();
    writeln!(w, "    delay[i] >= startOf(visit[i]) - aimed[i];").unwrap();
    writeln!(w).unwrap();

    // Cost binding constraints (Option C)
    writeln!(w, "  // Cost binding (linear/implication theo cost type)").unwrap();
    match cost_type {
        DelayCostType::Continuous => {
            // cost[i] >= delay[i]; minimizing → cost = delay
            writeln!(w, "  forall(i in VisitsWithAimed)").unwrap();
            writeln!(w, "    cost[i] >= delay[i];").unwrap();
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
            // cost * interval >= delay → cost = ceil(delay/interval) khi minimize
            writeln!(w, "  forall(i in VisitsWithAimed)").unwrap();
            writeln!(w, "    cost[i] * {} >= delay[i];", interval).unwrap();
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
            // Cho mỗi (threshold, cost_val):
            //   (delay > threshold) => (cost >= cost_val)
            for (threshold, cost_val) in &thr.thresholds {
                writeln!(w, "  forall(i in VisitsWithAimed)").unwrap();
                writeln!(
                    w,
                    "    (delay[i] > {}) => (cost[i] >= {});",
                    threshold, cost_val
                )
                .unwrap();
            }
        }
    }
    writeln!(w).unwrap();

    writeln!(w, "  // Precedence within train").unwrap();
    writeln!(w, "  forall(<i, j> in PrecArcs)").unwrap();
    writeln!(w, "    endBeforeStart(visit[i], visit[j]);").unwrap();
    writeln!(w).unwrap();
    writeln!(w, "  // Disjunctive resource constraints").unwrap();
    writeln!(w, "  forall(r in ResGroups)").unwrap();
    writeln!(w, "    noOverlap(all(i in VisitsOnResource[r]) visit[i]);").unwrap();
    writeln!(w, "}}").unwrap();

    writeln!(w).unwrap();
    // ───── Output block to print solution ─────
    writeln!(w, "// Print solution variables").unwrap();
    writeln!(w, "execute {{").unwrap();
    writeln!(w, "  writeln(\"OBJECTIVE: \", cp.getObjValue());").unwrap();
    writeln!(w, "  for (var i in Visits) {{").unwrap();
    writeln!(
        w,
        "    writeln(\"v[\", i, \"]: start=\", visit[i].start, \" end=\", visit[i].end);"
    )
    .unwrap();
    writeln!(w, "  }}").unwrap();
    writeln!(w, "}}").unwrap();
}

// ────────────────────────────────────────────────────────────
// Write OPL .dat file (instance data)
// ────────────────────────────────────────────────────────────
fn write_opl_dat(path: &str, problem: &Problem) {
    let f = File::create(path).expect("create .dat");
    let mut w = BufWriter::new(f);

    // Build flat visit list with mapping (train, visit_idx) → flat index (1-based)
    let mut visit_index: HashMap<(usize, usize), usize> = HashMap::new();
    let mut flat_visits: Vec<(usize, usize)> = Vec::new();
    for (ti, train) in problem.trains.iter().enumerate() {
        for (vi, _visit) in train.visits.iter().enumerate() {
            let idx = flat_visits.len() + 1; // OPL is 1-indexed
            visit_index.insert((ti, vi), idx);
            flat_visits.push((ti, vi));
        }
    }

    let n_visits = flat_visits.len();
    writeln!(w, "// TRP instance data, exported by export_opl.rs").unwrap();
    writeln!(w).unwrap();
    writeln!(w, "nVisits = {};", n_visits).unwrap();

    // travelTime
    write!(w, "travelTime = [").unwrap();
    for (i, &(ti, vi)) in flat_visits.iter().enumerate() {
        let tt = problem.trains[ti].visits[vi].travel_time;
        if i > 0 {
            write!(w, ", ").unwrap();
        }
        if i % 20 == 0 && i > 0 {
            writeln!(w).unwrap();
            write!(w, "  ").unwrap();
        }
        write!(w, "{}", tt).unwrap();
    }
    writeln!(w, "];").unwrap();

    // earliest
    write!(w, "earliest = [").unwrap();
    for (i, &(ti, vi)) in flat_visits.iter().enumerate() {
        let e = problem.trains[ti].visits[vi].earliest;
        if i > 0 {
            write!(w, ", ").unwrap();
        }
        if i % 20 == 0 && i > 0 {
            writeln!(w).unwrap();
            write!(w, "  ").unwrap();
        }
        write!(w, "{}", e).unwrap();
    }
    writeln!(w, "];").unwrap();

    // aimed (giá trị thực, kể cả âm; 0 cho visits không có aimed - sẽ không dùng)
    write!(w, "aimed = [").unwrap();
    let mut aimed_visit_indices: Vec<usize> = Vec::new();
    for (i, &(ti, vi)) in flat_visits.iter().enumerate() {
        let (a, has_aimed) = match problem.trains[ti].visits[vi].aimed {
            Some(x) => (x, true),
            None => (0, false),
        };
        if has_aimed {
            aimed_visit_indices.push(i + 1); // OPL 1-indexed
        }
        if i > 0 {
            write!(w, ", ").unwrap();
        }
        if i % 20 == 0 && i > 0 {
            writeln!(w).unwrap();
            write!(w, "  ").unwrap();
        }
        write!(w, "{}", a).unwrap();
    }
    writeln!(w, "];").unwrap();
    writeln!(w).unwrap();

    // Set of visit indices có aimed (cost contribution)
    write!(w, "VisitsWithAimed = {{").unwrap();
    for (idx, &v) in aimed_visit_indices.iter().enumerate() {
        if idx > 0 {
            write!(w, ", ").unwrap();
        }
        write!(w, "{}", v).unwrap();
    }
    writeln!(w, "}};").unwrap();
    writeln!(w).unwrap();

    // Precedence arcs (within train)
    writeln!(w, "// Precedence within train: visit i → visit i+1").unwrap();
    write!(w, "PrecArcs = {{").unwrap();
    let mut first_arc = true;
    for (ti, train) in problem.trains.iter().enumerate() {
        for vi in 0..train.visits.len().saturating_sub(1) {
            let u = visit_index[&(ti, vi)];
            let v = visit_index[&(ti, vi + 1)];
            if !first_arc {
                write!(w, ", ").unwrap();
            }
            write!(w, "<{}, {}>", u, v).unwrap();
            first_arc = false;
        }
    }
    writeln!(w, "}};").unwrap();
    writeln!(w).unwrap();

    // Build resource → visits map, only resources with self-conflict
    let mut resource_visits: HashMap<usize, Vec<usize>> = HashMap::new();
    for (ti, train) in problem.trains.iter().enumerate() {
        for (vi, visit) in train.visits.iter().enumerate() {
            resource_visits
                .entry(visit.resource_id)
                .or_default()
                .push(visit_index[&(ti, vi)]);
        }
    }

    let conflict_set: std::collections::HashSet<(usize, usize)> =
        problem.conflicts.iter().copied().collect();

    // Filter: only resources with (r, r) in conflicts (exclusive)
    let mut resource_groups: Vec<Vec<usize>> = Vec::new();
    let mut resource_ids: Vec<&usize> = resource_visits.keys().collect();
    resource_ids.sort();
    for &res_id in &resource_ids {
        if !conflict_set.contains(&(*res_id, *res_id)) {
            continue;
        }
        let visits = &resource_visits[res_id];
        if visits.len() < 2 {
            continue;
        }
        resource_groups.push(visits.clone());
    }

    writeln!(
        w,
        "// Disjunctive resource groups (one noOverlap each)"
    )
    .unwrap();
    writeln!(w, "nResourceGroups = {};", resource_groups.len()).unwrap();
    writeln!(w, "VisitsOnResource = [").unwrap();
    for (i, group) in resource_groups.iter().enumerate() {
        write!(w, "  {{").unwrap();
        for (j, &v) in group.iter().enumerate() {
            if j > 0 {
                write!(w, ", ").unwrap();
            }
            write!(w, "{}", v).unwrap();
        }
        if i + 1 < resource_groups.len() {
            writeln!(w, "}},").unwrap();
        } else {
            writeln!(w, "}}").unwrap();
        }
    }
    writeln!(w, "];").unwrap();
}
