use std::{any::Any, cell::RefCell, collections::HashSet, fmt::Write, time::Instant};

use ddd::{
    maxsatsolver, parser,
    problem::{self, DelayCostThresholds, DelayCostType, NamedProblem, Visit},
    solvers::{
        ddd as ddd_solvers,
        ladder::{maxsatddd_ladder, maxsatddd_ladder_abstract, maxsatddd_ladder_sc},
        legacy::{maxsat_ddd, maxsat_ti},
        milp::{bigm, milp_ti},
        util::{
            counting_solver,
            greedy::{self, default_heuristic},
            heuristic,
        },
        SolverError,
    },
};

use std::path::PathBuf;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(
    name = "trainscheduler",
    about = "Optimal train scheduling experiments."
)]
struct Opt {
    /// Activate debug mode
    #[structopt(short, long)]
    debug: bool,

    #[structopt(short, long)]
    solvers: Vec<String>,

    #[structopt(long)]
    xml_instances: bool,

    #[structopt(long)]
    txt_instances: bool,

    #[structopt(long)]
    instance_name_filter: Option<String>,

    #[structopt(long)]
    instance_name_exact: bool,

    #[structopt(long)]
    verify_instances: bool,

    #[structopt(long)]
    objective: Option<String>,

    #[structopt(long)]
    other_objective: Option<String>,

    #[structopt(long)]
    json_output: Option<String>,

    /// Toggle extended-precedence-graph + unary energetic-reasoning
    /// preprocessing in `sat_ddd*` solvers (true/false).
    #[structopt(long)]
    satddd_use_extended_precedence_graph: Option<bool>,

    /// `sat_ddd*` only: pre-allocate SAT vars + monotonicity clauses for
    /// every cost-step threshold time at INIT (before iteration 1). On
    /// stepped objectives (e.g. `InfiniteSteps180`) this expands to up to
    /// 100 thresholds per visit. `Continuous` returns empty → no effect.
    /// True/false. Default false (lazy).
    #[structopt(long)]
    satddd_prealloc_cost_thresholds: Option<bool>,

    /// `sat_ddd*` only: pre-seed fixed-precedence (travel-time) rows from
    /// each visit's earliest time point at INIT. Reduces "travel-time
    /// conflict" iterations at the cost of a larger initial CNF.
    /// True/false. Default false (lazy).
    #[structopt(long)]
    satddd_seed_precedence_from_earliest: Option<bool>,

    /// `sat_ddd*` only: pre-seed pairwise AMO conflict clauses for visit
    /// pairs whose earliest occupation intervals overlap by ≥ 180s. Adds
    /// ~7 clauses + 2 aux vars per pair. True/false. Default false (lazy).
    #[structopt(long)]
    satddd_seed_resource_conflicts: Option<bool>,

    /// `sat_ddd*` only: enable SC (Sequential Counter) AMO encoding from
    /// Truong/Kieu/To (ICAART 2025) for resource-clique AMOs of size > 5.
    /// If false, AMOs always use pairwise. True/false. Default true.
    #[structopt(long)]
    satddd_use_sc_amo: Option<bool>,

    /// Toggle eager chain expansion in `maxsat_ddd_ladder_sc`: expand long
    /// travel-time precedence chains into per-step 3-literal clauses instead
    /// of a single 2-literal implication (NOT the SC AMO encoding —
    /// see `add_sc_amo` for that). True/false.
    #[structopt(long)]
    maxsatddd_ladder_sc_use_eager_chain_expansion: Option<bool>,

    /// Toggle precedence-graph preprocessing/propagation in `maxsat_ddd_ladder_sc` (true/false).
    #[structopt(long)]
    maxsatddd_ladder_sc_use_precedence_graph: Option<bool>,

    /// Toggle interval-graph conflict encoding in `maxsat_ddd_ladder_sc` (true/false).
    #[structopt(long)]
    maxsatddd_ladder_sc_use_interval_graph: Option<bool>,

    /// Alias of `maxsatddd_ladder_sc_use_interval_graph`.
    #[structopt(long)]
    maxsatddd_ladder_sc_use_interval_tree: Option<bool>,

    /// Within interval-graph clique-cover encoding, use SC Sequential
    /// Counter AMO (Truong/Kieu/To ICAART 2025) for large cliques. If false,
    /// keep pairwise AMO regardless of clique size. No effect when interval-
    /// graph encoding is off. True/false. Default true.
    #[structopt(long)]
    maxsatddd_ladder_sc_use_sc_amo: Option<bool>,

    /// Lite clique-AMO aggregation for the pair-based conflict path. When on,
    /// the pair scan also accumulates (resource, tau) → visit-set; cliques
    /// of size ≥ 3 get a single AMO encoded after the scan. Effective only
    /// when `--use-interval-graph false`. True/false. Default false.
    #[structopt(long)]
    maxsatddd_ladder_sc_use_touched_clique_amo: Option<bool>,

    /// **Experimental**: enable TRUE SCAMO encoding (Truong/Kieu/To, ICAART
    /// 2025). Currently only Phase 1 (group detection + stats) is wired in;
    /// the encoding itself still uses `add_hybrid_amo`. Use this to inspect
    /// whether the staircase pattern is worth the rewrite.
    #[structopt(long)]
    maxsatddd_ladder_sc_use_scamo: Option<bool>,

    /// Pre-seed precedence rows from each visit's earliest time point at
    /// initialization. Eager: encodes the full forward travel-time chain
    /// before iteration 1; reduces "travel-time conflict" iterations at the
    /// cost of a larger initial CNF (more time points + cost variables).
    /// Effective only when `--use-precedence-graph` or
    /// `--use-eager-chain-expansion` is true. True/false. Default false.
    #[structopt(long)]
    maxsatddd_ladder_sc_seed_from_earliest: Option<bool>,

    /// Pre-allocate SAT vars + monotonicity clauses for every cost-step
    /// threshold time at INIT (before iteration 1). For stepped objectives
    /// (e.g. `InfiniteSteps180`) this can balloon the initial CNF — up to
    /// 100 thresholds per visit. `Continuous` returns empty so the flag has
    /// no effect. Match `maxsatddd_ladder` lazy behaviour with `false`.
    /// True/false. Default false.
    #[structopt(long)]
    maxsatddd_ladder_sc_prealloc_cost_thresholds: Option<bool>,

    /// Objective encoding for `sat_ddd*` solvers: `scpb`, `totalizer`, or `bit_totalizer` (`nsc` accepted as alias).
    #[structopt(long)]
    satddd_objective_encoding: Option<String>,
}

fn parse_delay_cost_type(value: &str) -> Option<DelayCostType> {
    let key = value.to_ascii_lowercase();
    match key.as_str() {
        "finsteps1_5min" => Some(DelayCostType::FiniteSteps1_5Min),
        "finsteps1_3min" => Some(DelayCostType::FiniteSteps1_3Min),
        "finsteps123" => Some(DelayCostType::FiniteSteps123),
        "finsteps12345" => Some(DelayCostType::FiniteSteps12345),
        "finsteps139" => Some(DelayCostType::FiniteSteps139),
        "infsteps60" => Some(DelayCostType::InfiniteSteps60),
        "infsteps180" => Some(DelayCostType::InfiniteSteps180),
        "infsteps360" => Some(DelayCostType::InfiniteSteps360),
        // Alias: "infsteps123" means extending finsteps123 (1/2/3 at 180s steps) without cap.
        "infsteps123" => Some(DelayCostType::InfiniteSteps180),
        "cont" => Some(DelayCostType::Continuous),
        _ => None,
    }
}

fn parse_delay_cost_type_or_panic(value: &str) -> DelayCostType {
    parse_delay_cost_type(value).unwrap_or_else(|| {
        panic!(
            "Unknown objective type '{}'. Supported: finsteps1_5min, finsteps1_3min, finsteps123, finsteps12345, finsteps139, infsteps60, infsteps180, infsteps360, infsteps123, cont",
            value
        )
    })
}

fn parse_sat_objective_encoding(value: &str) -> Option<ddd_solvers::incremental_sat::SatObjectiveEncoding> {
    let key = value.to_ascii_lowercase();
    match key.as_str() {
        "scpb" | "nsc" => Some(ddd_solvers::incremental_sat::SatObjectiveEncoding::Scpb),
        "totalizer" | "incremental_totalizer" | "inc_totalizer" => {
            Some(ddd_solvers::incremental_sat::SatObjectiveEncoding::IncrementalTotalizer)
        }
        "bit_totalizer" | "binary_totalizer" | "bit" | "binary" => {
            Some(ddd_solvers::incremental_sat::SatObjectiveEncoding::BitTotalizer)
        }
        _ => None,
    }
}

fn parse_sat_objective_encoding_or_panic(value: &str) -> ddd_solvers::incremental_sat::SatObjectiveEncoding {
    parse_sat_objective_encoding(value).unwrap_or_else(|| {
        panic!(
            "Unknown SAT objective encoding '{}'. Supported: scpb, totalizer, bit_totalizer",
            value
        )
    })
}

fn is_likely_oom_panic(payload: &(dyn Any + Send)) -> bool {
    let message = if let Some(msg) = payload.downcast_ref::<String>() {
        Some(msg.as_str())
    } else if let Some(msg) = payload.downcast_ref::<&str>() {
        Some(*msg)
    } else {
        None
    };

    let Some(message) = message else {
        return false;
    };

    let message = message.to_ascii_lowercase();
    message.contains("failed to extend generalizedtotalizer encoding")
        || message.contains("glucose reserve failed")
        || message.contains("glucose add_clause_ref failed")
        || message.contains("outofmemory")
        || message.contains("out of memory")
        || message.contains("memory allocation")
}

pub fn xml_instances(mut x: impl FnMut(String, NamedProblem)) {
    let a_instances = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    // let a_instances = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let b_instances = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
    #[allow(unused)]
    let c_instances = [21, 22, 23, 24];

    for instance_id in a_instances
        .into_iter()
        .chain(b_instances)
        .chain(c_instances)
    {
        let filename = format!("instances/Instance{}.xml", instance_id);
        println!("Reading {}", filename);
        #[allow(unused)]
        let problem = parser::read_xml_file(
            &filename,
            problem::DelayMeasurementType::FinalStationArrival,
        );
        x(format!("xml {}", instance_id), problem);
    }
}

pub fn txt_instances(mut x: impl FnMut(String, NamedProblem)) {
    // let a_instances = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    // let b_instances = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
    // #[allow(unused)]
    // let c_instances = [21, 22, 23, 24];

    for (dir, shortname) in [
        ("instances_original", "orig"),
        ("instances_addtracktime", "track"),
        ("instances_addstationtime", "station"),
    ] {
        let instances = ["A", "B"]
            .iter()
            .flat_map(move |n| (1..=12).map(move |i| (n, i)));

        // let instances = instances.skip(16).take(1);

        for (infrastructure, number) in instances {
            let filename = format!("{}/Instance{}{}.txt", dir, infrastructure, number);
            println!("Reading {}", filename);
            #[allow(unused)]
            let (problem, _) = parser::read_txt_file(
                &filename,
                problem::DelayMeasurementType::FinalStationArrival,
                false,
                None,
                |_| {},
            );
            x(
                format!("{}{}{}", shortname, infrastructure, number),
                problem,
            );
            // break;
        }
        // break;
    }
}

pub fn verify_instances(mut x: impl FnMut(String, NamedProblem, Vec<Vec<i32>>) -> Vec<Vec<i32>>) {
    let a_instances = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let b_instances = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
    #[allow(unused)]
    let c_instances = [21, 22, 23, 24];
    let instances = [20];

    for solvertype in ["BigMComplete", "BigMLazyCon"] {
        for instance_id in a_instances
            .iter()
            .chain(b_instances.iter())
            .chain(c_instances.iter())
        {
            let filename = format!("InstanceResults/{}Sol{}.txt", solvertype, instance_id);
            println!("Reading {}", filename);
            #[allow(unused)]
            let (problem, solution) = parser::read_txt_file(
                &filename,
                problem::DelayMeasurementType::FinalStationArrival,
                true,
                None,
                |_| {},
            );

            let new_solution = x(
                format!("{} {}", solvertype, instance_id),
                problem,
                solution.unwrap(),
            );

            // let mut f = std::fs::File::create(&format!("{}.bl.txt", filename)).unwrap();
            // use std::io::Write;
            // parser::read_txt_file(
            //     &filename,
            //     problem::DelayMeasurementType::FinalStationArrival,
            //     true,
            //     Some(new_solution),
            //     |l| {
            //         writeln!(f, "{}", l).unwrap();
            //     },
            // );
        }
    }
}

#[derive(Debug)]
enum SolverType {
    BigMEager,
    BigMLazy,
    MaxSatDddLadderRC2,
    MaxSatDddLadderSc,
    MaxSatDddLadderRC2Abstract,
    MaxSatDddLadderIpamir,
    MaxSatDddCadical,
    MaxSatIdl,
    SatDdd,
    SatDddInc,
    SatDddSc,
    SatDddScTotalizer,
    SatDddScInc,
    SatDddScAddClauses,
    SatDddScFreshAddClauses,
    MipDdd,
    MipHull,
    Greedy,
    GreedyFast,
    GreedyFStr,
    Cutting,
    BinarizedBigMEager10Sec,
    BinarizedBigMEager30Sec,
    BinarizedBigMEager60Sec,
    BinarizedBigMLazy10Sec,
    BinarizedBigMLazy30Sec,
    BinarizedBigMLazy60Sec,
    MaxSatTi,
    MipTi,
    MaxSatDddExternal,
    MaxSatDddIpamir,
    MaxSatDddIncremental,
    MaxSatDddIncrementalNoProp,
    MaxSatDddPairwiseCustomRc2,
    MaxSatDddPairwiseCustomRc2NoProp,
}

const TIMEOUT: f64 = 120.0;

fn mk_env() -> grb::Env {
    let mut env = grb::Env::new("").unwrap();
    env.set(grb::param::Threads, 4).unwrap();
    env.set(grb::param::OutputFlag, 0).unwrap();
    env.set(grb::param::Cuts, 0).unwrap();
    env.set(grb::param::Heuristics, 0.0).unwrap();
    env
}

fn main() {
    pretty_env_logger::env_logger::Builder::from_env(
        pretty_env_logger::env_logger::Env::default().default_filter_or("trace"),
    )
    .init();

    let opt = Opt::from_args();
    println!("{:?}", opt);
    let solvers = opt
        .solvers
        .iter()
        .map(|x| match x.as_str() {
            "bigm_eager" => SolverType::BigMEager,
            "bigm_lazy" => SolverType::BigMLazy,
            "maxsat_ddd" => SolverType::MaxSatDddLadderRC2,
            "maxsat_ddd_ladder" => SolverType::MaxSatDddLadderRC2,
            "maxsat_ddd_abstract" => SolverType::MaxSatDddLadderRC2Abstract,
            "maxsat_ddd_ladder_sc" => SolverType::MaxSatDddLadderSc,
            "maxsat_ddd_ladder_ipamir" => SolverType::MaxSatDddLadderIpamir,
            "maxsat_ddd_cdc" => SolverType::MaxSatDddCadical,
            "maxsat_idl" => SolverType::MaxSatIdl,
            "sat_ddd" => SolverType::SatDdd,
            "sat_ddd_inc" => SolverType::SatDddInc,
            "sat_ddd_sc" => SolverType::SatDddSc,
            "sat_ddd_sc_totalizer" => SolverType::SatDddScTotalizer,
            "sat_ddd_sc_inc" => SolverType::SatDddScInc,
            "sat_ddd_sc_addclauses" => SolverType::SatDddScAddClauses,
            "sat_ddd_sc_fresh_addclauses" => SolverType::SatDddScFreshAddClauses,
            "mip_ddd" => SolverType::MipDdd,
            "mip_hull" => SolverType::MipHull,
            "greedy" => SolverType::Greedy,
            "greedyfast" => SolverType::GreedyFast,
            "greedyfaststrong" => SolverType::GreedyFStr,
            "cutting" => SolverType::Cutting,
            "bin_bigm_eager_10s" => SolverType::BinarizedBigMEager10Sec,
            "bin_bigm_eager_30s" => SolverType::BinarizedBigMEager30Sec,
            "bin_bigm_eager_60s" => SolverType::BinarizedBigMEager60Sec,
            "bin_bigm_lazy_10s" => SolverType::BinarizedBigMLazy10Sec,
            "bin_bigm_lazy_30s" => SolverType::BinarizedBigMLazy30Sec,
            "bin_bigm_lazy_60s" => SolverType::BinarizedBigMLazy60Sec,
            "maxsat_ti" => SolverType::MaxSatTi,
            "mip_ti" => SolverType::MipTi,
            "maxsat_ddd_external" => SolverType::MaxSatDddExternal,
            "maxsat_ddd_ipamir" => SolverType::MaxSatDddIpamir,
            "maxsat_ddd_incremental" => SolverType::MaxSatDddIncremental,
            "maxsat_ddd_incremental_noprop" => SolverType::MaxSatDddIncrementalNoProp,
            "maxsat_ddd_pairwise_customrc2" => SolverType::MaxSatDddPairwiseCustomRc2,
            "maxsat_ddd_pairwise_customrc2_noprop" => SolverType::MaxSatDddPairwiseCustomRc2NoProp,
            _ => panic!("unknown solver type"),
        })
        .collect::<Vec<_>>();
    println!("Using solvers {:?}", solvers);

    let delay_cost_type = opt
        .objective
        .as_deref()
        .map(parse_delay_cost_type_or_panic)
        .unwrap_or(DelayCostType::FiniteSteps123);
    println!("Using delay cost type {:?}", delay_cost_type);

    let other_delay_cost_type = opt
        .other_objective
        .as_deref()
        .map(parse_delay_cost_type_or_panic);

    // Default config = Option B: precedence + touched-clique AMO + SC AMO,
    // with eager-chain-expansion and full interval-graph clique cover OFF.
    // Empirically best on this benchmark; can be overridden via the
    // matching --maxsatddd-ladder-sc-* flags.
    let maxsatddd_ladder_sc_settings = maxsatddd_ladder_sc::MaxSatDddLadderScSettings {
        use_precedence_graph: opt
            .maxsatddd_ladder_sc_use_precedence_graph
            .unwrap_or(true),
        use_eager_chain_expansion: opt
            .maxsatddd_ladder_sc_use_eager_chain_expansion
            .unwrap_or(false),
        use_interval_graph_conflicts: opt
            .maxsatddd_ladder_sc_use_interval_tree
            .or(opt.maxsatddd_ladder_sc_use_interval_graph)
            .unwrap_or(false),
        use_sc_amo: opt.maxsatddd_ladder_sc_use_sc_amo.unwrap_or(true),
        use_touched_clique_amo: opt
            .maxsatddd_ladder_sc_use_touched_clique_amo
            .unwrap_or(true),
        seed_sc_from_earliest: opt
            .maxsatddd_ladder_sc_seed_from_earliest
            .unwrap_or(false),
        use_scamo_encoding: opt.maxsatddd_ladder_sc_use_scamo.unwrap_or(false),
        prealloc_cost_thresholds: opt
            .maxsatddd_ladder_sc_prealloc_cost_thresholds
            .unwrap_or(false),
    };
    println!(
        "MaxSatDddLadderSc settings {:?}",
        maxsatddd_ladder_sc_settings
    );

    let satddd_objective_encoding = opt
        .satddd_objective_encoding
        .as_deref()
        .map(parse_sat_objective_encoding_or_panic)
        .unwrap_or(ddd_solvers::incremental_sat::SatObjectiveEncoding::Scpb);
    println!("SatDdd objective encoding {:?}", satddd_objective_encoding);
    let satddd_settings = ddd_solvers::incremental_sat::SatDddSettings {
        use_extended_precedence_graph: opt
            .satddd_use_extended_precedence_graph
            .unwrap_or(true),
        prealloc_cost_thresholds: opt
            .satddd_prealloc_cost_thresholds
            .unwrap_or(false),
        seed_precedence_from_earliest: opt
            .satddd_seed_precedence_from_earliest
            .unwrap_or(false),
        seed_resource_conflicts: opt
            .satddd_seed_resource_conflicts
            .unwrap_or(false),
        use_sc_amo: opt.satddd_use_sc_amo.unwrap_or(true),
    };
    println!("SatDdd settings {:?}", satddd_settings);

    let perf_out = RefCell::new(String::new());

    let needs_gurobi = solvers.iter().any(|solver| {
        matches!(
            solver,
            SolverType::Greedy
                | SolverType::GreedyFast
                | SolverType::GreedyFStr
                | SolverType::BigMEager
                | SolverType::BigMLazy
                | SolverType::MipDdd
                | SolverType::MipHull
                | SolverType::MaxSatTi
                | SolverType::MipTi
                | SolverType::MaxSatDddExternal
                | SolverType::MaxSatDddIpamir
                | SolverType::MaxSatDddIncremental
                | SolverType::MaxSatDddIncrementalNoProp
                | SolverType::MaxSatDddPairwiseCustomRc2
                | SolverType::MaxSatDddPairwiseCustomRc2NoProp
                | SolverType::BinarizedBigMEager10Sec
                | SolverType::BinarizedBigMEager30Sec
                | SolverType::BinarizedBigMEager60Sec
                | SolverType::BinarizedBigMLazy10Sec
                | SolverType::BinarizedBigMLazy30Sec
                | SolverType::BinarizedBigMLazy60Sec
        )
    });

    let env = if needs_gurobi {
        println!("Starting gurobi environment...");
        let env = mk_env();
        println!("...ok.");
        Some(env)
    } else {
        None
    };
    let get_env = || {
        env.as_ref().expect(
            "Gurobi environment unavailable; configure a Gurobi license or choose a non-Gurobi solver.",
        )
    };

    let matches_instance_filter = |name: &str| {
        opt.instance_name_filter
            .as_deref()
            .map(|filter| {
                if opt.instance_name_exact {
                    name == filter
                } else {
                    name.contains(filter)
                }
            })
            .unwrap_or(true)
    };

    let mut problems: Vec<serde_json::Value> = Default::default();

    let mut solve_it = |name: String, p: NamedProblem| -> Result<Vec<Vec<i32>>, SolverError> {
        if solvers.is_empty() {
            panic!("no solver specified");
        }

        let problemstats = print_problem_stats(&p.problem);

        let mut solves: Vec<serde_json::Value> = Default::default();

        let mut solution = Result::Err(SolverError::NoSolution);
        for solver in solvers.iter() {
            let solve_wall_start = Instant::now();
            hprof::start_frame();
            println!("Starting solver {:?}", solver);
            let mut solve_data = serde_json::Map::new();

            solution =
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match solver {
                    SolverType::Cutting => unimplemented!(),
                    // ddd::solvers::cutting::solve_cutting(
                    //     &p.problem,
                    //     delay_cost_type,
                    //     TIMEOUT,
                    //     &p.train_names,
                    //     &p.resource_names,
                    //     |k, v| {
                    //         solve_data.insert(k, v);
                    //     },
                    // ),
                    SolverType::Greedy => {
                        greedy::solve2(&p.problem, get_env(), delay_cost_type, default_heuristic)
                    }
                    SolverType::GreedyFast => heuristic::solve_heuristic_better(
                        get_env(),
                        &p.problem,
                        delay_cost_type,
                        false,
                        None,
                    )
                    .and_then(|e| e.ok_or(SolverError::NoSolution)),
                    SolverType::GreedyFStr => heuristic::solve_heuristic_better(
                        get_env(),
                        &p.problem,
                        delay_cost_type,
                        true,
                        None,
                    )
                    .and_then(|e| e.ok_or(SolverError::NoSolution)),
                    SolverType::BigMEager => bigm::solve_bigm(
                        get_env(),
                        &mk_env,
                        &p.problem,
                        delay_cost_type,
                        false,
                        TIMEOUT,
                        &p.train_names,
                        &p.resource_names,
                        |k, v| {
                            solve_data.insert(k, v);
                        },
                    ),
                    SolverType::MipHull => bigm::solve_hull(
                        get_env(),
                        &mk_env,
                        &p.problem,
                        delay_cost_type,
                        true,
                        TIMEOUT,
                        &p.train_names,
                        &p.resource_names,
                        |k, v| {
                            solve_data.insert(k, v);
                        },
                    ),
                    SolverType::BigMLazy => bigm::solve_bigm(
                        get_env(),
                        &mk_env,
                        &p.problem,
                        delay_cost_type,
                        true,
                        TIMEOUT,
                        &p.train_names,
                        &p.resource_names,
                        |k, v| {
                            solve_data.insert(k, v);
                        },
                    ),
                    SolverType::MaxSatTi => maxsat_ti::solve(
                        maxsatsolver::Incremental::new(),
                        get_env(),
                        &p.problem,
                        TIMEOUT,
                        delay_cost_type,
                        |k, v| {
                            solve_data.insert(k, v);
                        },
                        10,
                        900,
                    )
                    .map(|(v, _)| v),
                    SolverType::MipTi => milp_ti::solve(
                        get_env(),
                        &p.problem,
                        TIMEOUT,
                        delay_cost_type,
                        |k, v| {
                            solve_data.insert(k, v);
                        },
                        10,
                        900,
                    )
                    .map(|(v, _)| v),
                    SolverType::MaxSatDddExternal => maxsat_ddd::solve(
                        || maxsatsolver::External::new(/* "./uwrmaxsat" */),
                        get_env(),
                        &p.problem,
                        TIMEOUT,
                        delay_cost_type,
                        |k, v| {
                            solve_data.insert(k, v);
                        },
                    )
                    .map(|(v, _)| v),
                    SolverType::MaxSatDddIpamir => maxsat_ddd::solve(
                        || maxsatsolver::Incremental::new(),
                        get_env(),
                        &p.problem,
                        TIMEOUT,
                        delay_cost_type,
                        |k, v| {
                            solve_data.insert(k, v);
                        },
                    )
                    .map(|(v, _)| v),
                    SolverType::MaxSatDddIncremental => maxsat_ddd::solve_incremental(
                        || maxsatsolver::Incremental::new(),
                        get_env(),
                        &p.problem,
                        TIMEOUT,
                        delay_cost_type,
                        true,
                        |k, v| {
                            solve_data.insert(k, v);
                        },
                    )
                    .map(|(v, _)| v),
                    SolverType::MaxSatDddIncrementalNoProp => maxsat_ddd::solve_incremental(
                        || maxsatsolver::Incremental::new(),
                        get_env(),
                        &p.problem,
                        TIMEOUT,
                        delay_cost_type,
                        false,
                        |k, v| {
                            solve_data.insert(k, v);
                        },
                    )
                    .map(|(v, _)| v),
                    SolverType::MaxSatDddPairwiseCustomRc2 => maxsat_ddd::solve_incremental(
                        || {
                            maxsatsolver::CustomRC2Incremental::new(
                                satcoder::solvers::minisat::Solver::new(),
                            )
                        },
                        get_env(),
                        &p.problem,
                        TIMEOUT,
                        delay_cost_type,
                        true,
                        |k, v| {
                            solve_data.insert(k, v);
                        },
                    )
                    .map(|(v, _)| v),
                    SolverType::MaxSatDddPairwiseCustomRc2NoProp => maxsat_ddd::solve_incremental(
                        || {
                            maxsatsolver::CustomRC2Incremental::new(
                                satcoder::solvers::minisat::Solver::new(),
                            )
                        },
                        get_env(),
                        &p.problem,
                        TIMEOUT,
                        delay_cost_type,
                        false,
                        |k, v| {
                            solve_data.insert(k, v);
                        },
                    )
                    .map(|(v, _)| v),
                    SolverType::MaxSatDddLadderSc => {
                        // Reset the per-thread CNF size counters before
                        // invoking solve, then wrap the underlying solver
                        // in CountingSolver so every new_var/add_clause is
                        // counted. The solve function reads the final
                        // counts via counting_solver::get_counts() and
                        // emits them as num_vars_total / num_clauses_total.
                        counting_solver::reset_counts();
                        maxsatddd_ladder_sc::solve_with_settings(
                            &mk_env,
                            counting_solver::CountingSolver::new(
                                satcoder::solvers::minisat::Solver::new(),
                            ),
                            &p.problem,
                            TIMEOUT,
                            delay_cost_type,
                            maxsatddd_ladder_sc_settings,
                            |k, v| {
                                solve_data.insert(k, v);
                            },
                        )
                        .map(|(v, _)| v)
                    }
                    SolverType::MaxSatDddLadderRC2 => {
                        counting_solver::reset_counts();
                        maxsatddd_ladder::solve(
                            &mk_env,
                            counting_solver::CountingSolver::new(
                                satcoder::solvers::minisat::Solver::new(),
                            ),
                            &p.problem,
                            TIMEOUT,
                            delay_cost_type,
                            |k, v| {
                                solve_data.insert(k, v);
                            },
                        )
                        .map(|(v, _)| v)
                    }
                    SolverType::MaxSatDddLadderRC2Abstract => maxsatddd_ladder_abstract::solve(
                        &mk_env,
                        maxsatsolver::CustomRC2Incremental::new(
                            satcoder::solvers::minisat::Solver::new(),
                        ),
                        &p.problem,
                        TIMEOUT,
                        delay_cost_type,
                        |k, v| {
                            solve_data.insert(k, v);
                        },
                    )
                    .map(|(v, _)| v),
                    SolverType::MaxSatDddLadderIpamir => maxsatddd_ladder_abstract::solve(
                        &mk_env,
                        maxsatsolver::Incremental::new(),
                        &p.problem,
                        TIMEOUT,
                        delay_cost_type,
                        |k, v| {
                            solve_data.insert(k, v);
                        },
                    )
                    .map(|(v, _)| v),
                    SolverType::MaxSatDddCadical => maxsatddd_ladder::solve(
                        &mk_env,
                        satcoder::solvers::minisat::Solver::new(),
                        &p.problem,
                        TIMEOUT,
                        delay_cost_type,
                        |k, v| {
                            solve_data.insert(k, v);
                        },
                    )
                    .map(|(v, _)| v),
                    SolverType::SatDdd => ddd_solvers::incremental_sat::solve_with_encoding_and_settings(
                        &mk_env,
                        satcoder::solvers::rustsat_glucose::Solver::new(),
                        &p.problem,
                        TIMEOUT,
                        delay_cost_type,
                        satddd_objective_encoding,
                        satddd_settings,
                        |k, v| {
                            solve_data.insert(k, v);
                        },
                    )
                    .map(|(v, _)| v),
                    SolverType::MaxSatIdl => {
                        eprintln!("Error: IDL solver is not available in this build.");
                        Err(SolverError::NoSolution)
                    }
                    SolverType::SatDddInc => {
                        ddd_solvers::incremental_sat::solve_incremental_with_encoding_and_settings(
                            &mk_env,
                            satcoder::solvers::rustsat_glucose::Solver::new(),
                            &p.problem,
                            TIMEOUT,
                            delay_cost_type,
                            satddd_objective_encoding,
                            satddd_settings,
                            |k, v| {
                                solve_data.insert(k, v);
                            },
                        )
                        .map(|(v, _)| v)
                    }
                    SolverType::SatDddSc => {
                        ddd_solvers::incremental_sat::solve_sc_with_encoding_and_settings(
                            &mk_env,
                            satcoder::solvers::rustsat_glucose::Solver::new(),
                            &p.problem,
                            TIMEOUT,
                            delay_cost_type,
                            satddd_objective_encoding,
                            satddd_settings,
                            |k, v| {
                                solve_data.insert(k, v);
                            },
                        )
                        .map(|(v, _)| v)
                    }
                    SolverType::SatDddScTotalizer => {
                        ddd_solvers::incremental_sat::solve_sc_with_encoding_and_settings(
                            &mk_env,
                            satcoder::solvers::rustsat_glucose::Solver::new(),
                            &p.problem,
                            TIMEOUT,
                            delay_cost_type,
                            ddd_solvers::incremental_sat::SatObjectiveEncoding::IncrementalTotalizer,
                            satddd_settings,
                            |k, v| {
                                solve_data.insert(k, v);
                            },
                        )
                        .map(|(v, _)| v)
                    }
                    SolverType::SatDddScInc => {
                        ddd_solvers::incremental_sat::solve_incremental_sc_with_encoding_and_settings(
                            &mk_env,
                            satcoder::solvers::rustsat_glucose::Solver::new(),
                            &p.problem,
                            TIMEOUT,
                            delay_cost_type,
                            satddd_objective_encoding,
                            satddd_settings,
                            |k, v| {
                                solve_data.insert(k, v);
                            },
                        )
                        .map(|(v, _)| v)
                    }
                    SolverType::SatDddScAddClauses => {
                        // Pure AddClauses bound mode: bound enforced by hard
                        // clauses (permanent) instead of per-call assumptions.
                        // Same SC hybrid precedence + Glucose backend; cost
                        // encoding configurable via --satddd-objective-encoding.
                        ddd_solvers::incremental_sat::solve_sc_addclauses_with_encoding_and_settings(
                            &mk_env,
                            satcoder::solvers::rustsat_glucose::Solver::new(),
                            &p.problem,
                            TIMEOUT,
                            delay_cost_type,
                            satddd_objective_encoding,
                            satddd_settings,
                            |k, v| {
                                solve_data.insert(k, v);
                            },
                        )
                        .map(|(v, _)| v)
                    }
                    SolverType::SatDddScFreshAddClauses => {
                        // Fresh-solver-per-iteration AddClauses (puresat module):
                        // replays the logged formula into a brand-new Glucose
                        // instance at every DDD iteration, dropping all learned
                        // clauses and VSIDS state. Foundation for parallel SAT.
                        // puresat duplicates incremental_sat's config types, so
                        // translate the parsed enums/struct here.
                        let pure_encoding = match satddd_objective_encoding {
                            ddd_solvers::incremental_sat::SatObjectiveEncoding::Scpb => {
                                ddd_solvers::puresat::SatObjectiveEncoding::Scpb
                            }
                            ddd_solvers::incremental_sat::SatObjectiveEncoding::IncrementalTotalizer => {
                                ddd_solvers::puresat::SatObjectiveEncoding::IncrementalTotalizer
                            }
                            ddd_solvers::incremental_sat::SatObjectiveEncoding::BitTotalizer => {
                                ddd_solvers::puresat::SatObjectiveEncoding::BitTotalizer
                            }
                        };
                        // Forward the same `--satddd-*` flags to puresat. The
                        // `use_extended_precedence_graph` flag maps to puresat's
                        // simpler `use_precedence_graph` (chain-only, no ER) —
                        // semantically the same now that incremental_sat also
                        // uses `chain_earliest` instead of ER.
                        let pure_settings = ddd_solvers::puresat::SatDddSettings {
                            use_precedence_graph: satddd_settings.use_extended_precedence_graph,
                            prealloc_cost_thresholds: satddd_settings.prealloc_cost_thresholds,
                            seed_precedence_from_earliest: satddd_settings
                                .seed_precedence_from_earliest,
                            seed_resource_conflicts: satddd_settings.seed_resource_conflicts,
                            use_sc_amo: satddd_settings.use_sc_amo,
                        };
                        ddd_solvers::puresat::solve_sc_fresh_addclauses_with_encoding_and_settings(
                            &mk_env,
                            satcoder::solvers::rustsat_glucose::Solver::new(),
                            &p.problem,
                            TIMEOUT,
                            delay_cost_type,
                            pure_encoding,
                            pure_settings,
                            |k, v| {
                                solve_data.insert(k, v);
                            },
                        )
                        .map(|(v, _)| v)
                    }
                    SolverType::MipDdd => ddd::solvers::milp::mipdddpack::solve(
                        &mk_env,
                        get_env(),
                        &p.problem,
                        delay_cost_type,
                        TIMEOUT,
                        |k, v| {
                            solve_data.insert(k, v);
                        },
                    ),
                    SolverType::BinarizedBigMEager10Sec => {
                        ddd::solvers::milp::binarizedbigm::solve_binarized_bigm(
                            get_env(),
                            &p.problem,
                            delay_cost_type,
                            false,
                            10,
                            TIMEOUT,
                            &p.train_names,
                            &p.resource_names,
                            |k, v| {
                                solve_data.insert(k, v);
                            },
                        )
                    }
                    SolverType::BinarizedBigMEager30Sec => {
                        ddd::solvers::milp::binarizedbigm::solve_binarized_bigm(
                            get_env(),
                            &p.problem,
                            delay_cost_type,
                            false,
                            30,
                            TIMEOUT,
                            &p.train_names,
                            &p.resource_names,
                            |k, v| {
                                solve_data.insert(k, v);
                            },
                        )
                    }
                    SolverType::BinarizedBigMEager60Sec => {
                        ddd::solvers::milp::binarizedbigm::solve_binarized_bigm(
                            get_env(),
                            &p.problem,
                            delay_cost_type,
                            false,
                            60,
                            TIMEOUT,
                            &p.train_names,
                            &p.resource_names,
                            |k, v| {
                                solve_data.insert(k, v);
                            },
                        )
                    }
                    SolverType::BinarizedBigMLazy10Sec => {
                        ddd::solvers::milp::binarizedbigm::solve_binarized_bigm(
                            get_env(),
                            &p.problem,
                            delay_cost_type,
                            true,
                            10,
                            TIMEOUT,
                            &p.train_names,
                            &p.resource_names,
                            |k, v| {
                                solve_data.insert(k, v);
                            },
                        )
                    }
                    SolverType::BinarizedBigMLazy30Sec => {
                        ddd::solvers::milp::binarizedbigm::solve_binarized_bigm(
                            get_env(),
                            &p.problem,
                            delay_cost_type,
                            true,
                            30,
                            TIMEOUT,
                            &p.train_names,
                            &p.resource_names,
                            |k, v| {
                                solve_data.insert(k, v);
                            },
                        )
                    }
                    SolverType::BinarizedBigMLazy60Sec => {
                        ddd::solvers::milp::binarizedbigm::solve_binarized_bigm(
                            get_env(),
                            &p.problem,
                            delay_cost_type,
                            true,
                            60,
                            TIMEOUT,
                            &p.train_names,
                            &p.resource_names,
                            |k, v| {
                                solve_data.insert(k, v);
                            },
                        )
                    }
                })) {
                    Ok(result) => result,
                    Err(payload) => {
                        if is_likely_oom_panic(payload.as_ref()) {
                            Err(SolverError::OutOfMemory)
                        } else {
                            std::panic::resume_unwind(payload);
                        }
                    }
                };
            hprof::end_frame();
            let sol_time = solve_wall_start.elapsed().as_secs_f64() * 1000.0;
            let solver_name = format!("{:?}", solver);
            solve_data.insert("solver_name".to_string(), solver_name.clone().into());
            solve_data.insert(
                "delay_cost_type".to_string(),
                format!("{:?}", delay_cost_type).into(),
            );
            if let Some(other_delay_cost_type) = other_delay_cost_type {
                solve_data.insert(
                    "other_delay_cost_type".to_string(),
                    format!("{:?}", other_delay_cost_type).into(),
                );
            }
            let solve_status = match solution.as_ref() {
                Ok(_) => "ok",
                Err(SolverError::NoSolution) => "no_solution",
                Err(SolverError::Timeout) => "timeout",
                Err(SolverError::OutOfMemory) => "oom",
                Err(SolverError::GurobiError(_)) => "gurobi_error",
            };
            solve_data.insert("status".to_string(), solve_status.into());
            solve_data.insert("sol_time".to_string(), sol_time.into());

            let solve_stats = if let Ok(solution) = solution.as_ref() {
                let cost = p
                    .problem
                    .verify_solution(solution, delay_cost_type)
                    .unwrap();

                let other_cost =
                    other_delay_cost_type.map(|c| p.problem.verify_solution(solution, c).unwrap());

                solve_data.insert("cost".to_string(), cost.into());
                if let Some(other_cost) = other_cost {
                    solve_data.insert("other_cost".to_string(), other_cost.into());
                }

                hprof::profiler().print_timing();
                writeln!(
                    perf_out.borrow_mut(),
                    "{:>10} {:<25?} {:<12} {:>5} {:>10.0}",
                    name,
                    delay_cost_type,
                    solver_name,
                    cost,
                    sol_time,
                )
                .unwrap();

                if let Some(other_cost) = other_cost {
                    writeln!(
                        perf_out.borrow_mut(),
                        "{:>10} {:<25?} {:<12} {:>5} {:>10.0}",
                        name,
                        other_delay_cost_type,
                        solver_name,
                        other_cost,
                        sol_time,
                    )
                    .unwrap();
                }
            } else {
                writeln!(
                    perf_out.borrow_mut(),
                    "{:>10} {:<25?} {:<12} {:>5} {:>10.0}",
                    name,
                    delay_cost_type,
                    solver_name,
                    9999.0,
                    9999.0,
                )
                .unwrap();
                if other_delay_cost_type.is_some() {
                    writeln!(
                        perf_out.borrow_mut(),
                        "{:>10} {:<25?} {:<12} {:>5} {:>10.0}",
                        name,
                        delay_cost_type,
                        solver_name,
                        9999.0,
                        9999.0,
                    )
                    .unwrap();
                }
            };

            solves.push(solve_data.into());
        }

        let index = problems.len();
        problems.push(serde_json::json! {{
            "index": index,
            "name": name,
            "trains": problemstats.trains,
            "conflicts": problemstats.conflicts,
            "avg_tracks": problemstats.avg_tracks,
            "conflicting_visit_pairs": problemstats.conflicting_visit_pairs,
            "delay_cost_type": format!("{:?}", delay_cost_type),
            "solves" : serde_json::Value::Array(solves),
        }});
        solution
    };

    if opt.xml_instances {
        xml_instances(|name, p| {
            if matches_instance_filter(&name) {
                let _ = solve_it(name, p);
            }
        });
    }
    if opt.txt_instances {
        txt_instances(|name, p| {
            if matches_instance_filter(&name) {
                let _ = solve_it(name, p);
            }
        });
    }
    if opt.verify_instances {
        verify_instances(|name, p, solution| {
            if !matches_instance_filter(&name) {
                return solution;
            }
            let cost = p
                .problem
                .verify_solution(&solution, delay_cost_type)
                .unwrap();
            writeln!(perf_out.borrow_mut(), "{:>10} {:>5}", name, cost,).unwrap();
            solve_it(name, p).unwrap()
        })
    }
    println!("{}", perf_out.into_inner());

    if let Some(f) = opt.json_output {
        std::fs::write(&f, serde_json::to_string_pretty(&problems).unwrap()).unwrap();
        println!("Wrote to file {:?}", f);
    }
}

struct ProblemStats {
    trains: usize,
    conflicts: usize,
    avg_tracks: f32,
    conflicting_visit_pairs: usize,
}

fn print_problem_stats(problem: &problem::Problem) -> ProblemStats {
    let avg_tracks = problem
        .trains
        .iter()
        .map(|t| {
            t.visits
                .iter()
                .filter(|v| problem.conflicts.contains(&(v.resource_id, v.resource_id)))
                .count()
        })
        .sum::<usize>() as f32
        / problem.trains.len() as f32;
    let mut conflicting_visit_pairs = 0;
    for t1 in 0..problem.trains.len() {
        for t2 in (t1 + 1)..problem.trains.len() {
            for Visit {
                resource_id: r1, ..
            } in problem.trains[t1].visits.iter()
            {
                for Visit {
                    resource_id: r2, ..
                } in problem.trains[t2].visits.iter()
                {
                    if problem.conflicts.contains(&(*r1, *r2)) {
                        conflicting_visit_pairs += 1;
                    }
                }
            }
        }
    }

    let delays = 0;
    let avgdelay = 0;

    let trains = problem.trains.len();
    let conflicts = problem.conflicts.len();
    println!(
        "trains {} tracks {} avgtracks {:.2} trackpairs {} delays {} avgdelay{}",
        trains, conflicts, avg_tracks, conflicting_visit_pairs, delays, avgdelay,
    );
    println!(
        "{} & {} & {:.2} & {} & {} & {} \\\\",
        trains, conflicts, avg_tracks, conflicting_visit_pairs, delays, avgdelay,
    );
    ProblemStats {
        trains,
        conflicts,
        avg_tracks,
        conflicting_visit_pairs,
    }
}

#[cfg(test)]
mod tests {
    use ddd::problem::{DelayCostType, NamedProblem};

    #[test]
    fn objective_alias_infsteps123() {
        assert!(matches!(
            super::parse_delay_cost_type("infsteps123"),
            Some(DelayCostType::InfiniteSteps180)
        ));
    }

    #[test]
    pub fn testproblem_maxsatddd() {
        let mut env = grb::Env::new("").unwrap();
        env.set(grb::param::OutputFlag, 0).unwrap();

        let delay_cost_type = DelayCostType::FiniteSteps123;

        let problem = crate::problem::problem1_with_stations();
        let result = ddd::solvers::ladder::maxsatddd_ladder::solve(
            &crate::mk_env,
            satcoder::solvers::minisat::Solver::new(),
            &problem,
            30.0,
            DelayCostType::FiniteSteps123,
            |_, _| {},
        )
        .unwrap()
        .0;
        let score = problem.verify_solution(&result, delay_cost_type);
        assert!(score.is_some());
    }

    #[test]
    pub fn testproblem_mipdddpack() {
        let mut env = grb::Env::new("").unwrap();
        env.set(grb::param::OutputFlag, 0).unwrap();

        let delay_cost_type = DelayCostType::FiniteSteps123;
        let mut env = grb::Env::new("").unwrap();
        env.set(grb::param::OutputFlag, 0).unwrap();

        let problem = crate::problem::problem1_with_stations();
        let result = ddd::solvers::milp::mipdddpack::solve(
            &crate::mk_env,
            &env,
            &problem,
            delay_cost_type,
            120.0,
            |_, _| {},
        )
        .unwrap();
        let score = problem.verify_solution(&result, delay_cost_type);
        assert!(score.is_some());
    }

    #[test]
    pub fn samescore_trivial() {
        let mut env = grb::Env::new("").unwrap();
        env.set(grb::param::OutputFlag, 0).unwrap();

        let problem = crate::problem::problem1_with_stations();
        let delay_cost_type = DelayCostType::FiniteSteps123;

        let result = ddd::solvers::ladder::maxsatddd_ladder::solve(
            &crate::mk_env,
            satcoder::solvers::minisat::Solver::new(),
            &problem,
            30.0,
            delay_cost_type,
            |_, _| {},
        )
        .unwrap()
        .0;
        let first_score = problem.verify_solution(&result, delay_cost_type);

        for _ in 0..100 {
            let result = ddd::solvers::ladder::maxsatddd_ladder::solve(
                &crate::mk_env,
                satcoder::solvers::minisat::Solver::new(),
                &problem,
                30.0,
                DelayCostType::FiniteSteps123,
                |_, _| {},
            )
            .unwrap()
            .0;
            let score = problem.verify_solution(&result, delay_cost_type);
            assert!(score == first_score);
        }
        println!("ALL COSTS WERE {:?}", first_score);
    }

    #[test]
    pub fn samescore_all_instances() {
        let mut env = grb::Env::new("").unwrap();
        env.set(grb::param::OutputFlag, 0).unwrap();

        let delay_cost_type = DelayCostType::FiniteSteps123;
        for delaytype in [
            ddd::problem::DelayMeasurementType::AllStationArrivals,
            ddd::problem::DelayMeasurementType::FinalStationArrival,
        ] {
            for instance_number in
                // [                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,            ]
                [3, 4, 5]
            {
                println!("{}", instance_number);
                let NamedProblem { problem, .. } = crate::parser::read_xml_file(
                    &format!("instances/Instance{}.xml", instance_number,),
                    delaytype,
                );

                let result = ddd::solvers::ladder::maxsatddd_ladder::solve(
                    &crate::mk_env,
                    satcoder::solvers::minisat::Solver::new(),
                    &problem,
                    30.0,
                    delay_cost_type,
                    |_, _| {},
                )
                .unwrap()
                .0;
                let first_score = problem.verify_solution(&result, delay_cost_type);

                for iteration in 0..100 {
                    println!("iteration {} {}", instance_number, iteration);
                    let result = ddd::solvers::ladder::maxsatddd_ladder::solve(
                        &crate::mk_env,
                        satcoder::solvers::minisat::Solver::new(),
                        &problem,
                        30.0,
                        DelayCostType::FiniteSteps123,
                        |_, _| {},
                    )
                    .unwrap()
                    .0;
                    let score = problem.verify_solution(&result, delay_cost_type);
                    assert!(score == first_score);
                }

                println!("ALL COSTS WERE {:?}", first_score);
            }
        }
    }
}
