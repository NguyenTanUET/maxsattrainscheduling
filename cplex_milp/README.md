# CPLEX MILP cho TRP (Pure `.lp` workflow)

Triển khai 2 mô hình MILP (BigM, MILP-TI) cho bài toán Train Rescheduling, dùng **CPLEX CLI full edition** qua `.lp` file.

- **Build model**: Rust binary `export_lp` (không cần Python)
- **Solve**: CPLEX CLI (bypass Community Edition 1000-var limit)
- **Verify**: Python (parse XML + TI post-processing)

## Kiến trúc

```
cplex_milp/
├── benchmark.sh       # Orchestrator (Rust build → CPLEX → verify)
├── batch.sh           # Multi-instance + CSV summary
├── verify.py          # Parse .sol + TI post-processing
├── parser.py          # Đọc dataset .txt
├── cost.py            # Cost functions
├── utils.py           # Verify helpers
├── README.md
└── results/           # .lp, .sol, .log, .summary.txt

src/bin/
└── export_lp.rs       # Rust binary - build MILP, port 1:1 từ Rust gốc
```

## Yêu cầu

| Thành phần | Bắt buộc | Mục đích |
|---|---|---|
| **CPLEX Studio 22.2+** (Linux) | ✅ | CLI solve full edition |
| **Rust toolchain** | ✅ | Build Rust binary |
| **Python 3.10+** | ✅ | Verify (parse XML, post-processing) |

Không cần `docplex`, `numpy`, hay package Python nào khác.

## Setup

### 1. Env vars cho CPLEX

```bash
export CPLEX_HOME=/opt/ibm/ILOG/CPLEX_Studio222/cplex
export PATH=$CPLEX_HOME/bin/x86-64_linux:$PATH
export LD_LIBRARY_PATH=$CPLEX_HOME/bin/x86-64_linux:$LD_LIBRARY_PATH
```

### 2. Build Rust binary export_lp

```bash
cd /mnt/d/github/maxsattrainscheduling
cargo build --release --bin export_lp
ls -la target/release/export_lp
```

### 3. Verify CPLEX CLI

```bash
which cplex && cplex -c "quit"
```

## Sử dụng

### 1 instance

```bash
./benchmark.sh ../instances/original/InstanceA1.txt bigm finsteps123 120 4
#               ^                                    ^    ^           ^   ^
#               instance                             solver cost      timeout threads
```

Output:
- `results/InstanceA1_bigm_finsteps123.lp`
- `results/InstanceA1_bigm_finsteps123.sol`
- `results/InstanceA1_bigm_finsteps123.cplex.log`
- `results/InstanceA1_bigm_finsteps123.summary.txt`

### TI với custom params

```bash
TI_INTERVAL=30 TI_BIG_M=600 \
    ./benchmark.sh ../instances/original/InstanceA1.txt ti cont 180 8
```

### Skip verify (chỉ build + solve)

```bash
SKIP_VERIFY=1 ./benchmark.sh ...
```

### Batch toàn bộ instances

```bash
./batch.sh ../instances/original bigm finsteps123 120 4
./batch.sh ../instances/original ti infsteps180 120 4 InstanceA  # filter
```

→ CSV summary: `results/batch_<solver>_<cost>_<timestamp>.csv`

## Cost types

| Flag | Loại |
|---|---|
| `finsteps123` | 3 ngưỡng 0/180/360 → cost 1/2/3 |
| `finsteps12345`, `finsteps139` | Custom thresholds |
| `finsteps1_3min`, `finsteps1_5min` | FiniteSteps đơn giản |
| `infsteps60`, `infsteps180`, `infsteps360` | Infinite stairs |
| `cont` | Linear continuous |

## Workflow

### BigM (đơn giản)

```
.txt --[Rust export_lp]--> .lp --[CPLEX CLI]--> .sol --[verify.py]--> cost+valid
```

### TI (port 1:1 Rust với post-processing)

```
.txt --[Rust export_lp]--> .lp --[CPLEX CLI]--> .sol_ti (discretized, cost ~17)
                                                    │
                                                    ▼ verify.py extract priorities
                                          _ti_minimize.lp (LP)
                                                    │
                                                    ▼ CPLEX CLI
                                            _ti_minimize.sol
                                                    │
                                                    ▼ parse continuous schedule
                                                cost = 11 ✓ valid ✓
```

→ Cost TI = Cost BigM = Rust baseline (khớp 100%)

## Đặc trưng

| Khía cạnh | Trạng thái |
|---|---|
| Port 1:1 với Rust formulation | ✅ BigM + TI + minimize_solution |
| Không phụ thuộc Python build | ✅ Rust binary `export_lp` |
| CPLEX full edition | ✅ Qua CLI (bypass Community Edition) |
| Pipeline tối giản | ✅ 4 file Python + 2 bash + 1 Rust binary |
| Kết quả khớp baseline Rust/Gurobi | ✅ Đã verify trên InstanceA1 |

## So sánh hiệu năng (InstanceA1)

| Pha | Thời gian | Tool |
|---|---|---|
| Build BigM | 0.04s | Rust binary |
| Solve BigM | 0.33s | CPLEX CLI |
| Build TI | 0.12s | Rust binary |
| Solve TI MILP | 1.06s | CPLEX CLI |
| TI post-process LP | <0.5s | CPLEX CLI |
| **Tổng TI** | **~1.5s** | Pipeline |

## So sánh với Rust/Gurobi baseline

Cùng instance + cost type, cost giống nhau 1:1:
- BigM InstanceA1 finsteps123: **11** (Rust=11, CPLEX=11)
- TI InstanceA1 finsteps123: **11** (post-processed, khớp BigM)

Baseline ở: `../2026-05-16-Verified-Result-For-Graduation-Thesis/BigM_MILP/`

## Test nhanh

```bash
# Verify env
which cplex && cplex -c "quit"

# Build Rust binary
cd ..
cargo build --release --bin export_lp
cd cplex_milp

# Run BigM test
./benchmark.sh ../instances/original/InstanceA1.txt bigm finsteps123 60 4

# Run TI test (với post-processing)
TI_INTERVAL=30 TI_BIG_M=600 \
    ./benchmark.sh ../instances/original/InstanceA1.txt ti finsteps123 60 4
```

Kỳ vọng cả 2: `cost = 11, Valid: True (OK)`.
