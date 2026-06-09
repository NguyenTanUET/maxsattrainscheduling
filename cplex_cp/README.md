# CPLEX CP Optimizer cho TRP

Triển khai **Constraint Programming** cho bài toán Train Rescheduling, dùng **CP Optimizer CLI** qua `.cpo` file.

- **Build model**: Rust binary `export_cp` (port từ Rust gốc)
- **Solve**: `cpoptimizer` CLI (bundled với CPLEX Studio)
- **Verify**: Python (parse text output)

## Vì sao CP cho TRP?

CP có **primitives chuyên cho scheduling** mà MILP không có:

| Khái niệm | MILP | CP Optimizer |
|---|---|---|
| Thời gian visit | `t[v]` continuous + bounds | `intervalVar(start, size)` |
| Tiền tố trong tàu | `t[v+1] >= t[v] + travel` | `endBeforeStart(v, v+1)` |
| Xung đột tài nguyên | BigM + binary order vars | `noOverlap([v1, v2, ...])` |
| Số biến binary | O(n²) cho conflict pairs | 0 |

→ Model CP **gọn hơn 5-10×**, dùng propagation chuyên dụng cho disjunctive scheduling.

## Kiến trúc

```
cplex_cp/
├── benchmark.sh       # Orchestrator (Rust build → cpoptimizer → verify)
├── batch.sh           # Multi-instance + CSV
├── verify.py          # Parse cpoptimizer text output
├── parser.py          # Đọc dataset .txt
├── cost.py            # Cost functions (shared với cplex_milp)
├── utils.py           # Verify helpers (shared)
├── README.md
└── results/

src/bin/
└── export_cp.rs       # Rust binary - build CP model → .cpo file
```

## Yêu cầu

| Thành phần | Bắt buộc | Mục đích |
|---|---|---|
| **CPLEX Studio 22.2+** (Linux) | ✅ | CP Optimizer CLI (`cpoptimizer`) |
| **Rust toolchain** | ✅ | Build `export_cp` binary |
| **Python 3.10+** | ✅ | Verify (parse text + utils) |

## Setup

### 1. Env vars

```bash
export CPLEX_HOME=/opt/ibm/ILOG/CPLEX_Studio222/cplex
export PATH=$CPLEX_HOME/bin/x86-64_linux:$PATH

# CP Optimizer
export CPO_HOME=/opt/ibm/ILOG/CPLEX_Studio222/cpoptimizer
export PATH=$CPO_HOME/bin/x86-64_linux:$PATH
export LD_LIBRARY_PATH=$CPO_HOME/bin/x86-64_linux:$LD_LIBRARY_PATH
```

### 2. Build Rust binary

```bash
cd /mnt/d/github/maxsattrainscheduling
cargo build --release --bin export_cp
ls -la target/release/export_cp
```

### 3. Verify cpoptimizer

```bash
which cpoptimizer
cpoptimizer -help 2>&1 | head -5
```

## Sử dụng

### 1 instance

```bash
./benchmark.sh ../instances/original/InstanceA1.txt finsteps123 120 4
#               ^                                    ^           ^   ^
#               instance                             cost        timeout threads
```

Output:
- `results/InstanceA1_cp_finsteps123.cpo` — CP model
- `results/InstanceA1_cp_finsteps123.out` — cpoptimizer output (text)
- `results/InstanceA1_cp_finsteps123.cpo.log` — solver log
- `results/InstanceA1_cp_finsteps123.summary.txt`

### Custom CP workers

```bash
CPO_WORKERS=8 ./benchmark.sh ../instances/original/InstanceA1.txt cont 180 4
```

### Skip verify

```bash
SKIP_VERIFY=1 ./benchmark.sh ...
```

### Batch toàn bộ instances

```bash
./batch.sh ../instances/original finsteps123 120 4
./batch.sh ../instances/original infsteps180 120 4
./batch.sh ../instances/original cont 120 4
```

## Cost types hỗ trợ

| Flag | CP expression |
|---|---|
| `finsteps123` | `sum(c_k * (delay > thr_k))` cho mỗi visit |
| `finsteps12345`, `finsteps139` | Tương tự với thresholds khác |
| `infsteps60/180/360` | `(delay + interval - 1) / interval` |
| `cont` | `max(0, startOf(v) - aimed)` |

## CP Formulation

```
// Mỗi visit là một interval variable
intervalVar v_0_0 = intervalVar(start=0..43200, size=30);
intervalVar v_0_1 = intervalVar(start=180..43200, size=200);
// ...

constraints {
    // Precedence trong tàu
    endBeforeStart(v_0_0, v_0_1);
    endBeforeStart(v_0_1, v_0_2);
    // ...
    
    // Disjunctive resource (mỗi track 1 noOverlap)
    noOverlap([v_0_1, v_1_3, v_2_5, ...]);
    // ...
}

// Cost expressions
dexpr int delay_0_1 = max(0, startOf(v_0_1) - 1000);
dexpr int cost_0_1 = (1 * (delay_0_1 > 0)) + 
                     (1 * (delay_0_1 > 180)) + 
                     (1 * (delay_0_1 > 360));
// ...

minimize sum(cost_*_*);
```

## So sánh với MILP/SAT

Cùng instance, dự kiến:
- **CP**: cost giống MILP (same problem), thường **nhanh hơn** cho scheduling structure
- **MILP**: tốt cho continuous cost, có thể chậm hơn TI cho stepped
- **SAT/MaxSAT**: tốt cho stepped cost (objective bậc thang)

CP đặc biệt mạnh cho:
- Many `noOverlap` constraints (disjunctive scheduling)
- Few number of variables nhưng nhiều interactions

## Workflow

```
.txt --[Rust export_cp]--> .cpo --[cpoptimizer CLI]--> output --[verify.py]--> cost+valid
```

## Test nhanh

```bash
# Build Rust binary
cd ..
cargo build --release --bin export_cp
cd cplex_cp

# Test 1 instance
./benchmark.sh ../instances/original/InstanceA1.txt finsteps123 60 4
```

Kỳ vọng:
- Build (Rust): ~0.05s
- Solve (cpoptimizer): vài giây
- Verified cost = 11 (khớp MILP/SAT)
- Valid: True

## Đặc trưng

| Khía cạnh | Trạng thái |
|---|---|
| Port 1:1 với Rust formulation | ✅ Reuse `ddd::parser`, `ddd::problem` |
| Không phụ thuộc Python build | ✅ Rust binary `export_cp` |
| CP Optimizer full edition | ✅ Qua CLI (bypass mọi limit) |
| Pipeline gọn | ✅ 4 file Python + 2 bash + 1 Rust binary |
| Comparison với MILP/SAT | ✅ Cùng input, cùng cost type, cùng instance |
