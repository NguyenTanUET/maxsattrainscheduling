# Summary Table (120s timeout)

**Solved = proved optimal (lb = ub).** Avg time tính trên instances solved.

## Average time (ms) — proved-optimal instances

| Objective | Set | BigM (Gurobi) | TI (Gurobi) | BigM (CPLEX) | TI (CPLEX) | CP (CPLEX) |
|---|---|---|---|---|---|---|
| Linear Continuous | Original | 38 | 5,992 | 63 | 31,582 | 156 |
|  | Track time | 2,022 | 3,923 | 92 | 8,710 | 101 |
|  | Station time | 228 | 7,333 | 45 | 86,770 | 65 |
|  | All | 749 | 5,760 | 68 | 36,968 | 116 |
||||||||
| Rounded Linear | Original | 28 | 4,745 | 255 | 13,582 | 107 |
|  | Track time | 4,904 | 1,173 | 1,777 | 5,655 | 74 |
|  | Station time | 361 | 3,689 | 4,322 | 31,630 | 138 |
|  | All | 1,733 | 3,273 | 2,063 | 13,856 | 108 |
||||||||
| Stepwise | Original | 10 | 27 | 153 | 33,651 | 34 |
|  | Track time | 862 | 3,639 | 451 | 10,070 | 27 |
|  | Station time | 216 | 637 | 366 | 0 | 18 |
|  | All | 362 | 1,434 | 323 | 25,076 | 30 |
||||||||

## #Solved (proved optimal)

| Objective | Set | BigM (Gurobi) | TI (Gurobi) | BigM (CPLEX) | TI (CPLEX) | CP (CPLEX) |
|---|---|---|---|---|---|---|
| Linear Continuous | Original | 24 | 24 | 19 | 4 | 14 |
|  | Track time | 22 | 21 | 16 | 1 | 10 |
|  | Station time | 21 | 21 | 12 | 1 | 8 |
|  | All | 67 | 66 | 47 | 6 | 32 |
||||||||
| Rounded Linear | Original | 24 | 24 | 24 | 5 | 14 |
|  | Track time | 22 | 21 | 22 | 2 | 9 |
|  | Station time | 21 | 21 | 22 | 1 | 10 |
|  | All | 67 | 66 | 68 | 8 | 33 |
||||||||
| Stepwise | Original | 24 | 24 | 24 | 7 | 17 |
|  | Track time | 24 | 24 | 24 | 4 | 9 |
|  | Station time | 24 | 24 | 24 | 0 | 4 |
|  | All | 72 | 72 | 72 | 11 | 30 |
||||||||
