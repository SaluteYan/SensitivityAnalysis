# Problem 21 Remote Experiment Runner

This folder contains launch scripts for the strengthened Problem 21 OPMWADE experiments.

## Linux server

```bash
bash scripts/run_problem21_paper_experiments.sh
```

The default launcher settings are a balanced remote profile:

```text
ANCHOR_MAX_NFES=6400
SENSITIVITY_MAX_NFES=6400
PARETO_MAX_NFES=6400
ANCHOR_REPEATS=8
SENSITIVITY_REPEATS=6
PARETO_REPEATS=6
PARETO_GRID=4
THREADS_PER_WORKER=1
WORKERS=CPU cores minus 2 on servers with at least 8 cores
```

Common overrides:

```bash
WORKERS=32 THREADS_PER_WORKER=1 \
OUTPUT_DIR=results/problem21_opmwade_sensitivity_balanced \
bash scripts/run_problem21_paper_experiments.sh
```

For a heavier final-paper run:

```bash
WORKERS=32 THREADS_PER_WORKER=1 \
ANCHOR_REPEATS=10 SENSITIVITY_REPEATS=10 PARETO_REPEATS=10 \
PARETO_GRID=5 OUTPUT_DIR=results/problem21_opmwade_sensitivity_paper \
bash scripts/run_problem21_paper_experiments.sh
```

For a faster server-side pilot:

```bash
WORKERS=16 ANCHOR_REPEATS=3 SENSITIVITY_REPEATS=3 PARETO_REPEATS=3 \
ANCHOR_MAX_NFES=2000 SENSITIVITY_MAX_NFES=2000 PARETO_MAX_NFES=3000 \
PARETO_GRID=4 OUTPUT_DIR=results/problem21_opmwade_sensitivity_server_pilot \
bash scripts/run_problem21_paper_experiments.sh
```

## Windows / PowerShell

```powershell
.\scripts\run_problem21_paper_experiments.ps1 -Workers 16 -ThreadsPerWorker 1
```

## Parallelism

The experiment script parallelizes independent cases with multiple Python worker processes.
The launcher sets BLAS/OpenMP thread counts to `THREADS_PER_WORKER` to avoid oversubscription.
For this codebase, `THREADS_PER_WORKER=1` and a large `WORKERS` value is usually best because the bottleneck is the Python time-domain integration loop.

## Resume Behavior

Each case writes one JSON file under:

```text
<OUTPUT_DIR>/cache
```

The script resumes automatically. A cached case is reused only if its signature matches the current weights, constraints, seed, repeats, NFEs, initialization file, damping mode, and population factors.

## Important Outputs

```text
<OUTPUT_DIR>/problem21_opmwade_experiment_report.md
<OUTPUT_DIR>/data/all_case_results.csv
<OUTPUT_DIR>/data/objective_conflict_results.csv
<OUTPUT_DIR>/data/weight_sensitivity_results.csv
<OUTPUT_DIR>/data/pareto_epsilon_results.csv
<OUTPUT_DIR>/data/nondominated_summary.csv
<OUTPUT_DIR>/figures/*.png
<OUTPUT_DIR>/figures/*.pdf
```

In Pareto figures, gray crosses are infeasible epsilon attempts, orange squares are feasible epsilon-constraint solutions, and black rings mark feasible non-dominated points.
