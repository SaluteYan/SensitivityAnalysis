from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Process-level parallelism is the main speedup path. Keep BLAS/OpenMP
# libraries from oversubscribing CPU cores unless the launcher overrides them.
for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")

import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


OBJECTIVE_KEYS = ("J_y", "J_u", "J_e")
OBJECTIVE_LABELS = (r"$J_y$", r"$J_u$", r"$J_e$")
OBJECTIVE_UNITS = ("m^2 s", "J", "rad")
PROPOSED_WEIGHTS = (0.5, 0.25, 0.25)
WEIGHT_CASES = [
    ("W1", (1.0, 0.0, 0.0), "Vibration-only"),
    ("W2", (0.0, 1.0, 0.0), "Energy-only"),
    ("W3", (0.0, 0.0, 1.0), "Tracking-only"),
    ("W4", (0.6, 0.2, 0.2), "Vibration-priority"),
    ("W5", (0.2, 0.6, 0.2), "Energy-priority"),
    ("W6", (0.2, 0.2, 0.6), "Tracking-priority"),
    ("W7", (0.5, 0.3, 0.2), "Vibration-energy"),
    ("W8", (0.5, 0.2, 0.3), "Vibration-tracking"),
    ("W9", (0.3, 0.5, 0.2), "Energy-vibration"),
    ("W10", (0.2, 0.5, 0.3), "Energy-tracking"),
    ("W11", (0.3, 0.2, 0.5), "Tracking-vibration"),
    ("W12", (0.2, 0.3, 0.5), "Tracking-energy"),
    ("W13", (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), "Balanced"),
    ("W14", PROPOSED_WEIGHTS, "Proposed current setting"),
    ("W15", (0.4, 0.3, 0.3), "Mild vibration-priority"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run problem-21 OPMWADE objective-conflict, normalization, weight-sensitivity, "
            "and epsilon-constraint Pareto experiments."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=WORKSPACE_ROOT / "results" / "problem21_opmwade_sensitivity")
    parser.add_argument("--init-file", type=Path, default=WORKSPACE_ROOT / "init_data" / "PrG21InitData-target_1_05-none.npz")
    parser.add_argument("--damping-mode", choices=["none", "fixed", "adaptive"], default="none")
    parser.add_argument("--target-angle", type=float, default=1.05)
    parser.add_argument("--max-nfes", type=int, default=220, help="Per-repeat OPMWADE evaluation budget.")
    parser.add_argument("--repeats", type=int, default=1, help="Repeats per case. Use 5-10 for final paper runs.")
    parser.add_argument("--anchor-max-nfes", type=int, default=None, help="Override NFEs for W1-W3 normalization anchors.")
    parser.add_argument("--sensitivity-max-nfes", type=int, default=None, help="Override NFEs for normalization/weight-sensitivity cases.")
    parser.add_argument("--pareto-max-nfes", type=int, default=None, help="Override NFEs for epsilon-constraint Pareto cases.")
    parser.add_argument("--anchor-repeats", type=int, default=None, help="Override repeats for W1-W3 normalization anchors.")
    parser.add_argument("--sensitivity-repeats", type=int, default=None, help="Override repeats for normalization/weight-sensitivity cases.")
    parser.add_argument("--pareto-repeats", type=int, default=None, help="Override repeats for epsilon-constraint Pareto cases.")
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=0,
        help="Print per-repeat OPMWADE progress every N NFEs; 0 disables progress lines.",
    )
    parser.add_argument("--initial-np-factor", type=float, default=4.0)
    parser.add_argument("--min-np-factor", type=float, default=2.0)
    parser.add_argument("--pareto-grid", type=int, default=3, help="epsilon levels per constrained objective.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore cached case JSON files and recompute.")
    parser.add_argument("--enable-late-enhancements", action="store_true", help="Enable OPMWADE late-stage enhancements.")
    return parser.parse_args()


def safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": output_dir,
        "cache": output_dir / "cache",
        "data": output_dir / "data",
        "figures": output_dir / "figures",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def finite_span(min_values: np.ndarray, max_values: np.ndarray) -> np.ndarray:
    span = np.asarray(max_values, dtype=float) - np.asarray(min_values, dtype=float)
    fallback = np.maximum(np.abs(max_values), 1.0) * 1e-6
    return np.where(np.abs(span) > 1e-15, span, fallback)


def format_elapsed(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    whole = int(seconds)
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{seconds:.1f}s"


def normalize_with_scale(values: np.ndarray, min_values: np.ndarray, span: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=float) - min_values) / span


def stage_max_nfes(args: argparse.Namespace, stage: str) -> int:
    values = {
        "anchor": args.anchor_max_nfes,
        "sensitivity": args.sensitivity_max_nfes,
        "pareto": args.pareto_max_nfes,
    }
    return int(values[stage] if values[stage] is not None else args.max_nfes)


def stage_repeats(args: argparse.Namespace, stage: str) -> int:
    values = {
        "anchor": args.anchor_repeats,
        "sensitivity": args.sensitivity_repeats,
        "pareto": args.pareto_repeats,
    }
    return int(values[stage] if values[stage] is not None else args.repeats)


def case_signature(case: dict[str, Any], shared: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "case_id",
        "stage",
        "weights",
        "normalization",
        "seed",
        "max_nfes",
        "repeats",
        "min_values",
        "max_values",
        "constraint_limits",
    ]
    shared_keys = [
        "init_file",
        "damping_mode",
        "target_angle",
        "initial_np_factor",
        "min_np_factor",
        "enable_late_enhancements",
    ]
    return {
        "case": {key: case.get(key) for key in keys},
        "shared": {key: shared.get(key) for key in shared_keys},
    }


def cache_matches(cached: dict[str, Any], case: dict[str, Any], shared: dict[str, Any]) -> bool:
    return cached.get("run_signature") == case_signature(case, shared)


def annotate_constraint_status(result: dict[str, Any]) -> dict[str, Any]:
    limits = result.get("constraint_limits")
    normalized = result.get("normalized_components")
    if limits is None or normalized is None:
        result["epsilon_violation"] = 0.0
        result["epsilon_feasible"] = True
        result["overall_feasible"] = float(result.get("base_penalty", 0.0)) <= 0.0
        return result

    lim = np.asarray(limits, dtype=float)
    norm = np.asarray(normalized, dtype=float)
    active = np.isfinite(lim)
    if active.any():
        violation = float(np.max(np.maximum(norm[active] - lim[active], 0.0)))
    else:
        violation = 0.0
    base_feasible = float(result.get("base_penalty", 0.0)) <= 0.0
    result["epsilon_violation"] = violation
    result["epsilon_feasible"] = violation <= 1e-10
    result["overall_feasible"] = bool(base_feasible and result["epsilon_feasible"])
    return result


def make_case(
    *,
    case_id: str,
    stage: str,
    label: str,
    weights: tuple[float, float, float],
    normalization: str,
    seed: int,
    max_nfes: int,
    repeats: int,
    min_values: np.ndarray | None = None,
    max_values: np.ndarray | None = None,
    constraint_limits: tuple[float, float, float] | None = None,
    meaning: str = "",
) -> dict[str, Any]:
    case: dict[str, Any] = {
        "case_id": case_id,
        "stage": stage,
        "label": label,
        "meaning": meaning,
        "weights": list(map(float, weights)),
        "normalization": normalization,
        "seed": int(seed),
        "max_nfes": int(max_nfes),
        "repeats": int(repeats),
    }
    if min_values is not None and max_values is not None:
        case["min_values"] = np.asarray(min_values, dtype=float).tolist()
        case["max_values"] = np.asarray(max_values, dtype=float).tolist()
    if constraint_limits is not None:
        case["constraint_limits"] = [float(v) for v in constraint_limits]
    return case


def configure_problem21(case: dict[str, Any], shared: dict[str, Any]) -> None:
    from algorithms.common import (
        configure_problem_from_init_data,
        set_problem21_objective_config,
        set_problem21_target_angle,
        set_trajectory_damping_mode,
    )
    from algorithms import opmwade

    set_trajectory_damping_mode(shared["damping_mode"])
    set_problem21_target_angle(float(shared["target_angle"]))
    configure_problem_from_init_data(21, init_file=shared["init_file"])

    if case["normalization"] == "minmax":
        set_problem21_objective_config(
            weights=case["weights"],
            normalization="minmax",
            min_values=case["min_values"],
            max_values=case["max_values"],
            constraint_limits=case.get("constraint_limits"),
        )
    else:
        set_problem21_objective_config(
            weights=case["weights"],
            normalization=case["normalization"],
            constraint_limits=case.get("constraint_limits"),
        )

    opmwade.INITIAL_NP_DIM_FACTOR = float(shared["initial_np_factor"])
    opmwade.MIN_NP_DIM_FACTOR = float(shared["min_np_factor"])


def run_case(case: dict[str, Any], shared: dict[str, Any]) -> dict[str, Any]:
    from algorithms.common import (
        normalize_problem21_objectives,
        problem21_objective_components,
        scalarize_problem21_objectives,
    )
    from algorithms import opmwade

    cache_path = Path(shared["cache_dir"]) / f"{safe_id(case['case_id'])}.json"
    if shared["resume"] and cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        if cache_matches(cached, case, shared):
            cached["cache_hit"] = True
            return annotate_constraint_status(cached)
        print(f"[cache-skip] {case['case_id']} cache signature differs; recomputing.", flush=True)

    t0 = time.perf_counter()
    configure_problem21(case, shared)
    result = opmwade.run(
        evals_range=(21,),
        repeat_num=int(case["repeats"]),
        seed=int(case["seed"]),
        max_nfes=int(case["max_nfes"]),
        save=False,
        init_file=shared["init_file"],
        progress_interval=int(shared.get("progress_interval", 0)),
        progress_label=f"{case['case_id']}:{case['stage']}",
        enable_late_enhancements=bool(shared["enable_late_enhancements"]),
    )[0]

    best_individuals = np.asarray(result.diagnostics["best_individuals"], dtype=float)
    if best_individuals.ndim == 1:
        best_individuals = best_individuals.reshape(1, -1)
    summary_x = np.asarray(result.diagnostics["summary_best_individual"], dtype=float).reshape(-1)
    if summary_x.size == 0 and best_individuals.size:
        summary_x = best_individuals[-1].copy()

    repeat_components = []
    repeat_normalized = []
    repeat_scalar = []
    repeat_penalty = []
    for x in best_individuals:
        components, base_penalty = problem21_objective_components(x, damping_mode=shared["damping_mode"])
        normalized = normalize_problem21_objectives(components)
        repeat_components.append(components.tolist())
        repeat_normalized.append(normalized.tolist())
        repeat_scalar.append(float(scalarize_problem21_objectives(components)))
        repeat_penalty.append(float(base_penalty))

    summary_components, summary_base_penalty = problem21_objective_components(summary_x, damping_mode=shared["damping_mode"])
    summary_normalized = normalize_problem21_objectives(summary_components)
    elapsed = time.perf_counter() - t0

    out: dict[str, Any] = {
        **case,
        "algorithm": result.algorithm,
        "evals": int(result.evals),
        "cache_hit": False,
        "elapsed_wall_time": float(elapsed),
        "elapsed_time_mean": float(result.elapsed_time),
        "best_scalar": float(result.best),
        "median_scalar": float(result.median),
        "mean_scalar": float(result.mean),
        "worst_scalar": float(result.worst),
        "std_scalar": float(result.std),
        "fearate": float(result.fearate),
        "process": np.asarray(result.process, dtype=float).tolist(),
        "best_x": summary_x.tolist(),
        "base_penalty": float(summary_base_penalty),
        "components": summary_components.tolist(),
        "normalized_components": summary_normalized.tolist(),
        "repeat_components": repeat_components,
        "repeat_normalized_components": repeat_normalized,
        "repeat_scalar": repeat_scalar,
        "repeat_base_penalty": repeat_penalty,
        "run_signature": case_signature(case, shared),
    }
    annotate_constraint_status(out)

    tmp_path = cache_path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    tmp_path.replace(cache_path)
    return out


def run_cases(cases: list[dict[str, Any]], shared: dict[str, Any], workers: int) -> list[dict[str, Any]]:
    if not cases:
        return []

    results: list[dict[str, Any]] = []
    phase_start = time.perf_counter()
    print(f"Running {len(cases)} case(s) with {workers} worker(s)...", flush=True)
    if workers <= 1:
        for idx, case in enumerate(cases, 1):
            result = run_case(case, shared)
            results.append(result)
            source = "cache" if result.get("cache_hit") else "run"
            case_elapsed = "cache" if result.get("cache_hit") else format_elapsed(float(result.get("elapsed_wall_time", 0.0)))
            print(
                f"[{idx}/{len(cases)}] {case['case_id']} done ({source}) | "
                f"case_elapsed={case_elapsed} | phase_elapsed={format_elapsed(time.perf_counter() - phase_start)}",
                flush=True,
            )
        return results

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_case, case, shared): case for case in cases}
        for idx, future in enumerate(as_completed(futures), 1):
            case = futures[future]
            result = future.result()
            results.append(result)
            source = "cache" if result.get("cache_hit") else "run"
            case_elapsed = "cache" if result.get("cache_hit") else format_elapsed(float(result.get("elapsed_wall_time", 0.0)))
            print(
                f"[{idx}/{len(cases)}] {case['case_id']} done ({source}) | "
                f"case_elapsed={case_elapsed} | phase_elapsed={format_elapsed(time.perf_counter() - phase_start)}",
                flush=True,
            )
    return sorted(results, key=lambda row: row["case_id"])


def component_array(result: dict[str, Any]) -> np.ndarray:
    return np.asarray(result["components"], dtype=float)


def attach_global_normalization(results: list[dict[str, Any]], min_values: np.ndarray, span: np.ndarray) -> None:
    for result in results:
        annotate_constraint_status(result)
        values = component_array(result)
        result["global_normalized_components"] = normalize_with_scale(values, min_values, span).tolist()
        weights = np.asarray(result["weights"], dtype=float)
        result["global_weighted_score"] = float(np.dot(weights, np.asarray(result["global_normalized_components"], dtype=float)))


def annotate_dominance(results: list[dict[str, Any]], prefix: str = "") -> None:
    feasible_idx = [
        idx
        for idx, result in enumerate(results)
        if bool(result.get("overall_feasible", float(result.get("base_penalty", 0.0)) <= 0.0))
        and "global_normalized_components" in result
    ]
    if not feasible_idx:
        for result in results:
            result[f"{prefix}is_nondominated"] = False
            result[f"{prefix}dominated_by"] = []
        return

    points = np.asarray([results[idx]["global_normalized_components"] for idx in feasible_idx], dtype=float)
    nd = nondominated_mask(points)
    for result in results:
        result[f"{prefix}is_nondominated"] = False
        result[f"{prefix}dominated_by"] = []

    for local_i, global_i in enumerate(feasible_idx):
        dominators = []
        for local_j, global_j in enumerate(feasible_idx):
            if local_i == local_j:
                continue
            if np.all(points[local_j] <= points[local_i]) and np.any(points[local_j] < points[local_i]):
                dominators.append(str(results[global_j]["case_id"]))
        results[global_i][f"{prefix}is_nondominated"] = bool(nd[local_i])
        results[global_i][f"{prefix}dominated_by"] = dominators


def scalar_stats(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size <= 1:
        return float(np.mean(arr)), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def flatten_result_row(result: dict[str, Any]) -> dict[str, Any]:
    components = np.asarray(result["components"], dtype=float)
    normalized = np.asarray(result.get("global_normalized_components", result["normalized_components"]), dtype=float)
    repeat_components = np.asarray(result.get("repeat_components", []), dtype=float)
    if repeat_components.size == 0:
        repeat_components = components.reshape(1, 3)
    repeat_global = repeat_components.copy()
    row: dict[str, Any] = {
        "case_id": result["case_id"],
        "stage": result["stage"],
        "label": result["label"],
        "meaning": result.get("meaning", ""),
        "normalization": result["normalization"],
        "w_y": result["weights"][0],
        "w_u": result["weights"][1],
        "w_e": result["weights"][2],
        "seed": result["seed"],
        "repeats": result["repeats"],
        "max_nfes": result["max_nfes"],
        "best_scalar": result["best_scalar"],
        "mean_scalar": result["mean_scalar"],
        "std_scalar": result["std_scalar"],
        "fearate": result["fearate"],
        "elapsed_wall_time": result["elapsed_wall_time"],
        "base_penalty": result["base_penalty"],
        "epsilon_feasible": bool(result.get("epsilon_feasible", True)),
        "epsilon_violation": float(result.get("epsilon_violation", 0.0)),
        "overall_feasible": bool(result.get("overall_feasible", float(result.get("base_penalty", 0.0)) <= 0.0)),
        "is_nondominated": bool(result.get("is_nondominated", False)),
        "dominated_by": ";".join(result.get("dominated_by", [])),
        "J_y": components[0],
        "J_u": components[1],
        "J_e": components[2],
        "J_y_bar": normalized[0],
        "J_u_bar": normalized[1],
        "J_e_bar": normalized[2],
        "global_weighted_score": result.get("global_weighted_score", ""),
    }
    if "global_scale_min" in result and "global_scale_span" in result:
        mins = np.asarray(result["global_scale_min"], dtype=float)
        spans = np.asarray(result["global_scale_span"], dtype=float)
        repeat_global = normalize_with_scale(repeat_components, mins, spans)
    for idx, key in enumerate(OBJECTIVE_KEYS):
        mean_raw, std_raw = scalar_stats(repeat_components[:, idx])
        mean_norm, std_norm = scalar_stats(repeat_global[:, idx])
        row[f"{key}_mean"] = mean_raw
        row[f"{key}_std"] = std_raw
        row[f"{key}_bar_mean"] = mean_norm
        row[f"{key}_bar_std"] = std_norm
    return row


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_scale_csv(path: Path, min_values: np.ndarray, max_values: np.ndarray, span: np.ndarray) -> None:
    rows = []
    for idx, key in enumerate(OBJECTIVE_KEYS):
        rows.append(
            {
                "objective": key,
                "physical_meaning": ["Tip vibration deformation", "Control energy", "Maximum tracking error"][idx],
                "unit": OBJECTIVE_UNITS[idx],
                "min_scale": float(min_values[idx]),
                "max_scale": float(max_values[idx]),
                "span": float(span[idx]),
                "normalized_objective": f"{key}_bar",
            }
        )
    write_csv(path, rows)


def nondominated_mask(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    keep = np.ones(arr.shape[0], dtype=bool)
    for i in range(arr.shape[0]):
        if not keep[i]:
            continue
        dominated = np.all(arr <= arr[i], axis=1) & np.any(arr < arr[i], axis=1)
        dominated[i] = False
        if np.any(dominated):
            keep[i] = False
    return keep


def save_figure(fig: Any, figures_dir: Path, name: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(figures_dir / f"{name}.{suffix}", bbox_inches="tight", dpi=600)


def setup_matplotlib() -> Any:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "axes.linewidth": 0.9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def plot_conflict_heatmap(results_by_id: dict[str, dict[str, Any]], figures_dir: Path) -> None:
    plt = setup_matplotlib()
    ids = ["W1", "W2", "W3", "W14", "UNSCALED_W14"]
    ids = [case_id for case_id in ids if case_id in results_by_id]
    matrix = np.vstack([results_by_id[case_id]["global_normalized_components"] for case_id in ids])
    labels = [results_by_id[case_id]["label"] for case_id in ids]

    fig, ax = plt.subplots(figsize=(5.9, 2.9))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto", vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))
    ax.set_xticks(np.arange(3), OBJECTIVE_LABELS)
    ax.set_yticks(np.arange(len(ids)), labels)
    ax.set_title("Objective Conflict Under Single-Objective Optima")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] > np.nanmean(matrix) else "black"
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color=color, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.86)
    cbar.set_label("Min-max normalized value")
    save_figure(fig, figures_dir, "fig01_objective_conflict_heatmap")
    plt.close(fig)


def plot_normalization_scales(min_values: np.ndarray, max_values: np.ndarray, span: np.ndarray, figures_dir: Path) -> None:
    plt = setup_matplotlib()
    x = np.arange(3)
    width = 0.34
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.bar(x - width / 2, min_values, width, label="Lower scale", color="#4C78A8")
    ax.bar(x + width / 2, max_values, width, label="Upper scale", color="#F58518")
    ax.scatter(x, span, color="#54A24B", marker="D", s=32, label="Span")
    ax.set_yscale("log")
    ax.set_xticks(x, OBJECTIVE_LABELS)
    ax.set_ylabel("Raw objective value (log scale)")
    ax.set_title("Normalization Scales for Problem 21 Objectives")
    ax.grid(True, axis="y", which="both", alpha=0.25)
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    for i, unit in enumerate(OBJECTIVE_UNITS):
        ax.text(i, min(max_values[i] * 1.18, max_values[i] + 1e-12), unit, ha="center", va="bottom", fontsize=7)
    save_figure(fig, figures_dir, "fig02_normalization_scales")
    plt.close(fig)


def plot_weight_sensitivity(weight_results: list[dict[str, Any]], figures_dir: Path) -> None:
    plt = setup_matplotlib()
    weight_results = sorted(weight_results, key=lambda row: int(row["case_id"][1:]))
    x = np.arange(len(weight_results))
    matrix = np.vstack([row["global_normalized_components"] for row in weight_results])
    scores = np.asarray([row["global_weighted_score"] for row in weight_results], dtype=float)
    labels = [row["case_id"] for row in weight_results]
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    width = 0.24
    for idx, key in enumerate(OBJECTIVE_LABELS):
        ax.bar(x + (idx - 1) * width, matrix[:, idx], width, label=key, color=colors[idx], alpha=0.88)
    ax2 = ax.twinx()
    ax2.plot(x, scores, color="#B279A2", marker="o", linewidth=1.4, markersize=3.5, label="Weighted score")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("Normalized objective value")
    ax2.set_ylabel("Weighted normalized score")
    ax.set_title("Sensitivity of OPMWADE Solution to Weighting Factors")
    ax.grid(True, axis="y", alpha=0.25)
    handles, labels_ = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels_ + labels2, frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    save_figure(fig, figures_dir, "fig03_weight_sensitivity")
    plt.close(fig)


def barycentric_to_xy(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = np.asarray(weights, dtype=float)
    x = w[:, 1] + 0.5 * w[:, 2]
    y = (math.sqrt(3.0) / 2.0) * w[:, 2]
    return x, y


def plot_weight_ternary(weight_results: list[dict[str, Any]], figures_dir: Path) -> None:
    plt = setup_matplotlib()
    weights = np.asarray([row["weights"] for row in weight_results], dtype=float)
    scores = np.asarray([row["global_weighted_score"] for row in weight_results], dtype=float)
    x, y = barycentric_to_xy(weights)

    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    triangle = np.array([[0, 0], [1, 0], [0.5, math.sqrt(3.0) / 2.0], [0, 0]])
    ax.plot(triangle[:, 0], triangle[:, 1], color="0.2", linewidth=1.0)
    for g in np.linspace(0.2, 0.8, 4):
        ax.plot([g, 0.5 + 0.5 * g], [0, (math.sqrt(3.0) / 2.0) * (1 - g)], color="0.85", linewidth=0.6)
        ax.plot([1 - g, 0.5 * (1 - g)], [0, (math.sqrt(3.0) / 2.0) * (1 - g)], color="0.85", linewidth=0.6)
        ax.plot([0.5 * g, 1 - 0.5 * g], [(math.sqrt(3.0) / 2.0) * g, (math.sqrt(3.0) / 2.0) * g], color="0.85", linewidth=0.6)
    sc = ax.scatter(x, y, c=scores, cmap="magma_r", s=42, edgecolor="white", linewidth=0.5, zorder=3)
    for xi, yi, row in zip(x, y, weight_results):
        offsets = {
            "W1": (0.035, 0.050, "left", "bottom"),
            "W2": (-0.035, 0.050, "right", "bottom"),
            "W3": (0.000, -0.055, "center", "top"),
            "W8": (-0.025, 0.025, "right", "bottom"),
            "W14": (0.030, -0.022, "left", "top"),
        }
        dx, dy, ha, va = offsets.get(row["case_id"], (0.0, 0.024, "center", "bottom"))
        ax.text(xi + dx, yi + dy, row["case_id"], ha=ha, va=va, fontsize=6.8)
    ax.text(-0.045, -0.045, r"$w_y$", ha="right", va="top", fontsize=10)
    ax.text(1.045, -0.045, r"$w_u$", ha="left", va="top", fontsize=10)
    ax.text(0.5, math.sqrt(3.0) / 2.0 + 0.055, r"$w_e$", ha="center", va="bottom", fontsize=10)
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, math.sqrt(3.0) / 2.0 + 0.10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Weight-Space Map of Scalarized Performance", pad=14)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Weighted normalized score")
    save_figure(fig, figures_dir, "fig04_weight_ternary")
    plt.close(fig)


def plot_pareto_pairwise(weight_results: list[dict[str, Any]], pareto_results: list[dict[str, Any]], figures_dir: Path) -> None:
    plt = setup_matplotlib()
    weighted_rows = [row for row in weight_results if bool(row.get("overall_feasible", True))]
    pareto_feasible_rows = [row for row in pareto_results if bool(row.get("overall_feasible", False))]
    pareto_infeasible_rows = [row for row in pareto_results if not bool(row.get("overall_feasible", False))]
    weighted = np.vstack([row["global_normalized_components"] for row in weighted_rows]) if weighted_rows else np.empty((0, 3))
    pareto = np.vstack([row["global_normalized_components"] for row in pareto_feasible_rows]) if pareto_feasible_rows else np.empty((0, 3))
    pareto_bad = np.vstack([row["global_normalized_components"] for row in pareto_infeasible_rows]) if pareto_infeasible_rows else np.empty((0, 3))
    combined = np.vstack([arr for arr in (weighted, pareto) if arr.size]) if (weighted.size or pareto.size) else np.empty((0, 3))
    nd = nondominated_mask(combined) if combined.size else np.empty(0, dtype=bool)
    pairs = [(0, 1), (0, 2), (1, 2)]
    titles = [r"$\bar{J}_y$-$\bar{J}_u$", r"$\bar{J}_y$-$\bar{J}_e$", r"$\bar{J}_u$-$\bar{J}_e$"]

    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.8))
    for ax, (i, j), title in zip(axes, pairs, titles):
        if weighted.size:
            ax.scatter(weighted[:, i], weighted[:, j], s=30, color="#4C78A8", alpha=0.82, label="Weighted-sum")
        if pareto_bad.size:
            ax.scatter(pareto_bad[:, i], pareto_bad[:, j], s=34, marker="x", color="0.55", alpha=0.75, label="epsilon infeasible")
        if pareto.size:
            ax.scatter(pareto[:, i], pareto[:, j], s=34, marker="s", color="#F58518", alpha=0.82, label="epsilon feasible")
        if combined.size:
            ax.scatter(combined[nd, i], combined[nd, j], s=58, facecolors="none", edgecolors="black", linewidth=0.9, label="Feasible non-dominated")
        ax.set_xlabel(OBJECTIVE_LABELS[i] + " normalized")
        ax.set_ylabel(OBJECTIVE_LABELS[j] + " normalized")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    handles, labels = [], []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, frameon=False, ncols=min(4, len(labels)), loc="upper center", bbox_to_anchor=(0.5, 1.12))
    save_figure(fig, figures_dir, "fig05_pareto_pairwise")
    plt.close(fig)


def plot_pareto_3d(weight_results: list[dict[str, Any]], pareto_results: list[dict[str, Any]], figures_dir: Path) -> None:
    plt = setup_matplotlib()
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    weighted_rows = [row for row in weight_results if bool(row.get("overall_feasible", True))]
    pareto_feasible_rows = [row for row in pareto_results if bool(row.get("overall_feasible", False))]
    pareto_infeasible_rows = [row for row in pareto_results if not bool(row.get("overall_feasible", False))]
    weighted = np.vstack([row["global_normalized_components"] for row in weighted_rows]) if weighted_rows else np.empty((0, 3))
    pareto = np.vstack([row["global_normalized_components"] for row in pareto_feasible_rows]) if pareto_feasible_rows else np.empty((0, 3))
    pareto_bad = np.vstack([row["global_normalized_components"] for row in pareto_infeasible_rows]) if pareto_infeasible_rows else np.empty((0, 3))

    fig = plt.figure(figsize=(5.2, 4.3))
    ax = fig.add_subplot(111, projection="3d")
    if weighted.size:
        ax.scatter(weighted[:, 0], weighted[:, 1], weighted[:, 2], s=28, color="#4C78A8", alpha=0.82, label="Weighted-sum")
    if pareto_bad.size:
        ax.scatter(pareto_bad[:, 0], pareto_bad[:, 1], pareto_bad[:, 2], s=30, marker="x", color="0.55", alpha=0.70, label="epsilon infeasible")
    if pareto.size:
        ax.scatter(pareto[:, 0], pareto[:, 1], pareto[:, 2], s=34, marker="s", color="#F58518", alpha=0.86, label="epsilon feasible")
    ax.set_xlabel(r"$\bar{J}_y$")
    ax.set_ylabel(r"$\bar{J}_u$")
    ax.set_zlabel(r"$\bar{J}_e$")
    ax.view_init(elev=23, azim=-52)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Three-Objective Pareto Verification")
    save_figure(fig, figures_dir, "fig06_pareto_3d")
    plt.close(fig)


def response_for_cases(selected: list[dict[str, Any]], shared: dict[str, Any], data_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    from algorithms.common import configure_problem_from_init_data, problem21_solution_response, set_problem21_target_angle, set_trajectory_damping_mode

    set_trajectory_damping_mode(shared["damping_mode"])
    set_problem21_target_angle(float(shared["target_angle"]))
    configure_problem_from_init_data(21, init_file=shared["init_file"])

    responses: dict[str, dict[str, np.ndarray]] = {}
    npz_payload: dict[str, np.ndarray] = {}
    for result in selected:
        case_id = safe_id(result["case_id"])
        response = problem21_solution_response(np.asarray(result["best_x"], dtype=float), damping_mode=shared["damping_mode"])
        time_values = np.asarray(response["time"], dtype=float)
        max_points = 1600
        idx = np.linspace(0, time_values.size - 1, min(max_points, time_values.size)).astype(int)
        payload = {
            "time": time_values[idx],
            "tip_deflection": np.asarray(response["tip_deflection"], dtype=float)[idx],
            "tracking_error": np.asarray(response["tracking_error"], dtype=float)[idx],
            "torque": np.asarray(response["torque"], dtype=float)[idx],
        }
        responses[case_id] = payload
        for key, value in payload.items():
            npz_payload[f"{case_id}_{key}"] = value
    np.savez_compressed(data_dir / "selected_time_responses.npz", **npz_payload)
    return responses


def plot_responses(responses: dict[str, dict[str, np.ndarray]], figures_dir: Path) -> None:
    if not responses:
        return
    plt = setup_matplotlib()
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]

    fig, axes = plt.subplots(3, 1, figsize=(6.8, 5.2), sharex=True)
    for idx, (case_id, response) in enumerate(responses.items()):
        color = colors[idx % len(colors)]
        t = response["time"]
        axes[0].plot(t, response["tip_deflection"], linewidth=1.0, color=color, label=case_id)
        axes[1].plot(t, response["tracking_error"], linewidth=1.0, color=color)
        axes[2].plot(t, response["torque"], linewidth=1.0, color=color)
    axes[0].set_ylabel(r"$y(L,t)$")
    axes[1].set_ylabel(r"$\theta_s-\theta_r$")
    axes[2].set_ylabel(r"$u(t)$")
    axes[2].set_xlabel("Time (s)")
    axes[0].set_title("Representative Closed-Loop Responses")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False, ncols=min(4, len(responses)), loc="upper center", bbox_to_anchor=(0.5, 1.34))
    save_figure(fig, figures_dir, "fig07_representative_responses")
    plt.close(fig)


def build_pareto_cases(
    weight_results: list[dict[str, Any]],
    min_values: np.ndarray,
    max_values: np.ndarray,
    args: argparse.Namespace,
    seed_offset: int,
) -> list[dict[str, Any]]:
    grid = max(2, int(args.pareto_grid))
    normalized = np.vstack([row["global_normalized_components"] for row in weight_results])
    ju_values = normalized[:, 1]
    je_values = normalized[:, 2]
    q = np.linspace(0.25, 0.85, grid)
    eps_u = np.quantile(ju_values, q)
    eps_e = np.quantile(je_values, q)
    cases = []
    counter = 0
    for i, eu in enumerate(eps_u, 1):
        for j, ee in enumerate(eps_e, 1):
            counter += 1
            cases.append(
                make_case(
                    case_id=f"PARETO_E{i}{j}",
                    stage="pareto_epsilon",
                    label=fr"$\epsilon_u={eu:.2f}, \epsilon_e={ee:.2f}$",
                    meaning="epsilon-constraint Pareto verification",
                    weights=(1.0, 0.0, 0.0),
                    normalization="minmax",
                    min_values=min_values,
                    max_values=max_values,
                    constraint_limits=(math.inf, float(eu), float(ee)),
                    seed=args.seed + seed_offset + counter,
                    max_nfes=stage_max_nfes(args, "pareto"),
                    repeats=stage_repeats(args, "pareto"),
                )
            )
    return cases


def write_report(
    path: Path,
    args: argparse.Namespace,
    min_values: np.ndarray,
    max_values: np.ndarray,
    span: np.ndarray,
    conflict_rows: list[dict[str, Any]],
    weight_rows: list[dict[str, Any]],
    pareto_rows: list[dict[str, Any]],
    nondominated_rows: list[dict[str, Any]],
) -> None:
    pareto_feasible_count = sum(bool(row.get("overall_feasible", False)) for row in pareto_rows)
    proposed = next((row for row in weight_rows if row["case_id"] == "W14"), None)
    proposed_dominated_by = proposed.get("dominated_by", "") if proposed is not None else ""
    lines = [
        "# Problem 21 OPMWADE Sensitivity Experiments",
        "",
        f"- Damping mode: `{args.damping_mode}`",
        f"- Target angle: `{args.target_angle}` rad",
        f"- Initial population file: `{args.init_file}`",
        f"- Anchor repeats / NFEs: `{stage_repeats(args, 'anchor')}` / `{stage_max_nfes(args, 'anchor')}`",
        f"- Sensitivity repeats / NFEs: `{stage_repeats(args, 'sensitivity')}` / `{stage_max_nfes(args, 'sensitivity')}`",
        f"- Pareto repeats / NFEs: `{stage_repeats(args, 'pareto')}` / `{stage_max_nfes(args, 'pareto')}`",
        f"- Population factors: initial `{args.initial_np_factor}`, minimum `{args.min_np_factor}`",
        f"- Parallel workers: `{args.workers}`",
        "",
        "## Normalization Scales",
        "",
        "| Objective | Unit | Min | Max | Span |",
        "|---|---:|---:|---:|---:|",
    ]
    for idx, key in enumerate(OBJECTIVE_KEYS):
        lines.append(f"| {key} | {OBJECTIVE_UNITS[idx]} | {min_values[idx]:.8g} | {max_values[idx]:.8g} | {span[idx]:.8g} |")
    lines.extend(["", "## Objective Conflict Summary", "", "| Case | Jy | Ju | Je | Jy_bar | Ju_bar | Je_bar |", "|---|---:|---:|---:|---:|---:|---:|"])
    for row in conflict_rows:
        lines.append(
            f"| {row['case_id']} | {row['J_y']:.6g} | {row['J_u']:.6g} | {row['J_e']:.6g} | "
            f"{row['J_y_bar']:.3f} | {row['J_u_bar']:.3f} | {row['J_e_bar']:.3f} |"
        )
    lines.extend(["", "## Weight Sensitivity Summary", "", "| Case | wy | wu | we | Jy_bar | Ju_bar | Je_bar | Score | Non-dominated | Dominated by |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"])
    for row in weight_rows:
        lines.append(
            f"| {row['case_id']} | {row['w_y']:.3g} | {row['w_u']:.3g} | {row['w_e']:.3g} | "
            f"{row['J_y_bar']:.3f} | {row['J_u_bar']:.3f} | {row['J_e_bar']:.3f} | {row['global_weighted_score']:.3f} | "
            f"{row['is_nondominated']} | {row['dominated_by']} |"
        )
    lines.extend(
        [
            "",
            "## Pareto Verification",
            "",
            f"- epsilon-constraint cases: `{len(pareto_rows)}`",
            f"- epsilon-feasible and terminal-feasible cases: `{pareto_feasible_count}`",
            "- In the Pareto figures, gray crosses are infeasible epsilon attempts; orange squares are feasible epsilon-constraint solutions.",
            "",
            "| Case | epsilon feasible | overall feasible | Jy_bar | Ju_bar | Je_bar | epsilon violation |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in pareto_rows:
        lines.append(
            f"| {row['case_id']} | {row['epsilon_feasible']} | {row['overall_feasible']} | "
            f"{row['J_y_bar']:.3f} | {row['J_u_bar']:.3f} | {row['J_e_bar']:.3f} | {row['epsilon_violation']:.3g} |"
        )
    lines.extend(["", "## Non-Dominated Candidate Summary", "", "| Case | Stage | Jy_bar | Ju_bar | Je_bar |", "|---|---|---:|---:|---:|"])
    for row in nondominated_rows:
        lines.append(f"| {row['case_id']} | {row['stage']} | {row['J_y_bar']:.3f} | {row['J_u_bar']:.3f} | {row['J_e_bar']:.3f} |")
    if proposed_dominated_by:
        lines.extend(
            [
                "",
                "## Proposed Setting Check",
                "",
                f"- W14 is dominated by: `{proposed_dominated_by}` under the current run. Treat W14 as a sensitivity sample unless larger-budget runs remove this dominance.",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    dirs = ensure_dirs(args.output_dir)
    shared = {
        "cache_dir": str(dirs["cache"]),
        "resume": not args.no_resume,
        "init_file": str(args.init_file.resolve()),
        "damping_mode": args.damping_mode,
        "target_angle": float(args.target_angle),
        "initial_np_factor": float(args.initial_np_factor),
        "min_np_factor": float(args.min_np_factor),
        "progress_interval": int(args.progress_interval),
        "enable_late_enhancements": bool(args.enable_late_enhancements),
    }
    config_path = dirs["data"] / "run_config.json"
    config_path.write_text(json.dumps({**vars(args), "init_file": str(args.init_file), "output_dir": str(args.output_dir)}, indent=2, default=str), encoding="utf-8")

    print("Phase 1/4: objective-conflict anchor runs", flush=True)
    anchor_cases = [
        make_case(
            case_id=case_id,
            stage="conflict_single_objective",
            label=meaning,
            meaning=meaning,
            weights=weights,
            normalization="none",
            seed=args.seed + idx,
            max_nfes=stage_max_nfes(args, "anchor"),
            repeats=stage_repeats(args, "anchor"),
        )
        for idx, (case_id, weights, meaning) in enumerate(WEIGHT_CASES[:3], 1)
    ]
    anchor_results = run_cases(anchor_cases, shared, args.workers)
    anchor_components = np.vstack([component_array(row) for row in anchor_results])
    min_values = np.min(anchor_components, axis=0)
    max_values = np.max(anchor_components, axis=0)
    span = finite_span(min_values, max_values)
    max_values = min_values + span
    write_scale_csv(dirs["data"] / "normalization_scales.csv", min_values, max_values, span)

    print("Phase 2/4: normalization and weight-sensitivity runs", flush=True)
    unscaled_case = make_case(
        case_id="UNSCALED_W14",
        stage="normalization_comparison",
        label="Unnormalized proposed setting",
        meaning="Historical unnormalized weighted-sum",
        weights=PROPOSED_WEIGHTS,
        normalization="none",
        seed=args.seed + 100,
        max_nfes=stage_max_nfes(args, "sensitivity"),
        repeats=stage_repeats(args, "sensitivity"),
    )
    sensitivity_cases = [
        make_case(
            case_id=case_id,
            stage="weight_sensitivity",
            label=meaning,
            meaning=meaning,
            weights=weights,
            normalization="minmax",
            min_values=min_values,
            max_values=max_values,
            seed=args.seed + 200 + idx,
            max_nfes=stage_max_nfes(args, "sensitivity"),
            repeats=stage_repeats(args, "sensitivity"),
        )
        for idx, (case_id, weights, meaning) in enumerate(WEIGHT_CASES[3:], 1)
    ]
    phase2_results = run_cases([unscaled_case, *sensitivity_cases], shared, args.workers)
    weight_results = [*anchor_results, *[row for row in phase2_results if row["case_id"].startswith("W")]]
    all_results = [*anchor_results, *phase2_results]
    attach_global_normalization(all_results, min_values, span)
    for result in all_results:
        result["global_scale_min"] = min_values.tolist()
        result["global_scale_span"] = span.tolist()

    print("Phase 3/4: epsilon-constraint Pareto verification", flush=True)
    pareto_cases = build_pareto_cases(weight_results, min_values, max_values, args, seed_offset=1000)
    pareto_results = run_cases(pareto_cases, shared, args.workers)
    attach_global_normalization(pareto_results, min_values, span)
    for result in pareto_results:
        result["global_scale_min"] = min_values.tolist()
        result["global_scale_span"] = span.tolist()
    all_results.extend(pareto_results)
    annotate_dominance(all_results)

    rows = [flatten_result_row(row) for row in all_results]
    write_csv(dirs["data"] / "all_case_results.csv", rows)
    conflict_rows = [flatten_result_row(row) for row in all_results if row["case_id"] in {"W1", "W2", "W3", "W14", "UNSCALED_W14"}]
    weight_rows = [flatten_result_row(row) for row in sorted(weight_results, key=lambda r: int(r["case_id"][1:]))]
    pareto_rows = [flatten_result_row(row) for row in pareto_results]
    nondominated_rows = [
        flatten_result_row(row)
        for row in sorted(
            all_results,
            key=lambda r: (str(r["stage"]), str(r["case_id"])),
        )
        if bool(row.get("is_nondominated", False))
    ]
    write_csv(dirs["data"] / "objective_conflict_results.csv", conflict_rows)
    write_csv(dirs["data"] / "weight_sensitivity_results.csv", weight_rows)
    write_csv(dirs["data"] / "pareto_epsilon_results.csv", pareto_rows)
    write_csv(dirs["data"] / "nondominated_summary.csv", nondominated_rows)
    (dirs["data"] / "all_case_results.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    print("Phase 4/4: plotting journal-style figures", flush=True)
    results_by_id = {row["case_id"]: row for row in all_results}
    plot_conflict_heatmap(results_by_id, dirs["figures"])
    plot_normalization_scales(min_values, max_values, span, dirs["figures"])
    plot_weight_sensitivity(weight_results, dirs["figures"])
    plot_weight_ternary(weight_results, dirs["figures"])
    plot_pareto_pairwise(weight_results, pareto_results, dirs["figures"])
    plot_pareto_3d(weight_results, pareto_results, dirs["figures"])
    selected_ids = ["W1", "W2", "W3", "W14"]
    selected = [results_by_id[case_id] for case_id in selected_ids if case_id in results_by_id]
    responses = response_for_cases(selected, shared, dirs["data"])
    plot_responses(responses, dirs["figures"])

    write_report(
        dirs["root"] / "problem21_opmwade_experiment_report.md",
        args,
        min_values,
        max_values,
        span,
        conflict_rows,
        weight_rows,
        pareto_rows,
        nondominated_rows,
    )
    print(f"Done. Results written to: {dirs['root']}", flush=True)


if __name__ == "__main__":
    main()
