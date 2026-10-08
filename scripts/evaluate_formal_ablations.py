"""Independent common-sample evaluation for the formal ablation study.

The search uses 128 samples for affordable candidate ranking. This script is
called only after all four searches finish and evaluates their selected lenses
with the same 1024 manufacturing-error draws. It also verifies the comparison
contract (89 material states and the intended final optimization stage).
"""

from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from deeplens.geolens import GeoLens
from deeplens.geolens_pkg.differentiable_mtf import ray_fourier_mtf
from deeplens.geolens_pkg.manufacturing import (
    configure_tolerances,
    evaluate,
    material_price_report,
    process_cost_proxy,
    read_relative_prices,
)


design = importlib.import_module("8_manufacturing_design")
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/paper"
EVALUATION_TRIALS = 1024
COMMON_ERROR_SEED = 20260928
RUNS = [
    ("Full method", "full", "robust"),
    ("Nominal only", "nominal_only", "nominal"),
    ("No soft yield", "no_soft_yield", "robust"),
    ("Greedy glass", "greedy", "robust"),
]


def load_search_settings() -> dict:
    path = ROOT / "configs/paper/full.yml"
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = yaml.safe_load((ROOT / settings["base_config"]).read_text(encoding="utf-8"))
    base["sampling"].update(settings.get("sampling_overrides", {}))
    base["process_cost"] = settings["process_cost"]
    base["differentiable_mtf"] = settings["differentiable_mtf"]
    base["acceptance"]["max_rgb_rms_mm"] = float(settings["acceptance_max_rgb_rms_mm"])
    return base


def evaluate_lens(name: str, lens_path: Path, base: dict, device: str) -> dict:
    lens = GeoLens(filename=str(lens_path), device=device)
    sampling = base["sampling"]
    design.apply_fixed_field(lens, sampling)
    configure_tolerances(lens, base["tolerances"])
    prices = read_relative_prices(str(ROOT / base["glass_catalog"]))
    elements, missing = material_price_report(
        lens, prices, base["missing_price_policy"])
    if missing:
        raise RuntimeError(f"Missing material prices for {name}: {missing}")

    result = evaluate(
        lens,
        sampling,
        elements,
        base["cost_proxy"],
        base["tolerances"],
        float(base["acceptance"]["max_rgb_rms_mm"]),
        EVALUATION_TRIALS,
        COMMON_ERROR_SEED,
    )
    rules = base["process_cost"]
    result["process_cost_proxy"] = float(process_cost_proxy(lens, elements, rules).item())
    result["total_build_cost_proxy"] = (
        result["cost_proxy_hk9l_mm3"] + result["process_cost_proxy"])
    result["expected_good_unit_cost_proxy"] = (
        result["total_build_cost_proxy"] / result["yield_estimate"]
        if result["yield_estimate"] > 0 else None)

    mtf_settings = base["differentiable_mtf"]
    gpu_devices = [torch.cuda.current_device()] if lens.device.type == "cuda" else []
    with torch.no_grad(), torch.random.fork_rng(devices=gpu_devices):
        torch.manual_seed(int(mtf_settings.get("ray_seed", COMMON_ERROR_SEED + 1)))
        mtf = ray_fourier_mtf(lens, mtf_settings)
    result["mtf50_min_ts"] = float(mtf.min().item())
    result["method"] = name
    result["lens_json"] = str(lens_path.relative_to(ROOT))
    return result


def compact(result: dict, start_cost: float) -> dict:
    return {
        "method": result["method"],
        "manufacturing_cost_index": result["total_build_cost_proxy"],
        "normalized_manufacturing_cost": result["total_build_cost_proxy"] / start_cost,
        "nominal_rms_mm": result["nominal_rms_mm"],
        "rms_mean_mm": result["rms_mean_mm"],
        "rms_p90_mm": result["rms_p90_mm"],
        "hard_yield": result["yield_estimate"],
        "yield_wilson95_low": result["yield_wilson95"][0],
        "yield_wilson95_high": result["yield_wilson95"][1],
        "expected_accepted_unit_cost_index": result["expected_good_unit_cost_proxy"],
        "mtf50_min_ts": result["mtf50_min_ts"],
        "evaluation_trials": EVALUATION_TRIALS,
        "common_error_seed": COMMON_ERROR_SEED,
        "lens_json": result["lens_json"],
    }


def plot(rows: list[dict]) -> None:
    labels = [row["method"] for row in rows]
    colors = ["#0f766e", "#64748b", "#d97706", "#7c3aed"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))

    definitions = [
        ("normalized_manufacturing_cost", "Normalized manufacturing cost", "Search start = 1.0"),
        ("rms_p90_mm", "Tolerance RMS p90", "mm; lower is better"),
        ("hard_yield", "Monte Carlo pass fraction", "1024 common error draws"),
        ("mtf50_min_ts", "Differentiable MTF surrogate", "Center field, 50 lp/mm"),
    ]
    for ax, (key, title, ylabel) in zip(axes, definitions):
        values = [row[key] for row in rows]
        bars = ax.bar(labels, values, color=colors)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=18)
        if key == "hard_yield":
            lower = [row[key] - row["yield_wilson95_low"] for row in rows]
            upper = [row["yield_wilson95_high"] - row[key] for row in rows]
            ax.errorbar(range(len(rows)), values, yerr=[lower, upper], fmt="none",
                        color="#1f2937", capsize=4)
            ax.set_ylim(0, 1.08)
        for bar, value in zip(bars, values):
            label = f"{value:.1%}" if key == "hard_yield" else f"{value:.3f}"
            ax.annotate(label, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)
    fig.suptitle("Formal ablation study with matched search and tolerance budgets")
    fig.tight_layout()
    fig.savefig(OUTPUT / "formal_ablation_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("Formal 1024-sample evaluation requires the CUDA environment")
    base = load_search_settings()
    reports = {}
    for label, folder, expected_stage in RUNS:
        path = OUTPUT / folder / "hammer_report.json"
        if not path.exists():
            raise FileNotFoundError(f"Formal ablation is incomplete: {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        if int(report["unique_material_states_evaluated"]) != 89:
            raise RuntimeError(
                f"{label} evaluated {report['unique_material_states_evaluated']} states; expected 89")
        if report["final_best"]["stage"] != expected_stage:
            raise RuntimeError(
                f"{label} ended at {report['final_best']['stage']}; expected {expected_stage}")
        reports[label] = report

    start_path = Path(reports["Full method"]["history"][0]["best"]["pre_recovery"]["lens_json"])
    start = evaluate_lens("Search start", ROOT / start_path, base, device)
    evaluated = []
    for label, _, _ in RUNS:
        lens_path = ROOT / reports[label]["final_best"]["lens_json"]
        print(f"Independent evaluation: {label}", flush=True)
        evaluated.append(evaluate_lens(label, lens_path, base, device))

    rows = [compact(item, start["total_build_cost_proxy"]) for item in evaluated]
    output = {
        "protocol": {
            "acceptance_rms_mm": base["acceptance"]["max_rgb_rms_mm"],
            "evaluation_trials": EVALUATION_TRIALS,
            "common_error_seed": COMMON_ERROR_SEED,
            "material_states_per_method": 89,
            "mtf_metric": "central-field min(T,S) differentiable surrogate at 50 lp/mm",
        },
        "search_start": compact(start, start["total_build_cost_proxy"]),
        "methods": rows,
        "rms_samples_mm": {item["method"]: item["rms_samples_mm"] for item in evaluated},
    }
    (OUTPUT / "formal_ablation_report.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8")
    with (OUTPUT / "formal_ablation_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(rows)
    for row in rows:
        print(f"{row['method']}: cost={row['normalized_manufacturing_cost']:.3f}, "
              f"RMS p90={row['rms_p90_mm']:.5f} mm, "
              f"yield={row['hard_yield']:.1%}, MTF50={row['mtf50_min_ts']:.3f}")
    print(f"Saved: {OUTPUT / 'formal_ablation_comparison.png'}")


if __name__ == "__main__":
    main()
