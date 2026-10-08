"""Hammer-inspired multi-round real-glass beam search for spherical elements.

This is not a reproduction of Zemax Hammer. It keeps several feasible material
paths, permits bounded temporary cost increases, reoptimizes every substitution,
and stops only after configurable rounds without a new global best.
"""

import argparse
import copy
import csv
import hashlib
import importlib
import itertools
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

expert = importlib.import_module("17_manufacturing_expert")
glass = importlib.import_module("15_glass_substitution")
workflow = importlib.import_module("18_iterative_manufacturing_expert")
review = importlib.import_module("10_review_mtf")


def material_signature(lens_json):
    """A geometry-independent material identity for duplicate/cycle prevention."""
    return tuple((index, surface["mat2"].lower()) for index, surface in
                 enumerate(lens_json["surfaces"])
                 if surface["mat2"].lower() not in {"air", "vacuum", "occluder"})


def signature_id(signature):
    text = json.dumps(signature, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def material_changes(lens_json, reference_json):
    changes = []
    for index, (surface, reference) in enumerate(zip(lens_json["surfaces"], reference_json["surfaces"])):
        if (surface["mat2"].lower() not in {"air", "vacuum", "occluder"} and
                surface["mat2"].lower() != reference["mat2"].lower()):
            changes.append({"surface_idx": index, "from": reference["mat2"].lower(),
                            "to": surface["mat2"].lower()})
    return changes


def apply_material_overrides(lens_json, overrides, prices):
    """Apply explicit phase-start materials such as K26R and a high-cost D glass."""
    result = copy.deepcopy(lens_json)
    changes = []
    for item in overrides:
        index = int(item["surface_idx"])
        if not 0 <= index < len(result["surfaces"]):
            raise ValueError(f"Invalid forced material surface index: {index}")
        old = result["surfaces"][index]["mat2"].lower()
        new = str(item["to"]).lower()
        if new not in prices:
            raise ValueError(f"Forced material {new} has no relative cost entry")
        # Constructing Material validates that DeepLens has usable dispersion data.
        glass.Material(new)
        result["surfaces"][index]["mat2"] = new
        changes.append({"surface_idx": index, "from": old, "to": new,
                        "reason": item.get("reason", "configured phase-start material")})
    return result, changes


def apply_geometry_perturbations(lens_json, perturbations):
    """Create a documented stress-test start using deterministic shape changes.

    This is deliberately configuration driven so the perturbed start is exactly
    reproducible and cannot be confused with the original AutoLens prescription.
    """
    result = copy.deepcopy(lens_json)
    records = []
    for item in perturbations:
        index = int(item["surface_idx"])
        if not 0 <= index < len(result["surfaces"]):
            raise ValueError(f"Invalid geometry-perturbation surface index: {index}")
        surface = result["surfaces"][index]
        before = {key: copy.deepcopy(surface.get(key)) for key in ("roc", "d_next", "conic", "ai")}
        if "roc_scale" in item:
            if "roc" not in surface or surface["roc"] in (None, 0):
                raise ValueError(f"Surface {index} has no finite roc to perturb")
            surface["roc"] = float(surface["roc"]) * float(item["roc_scale"])
        if "d_next_delta_mm" in item:
            surface["d_next"] = float(surface["d_next"]) + float(item["d_next_delta_mm"])
            if surface["d_next"] <= 0:
                raise ValueError(f"Surface {index} perturbation makes d_next nonpositive")
        if "conic_delta" in item:
            surface["conic"] = float(surface.get("conic", 0.0)) + float(item["conic_delta"])
        if "ai_scale" in item:
            if not surface.get("ai"):
                raise ValueError(f"Surface {index} has no aspheric coefficients to scale")
            surface["ai"] = [float(value) * float(item["ai_scale"]) for value in surface["ai"]]
        after = {key: copy.deepcopy(surface.get(key)) for key in ("roc", "d_next", "conic", "ai")}
        records.append({"surface_idx": index, "before": before, "after": after,
                        "reason": item.get("reason", "configured stress-test perturbation")})
    return result, records


def save_layout_comparison(initial_path, final_path, output, settings):
    """Save layouts plus a parameter-change panel for small but important changes."""
    from deeplens.geolens import GeoLens
    depth = float(settings.get("depth", -1000.0))
    initial_png = output / "initial_lens_layout.png"
    final_png = output / "final_lens_layout.png"
    GeoLens(filename=str(initial_path), device="cpu").draw_layout(
        filename=str(initial_png), depth=depth,
        lens_title=settings.get("initial_title", "Search start"), show=False)
    GeoLens(filename=str(final_path), device="cpu").draw_layout(
        filename=str(final_png), depth=depth, lens_title="Final manufacturing-aware design", show=False)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), gridspec_kw={"width_ratios": [1.2, 1.2, 1.0]})
    for ax, path in zip(axes[:2], (initial_png, final_png)):
        ax.imshow(plt.imread(path))
        ax.axis("off")
    initial_json = json.loads(Path(initial_path).read_text(encoding="utf-8"))
    final_json = json.loads(Path(final_path).read_text(encoding="utf-8"))
    indices, curvature_change, spacing_change = [], [], []
    for index, (before, after) in enumerate(zip(initial_json["surfaces"], final_json["surfaces"])):
        before_roc, after_roc = before.get("roc"), after.get("roc")
        before_spacing, after_spacing = before.get("d_next"), after.get("d_next")
        if before_roc not in (None, 0) and after_roc not in (None, 0):
            before_curvature, after_curvature = 1.0 / float(before_roc), 1.0 / float(after_roc)
            curvature = 100.0 * (after_curvature - before_curvature) / max(abs(before_curvature), 1e-9)
        else:
            curvature = 0.0
        if before_spacing not in (None, 0) and after_spacing is not None:
            spacing = 100.0 * (float(after_spacing) - float(before_spacing)) / max(abs(float(before_spacing)), 1e-9)
        else:
            spacing = 0.0
        indices.append(index)
        curvature_change.append(curvature)
        spacing_change.append(spacing)
    x = torch.arange(len(indices)).cpu().numpy()
    axes[2].bar(x - 0.18, curvature_change, width=0.36, color="#2563eb", label="surface curvature")
    axes[2].bar(x + 0.18, spacing_change, width=0.36, color="#d97706", label="axial spacing")
    axes[2].axhline(0, color="#334155", linewidth=0.8)
    axes[2].set(title="Prescription change from search start", xlabel="DeepLens surface index",
                ylabel="Relative parameter change (%)", xticks=x)
    axes[2].grid(axis="y", alpha=.25)
    axes[2].legend(fontsize=8)
    fig.suptitle("2D optical layout and ray-path comparison")
    fig.tight_layout()
    fig.savefig(output / "initial_vs_final_lens_layout.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_matched_fixed_material_control(seed, selected, base, search, output,
                                       reference_json, process_rules):
    """Give an unchanged-material seed the same number of optimization cycles.

    This separates improvement caused by material search from improvement caused
    merely by receiving more geometry-optimization steps.
    """
    cycles = max(1, int(selected["round_created"]))
    current_lens = Path(seed["lens_json"])
    records = []
    for cycle in range(1, cycles + 1):
        folder = output / "matched_fixed_material_control" / f"cycle_{cycle}"
        nominal_config = copy.deepcopy(base)
        nominal_config["input_lens"] = str(current_lens)
        nominal_config["training"]["steps"] = int(search["nominal_steps_per_trial"])
        nominal_config["training"]["objective"] = "nominal_only"
        nominal_lens, nominal_report = workflow.run_design(
            nominal_config, folder / "nominal_recovery")
        robust_config = copy.deepcopy(base)
        robust_config["input_lens"] = str(nominal_lens)
        robust_config["training"]["steps"] = int(search["robust_steps_per_trial"])
        robust_config["training"]["objective"] = "expected_good_cost"
        robust_lens, robust_report = workflow.run_design(
            robust_config, folder / "robust_refinement")
        row = workflow.summary(f"fixed_material_cycle_{cycle}", robust_lens,
                               robust_report, None, process_rules)
        node = node_from_summary(row, robust_lens, reference_json,
                                 seed["node_id"], cycle, "control")
        records.append(node)
        current_lens = robust_lens
        (output / "matched_fixed_material_control" / "progress.json").write_text(
            json.dumps(records, indent=2), encoding="utf-8")
        print(f"Matched fixed-material control cycle {cycle}/{cycles} completed", flush=True)

    control = records[-1]
    report_path = output / "hammer_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["matched_fixed_material_control"] = {
        "purpose": "same optimization cycles without material replacement",
        "cycles": cycles, "start": seed, "history": records, "final": control,
        "selected_minus_control": {
            "manufacturing_cost_index_fraction": (
                selected["build_cost_proxy"] / control["build_cost_proxy"] - 1),
            "rms_p90_fraction": selected["rms_p90_mm"] / control["rms_p90_mm"] - 1,
            "nominal_rms_fraction": selected["nominal_rms_mm"] / control["nominal_rms_mm"] - 1,
        }}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    labels = ["Fixed-material\ncontrol", "Material-search\nselected"]
    axes[0].bar(labels, [1.0, selected["build_cost_proxy"] / control["build_cost_proxy"]],
                color=["#64748b", "#0d9488"])
    axes[0].set(title="Relative manufacturing cost index", ylabel="Control = 1.0")
    axes[1].bar(labels, [control["rms_p90_mm"], selected["rms_p90_mm"]],
                color=["#64748b", "#0d9488"])
    axes[1].set(title="RMS p90 after equal optimization budget", ylabel="mm")
    for ax in axes:
        ax.grid(axis="y", alpha=.25)
    fig.suptitle("Matched optimization-budget ablation")
    fig.tight_layout()
    fig.savefig(output / "matched_budget_ablation.png", dpi=180)
    plt.close(fig)
    return control


def proposal_groups(proposals, search):
    """Enumerate unseen single/multiple replacements before applying run budgets."""
    pool = proposals[:int(search["proposal_pool_size"])]
    groups = []
    for size in range(1, int(search.get("max_changes_per_trial", 1)) + 1):
        for group in itertools.combinations(pool, size):
            if len({item["surface_idx"] for item in group}) == size:
                groups.append(group)
    # Prefer optically nearby moves, then combinations with larger estimated savings.
    groups.sort(key=lambda group: (sum(item["max_abs_delta_n"] for item in group),
                                   -sum(item["estimated_blank_saving_proxy"] for item in group),
                                   tuple((item["surface_idx"], item["to"]) for item in group)))
    return groups


def feasibility_reasons(candidate, reference, settings, rms_limit, material_constraints=None):
    reasons = []
    if candidate["nominal_rms_mm"] > rms_limit:
        reasons.append("nominal RMS above limit")
    if candidate["demo_yield"] < float(settings["min_demo_yield"]):
        reasons.append("demo yield below limit")
    if candidate["good_unit_cost_proxy"] is None:
        reasons.append("good-unit cost undefined because yield is zero")
    if candidate["build_cost_proxy"] > reference["build_cost_proxy"] * float(
            settings.get("max_build_cost_ratio_to_reference", 1.0)):
        reasons.append("build cost exceeds exploration ceiling")
    if material_constraints:
        materials = dict(candidate["signature"])
        for index_text, pattern in material_constraints.items():
            index = int(index_text)
            material = materials.get(index)
            if material is None or not re.fullmatch(pattern, material):
                reasons.append(f"surface {index} material violates required process route")
    return reasons


def search_score(candidate, reference, weights):
    """Lower is better; used to keep a diverse finite beam, not as product qualification."""
    good = candidate["good_unit_cost_proxy"]
    if good is None:
        return float("inf")
    score = (float(weights["good_unit_cost"]) * good / reference["good_unit_cost_proxy"]
             + float(weights["rms_p90"]) * candidate["rms_p90_mm"] / reference["rms_p90_mm"]
             + float(weights["yield_shortfall"]) * (1.0 - candidate["demo_yield"]))
    mtf_weight = float(weights.get("mtf_quality", 0.0))
    candidate_mtf = candidate.get("differentiable_mtf_worst")
    reference_mtf = reference.get("differentiable_mtf_worst")
    if mtf_weight > 0 and candidate_mtf is not None and reference_mtf is not None:
        # Higher MTF is better, so its reciprocal is a minimization term.
        score += mtf_weight * max(float(reference_mtf), 1e-6) / max(float(candidate_mtf), 1e-6)
    return score


def node_from_summary(row, lens_path, reference_json, parent_id, round_number, stage):
    lens_json = json.loads(Path(lens_path).read_text(encoding="utf-8"))
    signature = material_signature(lens_json)
    row.update({"node_id": signature_id(signature), "signature": list(signature),
                "parent_id": parent_id, "round_created": round_number, "stage": stage,
                "material_changes_from_reference": material_changes(lens_json, reference_json)})
    return row


def select_beam(old_beam, new_nodes, width, exploration_slots):
    """Keep the best known state plus new exploration branches and score-ranked survivors."""
    unique = {node["node_id"]: node for node in old_beam + new_nodes}
    combined = sorted(unique.values(), key=lambda node: node["search_score"])
    selected = []
    if combined:
        selected.append(combined[0])
    for node in sorted(new_nodes, key=lambda item: item["search_score"]):
        if node["node_id"] not in {item["node_id"] for item in selected}:
            selected.append(node)
        if len(selected) >= min(width, 1 + exploration_slots):
            break
    for node in combined:
        if node["node_id"] not in {item["node_id"] for item in selected}:
            selected.append(node)
        if len(selected) >= width:
            break
    return selected[:width]


def save_plots(history, output, settings=None):
    """Save the three convergence panels used to interpret a search run.

    The third panel uses the differentiable MTF configured for the run.  The
    paper experiment uses ``field_fractions: [1.0]`` and 93 lp/mm, so the
    historically named ``differentiable_mtf_worst`` field is min(T, S) at the
    maximum field.  Older configs remain readable for reproducibility.
    """
    settings = settings or {}
    rounds = [row["round"] for row in history]
    best = [row["best"] for row in history]
    original = next((node for node in history[0]["beam"]
                     if node.get("name") == "original_optimized"), history[0]["beam"][-1])
    pre_recovery = history[0]["best"].get("pre_recovery")
    plotted_rounds = ([-1] + rounds) if pre_recovery is not None else rounds
    plotted_best = ([pre_recovery] + best) if pre_recovery is not None else best
    search_start = pre_recovery if pre_recovery is not None else original
    reference_is_original = original.get("name") == "original_optimized"
    reference_label = "original optimized reference" if reference_is_original else "search start"

    # “综合最优”和“最低成本”是两个不同目标。逐轮分别画出，避免把已经完成
    # 降本的种子当成原始处方，使读者误以为搜索从未降低成本。
    lowest_cost_history = []
    lowest = None
    for row in history:
        candidates = list(row.get("beam", []))
        candidates += [node for node in row.get("trials", [])
                       if not node.get("feasibility_reasons")]
        for node in candidates:
            value = node.get("good_unit_cost_proxy")
            if value is not None and (lowest is None or value < lowest["good_unit_cost_proxy"]):
                lowest = node
        lowest_cost_history.append(lowest)

    mtf_settings = settings.get("differentiable_mtf", {})
    mtf_frequency = float(mtf_settings.get("frequency_lp_per_mm", 50.0))
    field_fractions = list(mtf_settings.get("field_fractions", [0.0]))
    central_only = field_fractions == [0.0]
    maximum_only = field_fractions == [1.0]
    if maximum_only:
        mtf_title = f"Maximum-field min(T,S) differentiable MTF at {mtf_frequency:g} lp/mm"
    elif central_only:
        mtf_title = f"Central-field min(T,S) differentiable MTF at {mtf_frequency:g} lp/mm"
    else:
        mtf_title = f"Worst sampled-field min(T,S) differentiable MTF at {mtf_frequency:g} lp/mm"

    # Save the plotted values in a spreadsheet-friendly file.  This is updated
    # after every completed round, so an overnight run can be inspected without
    # reopening Python or parsing the larger hammer_report.json file.
    csv_path = output / "convergence_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "round", "stage", "normalized_manufacturing_cost",
            "manufacturing_cost_index", "rms_p90_mm",
            "evaluation_mtf_min_ts", "selected_lens_json"])
        writer.writeheader()
        for round_value, node in zip(plotted_rounds, plotted_best):
            writer.writerow({
                "round": round_value,
                "stage": "pre_recovery" if round_value == -1 else "selected",
                "normalized_manufacturing_cost": node["build_cost_proxy"] / search_start["build_cost_proxy"],
                "manufacturing_cost_index": node["build_cost_proxy"],
                "rms_p90_mm": node["rms_p90_mm"],
                "evaluation_mtf_min_ts": node.get("differentiable_mtf_worst"),
                "selected_lens_json": node.get("lens_json"),
            })

    metrics = [("build_cost_proxy", "Normalized manufacturing cost", True),
               ("rms_p90_mm", "Selected design RMS p90 (mm)", False)]
    # Release adaptation: RMS-only examples have no MTF value to plot.
    if all(node.get("differentiable_mtf_worst") is not None for node in plotted_best):
        metrics.append(("differentiable_mtf_worst", mtf_title, False))
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.6))
    for ax, (key, title, normalize) in zip(axes, metrics):
        scale = search_start[key] if normalize else 1.0
        ax.plot(plotted_rounds, [node[key] / scale for node in plotted_best], "o-", color="#0d9488",
                label="composite-score selection")
        for row in history:
            values = [node[key] / scale for node in row["beam"] if node.get(key) is not None]
            if values:
                ax.scatter([row["round"]] * len(values), values, s=24, alpha=.45,
                           color="#2563eb", label="active beam" if row["round"] == 0 else None)
        start_round = -1 if pre_recovery is not None else 0
        ax.scatter([start_round], [search_start[key] / scale], marker="X", s=90,
                   color="#dc2626", zorder=5,
                   label=reference_label)
        if key in {"good_unit_cost_proxy", "build_cost_proxy"}:
            cost_nodes = ([pre_recovery] + lowest_cost_history
                          if pre_recovery is not None else lowest_cost_history)
            ax.plot(plotted_rounds, [node[key] / scale for node in cost_nodes], "--", color="#d97706",
                    linewidth=2, label="lowest-cost feasible so far")
        if key == "differentiable_mtf_worst":
            # The composite-score solution may trade a small amount of MTF for a
            # larger cost/RMS gain.  The dashed envelope shows the best MTF that
            # has appeared among the retained composite selections without
            # hiding those physically meaningful trade-offs.
            running_maximum = []
            value = float("-inf")
            for node in plotted_best:
                current = node.get(key)
                value = max(value, float(current)) if current is not None else value
                running_maximum.append(value)
            ax.plot(plotted_rounds, running_maximum, "--", color="#d97706", linewidth=1.8,
                    label="running maximum of selected designs")
        ylabel = (f"Relative value ({'original optimized' if reference_is_original else 'search start'} = 1.0)"
                  if normalize else None)
        ax.set(title=title, xlabel="Beam-search round", xticks=plotted_rounds, ylabel=ylabel)
        if pre_recovery is not None:
            ax.set_xticklabels(["pre\nrecovery"] + [str(value) for value in rounds])
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "hammer_convergence.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    for row in history:
        for node in row.get("new_feasible", []):
            ax.scatter(node["build_cost_proxy"], node["rms_p90_mm"], c=row["round"],
                       cmap="viridis", vmin=0, vmax=max(rounds), s=50, alpha=.7)
    ax.scatter(search_start["build_cost_proxy"], search_start["rms_p90_mm"], marker="X",
               s=130, color="#7c3aed", label=reference_label)
    ax.scatter(lowest_cost_history[-1]["build_cost_proxy"], lowest_cost_history[-1]["rms_p90_mm"],
               marker="D", s=110, color="#d97706", label="lowest-cost feasible")
    ax.scatter(best[-1]["build_cost_proxy"], best[-1]["rms_p90_mm"], marker="*",
               s=180, color="#dc2626", label="final composite-score selection")
    ax.set(xlabel="Manufacturing cost index (relative units)", ylabel="RMS p90 (mm)",
           title="Hammer-style explored feasible states")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "hammer_search_cloud.png", dpi=180)
    plt.close(fig)


def write_report(output, settings, history, best, lowest_cost, seen_count, stopped_reason):
    report = {"method": "Hammer-inspired real-glass beam search; not Zemax Hammer",
              "settings": settings["search"], "history": history,
              "unique_material_states_evaluated": seen_count,
              "stopped_reason": stopped_reason,
              "final_lens_json": best["lens_json"], "final_best": best,
              "lowest_cost_feasible": lowest_cost,
              "warning": ("MTF-assisted RMS/yield research proxy and relative cost scenario; "
                          "not production qualification")}
    (output / "hammer_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_plots(history, output, settings)


def save_independent_mtf_comparison(output, settings, lenses):
    """Evaluate selected designs with the slower independent PSF MTF diagnostic."""
    if not settings or not settings.get("enabled", False):
        return None
    mtf_settings = yaml.safe_load(Path(settings["config"]).read_text(encoding="utf-8"))
    mtf_settings["device"] = settings.get("device", "cuda")
    if settings.get("frequency_lp_per_mm") is not None:
        mtf_settings["frequency_lp_per_mm"] = float(settings["frequency_lp_per_mm"])
    result = {}
    cache_root = output / "independent_mtf"
    cache_root.mkdir(parents=True, exist_ok=True)
    for name, lens_path in lenses.items():
        lens_cache = cache_root / name
        lens_cache.mkdir(parents=True, exist_ok=True)
        result[name] = workflow.evaluate_mtf(lens_path, lens_cache, mtf_settings)
    review.make_plot(result, mtf_settings, cache_root)
    worst = {name: expert.worst_mtf(value) for name, value in result.items()}
    maximum = {name: expert.maximum_field_mtf(value) for name, value in result.items()}
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    names = list(maximum)
    values = [maximum[name] for name in names]
    palette = ["#64748b", "#7c3aed", "#0d9488", "#d97706", "#dc2626"]
    bars = ax.bar(names, values, color=[palette[index % len(palette)]
                                       for index in range(len(names))])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}",
                ha="center", va="bottom", fontsize=9)
    upper = max(values) * 1.25 if max(values) > 0 else 1.0
    ax.set(title=f'Maximum-field min(T,S) MTF at {mtf_settings["frequency_lp_per_mm"]:g} lp/mm',
           ylabel="Smaller of tangential and sagittal MTF", ylim=(0, upper))
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(cache_root / f'maximum_field_{mtf_settings["frequency_lp_per_mm"]:g}lpmm_mtf.png', dpi=180)
    plt.close(fig)
    report_path = output / "hammer_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["independent_mtf_comparison"] = {
        "frequency_lp_per_mm": mtf_settings["frequency_lp_per_mm"],
        "metric": "independent geometric polychromatic PSF MTF proxy",
        "primary_comparison": "maximum field, smaller of tangential and sagittal MTF",
        "maximum_field_min_ts_mtf": maximum,
        "full_field_worst_mtf_auxiliary": worst,
        "results": result,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return maximum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/20_hammer_beam_search.yml")
    parser.add_argument("--replot-only", action="store_true",
                        help="rebuild convergence plots/CSV from an existing hammer_report.json")
    args = parser.parse_args()
    settings = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output = Path(settings["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if args.replot_only:
        report_path = output / "hammer_report.json"
        if not report_path.exists():
            raise FileNotFoundError(f"No saved search report: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        save_plots(report["history"], output, settings)
        print(f"Updated: {output / 'hammer_convergence.png'}")
        print(f"Updated: {output / 'convergence_metrics.csv'}")
        return
    search = settings["search"]
    stage_policy = str(search.get("candidate_stage_policy", "best_stage"))
    if stage_policy not in {"best_stage", "final_stage"}:
        raise ValueError("search.candidate_stage_policy must be best_stage or final_stage")
    max_unique_states = search.get("max_unique_material_states")
    if max_unique_states is not None and int(max_unique_states) < 1:
        raise ValueError("search.max_unique_material_states must be positive")
    device = str(settings.get("device", "auto"))
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but torch {torch.__version__} cannot use CUDA")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hammer-style search device: {device}; torch={torch.__version__}", flush=True)

    base = yaml.safe_load(Path(settings["base_config"]).read_text(encoding="utf-8"))
    # A search-level seed makes repeated strategy comparisons configuration
    # driven.  Paired Beam/greedy runs can therefore share exactly the same
    # tolerance draws while different pairs remain statistically independent.
    if settings.get("seed") is not None:
        base["seed"] = int(settings["seed"])
    base["device"] = device
    base["resume"] = bool(search.get("resume", True))
    base["process_cost"] = copy.deepcopy(settings["process_cost"])
    base["training"].update(settings.get("training_overrides", {}))
    base["sampling"].update(settings.get("sampling_overrides", {}))
    if settings.get("differentiable_mtf"):
        base["differentiable_mtf"] = copy.deepcopy(settings["differentiable_mtf"])
    base["evaluation"]["trials"] = int(search["evaluation_trials"])
    rms_limit = float(settings["acceptance_max_rgb_rms_mm"])
    base["acceptance"]["max_rgb_rms_mm"] = rms_limit
    base["training"]["nominal_max_rms_mm"] = rms_limit
    prices = expert.prices_from_csv(base["glass_catalog"])

    reference_path = Path(settings["reference_lens_json"])
    reference_json = json.loads(reference_path.read_text(encoding="utf-8"))
    seeds = []
    seen = set()
    configured_starting_lens = None
    for index, item in enumerate(settings["seed_lenses"]):
        seed_folder = output / "seeds" / f'{index + 1}_{item["name"]}'
        config = copy.deepcopy(base)
        seed_input = Path(item["path"])
        forced_changes = []
        geometry_changes = []
        if settings.get("seed_material_overrides"):
            source_json = json.loads(seed_input.read_text(encoding="utf-8"))
            source_json, forced_changes = apply_material_overrides(
                source_json, settings["seed_material_overrides"], prices)
            if settings.get("seed_geometry_perturbations"):
                source_json, geometry_changes = apply_geometry_perturbations(
                    source_json, settings["seed_geometry_perturbations"])
            seed_input = seed_folder / "forced_starting_lens.json"
            seed_input.parent.mkdir(parents=True, exist_ok=True)
            seed_input.write_text(json.dumps(source_json, indent=2), encoding="utf-8")
            if configured_starting_lens is None:
                configured_starting_lens = seed_input
        config["input_lens"] = str(seed_input)
        config["training"]["steps"] = int(search.get("seed_recovery_steps", 0))
        config["training"]["objective"] = "nominal_only"
        config["training"].update(copy.deepcopy(settings.get("seed_training_overrides", {})))
        if config.get("differentiable_mtf") and search.get("seed_mtf_weight") is not None:
            config["differentiable_mtf"]["weight"] = float(search["seed_mtf_weight"])
        lens_path, report = workflow.run_design(config, seed_folder)
        row = workflow.summary(item["name"], lens_path, report, None, settings["process_cost"])
        node = node_from_summary(row, lens_path, reference_json, None, 0, "seed")
        if "baseline" in report:
            baseline_report = copy.deepcopy(report)
            baseline_report["optimized"] = report["baseline"]
            node["pre_recovery"] = workflow.summary(
                "pre_recovery_search_start", seed_input, baseline_report, None,
                settings["process_cost"])
        node["forced_seed_material_changes"] = forced_changes
        node["forced_seed_geometry_changes"] = geometry_changes
        seeds.append(node)
        seen.add(tuple(tuple(value) for value in node["signature"]))
    reference = seeds[0]
    for node in seeds:
        node["feasibility_reasons"] = feasibility_reasons(
            node, reference, settings["screening"], rms_limit,
            settings.get("required_material_regex_by_surface"))
        node["search_score"] = search_score(node, reference, settings["score_weights"])
    feasible_seeds = [node for node in seeds if not node["feasibility_reasons"]]
    if not feasible_seeds:
        raise RuntimeError("No seed lens passes the configured RMS/yield constraints")
    beam = sorted(feasible_seeds, key=lambda node: node["search_score"])[:int(search["beam_width"])]
    selection_metric = search.get("global_best_metric", "composite_score")
    if selection_metric not in {"composite_score", "good_unit_cost"}:
        raise ValueError("search.global_best_metric must be composite_score or good_unit_cost")
    best_key = ((lambda node: node["search_score"]) if selection_metric == "composite_score"
                else (lambda node: node["good_unit_cost_proxy"]))
    best = min(feasible_seeds, key=best_key)
    lowest_cost = min(feasible_seeds, key=lambda node: node["good_unit_cost_proxy"])
    history = [{"round": 0, "beam": beam, "best": best, "new_feasible": [], "trials": []}]
    best_value = best_key(best)
    stale_rounds = 0
    stopped_reason = "maximum rounds reached"
    write_report(output, settings, history, best, lowest_cost, len(seen), "search running")

    for round_number in range(1, int(search["max_rounds"]) + 1):
        round_folder = output / f"round_{round_number}"
        round_folder.mkdir(exist_ok=True)
        trial_records, new_nodes = [], []
        global_runs = 0
        for parent in beam:
            if global_runs >= int(search["max_trials_per_round"]):
                break
            if max_unique_states is not None and len(seen) >= int(max_unique_states):
                break
            parent_json = json.loads(Path(parent["lens_json"]).read_text(encoding="utf-8"))
            balanced_proposals, all_proposals = expert.propose_real_glasses(
                parent_json, prices, settings["proposal"])
            proposal_source = (balanced_proposals if search.get("use_balanced_proposals", False)
                               else all_proposals)
            groups = proposal_groups(proposal_source, search)
            parent_runs = 0
            for group in groups:
                if parent_runs >= int(search["max_trials_per_parent"]) or global_runs >= int(
                        search["max_trials_per_round"]):
                    break
                if max_unique_states is not None and len(seen) >= int(max_unique_states):
                    break
                try:
                    candidate_json, changes = glass.make_lens(parent_json,
                        [{key: change[key] for key in ("surface_idx", "from", "to")} for change in group],
                        prices, settings["proposal"]["wavelengths_nm"],
                        float(settings["proposal"]["max_abs_delta_n"]))
                except ValueError:
                    continue
                signature = material_signature(candidate_json)
                if signature in seen:
                    continue
                seen.add(signature)
                parent_runs += 1
                global_runs += 1
                label = "__".join(f's{x["surface_idx"]}_{x["to"].replace("-", "_")}' for x in group)
                folder = round_folder / f'p_{parent["node_id"]}' / f'trial_{global_runs}_{label}'
                folder.mkdir(parents=True, exist_ok=True)
                starting = folder / "starting_lens.json"
                starting.write_text(json.dumps(candidate_json, indent=2), encoding="utf-8")
                stage_nodes = []
                nominal_config = copy.deepcopy(base)
                nominal_config["input_lens"] = str(starting)
                nominal_config["training"]["steps"] = int(search["nominal_steps_per_trial"])
                nominal_config["training"]["objective"] = "nominal_only"
                nominal_lens, nominal_report = workflow.run_design(nominal_config, folder / "nominal_recovery")
                nominal_row = workflow.summary(label + "_nominal", nominal_lens, nominal_report,
                                                None, settings["process_cost"])
                nominal = node_from_summary(nominal_row, nominal_lens, reference_json,
                                            parent["node_id"], round_number, "nominal")
                stage_nodes.append(nominal)
                if int(search["robust_steps_per_trial"]) > 0:
                    robust_config = copy.deepcopy(base)
                    robust_config["input_lens"] = str(nominal_lens)
                    robust_config["training"]["steps"] = int(search["robust_steps_per_trial"])
                    # Configurable for the soft-yield ablation. Existing configs
                    # retain the complete expected-good-cost objective by default.
                    robust_config["training"]["objective"] = search.get(
                        "robust_objective", "expected_good_cost")
                    robust_lens, robust_report = workflow.run_design(robust_config, folder / "robust_refinement")
                    robust_row = workflow.summary(label + "_robust", robust_lens, robust_report,
                                                  None, settings["process_cost"])
                    robust = node_from_summary(robust_row, robust_lens, reference_json,
                                               parent["node_id"], round_number, "robust")
                    stage_nodes.append(robust)
                for node in stage_nodes:
                    node["last_substitution"] = changes
                    node["feasibility_reasons"] = feasibility_reasons(
                        node, reference, settings["screening"], rms_limit,
                        settings.get("required_material_regex_by_surface"))
                    node["search_score"] = search_score(node, reference, settings["score_weights"])
                    trial_records.append(node)
                # For formal ablations, nominal recovery is an intermediate state and
                # only the final refinement stage may continue the material path.
                # Older experiments keep their historical best-stage behaviour.
                selectable_stages = ([stage_nodes[-1]] if stage_policy == "final_stage"
                                     else stage_nodes)
                feasible_stages = [node for node in selectable_stages
                                   if not node["feasibility_reasons"]]
                if feasible_stages:
                    new_nodes.append(min(feasible_stages, key=lambda node: node["search_score"]))
                (round_folder / "progress.json").write_text(json.dumps(trial_records, indent=2), encoding="utf-8")
                print(f"Round {round_number}: parent={parent['node_id']} {label}; "
                      f"feasible={len(feasible_stages)}/{len(stage_nodes)}", flush=True)

        beam = select_beam(beam, new_nodes, int(search["beam_width"]),
                           int(search.get("exploration_slots", 1)))
        feasible_records = [node for node in trial_records if not node["feasibility_reasons"]]
        if feasible_records:
            record_lowest = min(feasible_records, key=lambda node: node["good_unit_cost_proxy"])
            if record_lowest["good_unit_cost_proxy"] < lowest_cost["good_unit_cost_proxy"]:
                lowest_cost = record_lowest
        feasible_all = [node for node in beam + new_nodes if not node["feasibility_reasons"]]
        if feasible_all:
            round_lowest_cost = min(feasible_all, key=lambda node: node["good_unit_cost_proxy"])
            if round_lowest_cost["good_unit_cost_proxy"] < lowest_cost["good_unit_cost_proxy"]:
                lowest_cost = round_lowest_cost
            round_best = min(feasible_all, key=best_key)
            min_gain = float(search.get("global_improvement_fraction", 0.0001))
            round_value = best_key(round_best)
            if round_value < best_value * (1 - min_gain):
                best, best_value, stale_rounds = round_best, round_value, 0
            else:
                stale_rounds += 1
        else:
            stale_rounds += 1
        history.append({"round": round_number, "beam": beam, "best": best,
                        "new_feasible": new_nodes, "trials": trial_records,
                        "unique_states_seen": len(seen), "stale_rounds": stale_rounds})
        write_report(output, settings, history, best, lowest_cost, len(seen), "search running")
        if max_unique_states is not None and len(seen) >= int(max_unique_states):
            stopped_reason = f"maximum unique material states reached ({int(max_unique_states)})"
            break
        if global_runs == 0:
            stopped_reason = "no unseen material states remain for the active beam"
            break
        if stale_rounds >= int(search["patience_rounds"]):
            stopped_reason = f"no new global best for {stale_rounds} consecutive rounds"
            break

    write_report(output, settings, history, best, lowest_cost, len(seen), stopped_reason)
    control_settings = settings.get("matched_fixed_material_control", {})
    fixed_control = None
    if control_settings.get("enabled", False):
        fixed_control = run_matched_fixed_material_control(
            seeds[0], best, base, search, output, reference_json, settings["process_cost"])
    layout = settings.get("layout_comparison", {})
    if layout.get("enabled", False):
        try:
            initial_layout_lens = layout.get("initial_lens", settings["reference_lens_json"])
            if initial_layout_lens == "search_start":
                initial_layout_lens = configured_starting_lens or settings["reference_lens_json"]
            save_layout_comparison(initial_layout_lens,
                                   best["lens_json"], output, layout)
        except Exception as error:
            (output / "layout_comparison_error.txt").write_text(str(error), encoding="utf-8")
            print(f"Warning: 2D layout comparison failed: {error}", flush=True)
    mtf = settings.get("independent_mtf")
    if mtf and mtf.get("enabled", False):
        try:
            comparison_lenses = {
                "search start": str(configured_starting_lens or seeds[0]["lens_json"]),
                "selected": best["lens_json"],
                "lowest cost": lowest_cost["lens_json"],
            }
            if fixed_control is not None:
                comparison_lenses = {
                    "search start": comparison_lenses["search start"],
                    "fixed-material control": fixed_control["lens_json"],
                    "selected": comparison_lenses["selected"],
                    "lowest cost": comparison_lenses["lowest cost"],
                }
            worst = save_independent_mtf_comparison(output, mtf, comparison_lenses)
            print(f"Independent MTF comparison: {worst}", flush=True)
        except Exception as error:
            (output / "independent_mtf_error.txt").write_text(str(error), encoding="utf-8")
            print(f"Warning: independent MTF comparison failed: {error}", flush=True)
    print(f"Stopped: {stopped_reason}")
    print(f"Final selected lens ({selection_metric}): {best['lens_json']}")
    print(f"Lowest-cost feasible lens: {lowest_cost['lens_json']}")


if __name__ == "__main__":
    main()
