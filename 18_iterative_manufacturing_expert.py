"""多轮加工专家：真实玻璃邻域搜索 → 可微精调 → 独立公差与 MTF 验收。"""

import argparse
import copy
import hashlib
import importlib
import itertools
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

expert = importlib.import_module("17_manufacturing_expert")
glass = importlib.import_module("15_glass_substitution")
review = importlib.import_module("10_review_mtf")


def run_design(config, folder):
    """每次候选使用单独目录，保存完整配置、优化轨迹和独立误差样本。"""
    folder.mkdir(parents=True, exist_ok=True)
    config["output_dir"] = str(folder)
    config["visualization"] = {"lens_layout": False, "summary_plots": False}
    # 路径相同但输入处方内容改变时不能复用旧结果。
    config["_input_sha256"] = hashlib.sha256(Path(config["input_lens"]).read_bytes()).hexdigest()
    path = folder / "run_config.yml"
    rendered = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
    lens_path, report_path = folder / "manufacturing_lens.json", folder / "report.json"
    if (config.get("resume", True) and path.exists() and
            path.read_text(encoding="utf-8") == rendered and lens_path.exists() and report_path.exists()):
        print(f"Reusing completed stage: {folder}", flush=True)
        return lens_path, json.loads(report_path.read_text(encoding="utf-8"))
    path.write_text(rendered, encoding="utf-8")
    subprocess.run([sys.executable, "8_manufacturing_design.py", "--config", str(path)], check=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return lens_path, report


def summary(name, lens_path, report, mtf, process_rules):
    result = report["optimized"]
    return {"name": name, "lens_json": str(lens_path),
            "nominal_rms_mm": result["nominal_rms_mm"],
            "rms_p90_mm": result["rms_p90_mm"],
            "demo_yield": result["yield_estimate"],
            "yield_wilson95": result["yield_wilson95"],
            "blank_cost_proxy": result["cost_proxy_hk9l_mm3"],
            "process_cost_proxy": result.get("process_cost_proxy", 0.0),
            "build_cost_proxy": result.get("total_build_cost_proxy", result["cost_proxy_hk9l_mm3"]),
            "good_unit_cost_proxy": result["expected_good_unit_cost_proxy"],
            "differentiable_mtf_worst": (min(min(row) for row in result["fourier_mtf_proxy"])
                                         if "fourier_mtf_proxy" in result else None),
            "independent_mtf_worst": expert.worst_mtf(mtf) if mtf is not None else None,
            "process_scenario_volume": process_rules["production_volume"]}


def passes(candidate, current, settings, rms_limit):
    """验收只使用独立的硬公差样本和独立 PSF 诊断，不看训练损失。"""
    reasons = []
    if candidate["nominal_rms_mm"] > rms_limit:
        reasons.append("nominal RMS above limit")
    if candidate["demo_yield"] < settings["min_demo_yield"]:
        reasons.append("demo yield below limit")
    if settings.get("metric", "mtf") == "mtf":
        if candidate["independent_mtf_worst"] < current["independent_mtf_worst"] * settings["min_relative_mtf"]:
            reasons.append("independent MTF loss")
    if candidate["good_unit_cost_proxy"] is None:
        reasons.append("candidate good-unit cost undefined because yield is zero")
    elif (current["good_unit_cost_proxy"] is not None and
          candidate["good_unit_cost_proxy"] >= current["good_unit_cost_proxy"] *
          (1 - settings["min_cost_improvement_fraction"])):
        reasons.append("no sufficient good-unit cost reduction")
    return reasons


def candidate_combinations(proposals, search):
    """同时尝试单片和多片替换；轮与轮之间还会继续累计已接受的替换。"""
    limit = int(search["max_trials_per_round"])
    max_changes = int(search.get("max_changes_per_trial", 1))
    if limit < 1 or max_changes < 1:
        raise ValueError("Search trial and material-change limits must be positive")
    pool = proposals[:int(search.get("proposal_pool_size", len(proposals)))]
    singles = [(item,) for item in pool]
    multiple = []
    for size in range(2, max_changes + 1):
        for group in itertools.combinations(pool, size):
            if len({item["surface_idx"] for item in group}) != size:
                continue
            multiple.append(group)
    multiple.sort(key=lambda group: -sum(item["estimated_blank_saving_proxy"] for item in group))
    return (singles + multiple)[:limit]


def evaluate_mtf(lens_path, folder, settings):
    """独立 MTF 可选，并可在断点续跑时复用。"""
    if settings is None:
        return None
    cache = folder / "independent_mtf.json"
    fingerprint = hashlib.sha256(Path(lens_path).read_bytes() +
        json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()
    if cache.exists():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        if saved.get("fingerprint") == fingerprint:
            return saved["result"]
    result = review.evaluate_lens(lens_path, settings)
    cache.write_text(json.dumps({"fingerprint": fingerprint, "result": result}, indent=2), encoding="utf-8")
    return result


def plot_history(history, output, metric="mtf", frequency=50.0):
    rows = [history[0]["selected"]] + [round_["selected"] for round_ in history[1:]]
    x = list(range(len(rows)))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    metrics = [("build_cost_proxy", "Relative manufacturing cost index"),
               ("good_unit_cost_proxy", "Expected accepted-unit cost index"),
               (("independent_mtf_worst", f"Independent {frequency:g} lp/mm geometric MTF")
                if metric == "mtf" else ("rms_p90_mm", "RMS tolerance samples: 90th percentile (mm)")),
               ("demo_yield", "RMS sample pass fraction")]
    for ax, (key, label) in zip(axes.flat, metrics):
        ax.plot(x, [row[key] for row in rows], "o-", color="#0d9488")
        ax.set(title=label, xlabel="Search round (best retained design)", xticks=x)
        if key == "demo_yield":
            ax.set_ylim(0, 1.05)
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output / "iteration_progress.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 6))
    y_key = "independent_mtf_worst" if metric == "mtf" else "rms_p90_mm"
    for round_ in history[1:]:
        for candidate in round_["candidates"]:
            accepted = not candidate["rejection_reasons"]
            stage = candidate["name"].rsplit("_", 1)[-1]
            marker = {"substitution": "o", "nominal": "s", "robust": "^"}.get(stage, "o")
            ax.scatter(candidate["build_cost_proxy"], candidate[y_key],
                       c="#0d9488" if accepted else "#d97706", s=75, marker=marker,
                       edgecolors="black", linewidths=.4)
            if candidate["lens_json"] == round_["selected"]["lens_json"]:
                ax.annotate(f'Round {round_["round"]} selected: {candidate["name"]}',
                            (candidate["build_cost_proxy"], candidate[y_key]),
                            xytext=(7, 7), textcoords="offset points", fontsize=8)
    ax.scatter(rows[0]["build_cost_proxy"], rows[0][y_key],
               c="#2563eb", s=100, marker="*", label="Original optimized")
    ylabel = f"Independent {frequency:g} lp/mm geometric MTF" if metric == "mtf" else "RMS p90 (mm; lower is better)"
    ax.set(xlabel="Relative manufacturing cost index", ylabel=ylabel,
           title="Explored material and refinement candidates")
    from matplotlib.lines import Line2D
    legend = [Line2D([0], [0], marker="*", color="w", markerfacecolor="#2563eb", markersize=12,
                     label="Original optimized"),
              Line2D([0], [0], marker="o", color="w", markerfacecolor="#0d9488", markersize=9,
                     label="Accepted candidate"),
              Line2D([0], [0], marker="o", color="w", markerfacecolor="#d97706", markersize=9,
                     label="Rejected candidate"),
              Line2D([0], [0], marker="s", color="#475569", linestyle="None", label="Nominal recovery"),
              Line2D([0], [0], marker="^", color="#475569", linestyle="None", label="Robust refinement")]
    ax.legend(handles=legend, fontsize=8, loc="lower right")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output / "candidate_tradeoff.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/18_iterative_manufacturing_expert.yml")
    args = parser.parse_args()
    settings = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output = Path(settings["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    requested_device = str(settings.get("device", "auto")).lower()
    if requested_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested, but torch {torch.__version__} has no usable CUDA. "
                           "Install a CUDA PyTorch build before the overnight run.")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if requested_device == "auto" else requested_device
    print(f"Search device: {device}; torch={torch.__version__}; CUDA build={torch.version.cuda}", flush=True)
    metric = settings["screening"].get("metric", "mtf")
    if metric not in {"mtf", "rms"}:
        raise ValueError("screening.metric must be mtf or rms")
    base = yaml.safe_load(Path(settings["base_config"]).read_text(encoding="utf-8"))
    base["device"] = device
    base["resume"] = bool(settings["search"].get("resume", True))
    base["training"].update(settings.get("training_overrides", {}))
    base["sampling"].update(settings.get("sampling_overrides", {}))
    if "acceptance_max_rgb_rms_mm" in settings:
        demo_limit = float(settings["acceptance_max_rgb_rms_mm"])
        base["acceptance"]["max_rgb_rms_mm"] = demo_limit
        base["training"]["nominal_max_rms_mm"] = demo_limit
    if "evaluation_trials" in settings["search"]:
        base["evaluation"]["trials"] = int(settings["search"]["evaluation_trials"])
    base["process_cost"] = copy.deepcopy(settings["process_cost"])
    mtf_settings = None
    if metric == "mtf":
        base["differentiable_mtf"] = copy.deepcopy(settings["differentiable_mtf"])
        mtf_settings = yaml.safe_load(Path(settings["independent_mtf_config"]).read_text(encoding="utf-8"))
        mtf_settings["frequency_lp_per_mm"] = float(settings["differentiable_mtf"]["frequency_lp_per_mm"])
        mtf_settings["device"] = device
    prices = expert.prices_from_csv(base["glass_catalog"])
    root = Path(settings["autolens_output_json"])
    base["input_lens"] = str(root)
    # 先建立标称合格基线；材料替换后同样先恢复标称像质，再尝试制造精调。
    base["training"]["steps"] = int(settings["search"].get("baseline_steps", 0))
    base["training"]["objective"] = "nominal_only"
    if metric == "mtf":
        base["differentiable_mtf"]["weight"] = 0.0
    base_lens, base_report = run_design(copy.deepcopy(base), output / "round_0_original")
    current = summary("original", base_lens, base_report,
                      evaluate_mtf(base_lens, output / "round_0_original", mtf_settings), settings["process_cost"])
    history = [{"round": 0, "selected": current, "candidates": []}]
    rms_limit = float(base["acceptance"]["max_rgb_rms_mm"])
    proposal = settings["proposal"]
    for round_number in range(1, int(settings["search"]["max_rounds"]) + 1):
        source_json = json.loads(Path(current["lens_json"]).read_text(encoding="utf-8"))
        candidate_pool, all_proposals = expert.propose_real_glasses(source_json, prices, proposal)
        candidates = candidate_combinations(candidate_pool, settings["search"])
        round_folder = output / f"round_{round_number}"
        round_folder.mkdir(exist_ok=True)
        (round_folder / "proposals.json").write_text(json.dumps(all_proposals, indent=2), encoding="utf-8")
        tried = []
        for index, group in enumerate(candidates):
            name = "__".join(f's{change["surface_idx"]}_{change["to"].replace("-", "_")}' for change in group)
            folder = round_folder / f"trial_{index + 1}_{name}"
            folder.mkdir(exist_ok=True)
            candidate_json, changes = glass.make_lens(source_json,
                [{key: change[key] for key in ("surface_idx", "from", "to")} for change in group], prices,
                proposal["wavelengths_nm"], float(proposal["max_abs_delta_n"]))
            starting = folder / "starting_lens.json"
            starting.write_text(json.dumps(candidate_json, indent=2), encoding="utf-8")
            unchanged_config = copy.deepcopy(base)
            unchanged_config["input_lens"] = str(starting)
            unchanged_config["training"]["steps"] = 0
            unchanged_lens, unchanged_report = run_design(unchanged_config, folder / "substitution_only")
            unchanged = summary(name + "_substitution", unchanged_lens, unchanged_report,
                                evaluate_mtf(unchanged_lens, folder / "substitution_only", mtf_settings), settings["process_cost"])
            unchanged["substitution"] = changes
            unchanged["rejection_reasons"] = passes(unchanged, current, settings["screening"], rms_limit)
            tried.append(unchanged)
            nominal_config = copy.deepcopy(base)
            nominal_config["input_lens"] = str(starting)
            nominal_config["training"]["steps"] = int(settings["search"]["nominal_steps_per_trial"])
            nominal_lens, nominal_report = run_design(nominal_config, folder / "nominal_recovery")
            nominal = summary(name + "_nominal", nominal_lens, nominal_report,
                              evaluate_mtf(nominal_lens, folder / "nominal_recovery", mtf_settings), settings["process_cost"])
            nominal["substitution"] = changes
            nominal["rejection_reasons"] = passes(nominal, current, settings["screening"], rms_limit)
            tried.append(nominal)
            # 精调从已恢复标称像质的处方出发；不合格时保留标称阶段结果。
            if int(settings["search"]["robust_steps_per_trial"]) > 0:
                robust_config = copy.deepcopy(base)
                robust_config["input_lens"] = str(nominal_lens)
                robust_config["training"]["steps"] = int(settings["search"]["robust_steps_per_trial"])
                robust_config["training"]["objective"] = "expected_good_cost"
                if metric == "mtf":
                    robust_config["differentiable_mtf"] = copy.deepcopy(settings["differentiable_mtf"])
                robust_lens, robust_report = run_design(robust_config, folder / "robust_refinement")
                robust = summary(name + "_robust", robust_lens, robust_report,
                                 evaluate_mtf(robust_lens, folder / "robust_refinement", mtf_settings), settings["process_cost"])
                robust["substitution"] = changes
                robust["rejection_reasons"] = passes(robust, current, settings["screening"], rms_limit)
                tried.append(robust)
            for candidate in tried[-(3 if int(settings["search"]["robust_steps_per_trial"]) > 0 else 2):]:
                quality = (f'MTF={candidate["independent_mtf_worst"]:.4f}' if metric == "mtf"
                           else f'RMS p90={candidate["rms_p90_mm"]:.5f}')
                print(f'Round {round_number} {candidate["name"]}: build={candidate["build_cost_proxy"]:.2f}, '
                      f'good={candidate["good_unit_cost_proxy"]}, {quality}, '
                      f'reject={candidate["rejection_reasons"]}', flush=True)
            (round_folder / "candidate_progress.json").write_text(json.dumps(tried, indent=2), encoding="utf-8")
        feasible = [candidate for candidate in tried if not candidate["rejection_reasons"]]
        if feasible:
            current = min(feasible, key=lambda row: row["good_unit_cost_proxy"])
        history.append({"round": round_number, "selected": current,
                        "candidates": tried, "proposal_count": len(all_proposals)})
        (output / "iteration_report.json").write_text(json.dumps({"history": history,
            "final_lens_json": current["lens_json"],
            "caution": "Relative process scenarios and geometric MTF proxies; not supplier quotations or production yield"},
            indent=2), encoding="utf-8")
        plot_history(history, output, metric=metric,
                     frequency=float(mtf_settings["frequency_lp_per_mm"]) if mtf_settings else 0.0)
        if not feasible:
            print("No candidate passed this round; stopping.", flush=True)
            break
    print(f'Final accepted lens: {current["lens_json"]}')


if __name__ == "__main__":
    main()
