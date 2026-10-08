"""AutoLens 后的加工专家 v1：提出真实玻璃替换、重优化、独立验收。

借鉴 Zemax 的“替换玻璃后重新优化”工作流，不复刻 Hammer 算法。
该入口只操作 DeepLens JSON，不会修改 AutoLens 或 Zemax 原始文件。
运行：python 17_manufacturing_expert.py --config configs/17_manufacturing_expert.yml
"""

import argparse
import copy
import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import yaml

from deeplens.optics.materials import CDGM_data, Material


def prices_from_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return {row["glass"].lower(): row for row in csv.DictReader(handle)}


def propose_real_glasses(lens_json, prices, settings):
    """初筛真实材料：表面类型、价格、熔炼频率和三波长折射率邻域。

    这是启发式提案器。折射率接近只减少无效尝试，最终要完整追迹。
    默认仍限定球面 H 系玻璃；配置可以把候选限制到指定非球面和 D 系材料。
    光学常数与价格相同的熔炼批次别名不会被当成一次有效材料替换。
    """
    wavelengths = settings["wavelengths_nm"]
    allowed_types = set(settings.get("surface_types", ["Spheric"]))
    allowed_indices = ({int(value) for value in settings["surface_indices"]}
                       if settings.get("surface_indices") is not None else None)
    name_pattern = settings.get("candidate_name_regex", r"h-[a-z0-9-]+")
    # Individual elements can have stricter process routes.  For example, the
    # last glass asphere may be restricted to D-series molded blanks while the
    # spherical elements may search both H and D catalog families.
    patterns_by_surface = {int(index): pattern for index, pattern in
                           settings.get("candidate_name_regex_by_surface", {}).items()}
    equivalent_dn = float(settings.get("equivalent_max_abs_delta_n", 1e-5))
    equivalent_price_fraction = float(settings.get("equivalent_max_price_fraction", 1e-6))
    proposals = []
    surfaces = lens_json["surfaces"]
    for index, surface in enumerate(surfaces[:-1]):
        old = surface["mat2"].lower()
        if (surface["type"] not in allowed_types or
                (allowed_indices is not None and index not in allowed_indices) or
                old not in prices or old not in CDGM_data):
            continue
        old_price = float(prices[old]["relative_price"])
        old_material = Material(old)
        # 与已有毛坯代理一致，以材料价格差乘以近似圆柱体积排序。
        radius = max(float(surface["r"]), float(surfaces[index + 1]["r"]))
        thickness = max(float(surface["d_next"]), 0.0)
        blank_volume = math.pi * (radius + float(settings["radial_allowance_mm"]))**2 * (
            thickness + float(settings["axial_allowance_mm"]))
        for new, row in prices.items():
            if new == old or new not in CDGM_data:
                continue
            if row["melt_frequency"] not in settings["melt_frequency_allowlist"]:
                continue
            new_price = float(row["relative_price"])
            # 默认仍只找更便宜玻璃。Hammer/beam 模式可以允许小幅临时涨价，
            # 给后续多轮替换留出跳出局部最优的通道。
            max_price_ratio = float(settings.get("max_candidate_price_ratio", 0.999999))
            if new_price > old_price * max_price_ratio:
                continue
            # 默认使用 H 系；非球面阶段可配置为只允许 D 系模压玻璃。
            surface_pattern = patterns_by_surface.get(index, name_pattern)
            if not re.fullmatch(surface_pattern, new):
                continue
            if settings["same_family_prefix"] and not new.startswith(old[:int(settings["same_family_prefix"])]):
                continue
            candidate = Material(new)
            delta_n = max(abs(candidate.refractive_index(w / 1000.0) -
                              old_material.refractive_index(w / 1000.0))
                          for w in wavelengths)
            if delta_n > float(settings["max_abs_delta_n"]):
                continue
            # H-K9L/H-K9LA 等只是相同材料的不同熔炼批次。若光学常数和
            # 相对价格均等价，跳过该动作，避免用“换批次”换取额外优化步数。
            same_price = abs(new_price - old_price) <= max(
                1.0, abs(old_price)) * equivalent_price_fraction
            if settings.get("exclude_equivalent_materials", True) and delta_n <= equivalent_dn and same_price:
                continue
            saving = blank_volume * (old_price - new_price) / 10.0
            proposals.append({"surface_idx": index, "from": old, "to": new,
                              "candidate_melt_frequency": row["melt_frequency"],
                              "old_relative_price": old_price,
                              "new_relative_price": new_price,
                              "candidate_price_ratio": new_price / old_price,
                              "max_abs_delta_n": delta_n,
                              "estimated_blank_saving_proxy": saving})
    ranking = settings.get("ranking", "cost_first")
    if ranking == "cost_first":
        proposals.sort(key=lambda row: (-row["estimated_blank_saving_proxy"],
                                        row["max_abs_delta_n"], row["to"]))
    elif ranking == "optical_first":
        proposals.sort(key=lambda row: (row["max_abs_delta_n"],
                                        -row["estimated_blank_saving_proxy"], row["to"]))
    else:
        raise ValueError("proposal.ranking must be cost_first or optical_first")
    # 一轮只试少数候选，避免组合爆炸；下一轮可从入选处方继续提出候选。
    chosen = []
    used_surface = {}
    for proposal in proposals:
        surface_idx = proposal["surface_idx"]
        if used_surface.get(surface_idx, 0) >= int(settings.get("max_per_surface", 1)):
            continue
        chosen.append(proposal)
        used_surface[surface_idx] = used_surface.get(surface_idx, 0) + 1
        if len(chosen) >= int(settings["candidate_limit"]):
            break
    return chosen, proposals


def invoke(script, config_path):
    subprocess.run([sys.executable, script, "--config", str(config_path)], check=True)


def write_yaml(path, value):
    rendered = yaml.safe_dump(value, allow_unicode=True, sort_keys=False)
    unchanged = path.exists() and path.read_text(encoding="utf-8") == rendered
    path.write_text(rendered, encoding="utf-8")
    return unchanged


def worst_mtf(result):
    return min(screen["best_allowed_focus"]["worst_mtf"] for screen in result.values())


def central_field_mtf(result):
    """Return the smaller T/S value at the field sample nearest zero.

    This is useful for early method demonstrations where the full-field geometric
    PSF model drives every candidate close to zero at the edge of the field.
    """
    values = []
    for screen in result.values():
        fields = screen["best_allowed_focus"]["fields"]
        central = min(fields, key=lambda row: abs(float(row["field_fraction"])))
        values.append(min(float(central["tangential"]), float(central["sagittal"])))
    return min(values)


def maximum_field_mtf(result):
    """Return min(T, S) at the largest sampled field fraction.

    This is the paper's unified MTF reporting convention.  The spatial
    frequency is supplied by the evaluation config (93 lp/mm in the paper).
    """
    values = []
    for screen in result.values():
        fields = screen["best_allowed_focus"]["fields"]
        edge = max(fields, key=lambda row: float(row["field_fraction"]))
        values.append(min(float(edge["tangential"]), float(edge["sagittal"])))
    return min(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/17_manufacturing_expert.yml")
    args = parser.parse_args()
    settings = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output = Path(settings["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    source = Path(settings["autolens_output_json"])
    lens_json = json.loads(source.read_text(encoding="utf-8"))
    base = yaml.safe_load(Path(settings["base_config"]).read_text(encoding="utf-8"))
    base["input_lens"] = str(source)
    base["cost_proxy"]["radial_allowance_mm"] = float(settings["proposal"]["radial_allowance_mm"])
    base["cost_proxy"]["axial_allowance_mm"] = float(settings["proposal"]["axial_allowance_mm"])
    base_path = output / "generated_base.yml"
    base_unchanged = write_yaml(base_path, base)

    prices = prices_from_csv(base["glass_catalog"])
    candidates, all_proposals = propose_real_glasses(lens_json, prices, settings["proposal"])
    (output / "glass_proposals.json").write_text(json.dumps(all_proposals, indent=2), encoding="utf-8")
    variants = [{"name": "original", "substitutions": []}]
    for item in candidates:
        variants.append({"name": f's{item["surface_idx"]}_{item["to"].replace("-", "_")}',
                         "substitutions": [{key: item[key] for key in ("surface_idx", "from", "to")} ]})
    if not candidates:
        raise RuntimeError("No real-glass substitution passed the proposal filters")
    glass_config = {"base_config": str(base_path), "output_dir": str(output / "glass_trials"),
                    "wavelengths_nm": settings["proposal"]["wavelengths_nm"],
                    "max_abs_delta_n": settings["proposal"]["max_abs_delta_n"],
                    "variants": variants}
    glass_config_path = output / "generated_glass_trials.yml"
    glass_unchanged = write_yaml(glass_config_path, glass_config)
    print(f"Proposed {len(candidates)} substitution(s): {candidates}", flush=True)
    if not (base_unchanged and glass_unchanged and
            (output / "glass_trials" / "glass_substitution_report.json").exists()):
        invoke("15_glass_substitution.py", glass_config_path)
    else:
        print("Reusing completed glass trials with identical generated configs.", flush=True)

    mtf_config = yaml.safe_load(Path(settings["mtf_template_config"]).read_text(encoding="utf-8"))
    mtf_config["output_dir"] = str(output / "mtf_diagnostic")
    mtf_config["lenses"] = {item["name"]: str(output / "glass_trials" / item["name"] /
                                              "manufacturing_lens.json") for item in variants}
    mtf_config_path = output / "generated_mtf.yml"
    mtf_unchanged = write_yaml(mtf_config_path, mtf_config)

    frequency = mtf_config["frequency_lp_per_mm"]
    if not (mtf_unchanged and (output / "mtf_diagnostic" /
                               f"review_{frequency:g}lpmm_mtf.json").exists()):
        invoke("10_review_mtf.py", mtf_config_path)
    else:
        print("Reusing completed MTF diagnostic with identical config.", flush=True)
    screen_config = {"glass_report": str(output / "glass_trials" / "glass_substitution_report.json"),
                     "mtf_report": str(output / "mtf_diagnostic" /
                                       f"review_{frequency:g}lpmm_mtf.json"),
                     "output_dir": str(output / "screening"),
                     "reference_variant": "original",
                     "min_relative_mtf": settings["screening"]["min_relative_mtf"],
                     "min_demo_yield": settings["screening"]["min_demo_yield"]}
    screen_path = output / "generated_screening.yml"
    screen_unchanged = write_yaml(screen_path, screen_config)
    if not (screen_unchanged and (output / "screening" / "glass_screening.json").exists()):
        invoke("16_glass_screening.py", screen_path)
    else:
        print("Reusing completed screening with identical config.", flush=True)
    result = json.loads((output / "screening" / "glass_screening.json").read_text(encoding="utf-8"))
    selected = result["selected_for_next_validation"]
    result["final_lens_json"] = (str(output / "glass_trials" / selected /
                                     "manufacturing_lens.json") if selected else None)
    refinement = settings.get("manufacturing_refinement", {})
    if selected and refinement.get("enabled", False):
        # 第二步才做制造误差驱动的可微精调。设一道独立验收关卡，避免
        # 训练目标下降但真实硬良率或 MTF 退化的方案覆盖已通过的候选。
        before_report = json.loads((output / "glass_trials" / selected / "report.json").read_text(encoding="utf-8"))
        before = before_report["optimized"]
        refine_config = copy.deepcopy(base)
        refine_config["input_lens"] = result["final_lens_json"]
        refine_config["output_dir"] = str(output / "refinement")
        refine_config["training"]["objective"] = "expected_good_cost"
        refine_config["training"]["steps"] = int(refinement["steps"])
        refine_config["visualization"] = {"lens_layout": False, "summary_plots": False}
        refine_path = output / "generated_refinement.yml"
        write_yaml(refine_path, refine_config)
        invoke("8_manufacturing_design.py", refine_path)
        after_report = json.loads((output / "refinement" / "report.json").read_text(encoding="utf-8"))
        after = after_report["optimized"]
        refine_mtf_config = copy.deepcopy(mtf_config)
        refine_mtf_config["output_dir"] = str(output / "refinement_mtf")
        refine_mtf_config["lenses"] = {
            "before_refinement": result["final_lens_json"],
            "after_refinement": str(output / "refinement" / "manufacturing_lens.json")}
        refine_mtf_path = output / "generated_refinement_mtf.yml"
        write_yaml(refine_mtf_path, refine_mtf_config)
        invoke("10_review_mtf.py", refine_mtf_path)
        mtf_result = json.loads((output / "refinement_mtf" /
                                 f"review_{frequency:g}lpmm_mtf.json").read_text(encoding="utf-8"))
        before_mtf = worst_mtf(mtf_result["lenses"]["before_refinement"])
        after_mtf = worst_mtf(mtf_result["lenses"]["after_refinement"])
        before_good = before["expected_good_unit_cost_proxy"]
        after_good = after["expected_good_unit_cost_proxy"]
        reasons = []
        if after["nominal_rms_mm"] > after["rms_threshold_mm"]:
            reasons.append("nominal RMS exceeds demo limit")
        if after["yield_estimate"] < float(settings["screening"]["min_demo_yield"]):
            reasons.append("hard-sample yield below screen")
        if after_mtf < before_mtf * float(settings["screening"]["min_relative_mtf"]):
            reasons.append("50 lp/mm proxy MTF degraded")
        min_gain = float(refinement["min_good_cost_improvement_fraction"])
        if before_good is None or after_good is None or after_good > before_good * (1 - min_gain):
            reasons.append("good-unit cost improvement below configured minimum")
        accepted = not reasons
        result["manufacturing_refinement"] = {
            "accepted": accepted, "reasons": reasons,
            "before_good_unit_cost_proxy": before_good,
            "after_good_unit_cost_proxy": after_good,
            "before_demo_yield": before["yield_estimate"],
            "after_demo_yield": after["yield_estimate"],
            "before_50lpmm_proxy": before_mtf, "after_50lpmm_proxy": after_mtf}
        if accepted:
            result["final_lens_json"] = str(output / "refinement" / "manufacturing_lens.json")
    result["autolens_output_json"] = str(source)
    result["proposed_substitutions"] = candidates
    result["note"] = ("Version 1: real-glass proposals + differentiable geometry reoptimization + "
                      "independent yield/MTF gates; not a reproduction of Zemax Hammer")
    (output / "expert_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Manufacturing expert selected: {result['selected_for_next_validation']}")
    print(f"Final lens for next validation: {result['final_lens_json']}")


if __name__ == "__main__":
    main()
