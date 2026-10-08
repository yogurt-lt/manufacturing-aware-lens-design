"""Config-driven manufacturing-aware fine-tuning of a matched DeepLens design.

Run: python 8_manufacturing_design.py --config configs/8_manufacturing_design.yml
The acceptance metric is RGB spot RMS, not production MTF or actual yield.
"""

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from deeplens.geolens import GeoLens
from deeplens.geolens_pkg.manufacturing import (
    assembly_groups, blank_cost_proxy, configure_tolerances, evaluate, material_price_report,
    good_unit_cost, perturbed_lens, process_cost_proxy, read_relative_prices, rms_merit, soft_yield,
)
from deeplens.geolens_pkg.differentiable_mtf import mtf_shortfall, ray_fourier_mtf
from deeplens.geolens_pkg.manufacturing_plots import save_summary_plots


def apply_fixed_field(lens, sampling):
    """锁定比较用物方视场，避免每次更新焦距时悄悄改变测试视场。"""
    if "reference_foclen_mm" in sampling:
        # loss_rms 的理想像点由 foclen 生成；固定目标映射才是同一产品指标。
        lens.foclen = float(sampling["reference_foclen_mm"])
    if "fixed_vfov_rad" in sampling:
        lens.vfov = float(sampling["fixed_vfov_rad"])
    if "fixed_hfov_rad" in sampling:
        lens.hfov = float(sampling["fixed_hfov_rad"])
    if "pupil_mode" in sampling:
        if sampling["pupil_mode"] not in {"default", "paraxial"}:
            raise ValueError("sampling.pupil_mode must be default or paraxial")
        lens.ray_sampling_pupil_mode = sampling["pupil_mode"]
    if "launch_pupil_z_mm" in sampling and "launch_pupil_radius_mm" in sampling:
        # 固定照明光束的物理发射平面和半径；每个候选结构都用同一束光验证。
        lens.entr_pupilz = float(sampling["launch_pupil_z_mm"])
        lens.entr_pupilr = float(sampling["launch_pupil_radius_mm"])


def run(config_path):
    """一次完整运行：读配置 → 基线 → 优化 → 独立评估 → 保存图表。"""
    with Path(config_path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    requested_device = str(config.get("device", "auto")).lower()
    if requested_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but this PyTorch installation cannot use CUDA")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if requested_device == "auto" else requested_device
    logging.info("Manufacturing design device: %s; torch=%s", device, torch.__version__)
    lens = GeoLens(filename=config["input_lens"], device=device)
    prices = read_relative_prices(config["glass_catalog"])
    elements, missing = material_price_report(lens, prices, config["missing_price_policy"])
    configure_tolerances(lens, config["tolerances"])
    sampling = config["sampling"]
    apply_fixed_field(lens, sampling)
    cost_options = config["cost_proxy"]
    process_rules = config.get("process_cost")
    mtf_settings = config.get("differentiable_mtf")
    threshold = float(config["acceptance"]["max_rgb_rms_mm"])
    trial_count = int(config["evaluation"]["trials"])
    if trial_count < 1:
        raise ValueError("evaluation.trials must be positive")

    tolerance = config["tolerances"]
    before = evaluate(lens, sampling, elements, cost_options, tolerance, threshold,
                      trial_count, seed + 10000)
    if process_rules:
        before["process_cost_proxy"] = float(process_cost_proxy(lens, elements, process_rules).item())
        before["total_build_cost_proxy"] = before["cost_proxy_hk9l_mm3"] + before["process_cost_proxy"]
        before["expected_good_unit_cost_proxy"] = (before["total_build_cost_proxy"] / before["yield_estimate"]
                                                    if before["yield_estimate"] > 0 else None)
    lens.write_lens_json(str(output / "baseline_lens.json"))
    if config.get("visualization", {}).get("lens_layout", True):
        lens.draw_layout(filename=str(output / "baseline_layout.png"), show=False)
    baseline_cost = max(before.get("total_build_cost_proxy", before["cost_proxy_hk9l_mm3"]), 1e-9)

    train = config["training"]
    objective = train.get("objective", "legacy")
    if objective not in {"legacy", "expected_good_cost", "mean_robust_cost", "nominal_only"}:
        raise ValueError("Unknown training.objective: " + str(objective))
    steps = int(train["steps"])
    draws = int(train["tolerance_draws_per_step"])
    if steps < 0 or draws < 1:
        raise ValueError("training.steps must be nonnegative and draws must be positive")
    history = []
    if steps:
        optimizer = lens.get_optimizer(lrs=[float(x) for x in train["lrs"]],
                                       decay=float(train["decay"]), optim_mat=False)
        if bool(train.get("freeze_surface_positions", False)):
            # During recovery from a large material jump, moving every surface
            # axially at once can invert an air gap before the optical powers
            # have recovered. Freeze surface positions while retaining the
            # independently optimized sensor position for refocus.
            for surface in lens.surfaces:
                if hasattr(surface, "d"):
                    surface.d.requires_grad_(False)
        for step in range(steps):
            # 镜片更新后必须重算入瞳；用旧入瞳训练会产生虚假的像质收益。
            if step and "launch_pupil_z_mm" not in sampling:
                lens.post_computation()
                apply_fixed_field(lens, sampling)
            optimizer.zero_grad()
            # 标称像质和几何约束。联合模式将标称像质视为必须满足的约束。
            nominal = rms_merit(lens, sampling)
            regularization, _ = lens.loss_reg()
            cost = blank_cost_proxy(lens, elements, **cost_options)
            if process_rules:
                cost = cost + process_cost_proxy(lens, elements, process_rules)
            mtf_values = None
            if mtf_settings and float(mtf_settings.get("weight", 0)) > 0:
                # 固定光线采样，避免每一步的 MTF 梯度被蒙特卡罗抽样噪声淹没；
                # fork_rng 保留公差采样的随机数状态。
                gpu_devices = [torch.cuda.current_device()] if lens.device.type == "cuda" else []
                with torch.random.fork_rng(devices=gpu_devices):
                    torch.manual_seed(int(mtf_settings.get("ray_seed", seed + 20000)))
                    mtf_values = ray_fourier_mtf(lens, mtf_settings)
            mtf_penalty = (float(mtf_settings["weight"]) * mtf_shortfall(
                mtf_values, mtf_settings["minimum_proxy_mtf"]) if mtf_values is not None else 0.0)
            perturbed_values = []
            soft_pass = None
            if objective in {"expected_good_cost", "mean_robust_cost"}:
                merits = []
                for _ in range(draws):
                    # 同时对光学参数和制造误差求导；采样误差本身不优化。
                    with perturbed_lens(lens, tolerance):
                        merit = rms_merit(lens, sampling)
                        merits.append(merit)
                        perturbed_values.append(float(merit.detach().item()))
                nominal_limit = float(train.get("nominal_max_rms_mm", threshold))
                width = float(train["yield_transition_mm"])
                # 软约束只在接近或超过标称门槛时显著增加。
                nominal_penalty = F.softplus((nominal - nominal_limit) / width) * width
                relative_build_cost = cost / baseline_cost
                if objective == "expected_good_cost":
                    # 完整方法：用平滑合格概率近似不可微的硬良率，再最小化
                    # “制造成本 / 软良率”。该形式在合格门槛附近仍能提供梯度。
                    soft_pass = soft_yield(merits, threshold, float(train["yield_transition_mm"]))
                    objective_loss = good_unit_cost(relative_build_cost, soft_pass,
                                                    float(train.get("yield_floor", 0.01)))
                    robust_term = float(train["good_cost_weight"]) * objective_loss
                else:
                    # 消融实验：仍使用完全相同的制造误差样本，但不用软良率。
                    # 以扰动 RMS 均值代表传统稳健目标，并直接加入制造成本。
                    # 因而该组隔离的是“软合格概率”形式，而不是删掉公差采样。
                    mean_perturbed_rms = torch.stack(merits).mean()
                    robust_term = (
                        float(train.get("good_cost_weight", 0.0)) * relative_build_cost
                        + float(train.get("robust_mean_weight", 0.0))
                        * mean_perturbed_rms / nominal_limit
                    )
                loss = (robust_term
                        + float(train["nominal_constraint_weight"]) * nominal_penalty / nominal_limit
                        + float(train["regularization_weight"]) * regularization + mtf_penalty)
                loss.backward()
            else:
                base_loss = (nominal + float(train["regularization_weight"]) * regularization
                             + (float(train["cost_weight"]) * cost / baseline_cost
                                if objective == "legacy" else 0.0) + mtf_penalty)
                base_loss.backward()
                robustness_weight = float(train["robustness_weight"]) if objective == "legacy" else 0.0
                for _ in range(draws if robustness_weight > 0 else 0):
                    with perturbed_lens(lens, tolerance):
                        merit = rms_merit(lens, sampling)
                        penalty = F.softplus((merit - threshold) / threshold) * threshold
                        (robustness_weight * penalty / draws).backward()
                        perturbed_values.append(float(merit.detach().item()))
            # Large material substitutions can create very large early gradients.
            # Optional clipping keeps the seed-recovery phase stable without
            # changing the default behaviour of existing configurations.
            gradient_norm = None
            clip_norm = train.get("gradient_clip_norm")
            if clip_norm is not None:
                parameters = [parameter for group in optimizer.param_groups
                              for parameter in group["params"] if parameter.grad is not None]
                gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(parameters, float(clip_norm))
                gradient_norm = float(gradient_norm_tensor.detach().item())
                if not np.isfinite(gradient_norm):
                    raise RuntimeError(f"Non-finite gradient norm at step {step + 1}")
            optimizer.step()
            row = {"step": step + 1, "nominal_rms_mm": float(nominal.detach().item()),
                   "perturbed_rms_mean_mm": float(np.mean(perturbed_values)) if perturbed_values else None,
                   "cost_proxy_hk9l_mm3": float(cost.detach().item()),
                   "soft_yield": float(soft_pass.detach().item()) if soft_pass is not None else None,
                   "fourier_mtf_worst": float(mtf_values.detach().min().item()) if mtf_values is not None else None,
                   "gradient_norm_before_clip": gradient_norm}
            history.append(row)
            if (step + 1) % int(train["log_every"]) == 0 or step == 0:
                logging.info("%s", row)

    lens.write_lens_json(str(output / "manufacturing_lens.json"))
    # 重新加载落盘处方再评估：DeepLens 加载时会重新计算视场和入/出瞳。
    # 若直接在训练对象上评估，会把旧入瞳与新镜片混用，造成虚假的良率提升。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    saved_lens = GeoLens(filename=str(output / "manufacturing_lens.json"), device=device)
    apply_fixed_field(saved_lens, sampling)
    configure_tolerances(saved_lens, tolerance)
    saved_elements, _ = material_price_report(saved_lens, prices, config["missing_price_policy"])
    after = evaluate(saved_lens, sampling, saved_elements, cost_options, tolerance, threshold,
                     trial_count, seed + 10000)
    if process_rules:
        after["process_cost_proxy"] = float(process_cost_proxy(saved_lens, saved_elements, process_rules).item())
        after["total_build_cost_proxy"] = after["cost_proxy_hk9l_mm3"] + after["process_cost_proxy"]
        after["expected_good_unit_cost_proxy"] = (after["total_build_cost_proxy"] / after["yield_estimate"]
                                                   if after["yield_estimate"] > 0 else None)
    if mtf_settings:
        with torch.no_grad():
            before_lens = GeoLens(filename=str(output / "baseline_lens.json"), device=device)
            apply_fixed_field(before_lens, sampling)
            gpu_devices = [torch.cuda.current_device()] if saved_lens.device.type == "cuda" else []
            with torch.random.fork_rng(devices=gpu_devices):
                torch.manual_seed(int(mtf_settings.get("ray_seed", seed + 20000)))
                before["fourier_mtf_proxy"] = ray_fourier_mtf(before_lens, mtf_settings).tolist()
                torch.manual_seed(int(mtf_settings.get("ray_seed", seed + 20000)))
                after["fourier_mtf_proxy"] = ray_fourier_mtf(saved_lens, mtf_settings).tolist()
    if config.get("visualization", {}).get("lens_layout", True):
        saved_lens.draw_layout(filename=str(output / "manufacturing_layout.png"), show=False)
    report = {"source_config": str(config_path), "input_lens": config["input_lens"],
              "glass_catalog": config["glass_catalog"], "missing_glass_prices": missing,
              "price_policy": config["missing_price_policy"], "priced_elements": elements,
              "cost_definition": ("relative manufacturing cost index: cylindrical blank volume times "
                                  "CDGM relative glass price / H-K9L price, plus configured process scenario; "
                                  "not a currency quotation"),
              "training_objective": objective,
              "process_cost_rules": process_rules, "differentiable_mtf_settings": mtf_settings,
              "tolerance_terms_used": sorted(config["tolerances"]),
              "assembly_groups_surface_indices": assembly_groups(lens),
              "acceptance_definition": "RGB spot RMS at configured field grid <= threshold; no focus compensation",
              "baseline": before, "optimized": after, "history": history}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if config.get("visualization", {}).get("summary_plots", True):
        save_summary_plots(report, output, dpi=int(config.get("visualization", {}).get("dpi", 160)))
    # 原始样本保存在 report.json；终端只显示摘要，方便直接读运行结果。
    concise = lambda result: {key: value for key, value in result.items() if key != "rms_samples_mm"}
    print(json.dumps({"baseline": concise(before), "optimized": concise(after),
                      "missing_glass_prices": missing}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/8_manufacturing_design.yml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(args.config)
