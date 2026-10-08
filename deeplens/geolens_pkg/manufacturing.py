"""Manufacturing-aware objectives for an already matched GeoLens prescription.

玻璃库只给出相对价格，因此下方的成本是毛坯材料代理值，不是成品报价。
本模块除了已有的面型误差，还会把整组镜片的偏心和倾斜施加到光线追迹坐标系。
"""

import csv
import math
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def read_relative_prices(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["glass"].lower(): float(row["relative_price"]) for row in rows}


def material_price_report(lens, prices, missing_policy="error"):
    """Resolve one price per physical lens element; report every imputation."""
    known = sorted(prices.values())
    if not known:
        raise ValueError("The price catalog is empty")
    median = float(np.median(known))
    elements = []
    missing = set()
    for i, surface in enumerate(lens.surfaces[:-1]):
        name = surface.mat2.name.lower()
        if name in {"air", "vacuum", "occluder"}:
            continue
        if name not in prices:
            missing.add(name)
            if missing_policy == "error":
                continue
            if missing_policy != "median_proxy":
                raise ValueError(f"Unknown missing_price_policy: {missing_policy}")
        elements.append({"surface": i, "material": name,
                         "relative_price": prices.get(name, median),
                         "imputed": name not in prices})
    if missing and missing_policy == "error":
        raise ValueError("No relative price for: " + ", ".join(sorted(missing)) +
                         ". Add verified prices or set missing_price_policy: median_proxy for a demo.")
    return elements, sorted(missing)


def blank_cost_proxy(lens, elements, radial_allowance_mm=0.5, axial_allowance_mm=0.5):
    """Differentiable sum of cylindrical blank volumes times relative prices.

    Dividing by H-K9L's catalog price (10) gives H-K9L-equivalent mm^3.
    It omits yield, machining, tooling, molding, coatings and assembly.
    """
    cost = torch.zeros((), device=lens.device, dtype=lens.dtype)
    for item in elements:
        i = item["surface"]
        first, second = lens.surfaces[i:i + 2]
        radius = max(first.r, second.r) + radial_allowance_mm
        thickness = torch.clamp(second.d - first.d + axial_allowance_mm, min=0.0)
        cost = cost + math.pi * radius**2 * thickness * (item["relative_price"] / 10.0)
    return cost


def process_cost_proxy(lens, elements, rules):
    """可配置工艺情景成本，单位沿用 H-K9L 等效 mm³。

    系数是研究情景，不是供应商报价。每片按入口和出口两面识别非球面；
    模压玻璃、注塑塑料和普通研磨玻璃分别计价。固定费用按量产件数摊销。
    """
    cost = torch.zeros((), device=lens.device, dtype=lens.dtype)
    volume = int(rules.get("production_volume", 1))
    if volume < 1:
        raise ValueError("production_volume must be positive")
    for item in elements:
        i = item["surface"]
        surfaces = lens.surfaces[i:i + 2]
        aspheric = any("aspheric" in type(surface).__name__.lower() for surface in surfaces)
        material = item["material"].lower()
        plastic = any(token.lower() in material for token in rules.get("plastic_material_tokens", []))
        if plastic:
            kind = "plastic_asphere" if aspheric else "plastic_sphere"
        else:
            kind = "glass_asphere" if aspheric else "glass_sphere"
        coefficients = rules["per_element_proxy"]
        cost = cost + float(coefficients[kind])
        if aspheric and not plastic and material not in [x.lower() for x in rules.get("preferred_glass_aspheres", [])]:
            cost = cost + float(rules.get("nonpreferred_glass_asphere_penalty", 0.0))
    cost = cost + float(rules.get("tooling_proxy", 0.0)) / volume
    return cost


def assembly_groups(lens):
    """连续玻璃面直到出射空气面视为一个装调组；胶合片随同一组一起移动。"""
    groups = []
    start = None
    for i, surface in enumerate(lens.surfaces):
        glass = surface.mat2.name.lower() not in {"air", "vacuum", "occluder"}
        if glass and start is None:
            start = i
        elif not glass and start is not None:
            groups.append(list(range(start, i + 1)))
            start = None
    if start is not None:
        groups.append(list(range(start, len(lens.surfaces))))
    return groups


@contextmanager
def perturbed_lens(lens, tolerance):
    """生成一次制造样本，并在退出时恢复所有表面，防止误差累积。"""
    original = [(s.pos_x, s.pos_y, s.vec_local) for s in lens.surfaces]
    try:
        # 原始 DeepLens 的采样仅让半径、轴向位置、曲率和圆锥系数影响追迹。
        for surface in lens.surfaces:
            surface.sample_tolerance()
        # 组装误差按镜组共享，而不是对每个光学面独立抽样。
        # 两个横向偏心分量单位为 mm；两个倾角分量单位为角分。
        shift_sigma = float(tolerance.get("element_decenter_sigma_mm", 0.0))
        angle_sigma = float(tolerance.get("element_tilt_sigma_arcmin", 0.0)) * np.pi / (180.0 * 60.0)
        for group in assembly_groups(lens):
            dx, dy = np.random.normal(0.0, shift_sigma, size=2)
            tx, ty = np.random.normal(0.0, angle_sigma, size=2)
            for i in group:
                surface = lens.surfaces[i]
                surface.pos_x = original[i][0] + float(dx)
                surface.pos_y = original[i][1] + float(dy)
                # 小角度近似：绕 x/y 轴旋转法向量。组内面保持相对位置。
                tilt_vector = surface.vec_local.new_tensor([ty, -tx, 0.0])
                surface.vec_local = F.normalize(original[i][2] + tilt_vector, dim=0)
        yield lens
    finally:
        for surface, (px, py, direction) in zip(lens.surfaces, original):
            surface.pos_x, surface.pos_y, surface.vec_local = px, py, direction
            surface.zero_tolerance()


def configure_tolerances(lens, tolerance):
    supported = {"r_tole", "d_tole", "c_tole", "k_tole",
                 "element_decenter_sigma_mm", "element_tilt_sigma_arcmin"}
    unsupported = set(tolerance) - supported
    if unsupported:
        raise ValueError("Ray tracing does not implement these tolerance terms: " + ", ".join(sorted(unsupported)))
    defaults = {"r_tole": 0.0, "d_tole": 0.0, "c_tole": 0.0, "k_tole": 0.0,
                "center_thick_tole": 0.0, "decenter_tole": 0.0, "tilt_tole": 0.0,
                "mat2_n_tole": 0.0, "mat2_V_tole": 0.0}
    defaults.update({key: value for key, value in tolerance.items() if key in defaults})
    if any(float(value) < 0 for value in tolerance.values()):
        raise ValueError("Tolerance standard deviations must be nonnegative")
    lens.init_tolerance(defaults)


def rms_merit(lens, sampling):
    """Spot RMS in mm across RGB wavelengths and sampled fields."""
    grid = sampling["num_grid"]
    grid = (int(grid), int(grid)) if isinstance(grid, int) else tuple(int(x) for x in grid)
    return lens.loss_rms(num_grid=grid,
                         num_rays=int(sampling["num_rays"]),
                         depth=float(sampling["depth"]))


def soft_yield(merits, threshold_mm, transition_mm):
    """可微的合格概率估计：每个误差样本在验收线附近平滑过渡。

    merits 是同一设计的若干制造样本 RMS；过渡宽度不是公差标准差，
    它只控制训练时对硬合格判定的平滑程度。独立评价仍用硬阈值。
    """
    if transition_mm <= 0:
        raise ValueError("transition_mm must be positive")
    return torch.sigmoid((threshold_mm - torch.stack(merits)) / transition_mm).mean()


def good_unit_cost(build_cost, yield_fraction, yield_floor=0.01):
    """预期单件合格品成本 = 单次制造成本 / 合格概率。

    假设不返工、失败件报废。floor 只用于防止训练初期除零；
    报告中的蒙特卡罗良率和成本按原始估计值展示。
    """
    return build_cost / torch.clamp(yield_fraction, min=yield_floor)


def evaluate(lens, sampling, elements, cost_options, tolerance, threshold_mm, trials, seed):
    """Independent seeded Monte Carlo; no optimizer gradients are recorded."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        gpu_devices = [torch.cuda.current_device()] if lens.device.type == "cuda" else []
        with torch.random.fork_rng(devices=gpu_devices), torch.no_grad():
            torch.manual_seed(seed)
            nominal = float(rms_merit(lens, sampling).item())
            cost = float(blank_cost_proxy(lens, elements, **cost_options).item())
            samples = []
            for _ in range(trials):
                with perturbed_lens(lens, tolerance):
                    samples.append(float(rms_merit(lens, sampling).item()))
    finally:
        np.random.set_state(state)
    values = np.asarray(samples)
    accepted = int(np.count_nonzero(values <= threshold_mm))
    rate = accepted / trials
    # Wilson 95% interval, to avoid reporting a false exact yield.
    z = 1.96
    center = (rate + z*z/(2*trials)) / (1 + z*z/trials)
    half = z * math.sqrt(rate*(1-rate)/trials + z*z/(4*trials*trials)) / (1 + z*z/trials)
    return {"nominal_rms_mm": nominal, "cost_proxy_hk9l_mm3": cost,
            "rms_threshold_mm": threshold_mm, "accepted": accepted, "trials": trials,
            "yield_estimate": rate, "yield_wilson95": [center-half, center+half],
            "expected_good_unit_cost_proxy": cost / rate if rate > 0 else None,
            "rms_mean_mm": float(values.mean()), "rms_p90_mm": float(np.quantile(values, 0.9)),
            "rms_samples_mm": samples}
