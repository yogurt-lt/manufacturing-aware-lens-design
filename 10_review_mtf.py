"""用评审表的 93 lp/mm 指标给 DeepLens 镜头做独立几何 MTF 诊断。

这里的 PSF 来自几何光线追迹，尚未包含衍射、照明光路和 DMD 像素 MTF；
因此结果不能直接代替 Zemax FFT MTF 或量产验收数据。
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from deeplens.geolens import GeoLens


def screen_distance_mm(diagonal_inch, aspect_ratio, throw_ratio):
    """投射距离 = 投射比 × 屏幕宽度；屏幕对角线单位为 inch。"""
    width, height = aspect_ratio
    screen_width = diagonal_inch * 25.4 * width / math.hypot(width, height)
    return screen_width * throw_ratio


def mtf_at_field(lens, field, distance_mm, settings, sample_seed):
    """三个波长的 PSF 按评审表 3:5:2 加权，再求 T/S 两方向 MTF。"""
    psfs = []
    weights = settings["wavelength_weights"]
    for index, wavelength_nm in enumerate(settings["wavelengths_nm"]):
        # 各设计使用同一组随机光线，减少比较时的采样噪声。
        torch.manual_seed(sample_seed + index)
        psf = lens.psf(points=[0.0, -float(field), -float(distance_mm)],
                       ks=int(settings["psf_kernel_pixels"]),
                       wvln=float(wavelength_nm) / 1000.0,
                       spp=int(settings["rays_per_wavelength"]),
                       recenter=bool(settings["recenter_psf"]))
        psfs.append(psf)
    weight_tensor = torch.as_tensor(weights, device=lens.device, dtype=lens.dtype)
    combined = (torch.stack(psfs) * weight_tensor[:, None, None]).sum(0) / weight_tensor.sum()
    frequency, tangential, sagittal = lens.psf2mtf(combined, lens.pixel_size)
    target = float(settings["frequency_lp_per_mm"])
    if target < frequency.min() or target > frequency.max():
        raise ValueError(f"{target} lp/mm lies outside sampled MTF range; change PSF sampling")
    return {"tangential": float(np.interp(target, frequency, tangential)),
            "sagittal": float(np.interp(target, frequency, sagittal))}


def evaluate_lens(path, settings):
    # GeoLens 加载处方时会采样入瞳；每个候选镜头使用相同随机种子。
    np.random.seed(int(settings.get("seed", 43702)))
    torch.manual_seed(int(settings.get("seed", 43702)))
    device = settings.get("device")
    lens = GeoLens(filename=str(path), device=None if device in (None, "auto") else device)
    # DeepLens JSON 中的默认 1024 网格使 93 lp/mm 超过奈奎斯特频率；
    # 这里只做计算网格过采样，物理像面半径和镜片处方保持不变。
    resolution = int(settings["psf_grid_resolution"])
    lens.set_sensor_res((resolution, resolution))
    # 一个补偿量作用于全部视场；不能针对每个视场分别“调到最好”。
    original_sensor = lens.d_sensor
    output = {}
    gpu_devices = [torch.cuda.current_device()] if lens.device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=gpu_devices), torch.no_grad():
            for diagonal in settings["screen_diagonals_inch"]:
                distance = screen_distance_mm(float(diagonal), settings["screen_aspect_ratio"],
                                              float(settings["throw_ratio"]))
                candidates = []
                for offset in settings["focus_offsets_mm"]:
                    if abs(float(offset)) > float(settings["review_bf_adjust_limit_mm"]):
                        raise ValueError("A focus offset exceeds the review BF adjustment limit")
                    lens.d_sensor = original_sensor + float(offset)
                    fields = []
                    for i, fraction in enumerate(settings["field_fractions"]):
                        values = mtf_at_field(lens, fraction, distance, settings,
                                              sample_seed=7000 + int(diagonal)*100 + i*10)
                        fields.append({"field_fraction": float(fraction), **values})
                    worst = min(min(row["tangential"], row["sagittal"]) for row in fields)
                    candidates.append({"focus_offset_mm": float(offset),
                                       "worst_mtf": worst, "fields": fields})
                output[str(diagonal)] = {"screen_distance_mm": distance,
                                         "best_allowed_focus": max(candidates, key=lambda x: x["worst_mtf"]),
                                         "all_focus_candidates": candidates}
    finally:
        lens.d_sensor = original_sensor
    return output


def make_plot(results, settings, output_dir):
    screens = settings["screen_diagonals_inch"]
    fig, axes = plt.subplots(1, len(screens), figsize=(6 * len(screens), 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    palette = ["#64748b", "#0d9488", "#d97706", "#7c3aed", "#dc2626", "#0891b2"]
    for ax, screen in zip(axes, screens):
        for index, (name, lens_result) in enumerate(results.items()):
            rows = lens_result[str(screen)]["best_allowed_focus"]["fields"]
            x = [row["field_fraction"] for row in rows]
            color = palette[index % len(palette)]
            ax.plot(x, [row["tangential"] for row in rows], "o-", color=color,
                    label=f"{name} T")
            ax.plot(x, [row["sagittal"] for row in rows], "s--", color=color,
                    label=f"{name} S")
        if settings.get("nominal_min_mtf") is not None:
            ax.axhline(float(settings["nominal_min_mtf"]), color="#7c3aed", linestyle=":",
                       label=f'Reference nominal {float(settings["nominal_min_mtf"]):.0%}')
        if settings.get("tolerance_min_mtf") is not None:
            ax.axhline(float(settings["tolerance_min_mtf"]), color="#b91c1c", linestyle=":",
                       label=f'Reference tolerance {float(settings["tolerance_min_mtf"]):.0%}')
        ax.set(xlabel="Normalized field", title=f'{screen}" screen, {settings["frequency_lp_per_mm"]:g} lp/mm',
               xlim=(-0.05, 1.05), ylim=(0, 1))
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Geometric polychromatic MTF proxy")
    axes[-1].legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / f'review_{settings["frequency_lp_per_mm"]:g}lpmm_mtf.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/10_review_mtf.yml")
    args = parser.parse_args()
    settings = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output = Path(settings["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, path in settings["lenses"].items():
        print(f"Evaluating {name} ...", flush=True)
        results[name] = evaluate_lens(path, settings)
    report = {"review_source": settings["review_source"],
              "metric": f'geometric polychromatic PSF MTF proxy at {settings["frequency_lp_per_mm"]:g} lp/mm',
              "warning": "Same projector-lens project: 34 mm denotes lens length, whereas 54 mm was the DeepLens total-length target including back focus. Initial and Zemax-refined prescriptions require parameter matching; diffraction and DMD MTF are excluded here.",
              "lenses": results}
    (output / f'review_{settings["frequency_lp_per_mm"]:g}lpmm_mtf.json').write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    make_plot(results, settings, output)
    for name, lens_result in results.items():
        for screen, data in lens_result.items():
            best = data["best_allowed_focus"]
            print(f'{name}, {screen}" screen: worst {settings["frequency_lp_per_mm"]:g} lp/mm proxy MTF={best["worst_mtf"]:.3f}, '
                  f'focus offset={best["focus_offset_mm"]:+.3f} mm')


if __name__ == "__main__":
    main()
