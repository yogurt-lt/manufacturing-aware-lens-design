"""配置驱动的真实玻璃替换实验。

这里不把两种玻璃插值为不存在的中间材料。每个候选组合先生成一份
DeepLens JSON，再在完全相同的训练预算和制造误差样本下优化与比较。
运行：python 15_glass_substitution.py --config configs/15_glass_substitution.yml
"""

import argparse
import copy
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from deeplens.optics.materials import Material


def price_catalog(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return {row["glass"].lower(): row for row in csv.DictReader(handle)}


def optical_distance(old, new, wavelengths_nm):
    """替换前后的折射率差；用于筛选，而非代替重新追迹。"""
    original, candidate = Material(old), Material(new)
    rows = []
    for wavelength in wavelengths_nm:
        wave_um = wavelength / 1000.0
        old_n = original.refractive_index(wave_um)
        new_n = candidate.refractive_index(wave_um)
        rows.append({"wavelength_nm": wavelength, "old_n": old_n,
                     "candidate_n": new_n, "delta_n": new_n - old_n})
    return rows


def make_lens(source, substitutions, prices, wavelengths, max_abs_dn):
    lens = copy.deepcopy(source)
    changes = []
    touched = set()
    for replacement in substitutions:
        index = int(replacement["surface_idx"])
        if index in touched or not 0 <= index < len(lens["surfaces"]):
            raise ValueError(f"Duplicate or invalid surface_idx: {index}")
        touched.add(index)
        surface = lens["surfaces"][index]
        old = surface["mat2"].lower()
        expected = replacement["from"].lower()
        new = replacement["to"].lower()
        if old != expected:
            raise ValueError(f"Surface {index}: expected {expected}, found {old}")
        if old not in prices or new not in prices:
            raise ValueError(f"Missing relative price for {old} or {new}")
        # CDGM 的 A/MA 仅表示价格表中的熔炼频次类别，不等于供货承诺。
        if prices[new]["melt_frequency"] not in {"A", "MA"}:
            raise ValueError(f"{new} is not in the configured A/MA screening group")
        dispersion = optical_distance(old, new, wavelengths)
        largest = max(abs(row["delta_n"]) for row in dispersion)
        if largest > max_abs_dn:
            raise ValueError(f"{old} -> {new}: |delta n|={largest:.5f} exceeds {max_abs_dn}")
        surface["mat2"] = new
        changes.append({"surface_idx": index, "from": old, "to": new,
                        "old_relative_price": float(prices[old]["relative_price"]),
                        "new_relative_price": float(prices[new]["relative_price"]),
                        "candidate_melt_frequency": prices[new]["melt_frequency"],
                        "index_by_wavelength": dispersion})
    return lens, changes


def draw(results, output):
    names = list(results)
    reports = [results[name]["report"]["optimized"] for name in names]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    costs = [r["cost_proxy_hk9l_mm3"] for r in reports]
    rates = [r["yield_estimate"] for r in reports]
    rms = [r["nominal_rms_mm"] for r in reports]
    axes[0].bar(x, costs, color="#0d9488")
    axes[0].set(title="Relative glass-blank cost index", ylabel="Relative units",
                ylim=(0, max(costs) * 1.16))
    for i, cost in enumerate(costs):
        axes[0].text(i, cost * 1.01, f"{cost:,.0f}", ha="center")
    axes[1].bar(x, rates, color="#2563eb")
    axes[1].set(title="Same 128 tolerance samples", ylabel="RMS proxy pass fraction",
                ylim=(0, 1.10))
    for i, rate in enumerate(rates):
        axes[1].text(i, min(rate + .025, 1.04), f"{rate:.1%}", ha="center")
    axes[2].bar(x, rms, color="#d97706")
    axes[2].set(title="Nominal RGB spot RMS", ylabel="mm", ylim=(0, max(rms) * 1.18))
    for i, value in enumerate(rms):
        axes[2].text(i, value * 1.02, f"{value:.4f}", ha="center")
    for ax in axes:
        ax.set_xticks(x, names, rotation=15)
        ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "glass_substitution_comparison.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/15_glass_substitution.yml")
    args = parser.parse_args()
    settings = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base = yaml.safe_load(Path(settings["base_config"]).read_text(encoding="utf-8"))
    source = json.loads(Path(base["input_lens"]).read_text(encoding="utf-8"))
    prices = price_catalog(base["glass_catalog"])
    output = Path(settings["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for variant in settings["variants"]:
        name = variant["name"]
        if not name.replace("_", "").isalnum() or name in results:
            raise ValueError("Variant names must be unique letters/digits/underscores")
        folder = output / name
        folder.mkdir(exist_ok=True)
        lens, changes = make_lens(source, variant.get("substitutions", []), prices,
                                  settings["wavelengths_nm"],
                                  float(settings["max_abs_delta_n"]))
        lens_path = folder / "starting_lens.json"
        lens_path.write_text(json.dumps(lens, indent=2), encoding="utf-8")
        config = copy.deepcopy(base)
        config["input_lens"] = str(lens_path)
        config["output_dir"] = str(folder)
        config["training"]["objective"] = "nominal_only"
        config["visualization"] = {"lens_layout": False, "summary_plots": False}
        config_path = folder / "run_config.yml"
        config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
                               encoding="utf-8")
        print(f"Running {name}: {changes}", flush=True)
        subprocess.run([sys.executable, "8_manufacturing_design.py", "--config",
                        str(config_path)], check=True)
        report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        results[name] = {"substitutions": changes, "report": report}
    # 保存每种组合的完整报告路径和紧凑汇总，便于检查是否同时达标。
    summary = {"comparison": "real catalog glasses, same geometry initialization, training and Monte Carlo seeds",
               "variants": {name: {"substitutions": item["substitutions"],
                                    "baseline": {k: v for k, v in item["report"]["baseline"].items()
                                                 if k != "rms_samples_mm"},
                                    "optimized": {k: v for k, v in item["report"]["optimized"].items()
                                                  if k != "rms_samples_mm"}}
                            for name, item in results.items()}}
    (output / "glass_substitution_report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    draw(results, output)
    for name, item in results.items():
        r = item["report"]["optimized"]
        print(f"{name}: cost={r['cost_proxy_hk9l_mm3']:.2f}, "
              f"pass={r['yield_estimate']:.1%}, nominal RMS={r['nominal_rms_mm']:.5f}")


if __name__ == "__main__":
    main()
