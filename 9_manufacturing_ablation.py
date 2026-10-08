"""一键做三组消融实验，并用完全相同的误差样本做公平比较。

运行：python 9_manufacturing_ablation.py --config configs/9_manufacturing_ablation.yml
"""

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


def compact(result):
    """完整随机样本留在各组 report.json；总表只保存方便阅读的汇总。"""
    return {key: value for key, value in result.items() if key != "rms_samples_mm"}


def plot_comparison(results, output):
    """通过率、标称光斑和材料代理成本并排呈现。"""
    labels = list(results)
    reports = [results[name] for name in labels]
    colors = ["#64748b", "#0d9488", "#d97706"][:len(labels)]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    x = np.arange(len(labels))

    rates = [r["optimized"]["yield_estimate"] for r in reports]
    bounds = [r["optimized"]["yield_wilson95"] for r in reports]
    errors = [[max(0, rates[i] - bounds[i][0]) for i in range(len(labels))],
              [max(0, bounds[i][1] - rates[i]) for i in range(len(labels))]]
    axes[0].bar(x, rates, color=colors)
    axes[0].errorbar(x, rates, yerr=errors, fmt="none", color="#1f2937", capsize=4)
    axes[0].set(ylim=(0, 1.05), ylabel="RMS proxy pass fraction", title="Same tolerance samples")
    for i, value in enumerate(rates):
        axes[0].text(i, min(bounds[i][1] + 0.02, 1.02), f"{value:.1%}", ha="center")

    rms = [r["optimized"]["nominal_rms_mm"] for r in reports]
    axes[1].bar(x, rms, color=colors)
    axes[1].set(ylabel="RGB spot RMS (mm)", title="Nominal image quality")
    for i, value in enumerate(rms):
        axes[1].text(i, value + 0.001, f"{value:.4f}", ha="center")
    axes[1].set_ylim(0, max(rms) * 1.18)

    reference = reports[0]["optimized"]["cost_proxy_hk9l_mm3"]
    cost_delta = [(r["optimized"]["cost_proxy_hk9l_mm3"] / reference - 1) * 100
                  for r in reports]
    axes[2].bar(x, cost_delta, color=colors)
    axes[2].axhline(0, color="#334155", linewidth=0.8)
    axes[2].set(ylabel="Change vs nominal-only (%)", title="Glass blank cost proxy")
    for i, value in enumerate(cost_delta):
        axes[2].annotate(f"{value:+.4f}%", (i, value), xytext=(0, 5 if value >= 0 else -15),
                         textcoords="offset points", ha="center")
    span = max(max(abs(value) for value in cost_delta) * 1.8, 0.005)
    axes[2].set_ylim(-span, span)

    good_cost = [r["optimized"].get("expected_good_unit_cost_proxy") for r in reports]
    axes[3].bar(x, [v if v is not None else 0 for v in good_cost], color=colors)
    axes[3].set(ylabel="Build cost / hard pass fraction", title="Expected cost per good unit")
    for i, value in enumerate(good_cost):
        axes[3].text(i, (value or 0) * 1.01, f"{value:.0f}" if value is not None else "no pass",
                     ha="center", fontsize=8)
    axes[3].set_ylim(0, max([v or 0 for v in good_cost] + [1]) * 1.18)

    for ax in axes:
        ax.set_xticks(x, labels, rotation=18)
    fig.suptitle("Manufacturing objective ablation (demo RMS criterion)")
    fig.tight_layout()
    fig.savefig(output / "ablation_comparison.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/9_manufacturing_ablation.yml")
    args = parser.parse_args()
    settings = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base = yaml.safe_load(Path(settings["base_config"]).read_text(encoding="utf-8"))
    root = Path(settings["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    reports = {}

    for variant in settings["variants"]:
        name = variant["name"]
        if not name.replace("_", "").isalnum() or name in reports:
            raise ValueError(f"Variant names must be unique letters/digits/underscores: {name}")
        # 除两个目标权重外，初始镜头、优化步数和随机种子完全相同。
        config = copy.deepcopy(base)
        config["output_dir"] = str(root / name)
        config["training"]["robustness_weight"] = float(variant["robustness_weight"])
        config["training"]["cost_weight"] = float(variant["cost_weight"])
        if "objective" in variant:
            config["training"]["objective"] = variant["objective"]
        config["visualization"] = {"lens_layout": False, "summary_plots": False}
        variant_config = root / f"{name}_config.yml"
        variant_config.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
                                  encoding="utf-8")
        print(f"Running {name} ...", flush=True)
        # 用同一个 Python 环境启动原脚本，避免复制三份优化代码。
        subprocess.run([sys.executable, "8_manufacturing_design.py", "--config",
                        str(variant_config)], check=True)
        reports[name] = json.loads((root / name / "report.json").read_text(encoding="utf-8"))

    baseline_values = [r["baseline"]["nominal_rms_mm"] for r in reports.values()]
    if max(baseline_values) - min(baseline_values) > 1e-7:
        raise AssertionError("Variants did not start from the same baseline")

    summary = {"comparison": "same starting lens, seed, training length and evaluation samples",
               "variants": {name: {"objective": report.get("training_objective", "legacy"),
                   "training_weights": {
                   "robustness": float(next(v for v in settings["variants"] if v["name"] == name)["robustness_weight"]),
                   "cost": float(next(v for v in settings["variants"] if v["name"] == name)["cost_weight"])},
                   "baseline": compact(report["baseline"]),
                   "optimized": compact(report["optimized"])}
                   for name, report in reports.items()}}
    (root / "ablation_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_comparison(reports, root)
    for name, report in reports.items():
        result = report["optimized"]
        print(f"{name}: RMS={result['nominal_rms_mm']:.5f} mm, "
              f"pass={result['yield_estimate']:.1%}, "
              f"cost proxy={result['cost_proxy_hk9l_mm3']:.2f}")


if __name__ == "__main__":
    main()
