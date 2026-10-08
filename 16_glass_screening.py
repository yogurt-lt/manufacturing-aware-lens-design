"""合并玻璃替换、制造良率和独立 50 lp/mm 结果，筛选可继续验证的组合。

相对 MTF 门槛是相对于原结构的“不得明显退化”研发筛选线，
不是产品验收指标。最终仍需 Zemax 和真实量产公差验证。
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


def worst_mtf(lens_result):
    return min(screen["best_allowed_focus"]["worst_mtf"]
               for screen in lens_result.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/16_glass_screening.yml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    glass_report = json.loads(Path(config["glass_report"]).read_text(encoding="utf-8"))
    mtf_report = json.loads(Path(config["mtf_report"]).read_text(encoding="utf-8"))
    original = config["reference_variant"]
    base_cost = glass_report["variants"][original]["optimized"]["cost_proxy_hk9l_mm3"]
    base_mtf = worst_mtf(mtf_report["lenses"][original])
    min_relative = float(config["min_relative_mtf"])
    min_yield = float(config["min_demo_yield"])
    rows = []
    for name, item in glass_report["variants"].items():
        outcome = item["optimized"]
        mtf = worst_mtf(mtf_report["lenses"][name])
        rate = outcome["yield_estimate"]
        nominal = outcome["nominal_rms_mm"]
        reasons = []
        if rate < min_yield:
            reasons.append("RMS proxy sample pass fraction below screening line")
        if mtf < base_mtf * min_relative:
            reasons.append("50 lp/mm proxy MTF below relative non-inferiority line")
        if nominal > outcome["rms_threshold_mm"]:
            reasons.append("Nominal RMS exceeds demo limit")
        rows.append({"name": name, "relative_cost": outcome["cost_proxy_hk9l_mm3"] / base_cost,
                     "cost_proxy_hk9l_mm3": outcome["cost_proxy_hk9l_mm3"],
                     "worst_50lpmm_mtf_proxy": mtf,
                     "mtf_relative_to_original": mtf / base_mtf,
                     "demo_yield": rate, "nominal_rms_mm": nominal,
                     "passes_screen": not reasons, "reasons": reasons})
    eligible = [row for row in rows if row["passes_screen"]]
    chosen = min(eligible, key=lambda row: row["relative_cost"])["name"] if eligible else None
    result = {"reference_variant": original,
              "screening_rule": {"min_relative_50lpmm_mtf_proxy": min_relative,
                                 "min_demo_rms_yield": min_yield,
                                 "nominal_rms_must_pass": True},
              "warning": "Relative research screening only; not an absolute product MTF or production yield qualification",
              "selected_for_next_validation": chosen, "candidates": rows}
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "glass_screening.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 5))
    for row in rows:
        color = "#0d9488" if row["passes_screen"] else "#dc2626"
        ax.scatter(row["relative_cost"] * 100, row["worst_50lpmm_mtf_proxy"],
                   s=150, c=color, edgecolors="white", linewidth=1.2)
        ax.annotate(row["name"], (row["relative_cost"] * 100,
                                  row["worst_50lpmm_mtf_proxy"]),
                    xytext=(5, 6), textcoords="offset points", fontsize=9)
    ax.axhline(base_mtf * min_relative, color="#b45309", linestyle="--",
               label=f"Relative MTF screen: {min_relative:.0%} of original")
    ax.axvline(100, color="#64748b", linestyle=":", label="Original cost")
    ax.set(xlabel="Relative glass blank cost proxy (%)",
           ylabel="Worst field 50 lp/mm geometric MTF proxy",
           title="Real glass choices: cost and independent image quality")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "glass_screening.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"selected_for_next_validation": chosen, "candidates": rows}, indent=2))


if __name__ == "__main__":
    main()
