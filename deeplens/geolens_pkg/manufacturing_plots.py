"""把数值报告画成可直接查看的 PNG；不参与光学优化。"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_summary_plots(report, output_dir, dpi=160):
    """生成误差分布、通过率与成本、训练曲线三张图。"""
    output = Path(output_dir)
    baseline, optimized = report["baseline"], report["optimized"]
    threshold = baseline["rms_threshold_mm"]

    # 同一验收线下比较整个分布，而不只看平均值。
    fig, ax = plt.subplots(figsize=(8, 5))
    arrays = [baseline["rms_samples_mm"], optimized["rms_samples_mm"]]
    low, high = min(map(min, arrays)), max(map(max, arrays))
    bins = np.linspace(low, high, 25) if high > low else 10
    ax.hist(arrays[0], bins=bins, alpha=0.55, label="Before", color="#64748b")
    ax.hist(arrays[1], bins=bins, alpha=0.55, label="After", color="#0d9488")
    ax.axvline(threshold, color="#c2410c", linestyle="--", linewidth=2,
               label=f"Pass limit {threshold:.3f} mm")
    ax.set(xlabel="RGB spot RMS (mm)", ylabel="Monte Carlo count",
           title="Manufacturing error: RMS distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "rms_distribution.png", dpi=dpi)
    plt.close(fig)

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    rates = [baseline["yield_estimate"], optimized["yield_estimate"]]
    intervals = [baseline["yield_wilson95"], optimized["yield_wilson95"]]
    errors = [[rates[i] - intervals[i][0] for i in range(2)],
              [intervals[i][1] - rates[i] for i in range(2)]]
    left.bar(["Before", "After"], rates, color=["#64748b", "#0d9488"])
    left.errorbar([0, 1], rates, yerr=errors, fmt="none", color="#1f2937", capsize=5)
    left.set(ylim=(0, 1), ylabel="Pass fraction", title="RMS proxy pass rate")
    for i, rate in enumerate(rates):
        left.text(i, min(intervals[i][1] + 0.02, 0.98), f"{rate:.1%}", ha="center")
    costs = [baseline["cost_proxy_hk9l_mm3"], optimized["cost_proxy_hk9l_mm3"]]
    change = (costs[1] / costs[0] - 1) * 100
    # 成本差很小时，截断坐标轴会夸大变化；直接列值更诚实也更易读。
    right.axis("off")
    right.set_title("Relative glass-blank cost index")
    right.text(0.08, 0.70, f"Before: {costs[0]:,.2f}", transform=right.transAxes,
               fontsize=13, color="#475569")
    right.text(0.08, 0.53, f"After:  {costs[1]:,.2f}", transform=right.transAxes,
               fontsize=13, color="#0d9488")
    right.text(0.08, 0.33, f"Change: {change:+.3f}%", transform=right.transAxes,
               fontsize=14, weight="bold")
    right.text(0.08, 0.16, "H-K9L-equivalent mm³; not currency",
               transform=right.transAxes, fontsize=9)
    fig.tight_layout()
    fig.savefig(output / "pass_rate_and_cost.png", dpi=dpi)
    plt.close(fig)

    history = report["history"]
    if history:
        fig, ax = plt.subplots(figsize=(8, 5))
        steps = [row["step"] for row in history]
        ax.plot(steps, [row["nominal_rms_mm"] for row in history], label="Nominal RMS")
        if any(row["perturbed_rms_mean_mm"] is not None for row in history):
            ax.plot(steps, [row["perturbed_rms_mean_mm"] for row in history],
                    label="Sampled error RMS", alpha=0.8)
        ax.axhline(threshold, color="#c2410c", linestyle="--", label="Pass limit")
        ax.set(xlabel="Optimization step", ylabel="RGB spot RMS (mm)",
               title="Training progress")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "training_progress.png", dpi=dpi)
        plt.close(fig)
