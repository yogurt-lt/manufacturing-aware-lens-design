"""Bootstrap uncertainty for the saved 1024-sample formal ablation.

The four methods were evaluated with common random numbers, so confidence
intervals for method differences use paired resampling of sample indices.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/paper/formal_ablation_report.json"
OUTPUT = ROOT / "results/paper/bootstrap_p90_ci.json"
N_BOOT = 20000
SEED = 20260930


def main() -> None:
    report = json.loads(SOURCE.read_text(encoding="utf-8"))
    samples = {name: np.asarray(values, dtype=np.float64)
               for name, values in report["rms_samples_mm"].items()}
    sizes = {len(values) for values in samples.values()}
    if sizes != {1024}:
        raise RuntimeError(f"Expected four 1024-sample arrays, got {sizes}")

    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, 1024, size=(N_BOOT, 1024), endpoint=False)
    bootstrap = {
        name: np.quantile(values[indices], 0.90, axis=1, method="linear")
        for name, values in samples.items()
    }

    summary = {}
    for name, values in samples.items():
        p90 = float(np.quantile(values, 0.90, method="linear"))
        lo, hi = np.quantile(bootstrap[name], [0.025, 0.975], method="linear")
        summary[name] = {
            "rms_p90_mm": p90,
            "bootstrap_95_ci_mm": [float(lo), float(hi)],
        }

    comparisons = {}
    for other in ("Nominal only", "No soft yield", "Greedy glass"):
        # Positive values mean the full method has lower RMS p90.
        diff = bootstrap[other] - bootstrap["Full method"]
        lo, hi = np.quantile(diff, [0.025, 0.975], method="linear")
        comparisons[f"Full method vs {other}"] = {
            "point_difference_mm": summary[other]["rms_p90_mm"] - summary["Full method"]["rms_p90_mm"],
            "paired_bootstrap_95_ci_mm": [float(lo), float(hi)],
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
        }

    output = {
        "source": str(SOURCE.relative_to(ROOT)),
        "bootstrap_resamples": N_BOOT,
        "bootstrap_seed": SEED,
        "quantile": 0.90,
        "interval": "percentile bootstrap, 95%",
        "common_sample_pairing": True,
        "methods": summary,
        "paired_comparisons": comparisons,
    }
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
