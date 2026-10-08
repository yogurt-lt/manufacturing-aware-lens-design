"""Read-only progress monitor for the overnight manufacturing search."""

import argparse
import json
import os
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path

import yaml


STAGES = {
    "substitution_only": "direct substitution evaluation",
    "nominal_recovery": "nominal RMS recovery",
    "robust_refinement": "yield/cost robust refinement",
}


def duration_text(seconds):
    if seconds is None:
        return "estimating..."
    seconds = max(0, int(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}d {hours:02d}h {minutes:02d}m"
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes}m {seconds:02d}s"


def newest_trial(output):
    trials = list(output.glob("round_*/**/trial_*"))
    if not trials:
        return None
    return max(trials, key=lambda path: max((item.stat().st_mtime for item in path.rglob("*")), default=0))


def stage_status(trial, hammer_mode=False):
    if trial is None:
        return "waiting for the first candidate"
    stages = ({key: value for key, value in STAGES.items() if key != "substitution_only"}
              if hammer_mode else STAGES)
    for folder_name, label in stages.items():
        folder = trial / folder_name
        if not folder.exists():
            return f"preparing {label}"
        if not (folder / "report.json").exists():
            if (folder / "manufacturing_lens.json").exists():
                return f"{label}: optimization saved; evaluating tolerance samples"
            if (folder / "run_config.yml").exists():
                return f"{label}: optimizing"
            return f"{label}: initializing"
    return "all stages for this candidate finished; collecting results"


def gpu_line():
    try:
        command = ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                   "--format=csv,noheader,nounits"]
        value = subprocess.check_output(command, text=True, timeout=4).strip().splitlines()[0]
        name, utilization, used, total = [item.strip() for item in value.split(",")]
        return f"GPU: {name} | utilization {utilization}% | memory {used}/{total} MiB"
    except Exception:
        return "GPU: nvidia-smi unavailable (file-based monitoring still works)"


def timing_estimate(output, current_round, max_rounds, max_trials, stages_per_trial):
    config_files = list(output.glob("round_*/**/run_config.yml"))
    reports = list(output.glob("round_*/**/report.json"))
    all_files = [item for item in output.rglob("*") if item.is_file()]
    if not config_files:
        return {}
    now = time.time()
    start = min(item.stat().st_mtime for item in config_files)
    last_update = max((item.stat().st_mtime for item in all_files), default=start)
    durations = []
    for report in reports:
        config_path = report.parent / "run_config.yml"
        if config_path.exists():
            value = report.stat().st_mtime - config_path.stat().st_mtime
            if value > 0:
                durations.append(value)
    typical = statistics.median(durations) if durations else None
    current_folder = output / f"round_{current_round}"
    current_complete = len(list(current_folder.glob("**/trial_*/**/report.json")))
    total_trial_complete = len([path for path in reports if "trial_" in str(path)])
    current_remaining = max(0, max_trials * stages_per_trial - current_complete)
    total_budget = max_rounds * max_trials * stages_per_trial
    total_remaining = max(0, total_budget - total_trial_complete)
    return {
        "elapsed": now - start,
        "last_update_age": now - last_update,
        "typical_stage": typical,
        "current_round_eta": current_remaining * typical if typical is not None else None,
        "full_budget_eta": total_remaining * typical if typical is not None else None,
        "completed_stage_reports": total_trial_complete,
        "total_budget_stage_reports": total_budget,
    }


def show(config_path):
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = Path(config["output_dir"])
    search = config["search"]
    hammer_mode = "beam_width" in search
    max_rounds = int(search["max_rounds"])
    max_trials = int(search["max_trials_per_round"])
    stages_per_trial = ((2 if int(search.get("robust_steps_per_trial", 0)) > 0 else 1)
                        if hammer_mode else
                        (3 if int(search.get("robust_steps_per_trial", 0)) > 0 else 2))
    print(f"Updated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(gpu_line())
    print(f"Output: {output.resolve()}")
    if not output.exists():
        print("Status: output directory has not been created yet")
        return

    rounds = sorted(output.glob("round_*"), key=lambda path: int(path.name.split("_")[1]))
    current_round = max((int(path.name.split("_")[1]) for path in rounds), default=0)
    trial = newest_trial(output)
    timing = timing_estimate(output, current_round, max_rounds, max_trials, stages_per_trial)
    print(f"Elapsed: {duration_text(timing.get('elapsed'))} | "
          f"last file update {duration_text(timing.get('last_update_age'))} ago")
    print(f"Typical completed stage: {duration_text(timing.get('typical_stage'))}")
    print(f"Configured budget: up to {max_rounds} rounds x {max_trials} candidates x "
          f"{stages_per_trial} stages")
    print(f"Position: round {current_round}/{max_rounds}"
          + (f" | {trial.name}" if trial else ""))
    print(f"Active stage: {stage_status(trial, hammer_mode)}")

    progress_path = output / f"round_{current_round}" / (
        "progress.json" if hammer_mode else "candidate_progress.json")
    rows = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else []
    completed_trials = len(list((output / f"round_{current_round}").glob("**/trial_*"))) if hammer_mode else len(
        {row["name"].rsplit("_", 1)[0] for row in rows})
    print(f"Candidates summarized in this round: {completed_trials}/{max_trials} "
          "(the actual candidate pool may be smaller)")
    print(f"Completed candidate-stage reports: {timing.get('completed_stage_reports', 0)}/"
          f"{timing.get('total_budget_stage_reports', 0)} configured upper bound")
    print(f"ETA, current-round upper bound: {duration_text(timing.get('current_round_eta'))}")
    print(f"ETA, full configured budget: {duration_text(timing.get('full_budget_eta'))}")
    print("ETA is a rough upper-bound estimate from the median completed-stage time; "
          "early stopping or a smaller proposal pool can finish much sooner.")

    if rows:
        print("\nRecently completed stages:")
        print("  candidate stage                         cost proxy       yield    RMS p90/mm  center MTF   decision")
        for row in rows[-8:]:
            reasons = row.get("feasibility_reasons", row.get("rejection_reasons", []))
            decision = "PASS" if not reasons else "REJECT"
            mtf = row.get("differentiable_mtf_worst")
            mtf_text = f"{mtf:.4f}" if mtf is not None else "n/a"
            print(f"  {row['name'][:38]:<38} {row['build_cost_proxy']:>10.2f}"
                  f" {row['demo_yield']:>11.1%} {row['rms_p90_mm']:>12.5f}"
                  f" {mtf_text:>11}   {decision}")

    report_path = output / ("hammer_report.json" if hammer_mode else "iteration_report.json")
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        selected = report["final_best"] if hammer_mode else report["history"][-1]["selected"]
        selected_mtf = selected.get("differentiable_mtf_worst")
        mtf_suffix = f" | center MTF {selected_mtf:.4f}" if selected_mtf is not None else ""
        print("\nBest retained design from the latest completed round:")
        print(f"  {selected['name']} | cost {selected['build_cost_proxy']:.2f} | "
              f"yield {selected['demo_yield']:.1%} | RMS p90 {selected['rms_p90_mm']:.5f} mm"
              f"{mtf_suffix}")
        print(f"  lens: {selected['lens_json']}")
        if hammer_mode:
            cheapest = report.get("lowest_cost_feasible")
            if cheapest and cheapest.get("node_id") != selected.get("node_id"):
                print("  Lowest-cost feasible design found so far:")
                print(f"    {cheapest['name']} | cost {cheapest['build_cost_proxy']:.2f} | "
                      f"yield {cheapest['demo_yield']:.1%} | "
                      f"RMS p90 {cheapest['rms_p90_mm']:.5f} mm")
            last = report["history"][-1]
            print(f"  active beam paths: {len(last['beam'])} | unique material states: "
                  f"{report['unique_material_states_evaluated']} | stale rounds: "
                  f"{last.get('stale_rounds', 0)}/{search['patience_rounds']}")

    control_progress = output / "matched_fixed_material_control" / "progress.json"
    if config.get("matched_fixed_material_control", {}).get("enabled", False):
        completed = (len(json.loads(control_progress.read_text(encoding="utf-8")))
                     if control_progress.exists() else 0)
        selected_round = 0
        if report_path.exists():
            selected_round = json.loads(report_path.read_text(encoding="utf-8"))["final_best"]["round_created"]
        print(f"\nMatched fixed-material control: {completed}/{max(1, selected_round)} cycles completed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/19_overnight_rms_search.yml")
    parser.add_argument("--watch", action="store_true",
                        help="refresh continuously; Ctrl+C stops only this monitor")
    parser.add_argument("--interval", type=float, default=15.0)
    args = parser.parse_args()
    path = Path(args.config)
    while True:
        if args.watch:
            os.system("cls" if os.name == "nt" else "clear")
        show(path)
        if not args.watch:
            break
        print(f"\nRefreshing every {args.interval:g} seconds. Ctrl+C stops only this monitor; "
              "the training process keeps running.")
        try:
            time.sleep(max(args.interval, 2.0))
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
