"""Merge per-GPU CSV outputs into a single results directory."""

import argparse
import shutil
from pathlib import Path

import pandas as pd

OUTPUT_DIRS = {
    # generation experiment
    "sanity":               "results/generation_sanity",
    "thorough":             "results/generation_thorough",
    "full":                 "results/generation",
    "focused":              "results/generation_focused",
    # capping experiment
    "cap_sanity":           "results/capping_sanity",
    "cap_light":            "results/capping_light",
    "cap_full":             "results/capping_full",
    "cap_paper":            "results/capping_paper",
    "cap_light_raw":        "results/capping_light_raw",
    # cross-axis capping
    "cap_cross_sanity":     "results/capping_cross_sanity",
    "cap_cross_axis":       "results/capping_cross_axis",
    "cap_cross_full":       "results/capping_cross_full",
}

# Columns present in all experiments; missing ones are skipped gracefully
SORT_COLS = ["prompt_idx", "step"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=list(OUTPUT_DIRS), required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    args = parser.parse_args()

    tmp_dirs = [Path(f"_parallel_tmp/gpu{g}") for g in args.gpus]
    final_dir = Path(OUTPUT_DIRS[args.preset])
    final_dir.mkdir(parents=True, exist_ok=True)

    # Files that are sliced across GPUs and need concatenation
    merge_files = [
        "assistant_axis_generations.csv",
        "per_step_metrics.csv",
        "cross_axis_generations.csv",
        "cross_axis_per_step_metrics.csv",
    ]
    for fname in merge_files:
        parts = [pd.read_csv(d / fname) for d in tmp_dirs if (d / fname).exists()]
        if not parts:
            if fname in ("assistant_axis_generations.csv", "per_step_metrics.csv"):
                print(f"  WARNING: no data for {fname}")
            continue
        merged = pd.concat(parts, ignore_index=True)
        cols = [c for c in SORT_COLS if c in merged.columns]
        merged.sort_values(cols, inplace=True)
        merged.to_csv(final_dir / fname, index=False)
        print(f"  {fname}: {len(merged)} rows")

    # Capability eval files are identical across GPUs (same calibration prompts),
    # so just copy from the first available GPU
    for cap_fname in ["assistant_axis_capability_eval.csv", "cross_axis_capability_eval.csv"]:
        for d in tmp_dirs:
            cap_eval = d / cap_fname
            if cap_eval.exists():
                shutil.copy2(cap_eval, final_dir / cap_fname)
                print(f"  {cap_fname} copied from {d.name}")
                break

    version_src = tmp_dirs[0] / "version.json"
    if version_src.exists():
        shutil.copy2(version_src, final_dir / "version.json")
        print(f"  version.json copied")

    print(f"Results: {final_dir}/")


if __name__ == "__main__":
    main()
