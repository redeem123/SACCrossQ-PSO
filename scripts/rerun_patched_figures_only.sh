#!/bin/zsh
set -euo pipefail

ROOT="/Users/hust-hwashin621m/Desktop/vietanhpaper-2"
OUT_DIR="$ROOT/results/final/patchedfiguresonly"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$OUT_DIR/patched_figures_only_${TS}.log"

mkdir -p "$OUT_DIR"

run_case() {
  local group="$1"
  local scenario_filter="$2"
  local tag="$3"
  local tmp_dir="$OUT_DIR/_tmp_${tag}"

  rm -rf "$tmp_dir"
  mkdir -p "$tmp_dir"

  {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running ${tag}"
    export VIETANH_ALGO_GROUP="$group"
    export VIETANH_SCENARIO_FILTER="$scenario_filter"
    export VIETANH_NUM_RUNS=1
    export VIETANH_POP_SIZE=100
    export VIETANH_MAX_ITERATIONS=1000
    export VIETANH_PARPOOL_WORKERS=14
    export VIETANH_RESULTS_DIR="$tmp_dir"
    export APEXPSO_CONFIG_QUIET=1

    matlab -batch "cd('$ROOT'); run('run_comparison.m');"

    cp "$tmp_dir/plot_Algorithm_Comparison_Scenario1_3D.pdf" "$OUT_DIR/${tag}_3D.pdf"
    cp "$tmp_dir/plot_Algorithm_Comparison_Scenario1_TopView.pdf" "$OUT_DIR/${tag}_TopView.pdf"
    cp "$tmp_dir/Parameter_Tracking_Scenario1.pdf" "$OUT_DIR/${tag}_Parameter_Tracking.pdf"

    if [[ -f "$tmp_dir/Reward_Comparison_Scenario1.pdf" ]]; then
      cp "$tmp_dir/Reward_Comparison_Scenario1.pdf" "$OUT_DIR/${tag}_Reward_Comparison.pdf"
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished ${tag}"
  } >> "$LOG_FILE" 2>&1
}

run_case "rlbased" "chrismasterrain2 - simple" "rlbased_s1"
run_case "rlbased" "chrismasterrain2 - medium" "rlbased_s2"
run_case "others" "chrismasterrain2 - simple" "others_s1"
run_case "others" "chrismasterrain2 - medium" "others_s2"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All patched figure-only runs completed" >> "$LOG_FILE" 2>&1
