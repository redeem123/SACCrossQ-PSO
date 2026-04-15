#!/bin/zsh
set -euo pipefail

repo_root="/Users/hust-hwashin621m/Desktop/vietanhpaper-2"

if [ "$#" -ge 1 ]; then
  output_dir="$1"
else
  timestamp="$(date +%F_%H-%M-%S)"
  output_dir="$repo_root/outputs/research/autosearch_daemon_${timestamp}"
fi

mkdir -p "$output_dir"
log_path="$output_dir/background_run.log"

export APEXPSO_AUTOSEARCH_OUTPUT_DIR="$output_dir"
cd "$repo_root"

while [ ! -f "$APEXPSO_AUTOSEARCH_OUTPUT_DIR/objective_met.flag" ]; do
  set +e
  matlab -batch "cd('$repo_root'); addpath(genpath('.')); run_afsacpso_autosearch_loop();" >> "$log_path" 2>&1
  rc=$?
  set -e

  if [ -f "$APEXPSO_AUTOSEARCH_OUTPUT_DIR/objective_met.flag" ]; then
    exit 0
  fi

  printf "\n[wrapper] MATLAB exited with code %s at %s\n" "$rc" "$(date '+%F %T')" >> "$log_path"
  sleep 5
done
