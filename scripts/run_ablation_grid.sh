#!/bin/bash
set -euo pipefail

for config in baseline_no_cost.yaml online_latency_reward.yaml; do
  python training/train.py \
    --config config/base_config.yaml \
    --override_config "config/experiment_configs/${config}" \
    --output_dir "checkpoints/${config%.yaml}"
done
