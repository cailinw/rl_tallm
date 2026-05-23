#!/bin/bash
set -euo pipefail

CONFIG_PATH="${1:-config/experiment_configs/online_latency_reward.yaml}"

python training/train.py \
  --config config/base_config.yaml \
  --override_config "$CONFIG_PATH" \
  --output_dir checkpoints/latency_aware_rl
