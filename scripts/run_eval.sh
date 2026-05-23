#!/bin/bash
set -euo pipefail

RESULTS_DIR="${1:-checkpoints/latency_aware_rl}"

python evaluation/eval_correctness.py --results_dir "$RESULTS_DIR"
python evaluation/eval_efficiency.py --call_log_dir data/call_logs
python evaluation/eval_transfer.py --config config/base_config.yaml
python evaluation/eval_degradation.py --config config/base_config.yaml
python evaluation/pareto_plot.py --results_file "$RESULTS_DIR/train_metrics.json" --output figures/pareto.png
