#!/usr/bin/env bash
# Resume α ablation: α=0.2 adapter already trained, start from its eval.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source .venv/bin/activate

EVAL_TASKS="13,20,21,31,36,40,51,61"
TRAIN_TASKS="1,2,3,4,5,7,9,10,11,14,15,16,17,18,19,26"

TRAIN_ARGS="
  --model_name Qwen/Qwen2.5-3B-Instruct
  --train_task_ids $TRAIN_TASKS
  --train_steps 50
  --group_size 2
  --load_in_4bit
  --use_lora
  --latency_weight 0.1
  --n_eval 4
  --dtype bfloat16
  --seed 42
"

EVAL_ARGS="
  --model_name Qwen/Qwen2.5-3B-Instruct
  --load_in_4bit
  --latency_weight 0.1
  --eval_task_ids $EVAL_TASKS
  --n_eval 400
  --dtype bfloat16
  --no_force_exit
"

echo "=========================================="
echo " α = 0.2  — EVALUATION (327 samples)"
echo "=========================================="
python experiments/catp_online_grpo.py \
    $EVAL_ARGS \
    --alpha 0.2 \
    --load_adapter "experiments/results/catp_grpo_alpha_0.2_train/adapter" \
    --output_dir "experiments/results/catp_grpo_alpha_0.2_eval"

echo "=========================================="
echo " α = 0.8  — TRAINING"
echo "=========================================="
python experiments/catp_online_grpo.py \
    $TRAIN_ARGS \
    --alpha 0.8 \
    --output_dir "experiments/results/catp_grpo_alpha_0.8_train"

echo "=========================================="
echo " α = 0.8  — EVALUATION (327 samples)"
echo "=========================================="
python experiments/catp_online_grpo.py \
    $EVAL_ARGS \
    --alpha 0.8 \
    --load_adapter "experiments/results/catp_grpo_alpha_0.8_train/adapter" \
    --output_dir "experiments/results/catp_grpo_alpha_0.8_eval"

echo "=========================================="
echo " Aggregating α ablation results"
echo "=========================================="
MPLBACKEND=Agg python experiments/compare_alpha_ablation.py \
  --output_dir    experiments/results/alpha_ablation \
  --alpha_02_dir  experiments/results/catp_grpo_alpha_0.2_eval \
  --alpha_05_dir  experiments/results/catp_qwen_online_loaded_8tasks_327eval \
  --alpha_08_dir  experiments/results/catp_grpo_alpha_0.8_eval

echo ""
echo "All done. Results in experiments/results/alpha_ablation/"
