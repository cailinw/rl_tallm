#!/usr/bin/env bash
# Run online GRPO α ablation: train α=0.2 then α=0.8, eval each, then aggregate.
# Each α is a two-step process:
#   1. Train  → saves LoRA adapter to  results/catp_grpo_alpha_X.Y/adapter/
#   2. Eval   → loads adapter, evals on 8 tasks / 327 samples
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source .venv/bin/activate

TRAIN_TASKS="1,2,3,4,5,7,9,10,11,14,15,16,17,18,19,26"
EVAL_TASKS="13,20,21,31,36,40,51,61"

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

run_alpha() {
    local ALPHA=$1
    local TAG="catp_grpo_alpha_${ALPHA}"
    echo ""
    echo "=========================================="
    echo " α = ${ALPHA}  — TRAINING"
    echo "=========================================="
    python experiments/catp_online_grpo.py \
        $TRAIN_ARGS \
        --alpha "$ALPHA" \
        --output_dir "experiments/results/${TAG}_train"

    echo ""
    echo "=========================================="
    echo " α = ${ALPHA}  — EVALUATION (327 samples)"
    echo "=========================================="
    python experiments/catp_online_grpo.py \
        $EVAL_ARGS \
        --alpha "$ALPHA" \
        --load_adapter "experiments/results/${TAG}_train/adapter" \
        --output_dir "experiments/results/${TAG}_eval"
}

run_alpha 0.2
run_alpha 0.8

echo ""
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
