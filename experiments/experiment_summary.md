# Experiment Summary

All experiments are run from the repo root `/home/cailinw/rl_tallm` with the
virtualenv activated:

```bash
cd /home/cailinw/rl_tallm
source .venv/bin/activate
```

The main evaluation script is `experiments/catp_online_grpo.py`.  
The three-way comparison aggregation script is `experiments/compare_catp_methods.py`.  
The α-ablation aggregation script is `experiments/compare_alpha_ablation.py`.

---

## Experiment 1 — Three-way comparison (main result)

Compares Qwen zero-shot baseline, Qwen online GRPO (α=0.5), and CATP-LLM offline
on 8 evaluation tasks across 325–327 aligned samples.

### Results location

```
experiments/results/catp_three_way_compare_paperlike_qwen25_3b_8tasks_327eval/
  aligned_final_metrics.json      ← per-method summary numbers
  aligned_final_metrics.csv
  aligned_per_task_metrics.json   ← per-task breakdown
  aligned_per_task_metrics.csv
  aligned_final_metrics.png
  aligned_latency_cost_metrics.png
  aligned_score_latency_tradeoff.png
  aligned_per_task_score_reward.png
  aligned_per_task_latency.png
  catp_offline_training_loss.png
  training_progress.png
  training_curves.csv
```

### Key numbers (325 aligned samples, 8 eval tasks: 13,20,21,31,36,40,51,61)

| Method              | Score  | Time (ms) | Cost   | Reward  | Valid rate |
|---------------------|--------|-----------|--------|---------|------------|
| Qwen baseline       | 0.402  | 662       | 0.233  | 0.114   | 90.5%      |
| Qwen online GRPO    | 0.582  | 104       | 0.013  | 0.274   | 100%       |
| CATP-LLM offline    | 0.157  | 737       | 0.303  | −0.018  | 87.1%      |

### GRPO training config (adapter used for eval)

Adapter path: `experiments/results/catp_qwen_online_compare_full_16x50_164eval_retry/adapter`

Config (`config.json` in same dir):

| Parameter       | Value                                      |
|-----------------|--------------------------------------------|
| model           | Qwen/Qwen2.5-3B-Instruct                   |
| train_task_ids  | 1,2,3,4,5,7,9,10,11,14,15,16,17,18,19,26  |
| train_steps     | 50                                         |
| group_size (G)  | 2                                          |
| alpha           | 0.5                                        |
| latency_weight  | 0.1                                        |
| lr              | 5e-5                                       |
| use_lora        | true (QLoRA NF4 4-bit)                     |
| max_new_tokens  | 96                                         |
| max_tools       | 2                                          |
| seed            | 42                                         |

### CATP-LLM offline config

Config file: `external/OpenCATP-LLM/src/catpllm/data/config_data/qwen25_3b_compare.yaml`

| Parameter              | Value                                              |
|------------------------|----------------------------------------------------|
| model                  | Qwen/Qwen2.5-3B-Instruct                           |
| rank                   | 64                                                 |
| epochs                 | 2                                                  |
| episodes_per_epoch     | 1400                                               |
| gamma                  | 0.9                                                |
| alpha                  | 0.5                                                |
| lr                     | 1e-4                                               |
| batch_size             | 1 (grad_accum=4)                                   |
| train_plan_pool        | `src/catpllm/data/training_data/seq_plan_pool.pkl` |
| test_task_list         | [20, 21, 31, 36]                                   |
| scheduled_sampling_rate| 0.1                                                |

### How to reproduce

**Step 1 — Train GRPO (already done, adapter saved):**
```bash
python experiments/catp_online_grpo.py \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --load_in_4bit --use_lora \
  --train_steps 50 --group_size 2 \
  --alpha 0.5 --latency_weight 0.1 \
  --train_task_ids 1,2,3,4,5,7,9,10,11,14,15,16,17,18,19,26 \
  --n_train 16 --n_eval 4 \
  --lr 5e-5 --max_new_tokens 96 --max_tools 2 \
  --seed 42 --dtype bfloat16 \
  --output_dir experiments/results/catp_qwen_online_compare_full_16x50_164eval_retry
```

**Step 2 — Evaluate all three methods:**
```bash
python experiments/catp_online_grpo.py \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --load_in_4bit \
  --alpha 0.5 --latency_weight 0.1 \
  --eval_task_ids 13,20,21,31,36,40,51,61 \
  --n_eval 400 --dtype bfloat16 \
  --load_adapter experiments/results/catp_qwen_online_compare_full_16x50_164eval_retry/adapter \
  --output_dir experiments/results/catp_qwen_online_loaded_8tasks_327eval
```

**Step 3 — Run CATP-LLM offline** (run from OpenCATP-LLM directory):
```bash
cd external/OpenCATP-LLM
python -m src.catpllm.main \
  --config src/catpllm/data/config_data/qwen25_3b_compare.yaml
```

**Step 4 — Aggregate and plot:**
```bash
python experiments/compare_catp_methods.py \
  --grpo_dir   experiments/results/catp_qwen_online_loaded_8tasks_327eval \
  --catp_dir   external/OpenCATP-LLM/results/<run_dir> \
  --output_dir experiments/results/catp_three_way_compare_paperlike_qwen25_3b_8tasks_327eval
```

---

## Experiment 2 — α ablation (Online GRPO only)

Trains and evaluates Online GRPO with α ∈ {0.2, 0.5, 0.8}, all other settings
fixed (λ=0.1, 50 steps, seed=42, same 16 training tasks).

### Results location

| α   | Training dir                                                       | Eval dir                                           |
|-----|--------------------------------------------------------------------|----------------------------------------------------|
| 0.2 | `experiments/results/catp_grpo_alpha_0.2_train/`                   | `experiments/results/catp_grpo_alpha_0.2_eval/`    |
| 0.5 | `experiments/results/catp_qwen_online_compare_full_16x50_164eval_retry/` | `experiments/results/catp_qwen_online_loaded_8tasks_327eval/` |
| 0.8 | `experiments/results/catp_grpo_alpha_0.8_train/`                   | `experiments/results/catp_grpo_alpha_0.8_eval/`    |

Aggregated plots and per-task CSV:
```
experiments/results/alpha_ablation/
  alpha_ablation_overall.png        ← bar chart: score/time/cost/reward per α
  alpha_ablation_per_task_score.png ← per-task score breakdown
  alpha_ablation_per_task_time.png  ← per-task latency breakdown
  alpha_ablation_per_task.csv       ← raw per-task numbers
  training_curves_overlay.png       ← reward + loss curves for all three α
```

Each training/eval dir contains:
- `config.json` — full argument dump
- `train_metrics.csv` — step, mean_reward, policy_loss (training dirs only)
- `episodes.csv` — per-sample results with columns: policy, split, task_id, sample_id, valid, task_score, cost_price, exec_time, qop, reward
- `summary.json` — aggregate metrics
- `adapter/` — saved LoRA weights (training dirs only)

### Key numbers (327 aligned GRPO samples, 8 eval tasks: 13,20,21,31,36,40,51,61)

| α         | Score   | Time (ms) | Cost   | Reward  | Task 40 score |
|-----------|---------|-----------|--------|---------|---------------|
| 0.2       | −0.090  | 592       | 0.595  | −0.275  | 0.872         |
| 0.5 (main)| 0.582   | 104       | 0.013  | 0.274   | 0.332         |
| 0.8       | 0.205   | 708       | 0.364  | 0.179   | 0.872         |
| Baseline  | 0.402   | 662       | 0.233  | 0.114   | 0.881         |

### Shared training config (α=0.2 and α=0.8)

| Parameter       | Value                                      |
|-----------------|--------------------------------------------|
| model           | Qwen/Qwen2.5-3B-Instruct                   |
| train_task_ids  | 1,2,3,4,5,7,9,10,11,14,15,16,17,18,19,26  |
| train_steps     | 50                                         |
| group_size (G)  | 2                                          |
| latency_weight  | 0.1                                        |
| lr              | 5e-5                                       |
| use_lora        | true (QLoRA NF4 4-bit)                     |
| max_new_tokens  | 192                                        |
| max_tools       | 5                                          |
| seed            | 42                                         |

> **Note:** α=0.5 used `max_new_tokens=96` and `max_tools=2` (earlier experiment);
> α=0.2 and α=0.8 used `max_new_tokens=192` and `max_tools=5`. This is a minor
> confound in the ablation.

### How to reproduce

```bash
# Train α=0.2
python experiments/catp_online_grpo.py \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --load_in_4bit --use_lora \
  --train_steps 50 --group_size 2 \
  --alpha 0.2 --latency_weight 0.1 \
  --train_task_ids 1,2,3,4,5,7,9,10,11,14,15,16,17,18,19,26 \
  --n_train 4 --n_eval 4 --seed 42 --dtype bfloat16 \
  --output_dir experiments/results/catp_grpo_alpha_0.2_train

# Eval α=0.2 (load saved adapter)
python experiments/catp_online_grpo.py \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --load_in_4bit --alpha 0.2 --latency_weight 0.1 \
  --eval_task_ids 13,20,21,31,36,40,51,61 \
  --n_eval 400 --dtype bfloat16 \
  --load_adapter experiments/results/catp_grpo_alpha_0.2_train/adapter \
  --output_dir experiments/results/catp_grpo_alpha_0.2_eval

# Repeat with --alpha 0.8 and corresponding output dirs for α=0.8

# Aggregate and plot
MPLBACKEND=Agg python experiments/compare_alpha_ablation.py \
  --output_dir    experiments/results/alpha_ablation \
  --alpha_02_dir  experiments/results/catp_grpo_alpha_0.2_eval \
  --alpha_05_dir  experiments/results/catp_qwen_online_loaded_8tasks_327eval \
  --alpha_08_dir  experiments/results/catp_grpo_alpha_0.8_eval
```

---

## Paper figures

All figures used in the paper live under `paper/figs/`:

| File | Source experiment |
|------|-------------------|
| `training_progress.png` | Three-way comparison (Exp 1) |
| `catp_offline_training_loss.png` | Three-way comparison (Exp 1) |
| `aligned_final_metrics.png` | Three-way comparison (Exp 1) |
| `aligned_latency_cost_metrics.png` | Three-way comparison (Exp 1) |
| `aligned_score_latency_tradeoff.png` | Three-way comparison (Exp 1) |
| `aligned_per_task_score_reward.png` | Three-way comparison (Exp 1) |
| `aligned_per_task_latency.png` | Three-way comparison (Exp 1) |
| `alpha_ablation_overall.png` | α ablation (Exp 2) |
| `alpha_ablation_per_task_score.png` | α ablation (Exp 2) |
| `alpha_ablation_per_task_time.png` | α ablation (Exp 2) |
| `training_curves_overlay.png` | α ablation training curves (Exp 2) |

---

## Known limitations / confounds

- **No multiple seeds.** Each configuration was run once. No confidence intervals.
- **Train set = eval set.** The 8 evaluation tasks overlap with the 16 training tasks for Exp 1. No held-out test tasks.
- **α=0.5 config differs slightly** from α=0.2/0.8 (`max_new_tokens=96` vs 192, `max_tools=2` vs 5).
- **α=0.2 and α=0.8 did not converge** during training (reward flat at −1.0 for ~40 of 50 steps); their eval results are closer to zero-shot than to a properly trained policy.
- **CATP-LLM offline** was evaluated only on tasks 20,21,31,36 (its native test set); scores on the other 4 tasks (13,36,40,51,61) come from a separate generalization eval not part of its original setup.
