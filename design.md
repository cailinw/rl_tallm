# Latency-Aware RL for Tool-Use LLMs — Implementation Spec
## For use with Cursor / Codex / AI coding assistants

---

## Project Overview

**Goal**: Fine-tune a 7B LLM (Qwen2.5-7B-Instruct) using online GRPO (Group Relative Policy Optimization) to learn cost-aware tool selection policies. The model is rewarded both for task correctness AND for preferring low-latency tools when multiple tools can solve the task.

**Extends**: CATP-LLM (ICCV 2025) — which used offline RL with static cost attributes — by moving to online GRPO with real-time measured latency as a reward signal.

**Key novelty**:
1. Online (not offline) RL with live latency measurement per rollout
2. Latency injected as a contextual observation in the model's reasoning chain
3. Cross-tool-set transfer evaluation (train on categories A/B/C, test on D)
4. Adversarial degradation eval (train on stable APIs, test with artificially throttled latencies)

**Hardware**: RTX 4090 (24 GB VRAM) + GV100 (32 GB VRAM) as separate machines. Use GV100 for training, 4090 for parallel evaluation.

---

## Repository Structure

```
latency-aware-rl/
├── README.md
├── requirements.txt
├── config/
│   ├── base_config.yaml          # default hyperparameters
│   ├── experiment_configs/
│   │   ├── baseline_no_cost.yaml
│   │   ├── online_latency_reward.yaml
│   │   ├── ablation_offline_vs_online.yaml
│   │   └── transfer_eval.yaml
├── data/
│   ├── toolbench_loader.py       # loads ToolBench API subsets
│   ├── api_registry.py           # tool definitions with schema + category tags
│   ├── task_sampler.py           # samples tasks per training/eval split
│   └── splits/
│       train_categories.txt      # e.g. "weather,finance,social"
│       test_categories.txt       # e.g. "travel"
├── environment/
│   ├── tool_env.py               # core RL environment: step(), reset(), render()
│   ├── api_executor.py           # executes real or replayed API calls, logs latency
│   ├── latency_tracker.py        # per-tool rolling mean/std latency tracker
│   └── degradation_wrapper.py   # adversarial wrapper: artificially throttles APIs
├── model/
│   ├── policy.py                 # wraps HuggingFace model + tokenizer
│   ├── context_builder.py        # builds prompt with latency observations injected
│   └── tool_parser.py            # parses model output into structured tool calls
├── training/
│   ├── grpo_trainer.py           # custom GRPO training loop using trl
│   ├── reward.py                 # reward function: task_success + latency penalty
│   ├── rollout_buffer.py         # stores multi-turn trajectories
│   └── train.py                  # main training entry point
├── evaluation/
│   ├── eval_correctness.py       # task success rate on BFCL-v3 / ToolBench subset
│   ├── eval_efficiency.py        # avg tool calls, total latency per task
│   ├── eval_transfer.py          # run on held-out tool categories
│   ├── eval_degradation.py       # run with degradation_wrapper enabled
│   └── pareto_plot.py            # plots accuracy vs. latency Pareto curves
├── baselines/
│   ├── sft_baseline.py           # SFT fine-tune on correctness-only demonstrations
│   ├── correctness_only_rl.py    # GRPO with task_success reward only (no latency)
│   └── catp_offline_rl.py        # offline RL baseline replicating CATP-LLM approach
└── scripts/
    ├── run_training.sh
    ├── run_eval.sh
    └── run_ablation_grid.sh
```

---

## Dependencies

```
# requirements.txt
torch>=2.3.0
transformers>=4.47.0
trl>=0.12.0           # for GRPOTrainer
peft>=0.14.0          # for LoRA
accelerate>=1.0.0
datasets>=3.0.0
bitsandbytes>=0.43.0  # optional: for 4-bit quant fallback on 4090
vllm>=0.6.0           # for fast inference during rollout generation
requests>=2.31.0      # for real API calls
aiohttp>=3.9.0        # for async API calls during rollout
numpy>=1.26.0
pandas>=2.2.0
matplotlib>=3.8.0
wandb>=0.17.0         # experiment tracking
pyyaml>=6.0
```

---

## 1. Tool Environment (`environment/tool_env.py`)

```python
"""
Multi-turn tool-use RL environment.
- Wraps a task (query + available tools) as an RL episode
- Each step: model generates a tool call or final answer
- Environment executes the tool, returns result + latency
- Episode ends when model outputs final answer or max_steps reached
"""

class ToolEnv:
    def __init__(self, api_registry, latency_tracker, max_steps=8):
        """
        api_registry: APIRegistry — maps tool_name -> schema + executor
        latency_tracker: LatencyTracker — maintains rolling latency stats per tool
        max_steps: int — max tool calls per episode before forced termination
        """

    def reset(self, task: dict) -> dict:
        """
        task = {
            "query": str,
            "available_tools": List[str],  # tool names from api_registry
            "ground_truth": str,            # expected final answer (for reward)
            "required_tools": List[str],    # ground-truth minimal tool set (for reward)
            "category": str                 # API category tag, e.g. "weather"
        }
        Returns initial observation dict.
        """

    def step(self, action: dict) -> tuple[dict, float, bool, dict]:
        """
        action = {
            "type": "tool_call" | "final_answer",
            "tool_name": str,           # if type == tool_call
            "tool_args": dict,          # if type == tool_call
            "answer": str               # if type == final_answer
        }
        Returns: (observation, reward, done, info)

        observation includes:
          - tool_result: str (API response)
          - latency_ms: float (measured wall-clock time)
          - latency_tier: str ("FAST" | "MEDIUM" | "SLOW")  <- derived from rolling stats
          - latency_delta: float  <- observed - rolling_mean (positive = slower than usual)
          - step_count: int

        reward: float (0 during episode; final reward assigned at done=True)
        done: bool
        info: dict (diagnostic metadata)
        """

    def compute_episode_reward(self, trajectory: list) -> float:
        """
        Called at episode end.
        reward = r_task - w_latency * sum(normalized_latency_deltas) + r_timeout_penalty
        See reward.py for details.
        """
```

---

## 2. API Executor (`environment/api_executor.py`)

```python
"""
Executes tool calls against real ToolBench RapidAPI endpoints.
Falls back to replaying pre-logged latency traces for reproducibility.

Two modes:
  - LIVE: Makes real HTTP requests, logs latency to disk
  - REPLAY: Uses pre-logged latency distributions per tool (sampled at runtime)
            This avoids rate limits during large training runs while preserving
            realistic latency variance. Pre-log by running collect_latency_traces.py
            once before training.
"""

class APIExecutor:
    def __init__(self, mode: str = "replay", trace_dir: str = "data/latency_traces/"):
        """
        mode: "live" | "replay"
        trace_dir: path to pre-logged latency trace files (one JSON per tool)
        """

    async def call(self, tool_name: str, args: dict) -> dict:
        """
        Returns:
          {
            "result": str,      # API response text
            "latency_ms": float,
            "success": bool,    # False on timeout or HTTP error
            "error": str | None
          }
        Logs every call to data/call_logs/
        """

    def collect_traces(self, tool_names: list, n_calls_per_tool: int = 100):
        """
        Pre-run this ONCE before training to collect real latency distributions.
        Saves to data/latency_traces/{tool_name}.json
        Each file: {"latencies_ms": [...], "timeout_rate": float, "mean": float, "std": float}
        """
```

---

## 3. Latency Tracker (`environment/latency_tracker.py`)

```python
"""
Maintains per-tool rolling statistics for latency normalization.
Used to compute the latency_delta reward signal.
"""

class LatencyTracker:
    def __init__(self, window_size: int = 50):
        """
        window_size: rolling window for mean/std calculation
        Initialized from pre-logged traces if available.
        """

    def update(self, tool_name: str, latency_ms: float):
        """Add new observation to rolling window."""

    def get_tier(self, tool_name: str, latency_ms: float) -> str:
        """
        Returns "FAST" | "MEDIUM" | "SLOW" based on z-score:
          z < -0.5  -> FAST
          z > +0.5  -> SLOW
          else      -> MEDIUM
        """

    def get_delta(self, tool_name: str, latency_ms: float) -> float:
        """Returns (observed - rolling_mean) / rolling_std (z-score)."""

    def get_stats(self, tool_name: str) -> dict:
        """Returns {"mean": float, "std": float, "timeout_rate": float}"""
```

---

## 4. Degradation Wrapper (`environment/degradation_wrapper.py`)

```python
"""
Wraps APIExecutor to simulate API degradation for adversarial evaluation.
ONLY used during evaluation, never during training.
"""

class DegradationWrapper:
    def __init__(self, executor: APIExecutor, config: dict):
        """
        config = {
            "target_tools": List[str],   # which tools to degrade
            "latency_multiplier": float, # e.g. 5.0 = 5x slower
            "timeout_rate": float,       # e.g. 0.15 = 15% timeout rate
            "mode": "uniform" | "spike"  # uniform degradation or random spikes
        }
        """

    async def call(self, tool_name: str, args: dict) -> dict:
        """Applies degradation then calls underlying executor."""
```

---

## 5. Context Builder (`model/context_builder.py`)

```python
"""
Builds the prompt fed to the model at each turn.
Injects latency observations into the reasoning context so the model
can learn to condition its tool selection on cost information.
"""

SYSTEM_PROMPT = """You are a tool-using assistant. For each task, you may call available tools to gather information.
After each tool call, you will see the result along with how fast the tool responded.
Use this information to prefer faster tools when multiple tools can answer the question.
When you have enough information, provide a final answer.

Available tools and their typical speeds will be shown in the task context."""

def build_turn_prompt(
    query: str,
    available_tools: List[dict],   # list of tool schemas
    history: List[dict],           # prior turns: [{tool, args, result, latency_tier, latency_delta}]
    step: int
) -> str:
    """
    Formats the full prompt for the current turn.

    Tool call history is formatted as:
    [Turn 1] Called: web_search(query="...")
    Result: "..."
    ⏱ Response: FAST (0.04s | -1.2σ faster than usual)

    [Turn 2] Called: weather_api(city="...")
    Result: "..."
    ⏱ Response: SLOW (2.1s | +1.8σ slower than usual)

    The latency tier and sigma offset are the key contextual signals.
    """

def parse_model_output(text: str) -> dict:
    """
    Parses model output into structured action.
    Handles both <tool_call>...</tool_call> XML format and JSON format.
    Returns action dict compatible with ToolEnv.step()
    """
```

---

## 6. Reward Function (`training/reward.py`)

```python
"""
Episode-level reward combining task correctness, latency efficiency, and timeout penalties.
"""

def compute_reward(
    trajectory: list,       # list of step dicts from ToolEnv
    final_answer: str,
    ground_truth: str,
    required_tools: list,
    config: dict
) -> dict:
    """
    Returns {
        "total": float,
        "r_task": float,
        "r_latency": float,
        "r_timeout": float,
        "breakdown": dict    # for logging
    }

    Reward formula:
      r_total = r_task - w_latency * sum(latency_deltas_for_positive_turns) + r_timeout

    Where:
      r_task:
        +1.0  if final_answer matches ground_truth (exact or fuzzy match)
        +0.3  if answer is partially correct (semantic similarity > 0.7)
        0.0   otherwise

      r_latency:
        For each tool call turn i where r_task > 0 (i.e., successful trajectory):
          penalty_i = max(0, latency_delta_i)   <- only penalize slower-than-average calls
        r_latency = -w_latency * mean(penalty_i for i in trajectory)
        w_latency is a hyperparameter (sweep: 0.0, 0.1, 0.3, 0.5, 1.0)

      r_timeout:
        -2.0 per timeout event in the trajectory

    Note: latency penalty only applied on successful trajectories to avoid
    discouraging tool use on failed trajectories (which would cause under-calling).
    """

def fuzzy_answer_match(pred: str, gold: str) -> float:
    """
    Returns similarity score in [0, 1].
    Uses token overlap F1 as primary metric (SQuAD-style).
    Falls back to exact string match.
    """
```

---

## 7. GRPO Trainer (`training/grpo_trainer.py`)

```python
"""
Online GRPO training loop using HuggingFace trl's GRPOTrainer.
Key differences from standard GRPOTrainer:
  - Reward function is episode-level (not step-level), computed by ToolEnv
  - Multi-turn rollouts with real tool execution between turns
  - Latency context injected into each turn's prompt
  - Void-turn filtering: turns where tool call returned no useful result
    are excluded from the policy gradient update
"""

# Recommended GRPOConfig settings:
GRPO_CONFIG = {
    "num_generations": 8,         # group size G for relative advantage
    "max_new_tokens": 512,
    "temperature": 0.8,
    "learning_rate": 5e-6,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "kl_coeff": 0.04,             # KL penalty against reference model
    "max_grad_norm": 0.5,
    "num_train_epochs": 3,
    "logging_steps": 10,
    "save_steps": 100,
    "use_vllm": True,             # use vLLM for fast rollout generation
}

# LoRA config for 7B model on GV100 (32 GB):
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

def void_turn_filter(trajectory: list) -> list:
    """
    Removes turns where:
      - Tool returned an error or empty result
      - Tool call was syntactically invalid
      - Tool result was identical to a prior turn's result (redundant call)
    Returns filtered trajectory for policy gradient computation.
    This prevents gradient updates from tool failures polluting the policy.
    """
```

---

## 8. Data: ToolBench Loader (`data/toolbench_loader.py`)

```python
"""
Loads tasks from ToolBench dataset (HuggingFace: ToolBench/ToolBench).
Splits tools into train/test by API category.

Train categories (suggested): Social, Finance, Weather, News
Test (transfer eval) categories: Travel, Sports, Entertainment

For each task, provides:
  - query: user instruction
  - available_tools: list of tool schemas relevant to this task
  - ground_truth: expected answer or action sequence
  - required_tools: minimal set of tools needed (for reward)
  - category: API category string

Filters to tasks where:
  - At least 2 tools are available (to create genuine selection decisions)
  - At least one "cheap" tool (< 200ms avg) and one "expensive" tool (> 500ms avg) exist
    in the available set — this ensures the model faces a real cost-aware choice
"""
```

---

## 9. Baselines (`baselines/`)

### 9a. Correctness-Only GRPO (`baselines/correctness_only_rl.py`)
- Same setup as main training but `w_latency = 0.0`
- No latency context injected into prompts
- Direct ablation: does adding latency reward change behavior?

### 9b. SFT Baseline (`baselines/sft_baseline.py`)
- Fine-tune Qwen2.5-7B on ToolBench demonstrations using standard cross-entropy
- Demonstrations are "gold" trajectories with correct tool sequences
- No cost signal — purely imitation learning
- Use same LoRA config as GRPO for fair comparison

### 9c. CATP-Style Offline RL (`baselines/catp_offline_rl.py`)
- Collect a fixed dataset of N=5000 trajectories using the SFT baseline model
- Label each trajectory with static cost attributes (pre-logged avg latency per tool)
- Train a reward model on these labels
- Fine-tune Qwen2.5-7B via PPO against the learned reward model
- This replicates the offline RL paradigm of CATP-LLM for direct comparison

---

## 10. Evaluation Suite (`evaluation/`)

### Primary Metrics

| Metric | Description | Script |
|---|---|---|
| Task Success Rate | % tasks with correct final answer | eval_correctness.py |
| Avg Latency/Task | Mean total API latency per completed task (ms) | eval_efficiency.py |
| Avg Tool Calls/Task | Mean number of tool invocations | eval_efficiency.py |
| Timeout Rate | % tool calls that timed out | eval_efficiency.py |
| Pareto Efficiency | Accuracy vs. latency trade-off curve | pareto_plot.py |

### Transfer Evaluation (`eval_transfer.py`)
- Load model trained on train categories
- Run on 200 tasks from held-out test categories (travel, sports)
- Report all primary metrics
- Compare: latency-aware RL vs. correctness-only RL vs. SFT

### Degradation Evaluation (`eval_degradation.py`)
```python
DEGRADATION_CONFIGS = [
    {"target": "all",     "multiplier": 2.0, "timeout_rate": 0.0},   # 2x slower
    {"target": "all",     "multiplier": 5.0, "timeout_rate": 0.0},   # 5x slower
    {"target": "top50pct","multiplier": 1.0, "timeout_rate": 0.15},  # 15% timeouts on slow tools
    {"target": "top50pct","multiplier": 3.0, "timeout_rate": 0.10},  # combined
]
```
For each config: measure whether the RL-trained model re-routes to less-degraded tools more effectively than correctness-only baseline. Key metric: **tool re-routing rate** (% of tasks where model switches away from degraded tool vs. baseline).

---

## 11. Experiment Grid (Ablation Table for Paper)

```
Condition | w_latency | Latency in Context | Void Filter | Description
----------|-----------|--------------------|--------------------------
C1        | 0.0       | No                 | No          | SFT baseline
C2        | 0.0       | No                 | Yes         | Correctness-only GRPO
C3        | 0.3       | No                 | Yes         | Latency reward, no context
C4        | 0.0       | Yes                | Yes         | Context only, no reward
C5        | 0.3       | Yes                | Yes         | FULL: latency reward + context  ← proposed
C6        | 0.5       | Yes                | Yes         | Aggressive latency penalty
C7        | 1.0       | Yes                | Yes         | Maximum latency penalty
```

Run C1–C7 on training tool set. Evaluate all on: (a) train distribution, (b) transfer, (c) degradation.
This produces the 3 ablation tables needed for the paper.

---

## 12. Training Scripts

### `scripts/run_training.sh`
```bash
#!/bin/bash
# Run on GV100 (training machine)
# Usage: ./run_training.sh --config config/experiment_configs/online_latency_reward.yaml

python training/train.py \
  --config $1 \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --output_dir checkpoints/latency_aware_rl \
  --wandb_project latency-aware-rl \
  --seed 42
```

### `scripts/run_eval.sh`
```bash
#!/bin/bash
# Run on 4090 (eval machine) in parallel with training
# Usage: ./run_eval.sh --checkpoint checkpoints/latency_aware_rl/step_500

python evaluation/eval_correctness.py --checkpoint $1 --split test
python evaluation/eval_efficiency.py --checkpoint $1 --split test
python evaluation/eval_transfer.py --checkpoint $1
python evaluation/eval_degradation.py --checkpoint $1
python evaluation/pareto_plot.py --results_dir results/ --output figures/pareto.pdf
```

### `scripts/run_ablation_grid.sh`
```bash
#!/bin/bash
# Runs all 7 conditions sequentially on GV100
for config in baseline_no_cost online_latency_reward; do
  python training/train.py --config config/experiment_configs/$config.yaml
done
```

---

## 13. Config File Schema (`config/base_config.yaml`)

```yaml
model:
  name: "Qwen/Qwen2.5-7B-Instruct"
  lora_r: 16
  lora_alpha: 32
  load_in_4bit: false         # set true only if VRAM is tight

training:
  w_latency: 0.3              # latency penalty weight (ablate: 0.0, 0.1, 0.3, 0.5, 1.0)
  inject_latency_context: true
  void_turn_filter: true
  num_generations: 8          # GRPO group size
  max_steps_per_episode: 8
  learning_rate: 5e-6
  kl_coeff: 0.04
  total_train_steps: 2000
  eval_every_n_steps: 200

data:
  train_categories: ["Social", "Finance", "Weather", "News"]
  test_categories: ["Travel", "Sports"]
  min_tools_per_task: 2
  require_cost_variance: true  # only tasks with both cheap+expensive tools

environment:
  executor_mode: "replay"      # "live" or "replay"
  trace_dir: "data/latency_traces/"
  timeout_threshold_ms: 5000

reward:
  r_task_correct: 1.0
  r_task_partial: 0.3
  r_timeout_penalty: -2.0
  latency_penalty_only_on_success: true

eval:
  degradation_multipliers: [2.0, 5.0]
  degradation_timeout_rates: [0.0, 0.15]
  n_eval_tasks: 200
```

---

## 14. Pre-Run Checklist

Before starting training:

1. **Collect latency traces** (run once, takes ~30 min):
   ```bash
   python environment/api_executor.py --mode collect --n_calls 100
   ```

2. **Verify ToolBench access**:
   ```bash
   python data/toolbench_loader.py --test
   ```

3. **Smoke test environment** (5 episodes, no training):
   ```bash
   python environment/tool_env.py --smoke_test --n_episodes 5
   ```

4. **Verify GRPO loop** (10 steps, log rewards):
   ```bash
   python training/train.py --config config/base_config.yaml --debug_steps 10
   ```

5. **Check VRAM** on GV100:
   - Qwen2.5-7B + LoRA r=16 + vLLM: ~26–28 GB. Should fit on 32 GB.
   - If OOM: reduce `num_generations` from 8 to 4, or enable `load_in_4bit: true`

---

## 15. Expected Results & Paper Claims

Based on the design, anticipate the following findings (to be validated empirically):

| Finding | Measured By | Expected Direction |
|---|---|---|
| Latency-aware RL reduces avg task latency | eval_efficiency | Lower than correctness-only RL |
| Accuracy is maintained or slightly reduced | eval_correctness | Within 2-3% of correctness-only RL |
| Cost-aware policy transfers to unseen tool categories | eval_transfer | Better latency than SFT/correctness-RL on test categories |
| Latency-aware model re-routes under degradation | eval_degradation | Higher re-routing rate than baselines |
| Online > Offline RL for latency optimization | vs. CATP-offline | Lower latency at same accuracy |

If any finding goes against expectation, report it honestly — null results in ablations are publishable in this space.

---

## 16. Paper Outline (for Writing After Experiments)

```
1. Introduction
   - Motivation: real APIs have variable cost; correctness-only training ignores this
   - Problem: train an LLM to be simultaneously accurate AND cost-efficient
   - Contribution summary (3 bullet points)

2. Related Work
   - CATP-LLM (offline RL, static cost) — our direct predecessor
   - ToolRL / ReTool (online RL, correctness-only)
   - xRouter (cost-aware routing across models, not tools)
   - Distinguish: online + real latency + transfer + degradation robustness

3. Method
   3.1 Problem Formulation (MDP over tool-use episodes)
   3.2 Latency-Aware Context Injection
   3.3 Reward Function (equation)
   3.4 Online GRPO with Void-Turn Filtering

4. Experiments
   4.1 Setup (model, data, baselines, hardware)
   4.2 Main Result: Accuracy-Latency Pareto Curve (Fig 1)
   4.3 Ablation: which components drive efficiency gain (Table 1: C1-C7)
   4.4 Transfer Evaluation (Table 2)
   4.5 Degradation Robustness (Table 3 + qualitative examples)

5. Analysis
   - Tool preference ordering learned by the model (does it prefer fast tools?)
   - Does latency context actually influence the model's reasoning? (attention analysis or chain-of-thought inspection)

6. Conclusion + Limitations
   - Limitations: single model family, English-only tasks, limited tool categories
   - Future work: multi-agent extension, schema drift robustness
```
