# Latency-Aware RL for Tool-Use LLMs

Reference implementation scaffold for online GRPO-style training with latency-aware rewards.

## What this includes

- Multi-turn tool-use environment with latency observations
- API executor with `live` and `replay` modes
- Rolling latency tracker and degradation wrapper
- Reward computation with correctness, latency, and timeout components
- Task loading from:
  - `OpenCATP-LLM` dataset exports (local JSON/JSONL path)
  - `WildToolBench` (`wild-tool-bench/data/Wild-Tool-Bench.jsonl`)
- Training/evaluation entry points and experiment configs

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run a short debug train loop:

```bash
python training/train.py --config config/base_config.yaml --debug_steps 10
```

Switch benchmark source (CATP-LLM vs WildToolBench):

```yaml
data:
  benchmark: "catp_llm"         # or "wild_tool_bench"
  catp_llm_path: "/path/to/catp_dataset.jsonl"
  wild_tool_bench_path: "/path/to/Wild-Tool-Bench.jsonl"
```

Validate/inspect benchmark schema normalization:

```bash
python data/toolbench_loader.py --benchmark catp_llm --dataset_path /path/to/catp_dataset.jsonl --split train --inspect --strict
python data/toolbench_loader.py --benchmark wild_tool_bench --dataset_path /path/to/Wild-Tool-Bench.jsonl --split train --inspect --strict
```

## Notes

- This repository intentionally favors clear, extensible code over a fully optimized research training stack.
- GRPO integration points are included, but full-scale training settings (multi-GPU/distributed) should be adapted to your runtime.
