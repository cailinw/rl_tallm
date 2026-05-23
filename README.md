# Latency-Aware RL for Tool-Use LLMs

Reference implementation scaffold for online GRPO-style training with latency-aware rewards.

## What this includes

- Multi-turn tool-use environment with latency observations
- API executor with `live` and `replay` modes
- Rolling latency tracker and degradation wrapper
- Reward computation with correctness, latency, and timeout components
- Task loading from:
  - `ToolBench`
  - `Live API Bench` (arXiv: 2506.11266) via configurable dataset source
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

Switch benchmark source (ToolBench vs Live API Bench):

```yaml
data:
  benchmark: "toolbench"        # or "live_api_bench"
```

Validate/inspect benchmark schema normalization:

```bash
python data/toolbench_loader.py --benchmark toolbench --split train --inspect --strict
python data/toolbench_loader.py --benchmark live_api_bench --split train --inspect --strict
```

## Notes

- This repository intentionally favors clear, extensible code over a fully optimized research training stack.
- GRPO integration points are included, but full-scale training settings (multi-GPU/distributed) should be adapted to your runtime.
