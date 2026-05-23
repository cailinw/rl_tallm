from __future__ import annotations

import argparse
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug_steps", type=int, default=50)
    args = parser.parse_args()
    cmd = [
        "python",
        "training/train.py",
        "--config",
        "config/base_config.yaml",
        "--override_config",
        "config/experiment_configs/baseline_no_cost.yaml",
        "--debug_steps",
        str(args.debug_steps),
        "--output_dir",
        "checkpoints/correctness_only_rl",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
