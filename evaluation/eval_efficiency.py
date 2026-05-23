from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--call_log_dir", default="data/call_logs")
    args = parser.parse_args()

    log_dir = Path(args.call_log_dir)
    logs = []
    if log_dir.exists():
        for p in log_dir.glob("*.json"):
            try:
                logs.append(json.loads(p.read_text()))
            except Exception:
                continue

    latencies = [float(x.get("output", {}).get("latency_ms", 0.0)) for x in logs]
    timeouts = [int(bool(x.get("output", {}).get("timed_out", False))) for x in logs]
    print(
        json.dumps(
            {
                "n_calls": len(logs),
                "avg_latency_ms": (sum(latencies) / max(1, len(latencies))) if latencies else 0.0,
                "timeout_rate": (sum(timeouts) / max(1, len(timeouts))) if timeouts else 0.0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
