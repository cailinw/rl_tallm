from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    out_dir = Path("checkpoints/catp_offline_rl")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "placeholder",
        "note": "CATP-style offline RL pipeline entry point. Add trajectory collection + reward-model + PPO here.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
