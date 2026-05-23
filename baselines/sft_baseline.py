from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    out_dir = Path("checkpoints/sft_baseline")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "placeholder",
        "note": "SFT baseline wiring point. Integrate transformers Trainer for CE fine-tuning here.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
