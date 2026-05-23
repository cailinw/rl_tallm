from __future__ import annotations

from typing import Dict

from model.context_builder import parse_model_output


def parse_action(text: str) -> Dict:
    return parse_model_output(text)
