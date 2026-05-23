from __future__ import annotations

import json
import re
from typing import Dict, List


SYSTEM_PROMPT = """You are a tool-using assistant. For each task, you may call available tools to gather information.
After each tool call, you will see the result and how fast the tool responded.
Use this information to prefer faster tools when multiple tools can answer the question.
When you have enough information, provide a final answer.

Output either:
1) JSON: {"type":"tool_call","tool_name":"...","tool_args":{...}}
2) JSON: {"type":"final_answer","answer":"..."}
3) XML: <tool_call>{"tool_name":"...","tool_args":{...}}</tool_call>
"""


def build_turn_prompt(query: str, available_tools: List[dict], history: List[dict], step: int) -> str:
    tools_block = "\n".join(
        [
            f"- {tool['name']}: {tool.get('description', '')} "
            f"(typical {tool.get('default_latency_ms', 'n/a')} ms)"
            for tool in available_tools
        ]
    )
    history_lines: List[str] = []
    for idx, turn in enumerate(history, start=1):
        tool_name = turn.get("tool_name", "unknown_tool")
        args = turn.get("tool_args", {})
        result = str(turn.get("tool_result", ""))[:500]
        tier = turn.get("latency_tier", "MEDIUM")
        latency_s = float(turn.get("latency_ms", 0.0)) / 1000.0
        delta = float(turn.get("latency_delta", 0.0))
        history_lines.append(
            f"[Turn {idx}] Called: {tool_name}(args={json.dumps(args, ensure_ascii=True)})\n"
            f"Result: {result}\n"
            f"Response: {tier} ({latency_s:.2f}s | {delta:+.2f} sigma)"
        )
    history_block = "\n\n".join(history_lines) if history_lines else "No previous turns."

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Task: {query}\n"
        f"Step: {step}\n\n"
        f"Available tools:\n{tools_block}\n\n"
        f"History:\n{history_block}\n\n"
        "Pick one action now."
    )


def parse_model_output(text: str) -> Dict:
    text = text.strip()
    xml_match = re.search(r"<tool_call>(.*?)</tool_call>", text, flags=re.DOTALL)
    if xml_match:
        try:
            payload = json.loads(xml_match.group(1).strip())
            return {"type": "tool_call", "tool_name": payload["tool_name"], "tool_args": payload.get("tool_args", {})}
        except Exception:
            return {"type": "final_answer", "answer": text}
    try:
        obj = json.loads(text)
        if obj.get("type") in {"tool_call", "final_answer"}:
            return obj
    except Exception:
        pass
    return {"type": "final_answer", "answer": text}
