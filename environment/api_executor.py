from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.api_registry import APIRegistry
from data.api_registry import ToolSpec


class APIExecutor:
    def __init__(
        self,
        api_registry: APIRegistry,
        mode: str = "replay",
        trace_dir: str = "data/latency_traces",
        call_log_dir: str = "data/call_logs",
        timeout_threshold_ms: int = 5000,
    ):
        self.api_registry = api_registry
        self.mode = mode
        self.trace_dir = Path(trace_dir)
        self.call_log_dir = Path(call_log_dir)
        self.timeout_threshold_ms = timeout_threshold_ms
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.call_log_dir.mkdir(parents=True, exist_ok=True)

    async def call(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if self.mode == "live":
            out = await self._call_live(tool_name, args)
        else:
            out = self._call_replay(tool_name)
        self._log_call(tool_name, args, out)
        return out

    async def _call_live(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        import aiohttp

        tool = self.api_registry.get(tool_name)
        if not tool.endpoint:
            return {
                "result": f"No endpoint configured for {tool_name}",
                "latency_ms": float(tool.default_latency_ms),
                "success": False,
                "error": "missing_endpoint",
                "timed_out": False,
            }

        timeout_sec = self.timeout_threshold_ms / 1000.0
        started = time.perf_counter()
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_sec)) as session:
                async with session.get(tool.endpoint, params=args) as resp:
                    text = await resp.text()
                    latency_ms = (time.perf_counter() - started) * 1000.0
                    return {
                        "result": text[:4000],
                        "latency_ms": latency_ms,
                        "success": 200 <= resp.status < 300,
                        "error": None if 200 <= resp.status < 300 else f"http_{resp.status}",
                        "timed_out": False,
                    }
        except asyncio.TimeoutError:
            latency_ms = (time.perf_counter() - started) * 1000.0
            return {
                "result": "",
                "latency_ms": latency_ms,
                "success": False,
                "error": "timeout",
                "timed_out": True,
            }
        except Exception as exc:  # pragma: no cover - defensive
            latency_ms = (time.perf_counter() - started) * 1000.0
            return {
                "result": "",
                "latency_ms": latency_ms,
                "success": False,
                "error": str(exc),
                "timed_out": False,
            }

    def _call_replay(self, tool_name: str) -> Dict[str, Any]:
        trace_file = self.trace_dir / f"{tool_name}.json"
        if trace_file.exists():
            data = json.loads(trace_file.read_text())
            latencies = data.get("latencies_ms", [])
            timeout_rate = float(data.get("timeout_rate", 0.0))
            latency_ms = float(random.choice(latencies)) if latencies else 400.0
            timed_out = random.random() < timeout_rate
        else:
            tool = self.api_registry.get(tool_name)
            latency_ms = float(max(10.0, random.gauss(tool.default_latency_ms, 0.2 * tool.default_latency_ms)))
            timed_out = random.random() < 0.02

        return {
            "result": f"replay_result_for::{tool_name}",
            "latency_ms": latency_ms,
            "success": not timed_out,
            "error": "timeout" if timed_out else None,
            "timed_out": timed_out,
        }

    def _log_call(self, tool_name: str, args: Dict[str, Any], output: Dict[str, Any]) -> None:
        stamp = int(time.time() * 1000)
        out_path = self.call_log_dir / f"{stamp}_{tool_name}.json"
        out_path.write_text(
            json.dumps(
                {
                    "tool_name": tool_name,
                    "args": args,
                    "output": output,
                },
                ensure_ascii=True,
            )
        )

    def collect_traces(self, tool_names: list[str], n_calls_per_tool: int = 100) -> None:
        for tool_name in tool_names:
            latencies = []
            timeouts = 0
            for _ in range(n_calls_per_tool):
                out = self._call_replay(tool_name)
                latencies.append(out["latency_ms"])
                timeouts += int(out.get("timed_out", False))
            stats = {
                "latencies_ms": latencies,
                "timeout_rate": timeouts / max(1, n_calls_per_tool),
                "mean": sum(latencies) / max(1, len(latencies)),
                "std": (sum((x - (sum(latencies) / max(1, len(latencies)))) ** 2 for x in latencies) / max(1, len(latencies)))
                ** 0.5,
            }
            (self.trace_dir / f"{tool_name}.json").write_text(json.dumps(stats, ensure_ascii=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["collect", "replay"], default="collect")
    parser.add_argument("--n_calls", type=int, default=100)
    args = parser.parse_args()

    registry = APIRegistry()
    for tool_name in ["weather_fast", "weather_slow", "stock_quote_fast", "stock_quote_slow"]:
        registry.register(
            ToolSpec(
                name=tool_name,
                description=f"{tool_name} synthetic tool",
                schema={"type": "object", "properties": {"query": {"type": "string"}}},
                category="synthetic",
                default_latency_ms=120.0 if "fast" in tool_name else 900.0,
            )
        )
    executor = APIExecutor(registry, mode="replay")
    if args.mode == "collect":
        executor.collect_traces(registry.list_tools(), n_calls_per_tool=args.n_calls)
        print(f"Collected traces for {len(registry.list_tools())} tools.")
    else:
        out = asyncio.run(executor.call("weather_fast", {"query": "weather in SF"}))
        print(out)


if __name__ == "__main__":
    main()
