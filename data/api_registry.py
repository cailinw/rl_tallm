from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolSpec:
    name: str
    description: str
    schema: Dict[str, Any]
    category: str
    endpoint: Optional[str] = None
    default_latency_ms: float = 400.0
    executor: Optional[Callable[[Dict[str, Any]], Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class APIRegistry:
    def __init__(self, tool_specs: List[ToolSpec] | None = None):
        self._tools: Dict[str, ToolSpec] = {}
        for spec in tool_specs or []:
            self.register(spec)

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self._tools

    def get(self, tool_name: str) -> ToolSpec:
        if tool_name not in self._tools:
            raise KeyError(f"Unknown tool: {tool_name}")
        return self._tools[tool_name]

    def get_schema(self, tool_name: str) -> Dict[str, Any]:
        return self.get(tool_name).schema

    def list_tools(self) -> List[str]:
        return sorted(self._tools.keys())

    def to_prompt_schemas(self, tool_names: List[str]) -> List[Dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "schema": spec.schema,
                "category": spec.category,
                "default_latency_ms": spec.default_latency_ms,
            }
            for spec in (self.get(name) for name in tool_names if self.has_tool(name))
        ]
