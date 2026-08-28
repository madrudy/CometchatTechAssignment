from dataclasses import dataclass, field
from typing import Any

@dataclass
class Chunk:
    chunk_id: str
    filename: str
    heading: str
    text: str
    metadata: dict[str, Any]
    score: float = 0.0

    @property
    def source_label(self) -> str:
        return f"{self.filename} — {self.heading}"

@dataclass
class ToolResult:
    status: str
    data: dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentResult:
    answer: str
    sources: list[str] = field(default_factory=list)
    handoff: bool = False
    tool_called: bool = False
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    trace: dict[str, Any] = field(default_factory=dict)
