"""
Core data types for the BOB agent runtime.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class AgentConfig:
    """Configuration for a single agent run."""
    max_turns: int = 10
    agent_max_tokens: int = 1600
    temperature: float = 0.16
    context_size: int = 8192
    compact_trigger_ratio: float = 0.70
    compact_preserve_recent: int = 6
    compact_summary_tokens: int = 300
    supports_thinking: bool = False
    enable_auto_verify: bool = True
    enable_nudging: bool = True
    max_nudges: int = 2
    # chars per "tool result" turn allowed before trimming
    tool_result_max_chars: int = 3000
    # soft cap on the tool result payload per turn
    tool_result_budget_ratio: float = 0.25


@dataclass(frozen=True)
class NudgeState:
    """Tracks nudge counters so agent.py stays stateless across turns."""
    inspect_nudges: int = 0
    mutation_nudges: int = 0
    empty_draft_streak: int = 0

    def with_inspect_nudge(self) -> "NudgeState":
        return NudgeState(
            inspect_nudges=self.inspect_nudges + 1,
            mutation_nudges=self.mutation_nudges,
            empty_draft_streak=self.empty_draft_streak,
        )

    def with_mutation_nudge(self) -> "NudgeState":
        return NudgeState(
            inspect_nudges=self.inspect_nudges,
            mutation_nudges=self.mutation_nudges + 1,
            empty_draft_streak=self.empty_draft_streak,
        )

    def with_empty_streak(self, streak: int) -> "NudgeState":
        return NudgeState(
            inspect_nudges=self.inspect_nudges,
            mutation_nudges=self.mutation_nudges,
            empty_draft_streak=streak,
        )
