"""
Agent session state — tracks the message thread and memory summary.

Modelled on the open-source claude-code agent_session.py, but simplified
for direct llama-cpp-python use with local GGUF models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AgentMessage:
    """A single message in the conversation thread."""
    role: str   # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class AgentSession:
    """
    Full mutable conversation state for one agent run.

    Keeps the message list and a memory_summary that survives compaction.
    Compaction replaces old messages with a summary injected as a user turn.
    """

    messages: List[AgentMessage] = field(default_factory=list)
    memory_summary: str = ""

    # ── construction ────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        system_prompt: str,
        initial_user_content: str,
    ) -> "AgentSession":
        session = cls()
        session.messages.append(AgentMessage("system", system_prompt))
        session.messages.append(AgentMessage("user", initial_user_content))
        return session

    # ── mutation helpers ─────────────────────────────────────────────────────

    def add_assistant(self, content: str) -> None:
        self.messages.append(AgentMessage("assistant", content))

    def add_user(self, content: str) -> None:
        self.messages.append(AgentMessage("user", content))

    def replace_messages(
        self,
        new_messages: List[AgentMessage],
        new_summary: str,
    ) -> None:
        """Swap in compacted messages and update the memory summary."""
        self.messages = new_messages
        self.memory_summary = new_summary

    # ── serialisation ─────────────────────────────────────────────────────────

    def to_llm_messages(self) -> List[dict]:
        """Return a plain list of dicts for llama-cpp-python."""
        return [m.to_dict() for m in self.messages]

    # ── token estimation ──────────────────────────────────────────────────────

    def estimate_tokens(self) -> int:
        """Rough token count: 1 token ≈ 4 chars."""
        total = sum(len(m.content) for m in self.messages)
        return max(1, total // 4)

    def message_count(self) -> int:
        return len(self.messages)

    # ── context trimming ─────────────────────────────────────────────────────

    def trim_for_budget(
        self,
        max_tokens: int,
        context_size: int,
        preserve_system: bool = True,
    ) -> "AgentSession":
        """
        Return a version of this session that fits within the token budget.
        Drops the oldest non-system messages first.
        """
        input_budget = max(800, context_size - max_tokens - 256)
        if self.estimate_tokens() <= input_budget:
            return self

        trimmed = AgentSession(memory_summary=self.memory_summary)
        all_msgs = list(self.messages)

        # Always keep system message
        if preserve_system and all_msgs and all_msgs[0].role == "system":
            trimmed.messages.append(all_msgs[0])
            body = all_msgs[1:]
        else:
            body = all_msgs

        # Inject memory summary if present
        if self.memory_summary:
            trimmed.messages.append(
                AgentMessage(
                    "user",
                    "Compacted prior context:\n" + self.memory_summary[:2500],
                )
            )

        # Keep as many recent messages as fit
        recent = body[-5:]
        for msg in recent:
            trimmed.messages.append(
                AgentMessage(
                    msg.role,
                    msg.content[:5000],
                )
            )

        # Emergency: if still over budget, drop down to last 2 messages
        while (
            trimmed.estimate_tokens() >= input_budget
            and len(trimmed.messages) > 2
        ):
            trimmed.messages.pop(1)

        return trimmed
