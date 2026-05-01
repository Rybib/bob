"""
Conversation compaction for the BOB agent.

When the session grows past ~70 % of the model's context window, older messages
are summarised into a durable memory block and dropped.  The most recent messages
are always preserved so the model stays oriented.

Modelled on compact.py from the open-source claude-code agent, adapted for
direct llama-cpp-python inference (no HTTP API).
"""

from __future__ import annotations

from typing import Callable, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from bob.session import AgentSession
    from bob.types import AgentConfig

from bob.prompts import COMPACT_SYSTEM


GenerateFn = Callable[[List[dict], int, float], str]


def should_compact(session: "AgentSession", config: "AgentConfig") -> bool:
    """
    Return True if the session is large enough to warrant compaction.
    Threshold: 70 % of context_size (in estimated tokens) and at least 8 messages.
    """
    if session.message_count() <= 8:
        return False
    threshold = int(config.context_size * config.compact_trigger_ratio)
    return session.estimate_tokens() >= threshold


def compact_session(
    session: "AgentSession",
    generate_fn: GenerateFn,
    config: "AgentConfig",
) -> "AgentSession":
    """
    Summarise the oldest messages and return a new, shorter session.

    The system message and the most recent `compact_preserve_recent` messages
    are always kept.  Everything else is compressed into a summary that is
    injected as a synthetic user turn.
    """
    from bob.session import AgentMessage, AgentSession

    messages = session.messages
    preserve = config.compact_preserve_recent

    system_msg: Optional[AgentMessage] = None
    body: List[AgentMessage] = []

    if messages and messages[0].role == "system":
        system_msg = messages[0]
        body = messages[1:]
    else:
        body = list(messages)

    old_messages = body[:-preserve] if len(body) > preserve else []
    recent_messages = body[-preserve:] if len(body) > preserve else body

    if not old_messages:
        return session

    # ── build summary source ──────────────────────────────────────────────────
    parts: List[str] = []
    if session.memory_summary:
        parts.append("Previous memory:\n" + session.memory_summary)

    for msg in old_messages:
        role = msg.role.upper()
        content = msg.content[:2500]
        parts.append(f"{role}:\n{content}")

    source = "\n\n".join(parts)[:8000]

    # ── call the model for a summary ──────────────────────────────────────────
    try:
        new_summary = generate_fn(
            [
                {"role": "system", "content": COMPACT_SYSTEM},
                {"role": "user", "content": source},
            ],
            config.compact_summary_tokens,
            0.1,
        )
    except Exception:
        new_summary = ""

    if not new_summary:
        new_summary = session.memory_summary or _fallback_summary(old_messages)

    # ── assemble the compacted session ────────────────────────────────────────
    new_messages: List[AgentMessage] = []

    if system_msg:
        new_messages.append(system_msg)

    new_messages.append(
        AgentMessage(
            "user",
            "Compacted session memory:\n" + new_summary,
        )
    )

    new_messages.extend(recent_messages)

    new_session = AgentSession(
        messages=new_messages,
        memory_summary=new_summary,
    )
    return new_session


def _fallback_summary(messages: List) -> str:
    """Cheap text-only fallback when the LLM summary call fails."""
    parts = []
    for msg in messages[-4:]:
        role = getattr(msg, "role", "unknown").upper()
        content = str(getattr(msg, "content", ""))[:400]
        parts.append(f"{role}: {content}")
    return "\n".join(parts)
