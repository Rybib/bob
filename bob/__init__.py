"""
BOB agent package.

Provides the Claude Code-style agent architecture adapted for local GGUF models.
Import BobAgent and AgentConfig to integrate with the LLMEngine in bob.py.
"""

from bob.agent import BobAgent
from bob.types import AgentConfig

__all__ = ["BobAgent", "AgentConfig"]
