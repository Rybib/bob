"""
BobAgent — the main agent runtime.

Implements the full Claude Code-style turn loop adapted for local GGUF models:

  1. Build initial session (system prompt + bootstrapped user message)
  2. Run the turn loop:
       a. Compact if context is getting full
       b. Generate a response via the local LLM
       c. Parse any tool calls from the response
       d. If no tool calls → evaluate + return final answer
       e. Execute tools, feed results back, continue
  3. After max turns → fallback summarise or return None

Key design decisions vs the original bob.py _run_agentic_project_task_inner:
  - Proper AgentSession state management (session.py)
  - Token-based compaction trigger instead of character count
  - Multi-stage nudging: inspect → write → strong write with example
  - Actionable errors explicitly surfaced back to the model each turn
  - Clean separation between generation, parsing, and execution
  - auto-verify after HTML mutations (review_web_project)
  - Fallback chain on exit: clean text → tool-result summary
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Callable, List, Optional

from bob.compact import compact_session, should_compact
from bob.prompts import (
    QUICK_PLAN_SYSTEM,
    build_initial_user_message,
    build_system_prompt,
)
from bob.session import AgentSession
from bob.types import AgentConfig, NudgeState
from bob.workspace import (
    changed_project_roots,
    existing_projects_from_results,
    has_inspection,
    has_real_mutations,
    infer_existing_project,
    meaningful_project_name,
    requires_inspection,
    requires_mutation,
    summarize_tool_results,
)

# Type alias for the generate callable
GenerateFn = Callable[[List[dict], int, float], str]
OnTokenFn = Callable[[str], None]
OnSpeakFn = Optional[Callable[[str], None]]


class BobAgent:
    """
    Full agent loop for BOB's project-building capability.

    Dependencies are injected so the agent is decoupled from LLMEngine:
      - generate_fn:   calls the local LLM (messages, max_tokens, temperature) → str
      - harness:       ToolHarness instance from bob.py (parse + execute tools)
      - config:        AgentConfig (max turns, context size, etc.)
      - projects_dir:  Path to the projects/ folder (for summaries)
      - log_path:      Path where errors are appended
    """

    def __init__(
        self,
        generate_fn: GenerateFn,
        harness,
        config: AgentConfig,
        projects_dir: Path,
        log_path: Optional[Path] = None,
    ):
        self._gen = generate_fn
        self._harness = harness
        self._config = config
        self._projects_dir = projects_dir
        self._log_path = log_path
        self._system_prompt = build_system_prompt(projects_dir)
        # Mutable state across a single run (reset each call to run())
        self._active_project: Optional[str] = None

    # ── public entry points ──────────────────────────────────────────────────

    def run(
        self,
        user_text: str,
        on_token: OnTokenFn,
        on_speak: OnSpeakFn = None,
    ) -> Optional[str]:
        """
        Run the full agentic loop for a coding/file request.

        Returns the final spoken reply string, or None if no meaningful work
        was done (callers can then fall back to a simpler response path).
        """
        try:
            return self._run_inner(user_text, on_token, on_speak)
        except Exception as exc:
            msg = f"The agent hit an unexpected error and stopped safely: {exc}"
            on_token(msg)
            self._log_error()
            return None

    def quick_plan(self, user_text: str) -> str:
        """
        Generate a brief spoken plan of what the agent is about to do.
        Called before the main loop so the user gets immediate feedback.
        """
        try:
            return self._gen(
                [
                    {"role": "system", "content": QUICK_PLAN_SYSTEM},
                    {"role": "user", "content": user_text},
                ],
                80,
                0.4,
            )
        except Exception:
            return ""

    # ── internal loop ────────────────────────────────────────────────────────

    def _run_inner(
        self,
        user_text: str,
        on_token: OnTokenFn,
        on_speak: OnSpeakFn,
    ) -> Optional[str]:
        cfg = self._config

        # ── 0. Quick spoken plan ─────────────────────────────────────────────
        plan = self.quick_plan(user_text)
        if plan:
            if on_speak:
                on_speak(plan)
            else:
                on_token(plan)

        # ── 1. Bootstrap: discover existing projects ─────────────────────────
        on_token("Checking the projects folder before deciding what to do…")
        initial_results = self._harness.run_calls(
            [{"tool": "list_project_tree", "args": {"path": ".", "max_files": 220}}],
            on_update=on_token,
        )

        must_inspect = requires_inspection(user_text)
        must_mutate = requires_mutation(user_text)
        existing_projects = existing_projects_from_results(initial_results)
        focused_project = infer_existing_project(
            user_text, existing_projects, self._active_project
        )
        if focused_project:
            self._active_project = focused_project

        suggested = meaningful_project_name(user_text, self._projects_dir)

        # ── 2. Build initial session ─────────────────────────────────────────
        initial_user_msg = build_initial_user_message(
            user_text=user_text,
            suggested_folder=suggested,
            focused_project=focused_project,
            active_project=self._active_project,
            must_inspect=must_inspect,
            must_mutate=must_mutate,
            initial_tree_json=json.dumps(initial_results, ensure_ascii=False),
        )
        session = AgentSession.create(self._system_prompt, initial_user_msg)

        # ── 3. Turn loop ─────────────────────────────────────────────────────
        all_results: List[dict] = list(initial_results)
        inspected = not must_inspect
        nudge = NudgeState()

        for turn_index in range(cfg.max_turns):

            # ── 3a. Compact if needed ────────────────────────────────────────
            if should_compact(session, cfg):
                on_token("Compacting conversation memory to stay within context…")
                session = compact_session(session, self._gen, cfg)

            # ── 3b. Trim session to fit budget ───────────────────────────────
            trimmed = session.trim_for_budget(cfg.agent_max_tokens, cfg.context_size)

            # ── 3c. Generate ─────────────────────────────────────────────────
            draft = self._gen(
                trimmed.to_llm_messages(),
                cfg.agent_max_tokens,
                cfg.temperature,
            )

            if not draft:
                streak = nudge.empty_draft_streak + 1
                nudge = nudge.with_empty_streak(streak)
                if streak >= 2:
                    break
                continue
            else:
                nudge = nudge.with_empty_streak(0)

            # ── 3d. Stream status line to user ───────────────────────────────
            status = self._harness.parse_status(draft)
            if status:
                on_token(status)

            # ── 3e. Parse tool calls ─────────────────────────────────────────
            calls = self._harness.parse_tool_calls(draft)

            # ── 3f. No tool calls → decide what to do ───────────────────────
            if not calls:
                decision, nudge = self._handle_no_tool_calls(
                    draft=draft,
                    user_text=user_text,
                    session=session,
                    all_results=all_results,
                    must_inspect=must_inspect,
                    must_mutate=must_mutate,
                    inspected=inspected,
                    nudge=nudge,
                    on_token=on_token,
                )
                if decision is None:
                    # Nudge injected into session — continue loop
                    continue
                if decision is False:
                    # Hard stop — no useful output
                    break
                # decision is a string → final answer
                return decision

            # ── 3g. Execute tools ────────────────────────────────────────────
            tool_names = ", ".join(str(c.get("tool", "?")) for c in calls[:3])
            extra = "" if len(calls) <= 3 else f", plus {len(calls) - 3} more"
            on_token(f"Working in projects/: {tool_names}{extra}…")

            results = self._harness.run_calls(calls, on_update=on_token)
            all_results.extend(results)

            # Update active project from results
            roots = changed_project_roots(results)
            if roots:
                self._active_project = roots[0]

            # Track whether we have now inspected files
            if has_inspection(results):
                inspected = True

            # ── 3h. Auto-verify HTML mutations ───────────────────────────────
            if cfg.enable_auto_verify and has_real_mutations(results):
                verify_results = self._auto_verify(results, on_token)
                if verify_results:
                    all_results.extend(verify_results)
            else:
                verify_results = []

            # ── 3i. Feed results back into session ───────────────────────────
            session.add_assistant(draft)

            result_payload = results
            if verify_results:
                result_payload = results + [
                    {"tool": "auto_verify", "ok": True, "result": verify_results}
                ]

            actionable_errors = self._harness.actionable_errors(results)
            if actionable_errors:
                result_payload = result_payload + [
                    {
                        "tool": "error_summary",
                        "ok": False,
                        "error": "; ".join(actionable_errors[:3]),
                    }
                ]
                on_token(f"Tool issue: {actionable_errors[0].rstrip('.')}. Adjusting the next step…")

            # Budget-aware truncation of the tool result turn
            result_json = json.dumps(result_payload, ensure_ascii=False)
            current_ctx_chars = sum(len(m.content) for m in session.messages)
            remaining_budget = max(1500, (cfg.context_size * 3) - current_ctx_chars)
            result_cap = min(cfg.tool_result_max_chars, remaining_budget)

            session.add_user("Tool results:\n" + result_json[:result_cap])

        # ── 4. Max turns reached — check for useful output ───────────────────
        if not has_real_mutations(all_results):
            return None
        return summarize_tool_results(all_results, self._projects_dir)

    # ── no-tool-call decision logic ──────────────────────────────────────────

    def _handle_no_tool_calls(
        self,
        *,
        draft: str,
        user_text: str,
        session: AgentSession,
        all_results: List[dict],
        must_inspect: bool,
        must_mutate: bool,
        inspected: bool,
        nudge: NudgeState,
        on_token: OnTokenFn,
    ):
        """
        Called when the model produced no tool calls.

        Returns a tuple (decision, updated_nudge) where decision is:
          str   → final answer to return to the user
          None  → a nudge was injected; continue the loop
          False → hard stop (no useful output possible)
        """
        cfg = self._config

        # Nothing done yet at all
        if not all_results:
            return False, nudge

        # Model still hasn't inspected files when it should
        if must_inspect and not inspected and nudge.inspect_nudges < cfg.max_nudges:
            session.add_user(
                "You have not inspected the project files yet. "
                "Use list_project_tree or read_project_file to read the relevant "
                "files first, then make the requested changes."
            )
            return None, nudge.with_inspect_nudge()

        # Model has inspected but hasn't mutated anything
        if must_mutate and not has_real_mutations(all_results):
            if nudge.mutation_nudges == 0:
                session.add_user(
                    f'You have read the files but made NO actual changes yet. '
                    f'The user asked: "{user_text}". '
                    "You MUST now call write_project_file or edit_project_file "
                    "to make those changes. Do not re-read files. Do not explain "
                    "what you would do. Output a tool call with the new or updated "
                    "file content right now."
                )
                return None, nudge.with_mutation_nudge()
            elif nudge.mutation_nudges < cfg.max_nudges:
                session.add_user(
                    f'You still have not made any file edits. STOP. '
                    f'The user wants: "{user_text}". '
                    'Call write_project_file or edit_project_file NOW with the '
                    'actual changed content. Example:\n'
                    '{"tool_calls":[{"tool":"write_project_file",'
                    '"args":{"path":"project/style.css","content":"/* new CSS */"}}]}\n'
                    "Make the edit now — do not read any more files."
                )
                return None, nudge.with_mutation_nudge()

        # Everything looks good — the model gave a clean text answer
        return self._finalize(draft, all_results, user_text), nudge

    def _finalize(
        self,
        draft: str,
        all_results: List[dict],
        user_text: str,
    ) -> str:
        """
        Decide what the final reply string should be.

        Preference order:
          1. Clean prose text from the model (no JSON / code blocks)
          2. Summary derived from tool results
        """
        is_clean = (
            draft
            and "@@BOB_BUILD@@" not in draft
            and "```" not in draft
            and not self._harness.looks_like_tool_json(draft)
        )
        if is_clean:
            return draft.strip()

        return summarize_tool_results(all_results, self._projects_dir)

    # ── auto-verification ────────────────────────────────────────────────────

    def _auto_verify(self, results: List[dict], on_token: OnTokenFn) -> List[dict]:
        """Run review_web_project + list_project_tree after HTML mutations."""
        roots = changed_project_roots(results)[:3]
        if not roots:
            return []
        verify_calls = []
        for root in roots:
            verify_calls.append({"tool": "review_web_project", "args": {"path": root}})
            verify_calls.append(
                {"tool": "list_project_tree", "args": {"path": root, "max_files": 120}}
            )
        on_token("Reviewing and verifying the files that were changed…")
        return self._harness.run_calls(verify_calls, on_update=on_token)

    # ── error logging ────────────────────────────────────────────────────────

    def _log_error(self) -> None:
        if not self._log_path:
            return
        try:
            with self._log_path.open("a", encoding="utf-8") as fh:
                traceback.print_exc(file=fh)
        except Exception:
            pass
