#!/usr/bin/env python3
"""
BOB — Local AI Voice Assistant
Wake word → STT (Whisper) → LLM (GGUF) → TTS (Kokoro)
"""

import os, sys, time, queue, threading, re, warnings, tempfile, random, json, subprocess, shlex, shutil, contextlib, fnmatch, textwrap, faulthandler, traceback
import numpy as np
from pathlib import Path

# ── Point HuggingFace cache at our local models/ folder ─────────────────────
_HERE = Path(__file__).parent
LOGS_DIR = _HERE / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
_CRASH_LOG_HANDLE = None
try:
    _CRASH_LOG_HANDLE = (LOGS_DIR / "python-crash.log").open("a", encoding="utf-8")
    faulthandler.enable(file=_CRASH_LOG_HANDLE, all_threads=True)
except Exception:
    pass

os.environ["HUGGINGFACE_HUB_CACHE"] = str(_HERE / "models" / "hub")
os.environ["HF_HOME"]               = str(_HERE / "models")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
if os.environ.get("BOB_ALLOW_ONLINE", "").strip() != "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# Ensure Homebrew binaries (sox, etc.) are on PATH when run from IDLE or GUI
for _brew_bin in ("/opt/homebrew/bin", "/usr/local/bin"):
    if _brew_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _brew_bin + ":" + os.environ.get("PATH", "")
import sounddevice as sd
import soundfile as sf
from typing import Optional, List
from dataclasses import dataclass, field
from datetime import datetime

warnings.filterwarnings("ignore")

# ─────────────────────────────── Rich UI ────────────────────────────────────
from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich.table import Table
from rich.console import Group
from rich import box

console = Console()

# ─────────────────────────────── Config ─────────────────────────────────────
MODELS_DIR      = _HERE / "models"
PROJECTS_DIR    = _HERE / "projects"
CONFIG_PATH     = _HERE / "bob_config.json"
BOB_BUILD_VERSION = "project-builder-v5-kokoro"
ALLOW_ONLINE    = os.environ.get("BOB_ALLOW_ONLINE", "").strip() == "1"
LLM_GPU_LAYERS = int(os.environ.get("BOB_GPU_LAYERS", "-1"))

def _model_context_size(meta: dict) -> int:
    """Return context window to allocate, respecting BOB_LLM_CONTEXT override."""
    env_override = os.environ.get("BOB_LLM_CONTEXT", "").strip()
    if env_override:
        return int(env_override)
    return meta.get("context_size", 8192)

def _compact_trigger_tokens(context_size: int) -> int:
    """Tokens at which conversation history gets compacted (~70 % of context)."""
    return max(1200, int(context_size * 0.70) - 600)

def _compact_target_chars(context_size: int) -> int:
    """Max chars kept after compaction (~3.5 chars/token average for code)."""
    return max(6000, context_size * 3)

def _agent_max_tokens(meta: dict, context_size: int) -> int:
    """Max generation tokens for the agentic tool loop."""
    return min(meta.get("agent_max_tokens", 1600), context_size // 5)
SAMPLE_RATE     = 16_000
CHANNELS        = 1
FRAME_MS        = 30          # VAD frame size in ms
CHUNK_SAMPLES   = SAMPLE_RATE * FRAME_MS // 1000
SILENCE_SEC     = 1.3         # seconds of silence → stop recording
MIN_RECORD_SEC  = 1.8         # don't check for silence before this many seconds
MAX_RECORD_SEC  = 30
MIC_GAIN        = 3.0         # software boost for quiet microphones
LEVEL_GAIN      = 35.0        # visual meter boost; does not affect recording
WAKE_SILENCE_GATE = 0.0008    # quiet mic wake-word gate
RECORD_SILENCE_RMS = 0.0045   # quiet mic recording silence threshold
PARTIAL_STT_MIN_RMS = 0.0035  # quiet mic live transcript threshold
WHISPER_MODEL   = "base"      # tiny / base / small — base is the sweet spot
WAKE_WORDS      = {"bob", "hey bob", "okay bob", "hi bob"}

LLM_MODELS = {
    "gemma-e2b": {
        "label": "Gemma 4 E2B it Q4_K_M",
        "repo_id": "unsloth/gemma-4-E2B-it-GGUF",
        "filename": "gemma-4-E2B-it-Q4_K_M.gguf",
        "strengths": "Fastest and lightest. Good general assistant model for everyday local use.",
        "cache_name": "models--unsloth--gemma-4-E2B-it-GGUF",
        "context_size": 8192,
        "agent_max_tokens": 1600,
        "thinking": False,
    },
    "gemma-e4b": {
        "label": "Gemma 3n E4B it Q4_K_M",
        "repo_id": "himkhati22/gemma-3n-E4B-it-Q4_K_M-GGUF",
        "filename": "gemma-3n-e4b-it-q4_k_m.gguf",
        "strengths": "Better reasoning and writing than E2B while still reasonably local.",
        "cache_name": "models--himkhati22--gemma-3n-E4B-it-Q4_K_M-GGUF",
        "context_size": 8192,
        "agent_max_tokens": 1800,
        "thinking": False,
    },
    "qwen-27b": {
        "label": "Qwen3.6 27B Q4_K_M",
        "repo_id": "sm54/Qwen3.6-27B-Q4_K_M-GGUF",
        "filename": "qwen3.6-27b-q4_k_m.gguf",
        "strengths": "Best coding choice, strongest for larger builds, but very heavy and slower.",
        "cache_name": "models--sm54--Qwen3.6-27B-Q4_K_M-GGUF",
        "context_size": 16384,
        "agent_max_tokens": 2500,
        "thinking": True,   # Qwen3 supports <think>…</think> extended reasoning
    },
}
DEFAULT_LLM_MODEL = "gemma-e2b"

KOKORO_VOICES = {
    "am_michael": "Michael",
    "am_adam": "Adam",
    "am_liam": "Liam",
    "am_echo": "Echo",
    "am_onyx": "Onyx",
    "am_puck": "Puck",
    "af_bella": "Bella",
    "af_nova": "Nova",
    "af_sarah": "Sarah",
    "af_sky": "Sky",
    "bf_emma": "Emma",
    "bm_george": "George",
}
DEFAULT_KOKORO_VOICE = "am_echo"

def _first_snapshot(repo_cache_name: str) -> Optional[Path]:
    repo_dir = MODELS_DIR / "hub" / repo_cache_name / "snapshots"
    if not repo_dir.exists():
        return None
    snapshots = sorted(p for p in repo_dir.iterdir() if p.is_dir())
    return snapshots[-1] if snapshots else None


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {"llm_model": DEFAULT_LLM_MODEL, "kokoro_voice": DEFAULT_KOKORO_VOICE}
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if data.get("llm_model") not in LLM_MODELS:
            data["llm_model"] = DEFAULT_LLM_MODEL
        if data.get("kokoro_voice") not in KOKORO_VOICES:
            data["kokoro_voice"] = DEFAULT_KOKORO_VOICE
        return data
    except Exception:
        return {"llm_model": DEFAULT_LLM_MODEL, "kokoro_voice": DEFAULT_KOKORO_VOICE}


def save_config(config: dict):
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")


def selected_llm_key() -> str:
    return load_config().get("llm_model", DEFAULT_LLM_MODEL)


def selected_kokoro_voice() -> str:
    return load_config().get("kokoro_voice", DEFAULT_KOKORO_VOICE)


def local_llm_gguf(model_key: Optional[str] = None) -> Optional[Path]:
    meta = LLM_MODELS.get(model_key or selected_llm_key(), LLM_MODELS[DEFAULT_LLM_MODEL])
    snap = _first_snapshot(meta["cache_name"])
    if not snap:
        return None
    direct = snap / meta["filename"]
    if direct.exists():
        return direct
    matches = sorted(snap.glob("*.gguf"))
    return matches[0] if matches else None


def local_gemma_gguf() -> Optional[Path]:
    return local_llm_gguf("gemma-e2b")


def local_whisper_model(name: str) -> Optional[Path]:
    snap = _first_snapshot(f"models--Systran--faster-whisper-{name}")
    if snap and (snap / "model.bin").exists():
        return snap
    return None



LLM_SYSTEM = (
    "You are BOB, a friendly and helpful local AI voice assistant. "
    "If the user asks for code, a script, a website, an app, or any build/file request "
    "in any language or format (HTML, CSS, JavaScript, TypeScript, Python, Java, Rust, "
    "Go, C/C++, Ruby, PHP, SQL, shell scripts, config files, docs, etc.), "
    "do not print code in chat; the app will route that through the project-file builder. "
    "Keep responses concise and conversational — you are speaking aloud. "
    "Avoid markdown, bullet points, or lists unless explicitly asked. "
    "Be warm, natural, and direct."
)

TOOL_SYSTEM = f"""
You are BOB, a local assistant with a project-building tool harness.

You can do normal assistant conversation, but when the user asks you to build,
create, edit, inspect, reference, or manage files, use tools instead of merely
describing the work. All tool work is sandboxed inside this folder:
{PROJECTS_DIR}

Tool-call protocol:
- To call tools, reply with ONLY valid JSON shaped exactly like this:
  {{"tool_calls":[{{"tool":"tool_name","args":{{"key":"value"}}}}]}}
- You may include a short top-level "status" string in that same JSON to tell
  the user what you are doing, for example:
  {{"status":"Creating the HTML file now.","tool_calls":[{{"tool":"write_project_file","args":{{"path":"beautiful-site/index.html","content":"..."}}}}]}}
- You may call multiple tools at once when they are independent.
- After tool results are returned, either call more tools or give a concise
  finished answer for the user.
- Never wrap tool-call JSON in markdown.

Available tools:
- list_projects(): list top-level folders in projects/.
- list_project_tree(path=".", max_files=300): list files/folders under projects/.
- find_project_files(pattern="*", path=".", max_files=300): find files by glob.
- grep_project(pattern, path=".", glob="*", ignore_case=true, max_matches=80): search file contents.
- read_project_file(path, max_chars=12000, offset=1, limit=0): read a text file relative to projects/. Offset and limit are line-based.
- write_project_file(path, content): create or overwrite a text file relative to projects/.
- append_project_file(path, content): append to a text file relative to projects/.
- replace_in_project_file(path, old, new): replace exact text in a text file.
- edit_project_file(path, edits): make multiple exact replacements in one file. Each edit is {{"old":"...", "new":"..."}} or {{"oldText":"...", "newText":"..."}}.
- review_web_project(path): inspect a website folder and fix common incomplete styling issues, such as missing linked CSS.
- make_project_directory(path): create a folder relative to projects/.
- rename_project_path(old_path, new_path): rename/move a file or folder inside projects/.
- move_project_paths(paths, destination): move multiple files/folders into a destination folder inside projects/.
- delete_project_path(path): delete a file or folder relative to projects/.
- run_project_command(command, timeout=25): run a shell command with cwd locked to projects/.
- create_or_select_project(name): compatibility tool to create/select a top-level project folder.
- list_files(project, path="."): compatibility tool for listing inside a named project.
- read_file(project, path, max_chars=12000): compatibility tool for reading inside a named project.
- write_file(project, path, content): compatibility tool for writing inside a named project.
- append_file(project, path, content): compatibility tool for appending inside a named project.
- replace_in_file(project, path, old, new): compatibility tool for exact replacements.
- make_directory(project, path): compatibility tool for creating a directory.
- reference_workspace(max_files=120): list the main Bob repo and project files.
- read_workspace_file(path, max_chars=12000): read a text file from the main Bob repo.

Rules:
- Any request involving code, scripts, HTML, CSS, JavaScript, Python, apps,
  websites, or "build this" work MUST use the tool harness to create or edit
  real files in projects/. Do not answer those requests by printing source code
  in chat.
- You have full agency inside projects/: create named folders, inspect existing
  projects, edit files, delete files, and run commands when useful.
- For root-level project tools, path is always relative to projects/. To write
  an HTML file in a project, use:
  write_project_file(path="project-name/index.html", content="...")
- Do not pass a separate "project" argument to write_project_file. Put the
  project folder into the path instead.
- Choose meaningful folder names from the user's request, such as
  hello-world-site, snake-game, budget-tool, or weather-dashboard.
- Do not default to project-1/project-2 unless the user explicitly asks for a
  numbered scratch project or no meaningful name can be inferred.
- If the user asks to iterate on or modify an existing thing, inspect projects/
  first and edit the existing folder rather than creating a new one.
- Keep working in the selected/current project folder once you identify it.
  Do not switch to a new folder mid-task unless the user explicitly asks.
- Do not rename a project folder unless the user explicitly asks to rename it.
  Renaming is different from editing files inside a folder.
- When the user asks to move files into a folder, use move_project_paths for
  multiple files or rename_project_path for one file. If the destination is a
  folder, preserve each source file's basename inside that folder.
- For requests like "fix it", "change it", "update it", "make it better",
  "look at the code", or "reference the existing project", you MUST inspect
  projects/ with list_project_tree, find_project_files, grep_project, or
  read_project_file before answering.
- If you claim you fixed, changed, renamed, deleted, or improved something,
  you must have actually used a write/edit/delete/command tool in that turn.
- CRITICAL — improvement requests: when the user asks you to add, improve,
  enhance, update, or change ANYTHING (gradients, animations, content, styling,
  features, information, layout, colors, effects, etc.) you MUST call
  write_project_file or edit_project_file to make those actual changes.
  Reading files and saying "it looks fine" or "the project is complete" is
  NEVER acceptable as a response to an improvement request. You must output
  new/changed code using a tool call.
- review_web_project only checks for missing CSS links and bare structure
  issues. A result with no issues does NOT mean the user's requested
  improvements are done. Always make the improvements the user asked for even
  if review_web_project reports no problems.
- After inspecting an existing file, if the user asked for changes, your
  VERY NEXT action must be a write_project_file or edit_project_file call
  containing the requested improvements — not another read, not a text reply.
- You can create ANY type of text-based file: .py, .html, .css, .js, .ts,
  .tsx, .jsx, .java, .kt, .swift, .go, .rs, .c, .cpp, .h, .rb, .php, .sh,
  .sql, .r, .lua, .dart, .scala, .cs, .fs, .json, .yaml, .toml, .xml,
  .md, .txt, .csv, .env, .gitignore, Makefile, Dockerfile, and more.
  Use whichever extension matches what the user asked for.
- Prefer small complete apps: include runnable files, README notes, and obvious
  entry points when useful.
- For web projects, create polished responsive pages with meaningful CSS.
  If HTML links style.css, also write style.css.
  Prefer separate index.html and style.css unless the user asks for one file.
- For Python projects, create well-structured .py files. For CLI tools include
  argument parsing. For packages, create __init__.py and a README.
- For any project, create the files that make it actually runnable, not just
  a skeleton.
- If you create a folder for a build request, you must also create the actual
  requested file(s) before saying you are done.
- Before editing an existing file, inspect it first with read_project_file,
  grep_project, find_project_files, or list_project_tree unless the user clearly
  wants a fresh overwrite.
- Use edit_project_file for precise changes. Keep old text small but unique.
  If several edits are near each other, merge them into one replacement.
- Delete only what the user asked you to delete.
- Shell commands must be useful for the project and must not attempt to access
  files outside projects/.
- After the files are created or edited, give a short high-level summary:
  say the work is complete, name the folder(s), and list the important files or
  commands run. Do not include full source code unless the user explicitly asks
  to see a snippet.
- Never attempt to write outside projects/.
""".strip()

BUILD_MARKER = "@@BOB_BUILD@@"
BUILD_OUTPUT_SYSTEM = f"""
SYSTEM PROMPT:
You are BOB, a voice assistant. Act natural for normal conversation.

IMPORTANT CODING RULE:
If the user's request involves code, files, scripts, apps, or any programming/build
task in ANY language or format, you are NOT allowed to answer with code in chat.
You must output a machine-readable build signal so Bob can create real files.

OUTPUT FORMAT FOR CODING REQUESTS:
Output ONLY this exact marker, then valid JSON:

{BUILD_MARKER}
{{"project":"project-auto","files":[{{"path":"main.py","content":"<full file contents here>"}}]}}

Rules:
- Do not explain.
- Do not use markdown.
- Do not tell the user to copy and paste.
- Do not output anything before {BUILD_MARKER}.
- Generate complete, runnable file contents.
- The JSON must have a files array.
- Each file must have path and content.
- File paths must be relative, never absolute.
- The extension in path decides the file type — use whatever the user asked for:
  .py, .js, .ts, .html, .css, .java, .go, .rs, .cpp, .rb, .sh, .sql, .md, etc.
- For web projects: index.html + style.css + script.js (or whatever fits).
- For Python: main.py (or a named module).
- For Node/TS: index.js or index.ts with package.json if needed.
- For docs or config: README.md, .env, Dockerfile, Makefile, etc.
- For nested structures, use paths like src/main.py or lib/utils.ts.
""".strip()

# ─────────────────────────────── State ──────────────────────────────────────
class State:
    BOOT      = "boot"
    IDLE      = "idle"
    LISTENING = "listening"
    RECORDING = "recording"
    THINKING  = "thinking"
    SPEAKING  = "speaking"
    TYPING    = "typing"
    ERROR     = "error"

@dataclass
class Message:
    role:    str   # "user" | "bob"
    text:    str
    ts:      str   = field(default_factory=lambda: datetime.now().strftime("%H:%M"))

# ─────────────────────────────── UI ─────────────────────────────────────────
BOB_ART = """\
██████╗  ██████╗ ██████╗
██╔══██╗██╔═══██╗██╔══██╗
██████╔╝██║   ██║██████╔╝
██╔══██╗██║   ██║██╔══██╗
██████╔╝╚██████╔╝██████╔╝
╚═════╝  ╚═════╝ ╚═════╝"""

STATE_STYLE = {
    State.BOOT:      ("cyan",    "⏳", "Booting up…"),
    State.IDLE:      ("blue",    "💤", "Say  BOB  — or hold  SPACE  to talk — or press  T  to type"),
    State.LISTENING: ("green",   "👂", "Go ahead…"),
    State.RECORDING: ("green",   "🔴", "Listening — release SPACE when done"),
    State.THINKING:  ("yellow",  "🧠", "Thinking…"),
    State.SPEAKING:  ("magenta", "🔊", "Speaking…"),
    State.TYPING:    ("cyan",    "⌨️",  "Type your message — Enter to send  •  Esc to cancel"),
    State.ERROR:     ("red",     "⚠️",  "Error — check logs"),
}

# What to show in the live panel for each state
LIVE_LABEL = {
    State.BOOT:      ("cyan",    ""),
    State.IDLE:      ("blue",    ""),
    State.LISTENING: ("green",   ""),
    State.RECORDING: ("green",   "🎤  Hearing you:"),
    State.THINKING:  ("yellow",  "💭  BOB is thinking:"),
    State.SPEAKING:  ("magenta", "🔊  BOB is saying:"),
    State.TYPING:    ("cyan",    "⌨️   You are typing:"),
    State.ERROR:     ("red",     ""),
}

def build_ui(
    state: str,
    messages: List[Message],
    level_bar: str = "",
    status_line: str = "",
    live_text: str  = "",
    tts: str = "kokoro",
    settings_open: bool = False,
    settings_status: str = "",
    current_model: str = "",
    current_voice: str = "",
    settings_view: str = "main",
    settings_index: int = 0,
) -> Layout:
    color, icon, default_status = STATE_STYLE.get(state, ("white", "?", ""))
    status_text = status_line or default_status
    live_color, live_label = LIVE_LABEL.get(state, ("white", ""))

    # ── logo panel ──
    logo_text = Text(BOB_ART, style=f"bold {color}", justify="center")
    logo_panel = Panel(
        Align.center(logo_text),
        subtitle=f"[bold {color}]{icon}  {status_text}[/]",
        border_style=color,
        box=box.DOUBLE_EDGE,
        padding=(0, 2),
    )

    # ── audio level ──
    level_content = Text(level_bar, style=color) if level_bar else Text("")
    level_panel = Panel(
        level_content,
        title="[dim]mic[/]",
        border_style="dim",
        box=box.SIMPLE,
        height=3,
    )

    # ── live activity panel ──────────────────────────────────────────────────
    # Shows partial STT while recording, streaming tokens while thinking,
    # and BOB's words while speaking.
    if live_label and live_text:
        live_content = Text()
        live_content.append(f"{live_label}  ", style=f"bold {live_color}")
        live_content.append(live_text, style="bright_white")
    elif live_label and not live_text:
        live_content = Text(f"{live_label}  …", style=f"dim {live_color}")
    else:
        live_content = Text(
            "Say  BOB  or hold  SPACE  to talk" if state == State.IDLE else "",
            style="dim",
        )

    live_panel = Panel(
        live_content,
        title=f"[bold {live_color}]Live[/]" if live_label else "[dim]Live[/]",
        border_style=live_color if live_label else "dim",
        box=box.ROUNDED,
        height=4,
    )

    if settings_open:
        if settings_view == "voice":
            rows = [
                (
                    KOKORO_VOICES[key],
                    Text.assemble(
                        (key, "dim"),
                        ("  selected", "green") if KOKORO_VOICES[key] == current_voice else "",
                    ),
                )
                for key in KOKORO_VOICES
            ]
            title = "[bold cyan]Settings / Voice[/]"
            subtitle = "[dim]↑↓ move  Enter select  Esc back  S close[/]"
        elif settings_view == "model":
            rows = [
                (
                    meta["label"],
                    Text.assemble(
                        (meta["strengths"], "dim"),
                        ("  selected", "green") if meta["label"] == current_model else "",
                    ),
                )
                for meta in LLM_MODELS.values()
            ]
            title = "[bold cyan]Settings / AI Model[/]"
            subtitle = "[dim]↑↓ move  Enter select  Esc back  S close[/]"
        else:
            rows = [
                ("Voice", Text.assemble(("Current: ", "dim"), (current_voice, "bold"))),
                ("AI model", Text.assemble(("Current: ", "dim"), (current_model, "bold"))),
                ("Downloads", "Download/check selected model and core assets"),
                ("Close settings", "Return to Bob"),
            ]
            title = "[bold cyan]Settings[/]"
            subtitle = "[dim]↑↓ move  Enter open/select  Esc or S close[/]"
        chat_panel = Panel(
            Align.center(build_settings_content(rows, settings_index, settings_status), vertical="middle"),
            title=title,
            subtitle=subtitle,
            border_style="cyan",
            box=box.ROUNDED,
        )
    else:
        # ── conversation history ─────────────────────────────────────────────
        chat_width = max(40, console.width - 8)
        text_width = max(20, chat_width - 16)
        chat_height = max(6, console.height - 22)
        rendered_lines: List[tuple[str, str]] = []

        for m in messages:
            who = "YOU" if m.role == "user" else "BOB"
            style = "green" if m.role == "user" else color
            body_style = "white" if m.role == "user" else "bright_white"
            clean_text = re.sub(r"\s+", " ", (m.text or "").strip())
            wrapped = textwrap.wrap(
                clean_text,
                width=text_width,
                replace_whitespace=False,
                drop_whitespace=True,
            ) or [""]
            rendered_lines.append((f"{m.ts:>5}  {who:<3}  {wrapped[0]}", f"bold {style}" if m.role == "user" else body_style))
            for line in wrapped[1:]:
                rendered_lines.append((f"{'':>5}  {'':<3}  {line}", body_style))

        if not rendered_lines:
            rendered_lines.append(("No messages yet - say BOB to start.", "dim italic"))

        tail = rendered_lines[-chat_height:]
        chat_text = Text()
        for line, line_style in tail:
            if "  YOU  " in line:
                ts_part, rest = line[:5], line[7:]
                chat_text.append(ts_part, style="dim")
                chat_text.append("  ")
                chat_text.append("YOU", style="bold green")
                chat_text.append(rest[3:] + "\n", style="white")
            elif "  BOB  " in line:
                ts_part, rest = line[:5], line[7:]
                chat_text.append(ts_part, style="dim")
                chat_text.append("  ")
                chat_text.append("BOB", style=f"bold {color}")
                chat_text.append(rest[3:] + "\n", style="bright_white")
            else:
                chat_text.append(line + "\n", style=line_style)

        chat_panel = Panel(
            chat_text,
            title="[bold]Conversation[/]",
            border_style="dim",
            box=box.ROUNDED,
        )

    # ── footer ──
    footer = Text(
        "  Ctrl+C: quit  │  SPACE: push-to-talk  │  T: type  │  S: settings  ",
        style="dim",
        justify="center",
    )

    layout = Layout()
    layout.split_column(
        Layout(logo_panel,  name="logo",   size=10),
        Layout(level_panel, name="level",  size=3),
        Layout(live_panel,  name="live",   size=4),
        Layout(chat_panel,  name="chat"),
        Layout(footer,      name="footer", size=1),
    )
    return layout


def build_settings_content(
    rows: List[tuple],
    settings_index: int,
    settings_status: str = "",
) -> Group:
    lines = []
    max_name = max((len(str(name)) for name, _ in rows), default=0)
    rendered_rows = []
    for i, (name, detail) in enumerate(rows):
        selected = i == settings_index
        pointer = ">" if selected else " "
        style = "bold bright_white" if selected else "white"
        line = Text()
        line.append(pointer, style="bold cyan" if selected else "dim")
        line.append(" ")
        line.append(str(name).ljust(max_name), style=style)
        if detail:
            line.append("  ")
            if isinstance(detail, Text):
                line.append_text(detail)
            else:
                line.append(str(detail), style="white" if selected else "dim")
        rendered_rows.append(line)

    width = min(
        max((line.cell_len for line in rendered_rows), default=0),
        max(48, console.width - 10),
    )
    for line in rendered_rows:
        padded = Text()
        padded.append_text(line)
        if line.cell_len < width:
            padded.append(" " * (width - line.cell_len))
        lines.append(padded)

    block = [Align.center(Group(*lines))]
    if settings_status:
        block.append(Text(""))
        block.append(Align.center(Text(settings_status, style="yellow")))

    return Group(*block)


@contextlib.contextmanager
def quiet_terminal_input():
    """Prevent keypresses from being echoed under the Rich live UI."""
    if not sys.stdin.isatty():
        yield
        return
    try:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        new = old[:]
        new[3] = new[3] & ~(termios.ECHO | termios.ICANON)
        termios.tcsetattr(fd, termios.TCSADRAIN, new)
        try:
            yield
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except Exception:
        yield


def make_level_bar(amplitude: float, width: int = 50) -> str:
    """Convert 0‥1 amplitude to a unicode block bar."""
    bars   = int(amplitude * width)
    filled = "█" * bars
    empty  = "░" * (width - bars)
    pct    = f" {int(amplitude * 100):3d}%"
    return filled + empty + pct


# ─────────────────────────────── Audio helpers ──────────────────────────────
class AudioCapture:
    """Continuous microphone capture — stores frames in a queue."""

    def __init__(self):
        self.queue: queue.Queue = queue.Queue()
        self._stream: Optional[sd.InputStream] = None
        self.level: float = 0.0

    def start(self):
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()

    def _callback(self, indata, frames, time_info, status):
        chunk = np.clip(indata[:, 0].copy() * MIC_GAIN, -1.0, 1.0)
        self.level = float(np.abs(chunk).mean())
        self.queue.put(chunk)

    def read_chunk(self, timeout=0.1) -> Optional[np.ndarray]:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def flush(self):
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break


# ─────────────────────────────── Wake word ──────────────────────────────────
class WakeWordDetector:
    """Sliding 1.5-second window on a dedicated Whisper-tiny model."""

    CHECK_EVERY  = 15                                   # chunks between inferences
    BUF_FRAMES   = int(1.5 * SAMPLE_RATE / CHUNK_SAMPLES)
    SILENCE_GATE = WAKE_SILENCE_GATE

    def __init__(self):
        from faster_whisper import WhisperModel
        model_path = local_whisper_model("tiny")
        if model_path is None:
            if not ALLOW_ONLINE:
                raise RuntimeError(
                    "Whisper tiny is not cached locally. Run `python bob.py --setup` online once, "
                    "or set BOB_ALLOW_ONLINE=1 to allow downloads."
                )
            model_path = "tiny"
        self._whisper = WhisperModel(str(model_path), device="auto", compute_type="int8")
        self._buf: List[np.ndarray] = []
        self._ticks = 0

    def feed(self, chunk: np.ndarray) -> bool:
        self._buf.append(chunk)
        if len(self._buf) > self.BUF_FRAMES:
            self._buf.pop(0)
        self._ticks += 1
        if self._ticks < self.CHECK_EVERY:
            return False
        self._ticks = 0
        if len(self._buf) < self.BUF_FRAMES // 2:
            return False
        audio = np.concatenate(self._buf)
        if np.abs(audio).mean() < self.SILENCE_GATE:
            return False
        try:
            segments, _ = self._whisper.transcribe(
                audio, language="en", beam_size=1,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 100, "threshold": 0.3},
            )
            transcript = re.sub(r"[^a-z ]", "",
                                 " ".join(s.text for s in segments).lower().strip())
            if any(w in transcript for w in WAKE_WORDS):
                self._buf.clear()
                self._ticks = 0
                return True
        except Exception:
            pass
        return False


# ─────────────────────────────── Push-to-talk ───────────────────────────────
class PushToTalk:
    """Watches the keyboard globally via pynput (no terminal focus needed).

    Normal mode: space = push-to-talk, letter keys = settings shortcuts.
    Text mode: all printable keys build a message buffer; Enter submits it.
    """

    def __init__(self, on_key=None, on_text_char=None):
        self._held           = False
        self._press_event    = threading.Event()
        self._release_event  = threading.Event()
        self._listener       = None
        self._on_key         = on_key
        self._on_text_char   = on_text_char
        self._text_mode      = False

    @property
    def text_mode(self) -> bool:
        return self._text_mode

    @text_mode.setter
    def text_mode(self, value: bool):
        self._text_mode = value

    def start(self):
        from pynput import keyboard

        def on_press(key):
            if self._text_mode:
                # In text-input mode: route everything to the char handler.
                if key == keyboard.Key.esc and self._on_key:
                    threading.Thread(target=self._on_key, args=("esc",), daemon=True).start()
                elif key == keyboard.Key.enter and self._on_text_char:
                    threading.Thread(target=self._on_text_char, args=("\n",), daemon=True).start()
                elif key == keyboard.Key.backspace and self._on_text_char:
                    threading.Thread(target=self._on_text_char, args=("\x08",), daemon=True).start()
                elif key == keyboard.Key.space and self._on_text_char:
                    threading.Thread(target=self._on_text_char, args=(" ",), daemon=True).start()
                elif hasattr(key, "char") and key.char and self._on_text_char:
                    threading.Thread(target=self._on_text_char, args=(key.char,), daemon=True).start()
                return  # never fire PTT or settings keys while typing

            # Normal mode
            if key == keyboard.Key.space and not self._held:
                self._held = True
                self._release_event.clear()
                self._press_event.set()
            elif key == keyboard.Key.esc and self._on_key:
                threading.Thread(target=self._on_key, args=("esc",), daemon=True).start()
            elif key == keyboard.Key.up and self._on_key:
                threading.Thread(target=self._on_key, args=("up",), daemon=True).start()
            elif key == keyboard.Key.down and self._on_key:
                threading.Thread(target=self._on_key, args=("down",), daemon=True).start()
            elif key == keyboard.Key.enter and self._on_key:
                threading.Thread(target=self._on_key, args=("enter",), daemon=True).start()
            elif hasattr(key, "char") and key.char and self._on_key:
                threading.Thread(target=self._on_key, args=(key.char.lower(),), daemon=True).start()

        def on_release(key):
            if key == keyboard.Key.space and not self._text_mode:
                self._held = False
                self._press_event.clear()
                self._release_event.set()

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.daemon = True
        self._listener.start()

    def stop(self):
        if self._listener:
            self._listener.stop()

    @property
    def is_held(self) -> bool:
        return self._held

    def wait_for_press(self, timeout=0.1) -> bool:
        return self._press_event.wait(timeout=timeout)

    def wait_for_release(self, timeout=0.05) -> bool:
        return self._release_event.wait(timeout=timeout)


# ─────────────────────────────── STT ────────────────────────────────────────
class STTEngine:
    def __init__(self, whisper_model):
        self._whisper = whisper_model

    def transcribe(self, audio: np.ndarray) -> str:
        try:
            segments, _ = self._whisper.transcribe(
                audio,
                language="en",
                beam_size=3,
                vad_filter=True,
            )
            return " ".join(s.text for s in segments).strip()
        except Exception as e:
            console.print(f"[red]STT error: {e}[/]")
            return ""


# ─────────────────────────────── Tool harness ───────────────────────────────
class ProjectWorkspace:
    """Safe file tools for BOB's generated projects."""

    # Known binary formats that must never be written as text.
    # Everything else (including unknown extensions) is treated as text.
    BINARY_EXTENSIONS = {
        ".exe", ".dll", ".so", ".dylib", ".bin", ".pak", ".dat",
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".tiff", ".tif",
        ".mp3", ".mp4", ".wav", ".ogg", ".flac", ".avi", ".mov", ".mkv", ".webm",
        ".pdf", ".zip", ".tar", ".gz", ".bz2", ".xz", ".rar", ".7z", ".dmg", ".iso",
        ".db", ".sqlite", ".sqlite3",
        ".pyc", ".pyo", ".pyd", ".class", ".o", ".a", ".obj", ".lib", ".wasm",
        ".ttf", ".otf", ".woff", ".woff2", ".eot",
    }

    def __init__(self, root: Path = PROJECTS_DIR):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _slug(name: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9._ -]", "", (name or "default").strip())
        cleaned = re.sub(r"\s+", "-", cleaned).strip(".-").lower()
        return cleaned[:60] or "default"

    def _project_dir(self, project: str) -> Path:
        path = (self.root / self._slug(project)).resolve()
        if self.root not in path.parents and path != self.root:
            raise ValueError("Project path escaped the projects folder.")
        return path

    def _safe_root_path(self, path: str = ".") -> Path:
        raw = str(path or ".").replace("\\", "/").lstrip("/")
        target = (self.root / raw).resolve()
        if target != self.root and self.root not in target.parents:
            raise ValueError("Path escaped the projects folder.")
        return target

    def _safe_path(self, project: str, path: str = ".") -> Path:
        base = self._project_dir(project)
        target = (base / (path or ".")).resolve()
        if target != base and base not in target.parents:
            raise ValueError("Path escaped the selected project folder.")
        return target

    def _is_text_path(self, path: Path) -> bool:
        """Allow any file that is not a known binary format."""
        return path.suffix.lower() not in self.BINARY_EXTENSIONS

    def _unescape_generated_text(self, content: str) -> str:
        text = str(content or "")
        stripped = text.strip()
        if (
            len(stripped) >= 2
            and stripped[0] == stripped[-1]
            and stripped[0] in {'"', "'"}
        ):
            text = stripped[1:-1]

        # Models sometimes return JSON-string-looking content that skipped
        # JSON decoding. Turn visible escapes into real file formatting.
        if any(seq in text for seq in ("\\n", "\\t", '\\"', "\\/")):
            text = (
                text.replace("\\r\\n", "\n")
                .replace("\\n", "\n")
                .replace("\\t", "    ")
                .replace('\\"', '"')
                .replace("\\'", "'")
                .replace("\\/", "/")
            )
        return text

    def _normalize_html_content(self, content: str) -> str:
        html = self._unescape_generated_text(content)
        html = html.replace("│", "").replace("┃", "").replace("║", "")

        start = re.search(
            r"<!doctype\s+html\b|<\s*html\b|<\s*head\b|<\s*body\b",
            html,
            flags=re.IGNORECASE,
        )
        if start:
            html = html[start.start():]

        # If another file object got glued after the HTML, do not write that
        # JSON into index.html. Keep late <style>/<script> blocks by moving them
        # back inside the document first.
        close = re.search(r"</html\s*>", html, flags=re.IGNORECASE)
        if close:
            main = html[:close.end()]
            tail = html[close.end():]

            for style_block in re.findall(r"<style[\s\S]*?</style\s*>", tail, flags=re.IGNORECASE):
                if "</head>" in main.lower():
                    main = re.sub(r"</head\s*>", style_block.strip() + "\n</head>", main, count=1, flags=re.IGNORECASE)
                else:
                    main = style_block.strip() + "\n" + main

            for script_block in re.findall(r"<script[\s\S]*?</script\s*>", tail, flags=re.IGNORECASE):
                if "</body>" in main.lower():
                    main = re.sub(r"</body\s*>", script_block.strip() + "\n</body>", main, count=1, flags=re.IGNORECASE)
                else:
                    main = re.sub(r"</html\s*>", script_block.strip() + "\n</html>", main, count=1, flags=re.IGNORECASE)
            html = main

        html = re.sub(r"\n\s*\{?\s*\"path\"\s*:\s*\"[^\"]+\"[\s\S]*$", "", html).strip()

        if "<html" not in html.lower():
            html = "<!doctype html>\n<html lang=\"en\">\n<body>\n" + html.strip() + "\n</body>\n</html>"
        if "</body>" not in html.lower() and "<body" in html.lower():
            html += "\n</body>"
        if "</html>" not in html.lower():
            html += "\n</html>"
        if not html.lower().lstrip().startswith("<!doctype html"):
            html = "<!doctype html>\n" + html.lstrip()

        return html.strip() + "\n"

    def _normalize_generated_file_content(self, path: Path, content: str) -> str:
        suffix = path.suffix.lower()
        if suffix in {".html", ".htm"}:
            return self._normalize_html_content(content)
        text = self._unescape_generated_text(content)
        return text if text.endswith("\n") else text + "\n"

    def _safe_workspace_path(self, path: str) -> Path:
        target = (_HERE / (path or ".")).resolve()
        if target != _HERE.resolve() and _HERE.resolve() not in target.parents:
            raise ValueError("Path escaped the Bob workspace.")
        return target

    def create_or_select_project(self, name: str) -> dict:
        project_dir = self._project_dir(name)
        project_dir.mkdir(parents=True, exist_ok=True)
        return {"project": project_dir.name, "path": str(project_dir)}

    def list_projects(self) -> dict:
        projects = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        return {"projects": projects}

    def list_project_tree(self, path: str = ".", max_files: int = 300) -> dict:
        base = self._safe_root_path(path)
        if not base.exists():
            return {"path": str(Path(path or ".")), "items": [], "note": "Path does not exist."}
        if base.is_file():
            return {"path": str(base.relative_to(self.root)), "items": [str(base.relative_to(self.root))]}

        ignored = {".DS_Store", "__pycache__", ".git"}
        items = []
        for item in sorted(base.rglob("*")):
            rel_parts = item.relative_to(self.root).parts
            if any(part in ignored for part in rel_parts):
                continue
            suffix = "/" if item.is_dir() else ""
            items.append(str(item.relative_to(self.root)) + suffix)
        limit = max(20, min(int(max_files or 300), 800))
        return {"root": str(self.root), "items": items[:limit], "truncated": len(items) > limit}

    def find_project_files(self, pattern: str = "*", path: str = ".", max_files: int = 300) -> dict:
        base = self._safe_root_path(path)
        if not base.exists():
            return {"matches": [], "note": "Path does not exist."}
        ignored = {".DS_Store", "__pycache__", ".git"}
        pattern = str(pattern or "*")
        matches = []
        candidates = [base] if base.is_file() else sorted(base.rglob("*"))
        for item in candidates:
            if not item.is_file():
                continue
            rel_parts = item.relative_to(self.root).parts
            if any(part in ignored for part in rel_parts):
                continue
            rel = str(item.relative_to(self.root))
            if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(item.name, pattern):
                matches.append(rel)
        limit = max(20, min(int(max_files or 300), 800))
        return {"matches": matches[:limit], "truncated": len(matches) > limit}

    def grep_project(
        self,
        pattern: str,
        path: str = ".",
        glob: str = "*",
        ignore_case: bool = True,
        max_matches: int = 80,
    ) -> dict:
        base = self._safe_root_path(path)
        if not base.exists():
            return {"matches": [], "note": "Path does not exist."}
        try:
            flags = re.IGNORECASE if ignore_case else 0
            regex = re.compile(pattern, flags)
        except re.error as e:
            return {"error": f"Invalid regex: {e}"}

        matches = []
        limit = max(1, min(int(max_matches or 80), 300))
        candidates = [base] if base.is_file() else sorted(base.rglob("*"))
        for item in candidates:
            if len(matches) >= limit:
                break
            if not item.is_file() or item.name == ".DS_Store" or not self._is_text_path(item):
                continue
            rel = str(item.relative_to(self.root))
            if glob and glob != "*" and not (fnmatch.fnmatch(rel, glob) or fnmatch.fnmatch(item.name, glob)):
                continue
            try:
                lines = item.read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception:
                continue
            for line_no, line in enumerate(lines, start=1):
                if regex.search(line):
                    snippet = line[:240] + ("..." if len(line) > 240 else "")
                    matches.append({"path": rel, "line": line_no, "text": snippet})
                    if len(matches) >= limit:
                        break
        return {"matches": matches, "truncated": len(matches) >= limit}

    def read_project_file(self, path: str, max_chars: int = 12000, offset: int = 1, limit: int = 0) -> dict:
        target = self._safe_root_path(path)
        if not target.exists() or not target.is_file():
            return {"error": "File not found."}
        if not self._is_text_path(target):
            return {"error": "Refusing to read a non-text file."}
        text = target.read_text(encoding="utf-8", errors="replace")
        all_lines = text.splitlines()
        start = max(0, int(offset or 1) - 1)
        if start >= len(all_lines) and all_lines:
            return {"error": f"Offset {offset} is beyond end of file ({len(all_lines)} lines)."}
        if int(limit or 0) > 0:
            selected_lines = all_lines[start:start + int(limit)]
        else:
            selected_lines = all_lines[start:]
        selected = "\n".join(selected_lines)
        if text.endswith("\n") and selected_lines:
            selected += "\n"
        char_limit = max(1000, min(int(max_chars or 12000), 40000))
        content = selected[:char_limit]
        shown_lines = content.splitlines()
        next_offset = start + len(shown_lines) + 1
        truncated = len(selected) > char_limit or (start + len(selected_lines) < len(all_lines))
        return {
            "path": str(target.relative_to(self.root)),
            "content": content,
            "offset": start + 1,
            "total_lines": len(all_lines),
            "next_offset": next_offset if truncated and next_offset <= len(all_lines) else None,
            "truncated": truncated,
        }

    def write_project_file(self, path: str, content: str) -> dict:
        target = self._safe_root_path(path)
        if target == self.root:
            return {"error": "Refusing to overwrite the projects folder."}
        if not self._is_text_path(target):
            return {"error": "Refusing to write a non-text file extension."}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self._normalize_generated_file_content(target, content or ""), encoding="utf-8")
        return {"path": str(target.relative_to(self.root)), "absolute_path": str(target), "bytes": target.stat().st_size}

    def append_project_file(self, path: str, content: str) -> dict:
        target = self._safe_root_path(path)
        if not self._is_text_path(target):
            return {"error": "Refusing to append to a non-text file extension."}
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as f:
            f.write(self._unescape_generated_text(content or ""))
        return {"path": str(target.relative_to(self.root)), "absolute_path": str(target), "bytes": target.stat().st_size}

    def replace_in_project_file(self, path: str, old: str, new: str) -> dict:
        target = self._safe_root_path(path)
        if not target.exists() or not target.is_file():
            return {"error": "File not found."}
        if not self._is_text_path(target):
            return {"error": "Refusing to edit a non-text file."}
        text = target.read_text(encoding="utf-8", errors="replace")
        if old not in text:
            return {"error": "The old text was not found."}
        count = text.count(old)
        target.write_text(text.replace(old, self._unescape_generated_text(new or "")), encoding="utf-8")
        return {"path": str(target.relative_to(self.root)), "absolute_path": str(target), "replacements": count}

    def edit_project_file(self, path: str, edits: List[dict]) -> dict:
        target = self._safe_root_path(path)
        if not target.exists() or not target.is_file():
            return {"error": "File not found."}
        if not self._is_text_path(target):
            return {"error": "Refusing to edit a non-text file."}
        if not isinstance(edits, list) or not edits:
            return {"error": "edits must be a non-empty list."}

        text = target.read_text(encoding="utf-8", errors="replace")
        normalized_edits = []
        spans = []
        for i, edit in enumerate(edits, start=1):
            if not isinstance(edit, dict):
                return {"error": f"Edit {i} is not an object."}
            old = edit.get("old")
            if old is None:
                old = edit.get("oldText")
            new = edit.get("new")
            if new is None:
                new = edit.get("newText")
            if not isinstance(old, str) or old == "":
                return {"error": f"Edit {i} has empty old text."}
            if not isinstance(new, str):
                return {"error": f"Edit {i} has invalid new text."}
            count = text.count(old)
            if count != 1:
                return {"error": f"Edit {i} old text matched {count} times; it must match exactly once."}
            start = text.index(old)
            end = start + len(old)
            spans.append((start, end, i))
            normalized_edits.append((old, self._unescape_generated_text(new)))

        spans_sorted = sorted(spans)
        for (_, prev_end, prev_i), (next_start, _, next_i) in zip(spans_sorted, spans_sorted[1:]):
            if next_start < prev_end:
                return {"error": f"Edits {prev_i} and {next_i} overlap. Merge nearby changes into one edit."}

        updated = text
        for old, new in normalized_edits:
            updated = updated.replace(old, new, 1)
        target.write_text(updated, encoding="utf-8")
        return {
            "path": str(target.relative_to(self.root)),
            "absolute_path": str(target),
            "edits": len(normalized_edits),
            "bytes": target.stat().st_size,
        }

    def _default_website_css(self) -> str:
        return """/* BOB generated responsive website styling */
:root {
  --bg: #f6f7fb;
  --surface: #ffffff;
  --surface-soft: #eef3ff;
  --text: #172033;
  --muted: #5c677d;
  --primary: #3157d5;
  --primary-dark: #1f3fa8;
  --accent: #16a085;
  --border: rgba(23, 32, 51, 0.12);
  --shadow: 0 18px 45px rgba(23, 32, 51, 0.12);
}

* {
  box-sizing: border-box;
}

html {
  scroll-behavior: smooth;
}

body {
  margin: 0;
  font-family: Inter, "Segoe UI", Arial, sans-serif;
  color: var(--text);
  background:
    radial-gradient(circle at top left, rgba(49, 87, 213, 0.16), transparent 32rem),
    linear-gradient(180deg, #ffffff 0%, var(--bg) 45%, #edf2f7 100%);
  line-height: 1.6;
}

header {
  min-height: 58vh;
  display: grid;
  align-content: center;
  gap: 1.5rem;
  padding: 3rem clamp(1rem, 5vw, 5rem);
  background:
    linear-gradient(135deg, rgba(49, 87, 213, 0.94), rgba(22, 160, 133, 0.82)),
    linear-gradient(180deg, #3157d5, #16a085);
  color: white;
}

header h1 {
  max-width: 900px;
  margin: 0;
  font-size: clamp(2.4rem, 7vw, 5.5rem);
  line-height: 0.95;
}

nav ul {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  margin: 0;
  padding: 0;
  list-style: none;
}

nav a {
  display: inline-flex;
  align-items: center;
  min-height: 2.5rem;
  padding: 0.55rem 0.9rem;
  border: 1px solid rgba(255, 255, 255, 0.45);
  border-radius: 999px;
  color: white;
  text-decoration: none;
  font-weight: 700;
  background: rgba(255, 255, 255, 0.12);
}

main {
  width: min(1120px, calc(100% - 2rem));
  margin: -4rem auto 0;
  display: grid;
  gap: 1rem;
}

section {
  padding: clamp(1.25rem, 4vw, 2.5rem);
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--surface);
  box-shadow: var(--shadow);
}

section:nth-child(even) {
  background: var(--surface-soft);
}

h2 {
  margin: 0 0 0.75rem;
  font-size: clamp(1.5rem, 3vw, 2.25rem);
}

p {
  margin: 0;
  color: var(--muted);
  font-size: 1.05rem;
}

footer {
  margin-top: 3rem;
  padding: 2rem 1rem;
  text-align: center;
  color: white;
  background: #172033;
}

footer p {
  color: rgba(255, 255, 255, 0.78);
}

@media (max-width: 700px) {
  header {
    min-height: 48vh;
  }

  main {
    margin-top: -2rem;
  }

  nav ul {
    align-items: stretch;
    flex-direction: column;
  }

  nav a {
    justify-content: center;
  }
}
"""

    def review_web_project(self, path: str = ".") -> dict:
        base = self._safe_root_path(path)
        if not base.exists():
            return {"path": path, "issues": ["Path does not exist."], "fixes": []}
        root = base if base.is_dir() else base.parent
        html_files = [base] if base.is_file() and base.suffix.lower() in {".html", ".htm"} else sorted(root.rglob("*.html"))
        issues = []
        fixes = []
        css_files_seen = set()

        for html_path in html_files[:20]:
            if not html_path.is_file():
                continue
            rel_html = str(html_path.relative_to(self.root))
            html = html_path.read_text(encoding="utf-8", errors="replace")
            linked_css = re.findall(
                r"<link[^>]+rel=[\"']stylesheet[\"'][^>]*href=[\"']([^\"']+)[\"'][^>]*>",
                html,
                flags=re.IGNORECASE,
            )
            if not linked_css:
                css_rel_to_html = "style.css"
                css_path = html_path.parent / css_rel_to_html
                link_tag = f'    <link rel="stylesheet" href="{css_rel_to_html}">\n'
                if re.search(r"</head\s*>", html, flags=re.IGNORECASE):
                    html = re.sub(r"</head\s*>", link_tag + "</head>", html, count=1, flags=re.IGNORECASE)
                else:
                    html = html.replace("<body", "<head>\n" + link_tag + "</head>\n<body", 1)
                html_path.write_text(html, encoding="utf-8")
                linked_css = [css_rel_to_html]
                fixes.append(f"Added stylesheet link to {rel_html}.")

            for href in linked_css:
                if re.match(r"^(?:https?:)?//", href) or href.startswith("#"):
                    continue
                css_path = (html_path.parent / href.split("#", 1)[0].split("?", 1)[0]).resolve()
                if css_path != self.root and self.root not in css_path.parents:
                    issues.append(f"Stylesheet path escapes projects/: {href}")
                    continue
                css_files_seen.add(css_path)
                css_missing = not css_path.exists()
                css_tiny = False
                if css_path.exists():
                    try:
                        css_tiny = css_path.stat().st_size < 600
                    except Exception:
                        css_tiny = True
                if css_missing or css_tiny:
                    css_path.parent.mkdir(parents=True, exist_ok=True)
                    existing = ""
                    if css_path.exists():
                        existing = css_path.read_text(encoding="utf-8", errors="replace").strip()
                    css_content = self._default_website_css()
                    if existing and existing not in css_content:
                        css_content = existing + "\n\n" + css_content
                    css_path.write_text(css_content, encoding="utf-8")
                    reason = "Created" if css_missing else "Expanded"
                    fixes.append(f"{reason} stylesheet {css_path.relative_to(self.root)}.")

            if not re.search(r"<main\b", html, flags=re.IGNORECASE):
                issues.append(f"{rel_html} has no <main> landmark.")
            if len(re.findall(r"<section\b", html, flags=re.IGNORECASE)) < 2:
                issues.append(f"{rel_html} has very little page structure.")

        if not html_files:
            issues.append("No HTML files found.")

        return {
            "path": str(root.relative_to(self.root) if root != self.root else Path(".")),
            "html_files": [str(p.relative_to(self.root)) for p in html_files[:20]],
            "css_files": [str(p.relative_to(self.root)) for p in sorted(css_files_seen)],
            "issues": issues,
            "fixes": fixes,
        }

    def make_project_directory(self, path: str) -> dict:
        target = self._safe_root_path(path)
        target.mkdir(parents=True, exist_ok=True)
        return {"path": str(target.relative_to(self.root) if target != self.root else Path(".")), "absolute_path": str(target)}

    def rename_project_path(self, old_path: str, new_path: str) -> dict:
        old_target = self._safe_root_path(old_path)
        new_target = self._safe_root_path(new_path)
        if old_target == self.root or new_target == self.root:
            return {"error": "Refusing to rename the entire projects folder."}
        if not old_target.exists():
            return {"error": "Source path not found."}

        # If destination is an existing folder, treat this as "move into folder"
        # and preserve the source basename. This matches how users naturally ask
        # for file organization tasks.
        if new_target.exists() and new_target.is_dir():
            new_target = (new_target / old_target.name).resolve()
            if new_target != self.root and self.root not in new_target.parents:
                return {"error": "Destination escaped the projects folder."}
        if new_target.exists():
            return {"error": "Destination already exists."}
        new_target.parent.mkdir(parents=True, exist_ok=True)
        old_rel = str(old_target.relative_to(self.root))
        new_rel = str(new_target.relative_to(self.root))
        old_target.rename(new_target)
        return {
            "old_path": old_rel,
            "path": new_rel,
            "absolute_path": str(new_target),
            "renamed": True,
            "kind": "directory" if new_target.is_dir() else "file",
        }

    def move_project_paths(self, paths: List[str], destination: str) -> dict:
        if not isinstance(paths, list) or not paths:
            return {"error": "paths must be a non-empty list."}
        dest_dir = self._safe_root_path(destination)
        if dest_dir == self.root:
            return {"error": "Refusing to move files directly onto the projects root."}
        dest_dir.mkdir(parents=True, exist_ok=True)
        if not dest_dir.is_dir():
            return {"error": "Destination is not a directory."}

        moved = []
        errors = []
        for raw_path in paths:
            try:
                src = self._safe_root_path(str(raw_path))
                if src == self.root:
                    errors.append({"path": str(raw_path), "error": "Refusing to move the projects folder."})
                    continue
                if not src.exists():
                    errors.append({"path": str(raw_path), "error": "Source path not found."})
                    continue
                target = (dest_dir / src.name).resolve()
                if target != self.root and self.root not in target.parents:
                    errors.append({"path": str(raw_path), "error": "Destination escaped projects/."})
                    continue
                if target.exists():
                    errors.append({"path": str(raw_path), "error": f"Destination already exists: {target.relative_to(self.root)}"})
                    continue
                old_rel = str(src.relative_to(self.root))
                src.rename(target)
                moved.append({"old_path": old_rel, "path": str(target.relative_to(self.root))})
            except Exception as e:
                errors.append({"path": str(raw_path), "error": str(e)})

        return {
            "path": str(dest_dir.relative_to(self.root)),
            "moved": moved,
            "errors": errors,
            "ok_count": len(moved),
            "error_count": len(errors),
        }

    def delete_project_path(self, path: str) -> dict:
        target = self._safe_root_path(path)
        if target == self.root:
            return {"error": "Refusing to delete the entire projects folder."}
        if not target.exists():
            return {"path": str(Path(path or ".")), "deleted": False, "note": "Path does not exist."}
        rel = str(target.relative_to(self.root))
        if target.is_dir():
            shutil.rmtree(target)
            kind = "directory"
        else:
            target.unlink()
            kind = "file"
        return {"path": rel, "deleted": True, "kind": kind}

    def _command_is_safe(self, command: str) -> tuple[bool, str]:
        cmd = str(command or "").strip()
        if not cmd:
            return False, "Command is empty."
        lowered = cmd.lower()
        blocked_fragments = [
            "..", "~/", "$home", "${home}", "%userprofile%", "%homedrive%", " sudo ",
            "sudo ", " su ", "su ", "chmod 777", "chown ", "mkfs", "diskutil",
        ]
        padded = f" {lowered} "
        for fragment in blocked_fragments:
            if fragment in padded or fragment in lowered:
                return False, f"Command rejected because it may escape projects/: {fragment.strip()}"
        if re.search(r"(^|[\s=:])/(?!bin/|usr/bin/|usr/local/bin/|opt/homebrew/bin/)", cmd):
            return False, "Command rejected because it references an absolute path."
        return True, ""

    def run_project_command(self, command: str, timeout: int = 25) -> dict:
        ok, reason = self._command_is_safe(command)
        if not ok:
            return {"error": reason}
        limit = max(1, min(int(timeout or 25), 60))
        env = os.environ.copy()
        env["HOME"] = str(self.root)
        env["BOB_PROJECTS_DIR"] = str(self.root)
        env["BOB_PYTHON"] = sys.executable
        python_bin_dir = str(Path(sys.executable).parent)
        env["PATH"] = python_bin_dir + os.pathsep + env.get("PATH", "")
        shell_executable = "/bin/bash" if Path("/bin/bash").exists() else None
        completed = subprocess.run(
            command,
            cwd=str(self.root),
            env=env,
            shell=True,
            executable=shell_executable,
            text=True,
            capture_output=True,
            timeout=limit,
        )
        stdout = (completed.stdout or "")[-6000:]
        stderr = (completed.stderr or "")[-6000:]
        return {
            "command": command,
            "cwd": str(self.root),
            "returncode": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }

    def list_files(self, project: str, path: str = ".") -> dict:
        base = self._safe_path(project, path)
        if not base.exists():
            return {"files": [], "note": "Path does not exist."}
        if base.is_file():
            if base.name == ".DS_Store":
                return {"files": []}
            return {"files": [str(base.relative_to(self._project_dir(project)))]}

        files = []
        for item in sorted(base.rglob("*")):
            if item.is_file() and item.name != ".DS_Store":
                files.append(str(item.relative_to(self._project_dir(project))))
        return {"files": files[:200], "truncated": len(files) > 200}

    def read_file(self, project: str, path: str, max_chars: int = 12000) -> dict:
        target = self._safe_path(project, path)
        if not target.exists() or not target.is_file():
            return {"error": "File not found."}
        if not self._is_text_path(target):
            return {"error": "Refusing to read a non-text file."}
        text = target.read_text(encoding="utf-8", errors="replace")
        limit = max(1000, min(int(max_chars or 12000), 40000))
        return {
            "path": str(target.relative_to(self._project_dir(project))),
            "content": text[:limit],
            "truncated": len(text) > limit,
        }

    def write_file(self, project: str, path: str, content: str) -> dict:
        target = self._safe_path(project, path)
        if not self._is_text_path(target):
            return {"error": "Refusing to write a non-text file extension."}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self._normalize_generated_file_content(target, content or ""), encoding="utf-8")
        return {"path": str(target), "bytes": target.stat().st_size}

    def append_file(self, project: str, path: str, content: str) -> dict:
        target = self._safe_path(project, path)
        if not self._is_text_path(target):
            return {"error": "Refusing to append to a non-text file extension."}
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as f:
            f.write(content or "")
        return {"path": str(target), "bytes": target.stat().st_size}

    def replace_in_file(self, project: str, path: str, old: str, new: str) -> dict:
        target = self._safe_path(project, path)
        if not target.exists() or not target.is_file():
            return {"error": "File not found."}
        if not self._is_text_path(target):
            return {"error": "Refusing to edit a non-text file."}
        text = target.read_text(encoding="utf-8", errors="replace")
        if old not in text:
            return {"error": "The old text was not found."}
        count = text.count(old)
        target.write_text(text.replace(old, new), encoding="utf-8")
        return {"path": str(target), "replacements": count}

    def make_directory(self, project: str, path: str) -> dict:
        target = self._safe_path(project, path)
        target.mkdir(parents=True, exist_ok=True)
        return {"path": str(target)}

    def reference_workspace(self, max_files: int = 120) -> dict:
        ignored = {".venv", "__pycache__", ".git", "models"}
        files = []
        for item in sorted(_HERE.rglob("*")):
            if any(part in ignored for part in item.relative_to(_HERE).parts):
                continue
            if item.is_file():
                files.append(str(item.relative_to(_HERE)))
        limit = max(20, min(int(max_files or 120), 300))
        return {"root": str(_HERE), "files": files[:limit], "truncated": len(files) > limit}

    def read_workspace_file(self, path: str, max_chars: int = 12000) -> dict:
        target = self._safe_workspace_path(path)
        if not target.exists() or not target.is_file():
            return {"error": "File not found."}
        if not self._is_text_path(target):
            return {"error": "Refusing to read a non-text file."}
        text = target.read_text(encoding="utf-8", errors="replace")
        limit = max(1000, min(int(max_chars or 12000), 40000))
        return {
            "path": str(target.relative_to(_HERE)),
            "content": text[:limit],
            "truncated": len(text) > limit,
        }


class ToolHarness:
    """Runs structured Gemma tool requests against ProjectWorkspace."""

    def __init__(self, workspace: Optional[ProjectWorkspace] = None):
        self.workspace = workspace or ProjectWorkspace()
        self._tools = {
            "list_project_tree": self.workspace.list_project_tree,
            "find_project_files": self.workspace.find_project_files,
            "grep_project": self.workspace.grep_project,
            "read_project_file": self.workspace.read_project_file,
            "write_project_file": self.workspace.write_project_file,
            "append_project_file": self.workspace.append_project_file,
            "replace_in_project_file": self.workspace.replace_in_project_file,
            "edit_project_file": self.workspace.edit_project_file,
            "review_web_project": self.workspace.review_web_project,
            "make_project_directory": self.workspace.make_project_directory,
            "rename_project_path": self.workspace.rename_project_path,
            "move_project_paths": self.workspace.move_project_paths,
            "delete_project_path": self.workspace.delete_project_path,
            "run_project_command": self.workspace.run_project_command,
            "create_or_select_project": self.workspace.create_or_select_project,
            "list_projects": self.workspace.list_projects,
            "list_files": self.workspace.list_files,
            "read_file": self.workspace.read_file,
            "write_file": self.workspace.write_file,
            "append_file": self.workspace.append_file,
            "replace_in_file": self.workspace.replace_in_file,
            "make_directory": self.workspace.make_directory,
            "reference_workspace": self.workspace.reference_workspace,
            "read_workspace_file": self.workspace.read_workspace_file,
        }

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
            stripped = re.sub(r"\s*```$", "", stripped)
        try:
            return json.loads(stripped)
        except Exception:
            pass

        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
        return None

    @staticmethod
    def _scan_json_string_value(text: str, key: str) -> str:
        """Pull one JSON string value out of otherwise messy model text."""
        match = re.search(rf'"{re.escape(key)}"\s*:\s*"', text or "")
        if not match:
            return ""
        try:
            value, _ = json.decoder.scanstring(text, match.end())
            return value
        except Exception:
            # Fall back to a conservative non-greedy grab. This is less exact,
            # but it keeps tool calls from leaking to chat if JSON is imperfect.
            tail = text[match.end():]
            end = re.search(r'"\s*(?:[,}]|\]\s*})', tail)
            return tail[:end.start()] if end else tail

    def _salvage_tool_calls(self, text: str) -> List[dict]:
        raw = text or ""
        if not re.search(r'"?(?:tool_calls|tool|name)"?\s*:', raw):
            return []

        tool = self._scan_json_string_value(raw, "tool") or self._scan_json_string_value(raw, "name")
        if not tool:
            tool_match = re.search(r'"(?:tool|name)"\s*:\s*"?([a-zA-Z0-9_.-]+)"?', raw)
            tool = tool_match.group(1) if tool_match else ""
        if not tool:
            return []

        args = {}
        for key in ("project", "path", "content", "old", "new", "command"):
            value = self._scan_json_string_value(raw, key)
            if value:
                args[key] = value

        timeout_match = re.search(r'"timeout"\s*:\s*(\d+)', raw)
        if timeout_match:
            args["timeout"] = int(timeout_match.group(1))

        return [self._normalize_call({"tool": tool, "args": args})]

    def parse_tool_calls(self, text: str) -> List[dict]:
        data = self._extract_json(text)
        if not isinstance(data, dict):
            return self._salvage_tool_calls(text)
        calls = data.get("tool_calls", [])
        if isinstance(calls, dict):
            calls = [calls]
        if not isinstance(calls, list):
            single_tool = data.get("tool") or data.get("name")
            if single_tool:
                calls = [{"tool": single_tool, "args": data.get("args") or data.get("arguments") or {}}]
            else:
                return []
        return [self._normalize_call(call) for call in calls if isinstance(call, dict)]

    def parse_status(self, text: str) -> str:
        data = self._extract_json(text)
        if not isinstance(data, dict):
            return ""
        status = data.get("status") or data.get("thought") or data.get("message")
        if not isinstance(status, str):
            return ""
        status = re.sub(r"\s+", " ", status).strip()
        return status[:180]

    def looks_like_tool_json(self, text: str) -> bool:
        raw = text or ""
        data = self._extract_json(raw)
        if isinstance(data, dict) and (
            "tool_calls" in data or "tool" in data or "name" in data or "args" in data or "arguments" in data
        ):
            return True
        return bool(re.search(r'"?(?:tool_calls|tool|name|args|arguments)"?\s*:', raw))

    def _normalize_call(self, call: dict) -> dict:
        tool_name = call.get("tool") or call.get("name") or call.get("function")
        args = call.get("args") or call.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        if not isinstance(args, dict):
            args = {}

        aliases = {
            "create_file": "write_project_file",
            "write_file_in_project": "write_project_file",
            "edit_file": "edit_project_file",
            "multi_edit": "edit_project_file",
            "grep": "grep_project",
            "search": "grep_project",
            "find": "find_project_files",
            "glob": "find_project_files",
            "read_project": "read_project_file",
            "list_tree": "list_project_tree",
            "ls": "list_project_tree",
            "mkdir": "make_project_directory",
            "rename": "rename_project_path",
            "move": "rename_project_path",
            "move_file": "rename_project_path",
            "move_files": "move_project_paths",
            "organize_files": "move_project_paths",
            "delete_file": "delete_project_path",
            "delete_directory": "delete_project_path",
            "run_command": "run_project_command",
            "bash": "run_project_command",
        }
        tool_name = aliases.get(str(tool_name or ""), tool_name)

        for generic_key in ("name", "folder", "directory", "dir"):
            if generic_key in args and "path" not in args:
                args["path"] = args[generic_key]

        if tool_name == "make_project_directory":
            if "path" not in args:
                for alias in ("project", "name", "folder", "directory", "dir"):
                    if alias in args:
                        args["path"] = args[alias]
                        break
            if "path" not in args or not str(args.get("path") or "").strip():
                args["path"] = "untitled-project"

        if tool_name == "rename_project_path":
            if "old_path" not in args:
                for alias in ("source", "src", "from", "from_path", "old", "path"):
                    if alias in args:
                        args["old_path"] = args[alias]
                        break
            if "new_path" not in args:
                for alias in ("destination", "dest", "dst", "to", "to_path", "new"):
                    if alias in args:
                        args["new_path"] = args[alias]
                        break

        if tool_name == "move_project_paths":
            if "paths" not in args:
                for alias in ("sources", "files", "items", "paths_to_move"):
                    if alias in args:
                        args["paths"] = args[alias]
                        break
                if "source" in args:
                    args["paths"] = [args["source"]]
            if "destination" not in args:
                for alias in ("dest", "dst", "to", "to_path", "folder", "directory"):
                    if alias in args:
                        args["destination"] = args[alias]
                        break

        if "project" in args and tool_name == "rename_project_path":
            project = str(args.pop("project") or "").strip().strip("/\\")
            if project:
                if "old_path" in args:
                    old_path = str(args["old_path"]).strip().strip("/\\")
                    args["old_path"] = project if old_path in {"", "."} else f"{project}/{old_path}"
                if "new_path" in args:
                    new_path = str(args["new_path"]).strip().strip("/\\")
                    args["new_path"] = project if new_path in {"", "."} else f"{project}/{new_path}"

        if "project" in args and tool_name == "move_project_paths":
            project = str(args.pop("project") or "").strip().strip("/\\")
            if project:
                paths = args.get("paths")
                if isinstance(paths, list):
                    clean_paths = [str(path).strip().strip("/\\") for path in paths]
                    args["paths"] = [
                        project if clean_path in {"", "."} else f"{project}/{clean_path}"
                        for clean_path in clean_paths
                    ]
                if "destination" in args:
                    destination = str(args["destination"]).strip().strip("/\\")
                    args["destination"] = project if destination in {"", "."} else f"{project}/{destination}"

        # Be forgiving when the model mixes old "project" args with new
        # root-level project tools. Convert project+path into one safe path.
        root_path_tools = {
            "read_project_file",
            "write_project_file",
            "append_project_file",
            "replace_in_project_file",
            "edit_project_file",
            "make_project_directory",
            "rename_project_path",
            "delete_project_path",
        }
        if tool_name in root_path_tools and "project" in args:
            project = str(args.pop("project") or "").strip().strip("/\\")
            path = str(args.get("path") or ".").strip().strip("/\\")
            if project:
                args["path"] = project if path in {"", "."} else f"{project}/{path}"

        if tool_name == "edit_project_file" and "edits" not in args:
            for _alias in ("changes", "replacements", "patches", "modifications", "updates", "operations", "diffs", "hunks", "edits_list"):
                if _alias in args:
                    args["edits"] = args.pop(_alias)
                    break

        return {"tool": tool_name, "args": args}

    def _validate_call_args(self, tool_name: str, args: dict) -> Optional[str]:
        required = {
            "read_project_file": ["path"],
            "write_project_file": ["path", "content"],
            "append_project_file": ["path", "content"],
            "replace_in_project_file": ["path", "old", "new"],
            "edit_project_file": ["path", "edits"],
            "review_web_project": ["path"],
            "make_project_directory": ["path"],
            "rename_project_path": ["old_path", "new_path"],
            "move_project_paths": ["paths", "destination"],
            "delete_project_path": ["path"],
            "run_project_command": ["command"],
            "read_file": ["project", "path"],
            "write_file": ["project", "path", "content"],
            "append_file": ["project", "path", "content"],
            "replace_in_file": ["project", "path", "old", "new"],
            "make_directory": ["project", "path"],
        }
        missing = [name for name in required.get(tool_name, []) if name not in args or args.get(name) in (None, "")]
        if missing:
            return f"Missing required argument(s): {', '.join(missing)}."
        return None

    @staticmethod
    def _friendly_exception(error: Exception) -> str:
        message = str(error)
        if "missing 1 required positional argument" in message:
            match = re.search(r"argument: '([^']+)'", message)
            arg = match.group(1) if match else "a required value"
            return f"Missing required argument: {arg}."
        if "got an unexpected keyword argument" in message:
            match = re.search(r"keyword argument '([^']+)'", message)
            arg = match.group(1) if match else "an unsupported argument"
            return f"Unsupported argument: {arg}."
        return message

    def run_calls(self, calls: List[dict], on_update=None) -> List[dict]:
        results = []
        for call in calls:
            call = self._normalize_call(call) if isinstance(call, dict) else {}
            tool_name = call.get("tool")
            args = call.get("args") or {}
            if not isinstance(args, dict):
                args = {}

            if on_update:
                on_update(f"Using tool: {tool_name}…")

            fn = self._tools.get(tool_name)
            if not fn:
                results.append({"tool": tool_name, "ok": False, "error": "Unknown tool."})
                continue
            arg_error = self._validate_call_args(str(tool_name), args)
            if arg_error:
                results.append({"tool": tool_name, "ok": False, "error": arg_error})
                continue
            try:
                result = fn(**args)
                ok = not (isinstance(result, dict) and result.get("error"))
                entry = {"tool": tool_name, "ok": ok, "result": result}
                if not ok:
                    entry["error"] = str(result.get("error"))
                results.append(entry)
            except Exception as e:
                results.append({"tool": tool_name, "ok": False, "error": self._friendly_exception(e)})
        return results

    @staticmethod
    def actionable_errors(results: List[dict]) -> List[str]:
        errors = []
        for entry in results:
            tool = entry.get("tool", "tool")
            if not entry.get("ok"):
                errors.append(f"{tool}: {entry.get('error', 'unknown error')}")
                continue

            result = entry.get("result")
            if not isinstance(result, dict):
                continue

            # Some tools return diagnostic notes/issues inside a successful
            # result. Only treat them as errors when no useful work happened.
            if result.get("error"):
                errors.append(f"{tool}: {result.get('error')}")
                continue

            error_count = result.get("error_count")
            ok_count = result.get("ok_count", 0)
            if isinstance(error_count, int) and error_count > 0 and ok_count == 0:
                nested = result.get("errors") or []
                if nested:
                    detail = nested[0].get("error") if isinstance(nested[0], dict) else str(nested[0])
                    errors.append(f"{tool}: {detail}")
                else:
                    errors.append(f"{tool}: {error_count} operation(s) failed")
        return errors


# ─────────────────────────────── LLM ────────────────────────────────────────
class LLMEngine:
    def __init__(self):
        from llama_cpp import Llama
        self._model_key = selected_llm_key()
        self._model_meta = LLM_MODELS.get(self._model_key, LLM_MODELS[DEFAULT_LLM_MODEL])
        # Per-model context parameters derived from model metadata
        self._context_size         = _model_context_size(self._model_meta)
        self._compact_trigger      = _compact_trigger_tokens(self._context_size)
        self._compact_target_chars = _compact_target_chars(self._context_size)
        self._agent_max_tokens     = _agent_max_tokens(self._model_meta, self._context_size)
        self._supports_thinking    = bool(self._model_meta.get("thinking", False))
        console.print(
            f"[dim]Context window: {self._context_size} tokens  |  "
            f"compact trigger: {self._compact_trigger}  |  "
            f"agent gen: {self._agent_max_tokens}  |  "
            f"thinking: {self._supports_thinking}[/]"
        )
        model_path = local_llm_gguf(self._model_key)
        if model_path is not None:
            console.print(f"[dim]Loading {self._model_meta['label']} from local cache: {model_path}[/]")
            self._llm = Llama(
                model_path=str(model_path),
                n_ctx=self._context_size,
                n_gpu_layers=LLM_GPU_LAYERS,
                verbose=False,
            )
        elif ALLOW_ONLINE:
            console.print(f"[dim]Loading {self._model_meta['label']} from Hugging Face…[/]")
            self._llm = Llama.from_pretrained(
                repo_id=self._model_meta["repo_id"],
                filename=self._model_meta["filename"],
                n_ctx=self._context_size,
                n_gpu_layers=LLM_GPU_LAYERS,
                verbose=False,
            )
        else:
            raise RuntimeError(
                f"{self._model_meta['label']} is not cached locally. Run `python bob.py --setup` online once, "
                "or set BOB_ALLOW_ONLINE=1 to allow downloads."
            )
        self._history: List[dict] = []
        self._harness = ToolHarness()
        self._active_project: Optional[str] = None
        self._memory_summary = ""

    @property
    def model_label(self) -> str:
        return self._model_meta["label"]

    def _approx_tokens(self, messages_or_text) -> int:
        if isinstance(messages_or_text, str):
            text = messages_or_text
        else:
            text = "\n".join(str(m.get("content", "")) for m in messages_or_text)
        return max(1, len(text) // 4)

    def _shrink_text(self, text: str, max_chars: int = 4000) -> str:
        text = str(text or "")
        if len(text) <= max_chars:
            return text
        head = max_chars // 3
        tail = max_chars - head
        return text[:head] + "\n...[context compacted]...\n" + text[-tail:]

    @staticmethod
    def _strip_thinking_blocks(text: str) -> str:
        """Remove <think>…</think> reasoning blocks so they never reach the user."""
        cleaned = re.sub(r"<think>[\s\S]*?</think>\s*", "", text or "", flags=re.IGNORECASE).strip()
        return cleaned if cleaned else (text or "").strip()

    def _call_llm(self, messages: List[dict], max_tokens: int, temperature: float,
                  stream: bool = False, thinking: bool = False):
        """Unified LLM call.  Enables extended reasoning for supported models."""
        call_kw = dict(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            stop=["<end_of_turn>", "<eos>"],
            stream=stream,
        )
        if thinking and self._supports_thinking:
            try:
                return self._llm.create_chat_completion(
                    **call_kw,
                    chat_template_kwargs={"enable_thinking": True},
                )
            except TypeError:
                # Older llama-cpp-python that doesn't support chat_template_kwargs;
                # fall through to the standard call below.
                pass
        return self._llm.create_chat_completion(**call_kw)

    def _summarize_history_chunk(self, old_messages: List[dict]) -> str:
        compact_source = []
        for msg in old_messages:
            role = msg.get("role", "unknown")
            content = self._shrink_text(str(msg.get("content", "")), 2500)
            compact_source.append(f"{role.upper()}:\n{content}")
        source = "\n\n".join(compact_source)
        if self._memory_summary:
            source = "Previous memory summary:\n" + self._memory_summary + "\n\nConversation to merge:\n" + source
        source = self._shrink_text(source, self._compact_target_chars)

        try:
            response = self._llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Compress this assistant session into durable memory for a coding agent. "
                            "Keep user goals, active project/folder names, files changed, important decisions, "
                            "known errors, and next steps. Be concise."
                        ),
                    },
                    {"role": "user", "content": source},
                ],
                max_tokens=260,
                temperature=0.1,
                top_p=0.9,
                stop=["<end_of_turn>", "<eos>"],
                stream=False,
            )
            summary = response["choices"][0]["message"].get("content", "").strip()
            if summary:
                return summary
        except Exception:
            pass
        return self._shrink_text(source, 1800)

    def _compact_long_history(self, on_update=None):
        if len(self._history) <= 8 and self._approx_tokens(self._history) < self._compact_trigger:
            return
        if self._approx_tokens(self._history) < self._compact_trigger:
            return
        old = self._history[:-6]
        recent = self._history[-6:]
        if not old:
            return
        if on_update:
            on_update("Compacting conversation memory so the model stays within context...")
        self._memory_summary = self._summarize_history_chunk(old)
        self._history = recent

    def _prepare_messages_for_context(self, messages: List[dict], max_tokens: int, on_update=None) -> List[dict]:
        input_budget = max(1200, self._context_size - max_tokens - 256)
        if self._approx_tokens(messages) < min(self._compact_trigger, input_budget):
            return messages
        if on_update:
            on_update("Compacting prompt context before the next model call...")

        system = messages[0] if messages and messages[0].get("role") == "system" else None
        body = messages[1:] if system else messages[:]
        recent = body[-5:]
        older = body[:-5]

        summary = self._summarize_history_chunk(older) if older else self._memory_summary
        compacted = []
        if system:
            compacted.append(system)
        if summary:
            compacted.append({
                "role": "user",
                "content": "Compacted prior context for continuity:\n" + self._shrink_text(summary, 2500),
            })
        for msg in recent:
            compacted.append({
                "role": msg.get("role", "user"),
                "content": self._shrink_text(str(msg.get("content", "")), 5000),
            })
        while len(compacted) > 2 and self._approx_tokens(compacted) >= input_budget:
            compacted.pop(1)
        if self._approx_tokens(compacted) >= input_budget:
            compacted = [
                {
                    "role": msg.get("role", "user"),
                    "content": self._shrink_text(str(msg.get("content", "")), max(1200, input_budget * 2)),
                }
                for msg in compacted[-2:]
            ]
        return compacted

    def _completion_text(self, messages: List[dict], max_tokens: int = 900, temperature: float = 0.35, on_update=None, thinking: bool = False) -> str:
        effective_max = (
            min(max_tokens * 2 + 1024, self._context_size // 2)
            if thinking and self._supports_thinking
            else max_tokens
        )
        effective_max = max(64, min(effective_max, max(256, self._context_size // 3)))
        messages = self._prepare_messages_for_context(messages, effective_max, on_update=on_update)
        try:
            response = self._call_llm(messages, effective_max, temperature, thinking=thinking)
        except Exception as e:
            if "context" not in str(e).lower() and "token" not in str(e).lower():
                raise
            if on_update:
                on_update("The prompt was too large, so I compressed it and retried...")
            retry_max = max(128, effective_max // 2)
            messages = self._prepare_messages_for_context(messages[-4:], retry_max, on_update=on_update)
            try:
                response = self._call_llm(messages, retry_max, temperature, thinking=thinking)
            except Exception:
                if on_update:
                    on_update("Context still too large after compression — skipping this step.")
                return ""
        raw = response["choices"][0]["message"].get("content", "").strip()
        return self._strip_thinking_blocks(raw) if thinking else raw

    def _should_use_tools(self, text: str) -> bool:
        return bool(re.search(
            r"\b(build|create|make|generate|design|edit|write|code|coding|program|file|folder|project"
            r"|html|css|javascript|js|typescript|ts|python|py|java|kotlin|swift|golang|go|rust|ruby|php"
            r"|cpp|c\+\+|csharp|cs|scala|lua|dart|bash|shell|sh|sql|r\b|react|vue|angular|node|flask"
            r"|django|fastapi|spring|script|app|website|webpage|page|api|server|cli|library|package"
            r"|read|reference|fix|repair|improve|update|change|modify|rename|delete|remove|debug|error"
            r"|issue|style|restyle|refactor|look at|inspect|dockerfile|makefile|config|yaml|toml|json)\b",
            text.lower(),
        ))

    def _requires_project_inspection(self, text: str) -> bool:
        return bool(re.search(
            r"\b(fix|repair|improve|update|change|modify|rename|delete|remove|debug|error|issue|reference|look at|inspect|existing|current|that|it"
            r"|add|include|insert|enhance|upgrade|more|better|gradient|animation|effect|content|information|section|layout|style|theme|feature)\b",
            text.lower(),
        ))

    def _requires_mutation(self, text: str) -> bool:
        return bool(re.search(
            r"\b(fix|repair|improve|update|change|modify|rename|delete|remove|build|create|make|generate|design|write|style|restyle|refactor"
            r"|add|include|insert|put|give|show|display|append|attach|enhance|upgrade|polish|beautify|animate|gradient|animation|transition"
            r"|color|colours?|theme|layout|font|icon|image|hero|card|button|nav|header|footer|section|feature|effect|element|detail|content"
            r"|information|text|copy|responsive|mobile|dark|light|better|more|cool|nice|pretty|modern|clean|fresh|slick|bold|vibrant)\b",
            text.lower(),
        ))

    def _is_new_build_request(self, text: str) -> bool:
        return bool(re.search(
            r"\b(build|create|make|generate|design|write)\b.*\b(app|website|webpage|page|script|file|project|html|css|javascript|python)\b|\b(html|website|webpage|python script)\b",
            text.lower(),
        ))

    def _build_generation_prompt(self, user_text: str) -> str:
        return (
            f"{BUILD_OUTPUT_SYSTEM}\n\n"
            "USER REQUEST:\n"
            f"{user_text}\n\n"
            "Remember: for coding/build requests, output only the build marker and JSON."
        )

    def _summarize_tool_results(self, results: List[dict]) -> str:
        changed_files = []
        projects = set()

        for entry in results:
            result = entry.get("result") if entry.get("ok") else None
            if not isinstance(result, dict):
                continue

            project = result.get("project")
            if project:
                projects.add(project)

            moved = result.get("moved")
            if isinstance(moved, list):
                for move in moved:
                    if isinstance(move, dict) and move.get("path"):
                        move_path = str(move["path"]).replace("\\", "/").strip("/")
                        if move_path:
                            projects.add(move_path.split("/", 1)[0])
                            changed_files.append(move_path)

            path = result.get("path")
            if not path:
                continue
            if entry.get("tool") not in {
                "write_file", "append_file", "replace_in_file",
                "write_project_file", "append_project_file", "replace_in_project_file",
                "edit_project_file", "rename_project_path", "move_project_paths", "delete_project_path", "make_project_directory", "run_project_command",
            }:
                continue

            try:
                p = Path(path)
                if p.is_absolute() and PROJECTS_DIR.resolve() in p.resolve().parents:
                    rel = p.resolve().relative_to(PROJECTS_DIR.resolve())
                    parts = rel.parts
                    if parts:
                        projects.add(parts[0])
                    changed_files.append(str(rel))
                else:
                    clean_rel = str(path).replace("\\", "/").strip("/")
                    first_part = clean_rel.split("/", 1)[0]
                    if first_part and first_part not in {".", str(PROJECTS_DIR)}:
                        projects.add(first_part)
                    changed_files.append(str(path))
            except Exception:
                changed_files.append(str(path))

        changed_files = sorted(dict.fromkeys(changed_files))
        projects = sorted(projects)

        if projects:
            location = f"projects/{projects[0]}" if len(projects) == 1 else "the projects folder"
        else:
            location = "the projects folder"

        if changed_files:
            important = ", ".join(changed_files[:5])
            extra = "" if len(changed_files) <= 5 else f", plus {len(changed_files) - 5} more"
            return (
                f"Done. I finished it in {location}. "
                f"I changed {len(changed_files)} path"
                f"{'' if len(changed_files) == 1 else 's'}: {important}{extra}. "
                "The files are ready in your projects folder."
            )

        return "Done. I used the project harness and finished the requested work in your projects folder."

    def _has_file_write(self, results: List[dict]) -> bool:
        return any(
            entry.get("ok") and entry.get("tool") in {
                "write_file", "append_file", "replace_in_file",
                "write_project_file", "append_project_file", "replace_in_project_file",
                "edit_project_file", "rename_project_path", "move_project_paths", "delete_project_path", "make_project_directory",
            }
            for entry in results
        )

    def _next_project_name(self) -> str:
        root = self._harness.workspace.root
        existing = {p.name for p in root.iterdir() if p.is_dir()}
        n = 1
        while f"project-{n}" in existing:
            n += 1
        return f"project-{n}"

    def _meaningful_project_name(self, user_text: str) -> str:
        text = re.sub(r"['\"`]", "", user_text.lower())
        quoted = re.findall(r'"([^"]{3,60})"|\'([^\']{3,60})\'|`([^`]{3,60})`', user_text)
        for groups in quoted:
            candidate = next((g for g in groups if g), "")
            if candidate:
                return self._harness.workspace._slug(candidate)
        text = re.sub(
            r"\b(?:please|can you|could you|would you|build|create|make|generate|design|write|code|an?|the|for me|simple|basic|new|project|app|website|webpage|page|script|python|html|css|javascript|js)\b",
            " ",
            text,
        )
        words = re.findall(r"[a-z0-9]+", text)
        stop = {
            "that", "just", "says", "say", "with", "and", "or", "to", "in", "of",
            "inside", "folder", "file", "files", "want", "need", "called", "named",
        }
        words = [w for w in words if w not in stop]
        name = "-".join(words[:5]).strip("-")
        if name and len(name) >= 3:
            return name[:60]
        return self._next_project_name()

    def _existing_projects_from_tree_result(self, result: List[dict]) -> List[str]:
        projects = []
        for entry in result:
            payload = entry.get("result") if entry.get("ok") else None
            if not isinstance(payload, dict):
                continue
            items = payload.get("items")
            if isinstance(items, list):
                for item in items:
                    item = str(item)
                    if item.endswith("/") and "/" not in item.strip("/"):
                        projects.append(item.strip("/"))
        return sorted(dict.fromkeys(projects))

    def _infer_existing_project(self, user_text: str, projects: List[str]) -> Optional[str]:
        if not projects:
            return None
        lower = user_text.lower()
        for project in projects:
            readable = project.replace("-", " ").lower()
            if project.lower() in lower or readable in lower:
                return project
        if self._active_project in projects and re.search(r"\b(it|that|this|current|existing|same|there)\b", lower):
            return self._active_project
        if len(projects) == 1 and self._requires_project_inspection(user_text):
            return projects[0]
        return self._active_project if self._active_project in projects else None

    def _update_active_project_from_results(self, results: List[dict]):
        roots = self._changed_project_roots(results)
        if roots:
            self._active_project = roots[0]

    def _agent_result_message(self, user_text: str, transcript: List[dict], final_text: str) -> str:
        if (
            final_text
            and BUILD_MARKER not in final_text
            and "```" not in final_text
            and not self._harness.looks_like_tool_json(final_text)
            and not self._detect_generated_code_files(final_text)
        ):
            return final_text.strip()

        all_results = []
        for msg in transcript:
            if msg.get("role") == "user" and "Tool results:" in msg.get("content", ""):
                try:
                    payload = msg["content"].split("Tool results:", 1)[1].strip()
                    parsed = json.loads(payload)
                    if isinstance(parsed, list):
                        all_results.extend(parsed)
                except Exception:
                    pass
        return self._summarize_tool_results(all_results)

    def _has_real_build_output(self, results: List[dict]) -> bool:
        meaningful_tools = {
            "write_project_file", "append_project_file", "replace_in_project_file",
            "edit_project_file", "write_file", "append_file", "replace_in_file",
            "rename_project_path", "move_project_paths", "delete_project_path", "run_project_command",
        }
        return any(entry.get("ok") and entry.get("tool") in meaningful_tools for entry in results)

    def _changed_project_roots(self, results: List[dict]) -> List[str]:
        roots = []
        for entry in results:
            if not entry.get("ok"):
                continue
            result = entry.get("result")
            if not isinstance(result, dict):
                continue
            path = result.get("path")
            project = result.get("project")
            if project:
                roots.append(str(project).strip("/"))
            elif path:
                clean = str(path).replace("\\", "/").strip("/")
                if clean and clean != ".":
                    roots.append(clean.split("/", 1)[0])
        return sorted(dict.fromkeys(root for root in roots if root and root != "."))

    def _auto_verify_after_mutations(self, results: List[dict], on_token) -> List[dict]:
        if not self._has_real_build_output(results):
            return []
        roots = self._changed_project_roots(results)[:3]
        if not roots:
            return []
        verify_calls = []
        for root in roots:
            verify_calls.append({"tool": "review_web_project", "args": {"path": root}})
            verify_calls.append({"tool": "list_project_tree", "args": {"path": root, "max_files": 120}})
        on_token("Reviewing and verifying the files that were changed…")
        return self._harness.run_calls(verify_calls, on_update=on_token)

    def _quick_plan(self, user_text: str) -> str:
        """Generate a short spoken plan for what the agent is about to do — no thinking, fast."""
        try:
            return self._completion_text(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are BOB, a voice assistant. In 1-2 short conversational sentences, "
                            "state confidently what you are about to do to fulfill the user's request. "
                            "NEVER ask a question. NEVER ask for preferences or clarification. "
                            "Make a definitive statement of intent — just say what you will build or change. "
                            "Speak naturally as if talking aloud. No bullet points, no tool names."
                        ),
                    },
                    {"role": "user", "content": user_text},
                ],
                max_tokens=80,
                temperature=0.4,
                thinking=False,
            )
        except Exception:
            return ""

    def _run_agentic_project_task(self, user_text: str, on_token, on_speak=None) -> Optional[str]:
        """Let the model use bounded project tools for multi-step coding work."""
        try:
            return self._run_agentic_project_task_inner(user_text, on_token, on_speak=on_speak)
        except Exception as e:
            on_token(f"Agentic task hit an unexpected error and was stopped safely: {e}")
            with (LOGS_DIR / "agentic-error.log").open("a", encoding="utf-8") as _f:
                traceback.print_exc(file=_f)
            return None

    def _run_agentic_project_task_inner(self, user_text: str, on_token, on_speak=None) -> Optional[str]:
        # Quick spoken plan before tools run
        plan = self._quick_plan(user_text)
        if plan:
            if on_speak:
                on_speak(plan)
            else:
                on_token(plan)
        suggested = self._meaningful_project_name(user_text)
        on_token("Checking the projects folder before deciding what to do…")
        initial_results = self._harness.run_calls([
            {"tool": "list_project_tree", "args": {"path": ".", "max_files": 220}}
        ], on_update=on_token)
        must_inspect = self._requires_project_inspection(user_text)
        must_mutate = self._requires_mutation(user_text)
        existing_projects = self._existing_projects_from_tree_result(initial_results)
        focused_project = self._infer_existing_project(user_text, existing_projects)
        if focused_project:
            self._active_project = focused_project
        messages = [
            {"role": "system", "content": TOOL_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"User request: {user_text}\n\n"
                    f"Suggested folder name if creating something new: {suggested}\n"
                    f"Current selected project, if editing existing work: {focused_project or self._active_project or 'none'}\n"
                    f"Must inspect existing project files before answering: {'yes' if must_inspect else 'no'}\n"
                    f"Must make a real file change before claiming completion: {'yes' if must_mutate else 'no'}\n"
                    + (
                        "ACTION REQUIRED: Read the relevant files, then immediately call write_project_file or "
                        "edit_project_file with the actual requested changes. Do not stop after reading — make the edits.\n"
                        if must_mutate else ""
                    )
                    + "\nInitial projects/ tree:\n"
                    f"{json.dumps(initial_results, ensure_ascii=False)[:3000]}\n\n"
                    "Use tools to perform the work inside projects/. If the request refers to existing work, "
                    "read the relevant files then edit them. Keep all edits inside the selected project unless the user explicitly asks to create or rename another folder. "
                    "If you cannot find the right project, say what you found and ask a short clarification."
                ),
            },
        ]

        all_results: List[dict] = list(initial_results)
        inspected = not must_inspect
        _empty_draft_streak = 0
        _mutation_nudge_count = 0
        for turn in range(10):
            messages = self._prepare_messages_for_context(messages, max_tokens=self._agent_max_tokens, on_update=on_token)
            draft = self._completion_text(messages, max_tokens=self._agent_max_tokens, temperature=0.16, on_update=on_token, thinking=True)
            if not draft:
                _empty_draft_streak += 1
                if _empty_draft_streak >= 2:
                    break
            else:
                _empty_draft_streak = 0
            status = self._harness.parse_status(draft)
            if status:
                on_token(status)
            calls = self._harness.parse_tool_calls(draft)
            if not calls:
                if all_results:
                    if must_inspect and not inspected:
                        messages.append({
                            "role": "user",
                            "content": (
                                "You have not inspected the project files yet. "
                                "Use list_project_tree or read_project_file to read the relevant files first, "
                                "then make the requested changes."
                            ),
                        })
                        continue
                    if must_mutate and not self._has_real_build_output(all_results):
                        _mutation_nudge_count += 1
                        if _mutation_nudge_count == 1:
                            nudge = (
                                f"You have read the files but made NO actual changes yet. "
                                f"The user asked: \"{user_text}\". "
                                "You MUST now call write_project_file or edit_project_file to make those changes. "
                                "Do not re-read files. Do not explain what you would do. "
                                "Output a tool call with the new or updated file content right now."
                            )
                        else:
                            nudge = (
                                f"You still have not made any file edits. STOP. "
                                f"The user wants: \"{user_text}\". "
                                "Call write_project_file or edit_project_file NOW with the actual changed content. "
                                "Example: {{\"tool_calls\":[{{\"tool\":\"write_project_file\","
                                "\"args\":{{\"path\":\"project/style.css\",\"content\":\"/* new CSS here */\"}}}}]}} "
                                "Make the edit now — do not read any more files."
                            )
                        messages.append({"role": "user", "content": nudge})
                        continue
                    return self._agent_result_message(user_text, messages, draft)
                return None

            tool_names = ", ".join(str(c.get("tool")) for c in calls[:3])
            extra = "" if len(calls) <= 3 else f", plus {len(calls) - 3} more"
            on_token(f"Working in projects/: {tool_names}{extra}…")
            results = self._harness.run_calls(calls, on_update=on_token)
            all_results.extend(results)
            self._update_active_project_from_results(results)
            if any(r.get("tool") in {"list_project_tree", "find_project_files", "grep_project", "read_project_file", "read_file", "list_files"} and r.get("ok") for r in results):
                inspected = True
            verification = self._auto_verify_after_mutations(results, on_token)
            if verification:
                all_results.extend(verification)
            messages.append({"role": "assistant", "content": draft})
            result_payload = results
            if verification:
                result_payload = results + [{"tool": "auto_verify", "ok": True, "result": verification}]
            errors_for_model = self._harness.actionable_errors(results)
            if errors_for_model:
                result_payload = result_payload + [{
                    "tool": "error_summary",
                    "ok": False,
                    "error": "; ".join(errors_for_model[:3]),
                }]
            # Budget-aware truncation: each tool-result turn shares the remaining
            # context headroom. Cap at 3000 chars to leave room for future turns.
            tool_result_json = json.dumps(result_payload, ensure_ascii=False)
            current_ctx_chars = sum(len(str(m.get("content", ""))) for m in messages)
            remaining_budget = max(1500, (self._context_size * 3) - current_ctx_chars)
            tool_result_cap = min(3000, remaining_budget)
            messages.append({
                "role": "user",
                "content": "Tool results:\n" + tool_result_json[:tool_result_cap],
            })

            actionable_errors = self._harness.actionable_errors(results)
            if actionable_errors:
                first_error = actionable_errors[0]
                on_token(f"Tool issue: {first_error}. Adjusting the next step…")

        if not self._has_real_build_output(all_results):
            return None
        return self._summarize_tool_results(all_results)

    def _extract_code_from_direct_answer(self, draft: str, user_text: str) -> tuple[str, str]:
        """Recover useful code if Gemma ignored the tool protocol."""
        text = draft or ""

        html_code = self._extract_html_like_code(text)
        if html_code:
            return "index.html", html_code

        fenced = re.search(r"```([a-zA-Z0-9_-]*)\s*([\s\S]*?)```", text)
        if fenced:
            lang = fenced.group(1).lower()
            code = fenced.group(2).strip() + "\n"
            if lang in {"html", "htm"}:
                return "index.html", code
            if lang in {"python", "py"}:
                return "main.py", code
            if lang in {"javascript", "js"}:
                return "script.js", code
            if lang == "css":
                return "styles.css", code
            return "main.txt", code

        python_code = self._extract_python_like_code(text)
        if python_code:
            return "main.py", python_code

        wanted = "Hello World"
        says_match = re.search(
            r"\b(?:says?|display(?:s)?|show(?:s)?)\s+['\"]?([^.'\"]+)",
            user_text,
            flags=re.IGNORECASE,
        )
        if says_match:
            wanted = says_match.group(1).strip().capitalize()

        lower = user_text.lower()
        if any(word in lower for word in ("html", "website", "webpage", "page")):
            return "index.html", (
                "<!doctype html>\n"
                "<html lang=\"en\">\n"
                "<head>\n"
                "  <meta charset=\"utf-8\">\n"
                "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
                f"  <title>{wanted}</title>\n"
                "</head>\n"
                "<body>\n"
                f"  <h1>{wanted}</h1>\n"
                "</body>\n"
                "</html>\n"
            )
        if any(word in lower for word in ("python", "script", ".py")):
            return "main.py", f"print({wanted!r})\n"

        return "README.md", (
            "# Generated Project\n\n"
            "Bob created this project folder for your request. "
            "Ask Bob to add specific Python, HTML, CSS, or JavaScript files here.\n"
        )

    def _strip_terminal_artifacts(self, text: str) -> str:
        lines = []
        for line in (text or "").splitlines():
            cleaned = line.replace("│", "").replace("┃", "").replace("║", "")
            cleaned = re.sub(r"^\s*\d{1,2}:\d{2}\s+(?:BOB|YOU)\s+", "", cleaned)
            lines.append(cleaned.rstrip())
        return "\n".join(lines).strip()

    def _extract_html_like_code(self, text: str) -> str:
        cleaned = self._strip_terminal_artifacts(text)
        if not re.search(r"<\s*/?\s*(?:!doctype|html|head|title|body|h[1-6]|div|p|span|script|style|link|meta)\b", cleaned, flags=re.IGNORECASE):
            return ""

        tag_match = re.search(r"<\s*(?:!doctype|html|head|title|body|h[1-6]|div|p|span|script|style|link|meta)\b[\s\S]*", cleaned, flags=re.IGNORECASE)
        html = tag_match.group(0).strip() if tag_match else cleaned

        chatter_cuts = [
            r"\n\s*(?:That's it|That is it|Let me know|This will display|Save it as|Copy and paste)\b[\s\S]*$",
            r"\n\s*Here(?:'s| is)\b[\s\S]*$",
        ]
        for pattern in chatter_cuts:
            html = re.sub(pattern, "", html, flags=re.IGNORECASE).strip()

        if "<html" not in html.lower():
            html = "<!doctype html>\n<html lang=\"en\">\n<body>\n" + html + "\n</body>\n</html>"
        else:
            if "</body>" not in html.lower() and "<body" in html.lower():
                html += "\n</body>"
            if "</html>" not in html.lower():
                html += "\n</html>"
        return html.strip() + "\n"

    def _extract_python_like_code(self, text: str) -> str:
        cleaned = self._strip_terminal_artifacts(text)
        python_lines = []
        started = False
        for line in cleaned.splitlines():
            if re.match(r"\s*(?:from\s+\w[\w.]*\s+import\s+|import\s+\w|def\s+\w+\s*\(|class\s+\w+\s*[:(]|print\s*\(|if\s+__name__\s*==)", line):
                started = True
            if started:
                if re.match(r"\s*(?:Here(?:'s| is)|Copy and paste|Let me know|That's it)\b", line, flags=re.IGNORECASE):
                    break
                python_lines.append(line)
        code = "\n".join(python_lines).strip()
        return code + "\n" if code else ""

    def _path_for_code_block(self, lang: str, content: str, index: int, type_count: int = 1) -> str:
        lang = (lang or "").strip().lower()
        sample = (content or "").lstrip().lower()
        n = type_count

        if lang in {"html", "htm"} or sample.startswith("<!doctype html") or sample.startswith("<html"):
            return "index.html" if n == 1 else f"page-{n}.html"
        if lang in {"python", "py"}:
            return "main.py" if n == 1 else f"script-{n}.py"
        if lang in {"javascript", "js", "node"}:
            return "script.js" if n == 1 else f"script-{n}.js"
        if lang in {"css", "style"}:
            return "styles.css" if n == 1 else f"styles-{n}.css"
        if lang in {"json"}:
            return "data.json" if n == 1 else f"data-{n}.json"
        if lang in {"bash", "sh", "shell", "zsh"}:
            return "script.sh" if n == 1 else f"script-{n}.sh"
        if lang in {"markdown", "md"}:
            return "README.md" if n == 1 else f"notes-{n}.md"

        if re.search(r"\b(def|class|import|from)\s+\w+", content):
            return "main.py" if n == 1 else f"script-{n}.py"
        if re.search(r"\b(function|const|let|var)\s+\w+", content):
            return "script.js" if n == 1 else f"script-{n}.js"
        if "{" in content and ":" in content and ";" in content:
            return "styles.css" if n == 1 else f"styles-{n}.css"
        return "main.txt" if n == 1 else f"main-{n}.txt"

    def _detect_generated_code_files(self, text: str) -> List[dict]:
        """Hard-coded catch: convert generated code blocks/raw HTML into files."""
        if not text:
            return []

        files: List[dict] = []
        seen_paths = set()
        type_counts: dict = {}

        fences = list(re.finditer(r"```([a-zA-Z0-9_+.-]*)\s*\n?([\s\S]*?)```", text))
        for i, match in enumerate(fences, start=1):
            lang = match.group(1)
            content = match.group(2).strip()
            if not content or len(content) < 8:
                continue
            first_path = self._path_for_code_block(lang, content, i, 1)
            ext = Path(first_path).suffix or ".txt"
            type_counts[ext] = type_counts.get(ext, 0) + 1
            path = self._path_for_code_block(lang, content, i, type_counts[ext])
            while path in seen_paths:
                stem = Path(path).stem
                suffix = Path(path).suffix or ".txt"
                path = f"{stem}-{len(seen_paths) + 1}{suffix}"
            seen_paths.add(path)
            files.append({"path": path, "content": content + "\n"})

        if files:
            return files

        html_code = self._extract_html_like_code(text)
        if html_code:
            return [{"path": "index.html", "content": html_code}]

        python_code = self._extract_python_like_code(text)
        if python_code:
            return [{"path": "main.py", "content": python_code}]

        return []

    def _materialize_code_if_present(self, user_text: str, model_text: str, on_token) -> Optional[str]:
        calls = self._harness.parse_tool_calls(model_text)
        if calls:
            on_token("I detected a tool call, so I am executing it in projects/ instead of showing it.")
            results = self._harness.run_calls(calls, on_update=on_token)
            summary = self._summarize_tool_results(results)
            return self._explain_completed_build(user_text, summary)

        if self._harness.looks_like_tool_json(model_text):
            on_token("I caught an incomplete tool call, so I am using the fallback file builder.")
            summary = self._fallback_build_with_files(user_text, model_text, on_token)
            return self._explain_completed_build(user_text, summary)

        files = self._detect_generated_code_files(model_text)
        if not files:
            return None
        on_token("I detected generated code, so I am saving it into project files…")
        summary = self._write_build_files(files, on_token, user_text=user_text)
        return self._explain_completed_build(user_text, summary)

    def _parse_build_marker(self, text: str) -> List[dict]:
        if BUILD_MARKER not in (text or ""):
            return []
        payload = text.split(BUILD_MARKER, 1)[1].strip()
        payload = re.sub(r"^```(?:json)?\s*", "", payload)
        payload = re.sub(r"\s*```$", "", payload)
        try:
            data = json.loads(payload)
        except Exception:
            match = re.search(r"\{[\s\S]*\}", payload)
            if not match:
                return []
            try:
                data = json.loads(match.group(0))
            except Exception:
                return []

        files = data.get("files") if isinstance(data, dict) else None
        if not isinstance(files, list):
            return []

        clean_files = []
        for file_info in files:
            if not isinstance(file_info, dict):
                continue
            path = str(file_info.get("path") or "").strip().lstrip("/")
            content = file_info.get("content")
            if not path or not isinstance(content, str):
                continue
            path = path.replace("\\", "/")
            if path.endswith("/"):
                continue
            if Path(path).suffix == "":
                kind = str(file_info.get("type") or "").lower()
                if kind in {"html", "website", "webpage"}:
                    path += ".html"
                elif kind in {"python", "py", "script"}:
                    path += ".py"
                elif kind in {"css", "style"}:
                    path += ".css"
                elif kind in {"javascript", "js"}:
                    path += ".js"
                elif kind in {"markdown", "md", "docs"}:
                    path += ".md"
                else:
                    path += ".txt"
            clean_files.append({"path": path, "content": content})
        return clean_files

    def _write_build_files(self, files: List[dict], on_token, user_text: str = "") -> str:
        project = self._meaningful_project_name(user_text) if user_text else self._next_project_name()
        on_token(f"Creating files in projects/{project}…")
        calls = [{"tool": "make_project_directory", "args": {"path": project}}]
        for file_info in files:
            rel_path = f"{project}/{str(file_info['path']).lstrip('/')}"
            calls.append({
                "tool": "write_project_file",
                "args": {
                    "path": rel_path,
                    "content": file_info["content"],
                },
            })
        results = self._harness.run_calls(calls, on_update=on_token)
        return self._summarize_tool_results(results)

    def _fallback_build_with_files(self, user_text: str, draft: str, on_token) -> str:
        project = self._meaningful_project_name(user_text)
        filename, content = self._extract_code_from_direct_answer(draft, user_text)
        on_token(f"Creating {project}/{filename} with the project harness…")
        results = self._harness.run_calls([
            {"tool": "make_project_directory", "args": {"path": project}},
            {"tool": "write_project_file", "args": {"path": f"{project}/{filename}", "content": content}},
        ], on_update=on_token)
        return self._summarize_tool_results(results)

    def _explain_completed_build(self, user_text: str, summary: str) -> str:
        prompt = (
            "The requested files have already been written to disk. "
            "Give the user a short spoken summary only. Do not include source code. "
            "Mention the project folder and the important files.\n\n"
            f"User request: {user_text}\n"
            f"Build result: {summary}"
        )
        try:
            reply = self._completion_text(
                [{"role": "system", "content": LLM_SYSTEM}, {"role": "user", "content": prompt}],
                max_tokens=160,
                temperature=0.4,
            )
            if (
                reply
                and BUILD_MARKER not in reply
                and "<html" not in reply.lower()
                and "```" not in reply
                and not self._harness.looks_like_tool_json(reply)
                and not self._detect_generated_code_files(reply)
            ):
                return reply.strip()
        except Exception:
            pass
        return summary

    def chat_stream(self, user_text: str, on_token, on_speak=None) -> str:
        """Stream reply tokens, calling on_token(partial_text) as they arrive."""
        try:
            return self._chat_stream_inner(user_text, on_token, on_speak=on_speak)
        except Exception as e:
            msg = f"Something went wrong during that response. Please try again. ({type(e).__name__})"
            on_token(msg)
            with (LOGS_DIR / "llm-error.log").open("a", encoding="utf-8") as _f:
                traceback.print_exc(file=_f)
            return msg

    def _chat_stream_inner(self, user_text: str, on_token, on_speak=None) -> str:
        self._history.append({"role": "user", "content": user_text})
        self._compact_long_history(on_update=on_token)
        use_tools = self._should_use_tools(user_text)

        if use_tools:
            on_token("Opening the projects workspace and using tools…")
            agent_reply = self._run_agentic_project_task(user_text, on_token, on_speak=on_speak)
            if agent_reply:
                recovered_reply = self._materialize_code_if_present(user_text, agent_reply, on_token)
                if recovered_reply:
                    agent_reply = recovered_reply
                on_token(agent_reply)
                self._history.append({"role": "assistant", "content": agent_reply})
                return agent_reply

            if self._requires_project_inspection(user_text) and not self._is_new_build_request(user_text):
                reply = (
                    "I checked the projects folder, but I could not confidently identify the exact files to change. "
                    "Tell me the project or file name and I will inspect it, edit it, and verify the result."
                )
                on_token(reply)
                self._history.append({"role": "assistant", "content": reply})
                return reply

            on_token("Using the fallback file builder…")
            build_prompt = self._build_generation_prompt(user_text)
            draft = self._completion_text(
                [
                    {
                        "role": "system",
                        "content": (
                            "Follow the instructions that appear before USER REQUEST exactly. "
                            "For code/build requests, output only the required build marker and JSON."
                        ),
                    },
                    {"role": "user", "content": build_prompt},
                ],
                max_tokens=1800,
                temperature=0.2,
                on_update=on_token,
            )
            files = self._parse_build_marker(draft)
            if files:
                summary = self._write_build_files(files, on_token, user_text=user_text)
            else:
                detected_reply = self._materialize_code_if_present(user_text, draft, on_token)
                if detected_reply:
                    on_token(detected_reply)
                    self._history.append({"role": "assistant", "content": detected_reply})
                    return detected_reply
                summary = self._fallback_build_with_files(user_text, draft, on_token)

            reply = self._explain_completed_build(user_text, summary)
            recovered_reply = self._materialize_code_if_present(user_text, reply, on_token)
            if recovered_reply:
                reply = recovered_reply
            on_token(reply)
            self._history.append({"role": "assistant", "content": reply})
            return reply

        messages = [{"role": "system", "content": LLM_SYSTEM}]
        if self._memory_summary:
            messages.append({"role": "user", "content": "Memory summary:\n" + self._memory_summary})
        messages += self._history[-10:]
        stream_max_tokens = min(350, max(128, self._context_size // 20))
        messages = self._prepare_messages_for_context(messages, max_tokens=stream_max_tokens, on_update=on_token)

        full_reply = ""
        try:
            stream = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=stream_max_tokens,
                temperature=0.7,
                top_p=0.9,
                stop=["<end_of_turn>", "<eos>"],
                stream=True,
            )
            for chunk in stream:
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    full_reply += delta
                    on_token(full_reply)
        except Exception as e:
            full_reply = f"Sorry, I ran into an error: {e}"
            on_token(full_reply)

        full_reply = self._strip_thinking_blocks(full_reply)
        if full_reply:
            on_token(full_reply)

        materialized_reply = self._materialize_code_if_present(user_text, full_reply, on_token)
        if materialized_reply:
            on_token(materialized_reply)
            self._history.append({"role": "assistant", "content": materialized_reply})
            return materialized_reply

        if self._harness.looks_like_tool_json(full_reply):
            safe_reply = "I caught an internal tool call before showing it. Please ask me that build request again and I will write it into the projects folder."
            on_token(safe_reply)
            self._history.append({"role": "assistant", "content": safe_reply})
            return safe_reply

        self._history.append({"role": "assistant", "content": full_reply})
        return full_reply.strip()

    def reset(self):
        self._history.clear()


# ─────────────────────────────── TTS helpers ────────────────────────────────
def _split_sentences(text: str) -> List[str]:
    """Break reply into speakable chunks at sentence boundaries."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    out = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) > 110:
            sub = re.split(r',\s+', part)
            out.extend(s.strip() for s in sub if s.strip())
        else:
            out.append(part)
    return [s for s in out if len(s) > 2]


def _prepare_audio(raw_wav, sr_in: int):
    """Pull off GPU/MPS, normalise, fade, resample to 44.1 kHz."""
    from scipy.signal import resample_poly
    from math import gcd

    wav = raw_wav
    if hasattr(wav, "cpu"):   wav = wav.cpu()
    if hasattr(wav, "numpy"): wav = wav.numpy()
    audio = np.ascontiguousarray(np.array(wav, dtype=np.float32))

    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio * (0.80 / peak)

    fade = int(sr_in * 0.010)
    if len(audio) > fade * 2:
        audio[:fade]  *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
        audio[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)

    target_sr = 44_100
    if sr_in != target_sr:
        g = gcd(sr_in, target_sr)
        audio = resample_poly(audio, target_sr // g, sr_in // g).astype(np.float32)

    return audio, target_sr


# ─────────────────────────────── Kokoro TTS ─────────────────────────────────
class KokoroTTSEngine:
    """Fast TTS via kokoro-onnx.  Uses ONNX runtime — no Python version limits."""

    MODEL_PATH  = _HERE / "models" / "kokoro" / "kokoro-v1.0.onnx"
    VOICES_PATH = _HERE / "models" / "kokoro" / "voices-v1.0.bin"

    def __init__(self, voice: Optional[str] = None):
        from kokoro_onnx import Kokoro
        self.voice = voice if voice in KOKORO_VOICES else DEFAULT_KOKORO_VOICE
        console.print("[dim]Loading Kokoro TTS (fast mode)…[/]")
        self._kokoro = Kokoro(str(self.MODEL_PATH), str(self.VOICES_PATH))
        console.print(
            f"[green]✓[/] Kokoro TTS loaded — voice [bold]{KOKORO_VOICES[self.voice]}[/] "
            f"[dim]({self.voice})[/]"
        )

    def speak(self, text: str, stop_event: Optional[threading.Event] = None):
        if not text:
            return
        sentences = _split_sentences(text)
        if not sentences:
            return

        # Generate all sentences first, then concatenate and play once.
        # Kokoro is fast (~0.1s/sentence) so this adds minimal latency while
        # completely eliminating the per-sentence gaps that cause glitchiness.
        chunks: List[np.ndarray] = []
        out_sr = 44_100
        for sentence in sentences:
            if stop_event and stop_event.is_set():
                return
            try:
                samples, sr = self._kokoro.create(
                    sentence, voice=self.voice, speed=1.0, lang="en-us"
                )
                audio, out_sr = _prepare_audio(samples, sr)
                chunks.append(audio)
            except Exception as e:
                console.print(f"[dim]Kokoro sentence skipped: {e}[/]")

        if not chunks or (stop_event and stop_event.is_set()):
            return

        combined = np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)
        duration = max(0.1, len(combined) / float(out_sr))
        try:
            sd.stop()
            sd.play(combined, samplerate=out_sr, blocksize=4096, blocking=False)
            deadline = time.time() + duration + 1.0
            while time.time() < deadline:
                if stop_event and stop_event.is_set():
                    sd.stop()
                    return
                time.sleep(0.04)
            sd.wait()
        except Exception as e:
            console.print(f"[dim]Audio playback skipped: {e}[/]")
        finally:
            if stop_event and stop_event.is_set():
                with contextlib.suppress(Exception):
                    sd.stop()


# ─────────────────────────────── BOB Core ───────────────────────────────────
class BOB:
    TTS_ORDER = ["kokoro"]

    def __init__(self, tts: str = "kokoro", voice: Optional[str] = None):
        self.tts_backend  = tts
        self.tts_voice    = voice if voice in KOKORO_VOICES else selected_kokoro_voice()
        self.llm_model_key = selected_llm_key()
        self._tts_cache: dict = {}   # name → loaded engine, so switching is instant
        self.state     = State.BOOT
        self.messages: List[Message] = []
        self._lock     = threading.Lock()
        self._running  = True
        self._level    = 0.0
        self._status   = ""
        self._live_text = ""
        self._stt_lock  = threading.Lock()
        self.settings_open = False
        self._settings_view = "main"
        self._settings_index = 0
        self._settings_status = ""
        self._settings_lock = threading.Lock()
        # ── text input ──
        self._text_input_mode   = False
        self._text_input_buffer = ""
        self._text_input_queue: queue.Queue = queue.Queue()
        self._ptt = None                     # set in run() after PTT is created

    # ── engines ──────────────────────────────────────────────────────────────
    def _make_tts(self, name: str):
        return KokoroTTSEngine(voice=self.tts_voice)

    def _load_engines(self):
        from faster_whisper import WhisperModel

        console.print("[bold cyan]Loading Whisper tiny (wake word)…[/]")
        self._wake = WakeWordDetector()

        console.print("[bold cyan]Loading Whisper base (STT)…[/]")
        whisper_path = local_whisper_model(WHISPER_MODEL)
        if whisper_path is None:
            if not ALLOW_ONLINE:
                raise RuntimeError(
                    f"Whisper {WHISPER_MODEL} is not cached locally. Run `python bob.py --setup` online once, "
                    "or set BOB_ALLOW_ONLINE=1 to allow downloads."
                )
            whisper_path = Path(WHISPER_MODEL)
        self._whisper = WhisperModel(str(whisper_path), device="auto", compute_type="int8")
        self._stt = STTEngine(self._whisper)

        console.print("[bold cyan]Loading Gemma 4 E2B LLM…[/]")
        self._llm = LLMEngine()

        console.print("[bold cyan]Loading Kokoro TTS…[/]")
        engine = self._make_tts(self.tts_backend)
        self._tts_cache[self.tts_backend] = engine
        self._tts = engine

    def _switch_tts(self):
        """Kokoro is the only TTS backend now."""
        self._set_state(self.state, "Kokoro TTS is already selected")
        time.sleep(1.2)
        self._status = ""

    def _current_model_label(self) -> str:
        return LLM_MODELS.get(self.llm_model_key, LLM_MODELS[DEFAULT_LLM_MODEL])["label"]

    def _current_voice_label(self) -> str:
        return KOKORO_VOICES.get(self.tts_voice, self.tts_voice)

    def _save_runtime_config(self):
        save_config({"llm_model": self.llm_model_key, "kokoro_voice": self.tts_voice})

    def _cycle_voice(self):
        keys = list(KOKORO_VOICES.keys())
        idx = keys.index(self.tts_voice) if self.tts_voice in keys else 0
        self.tts_voice = keys[(idx + 1) % len(keys)]
        self._save_runtime_config()
        self._settings_status = f"Voice changed to {self._current_voice_label()}. Reloading Kokoro..."
        try:
            self._tts_cache.clear()
            self._tts = self._make_tts(self.tts_backend)
            self._tts_cache[self.tts_backend] = self._tts
            self._settings_status = f"Voice changed to {self._current_voice_label()}."
        except Exception as e:
            self._settings_status = f"Voice load failed: {e}"

    def _select_llm_model(self, model_key: str):
        if model_key not in LLM_MODELS:
            return
        self.llm_model_key = model_key
        self._save_runtime_config()
        label = self._current_model_label()
        if local_llm_gguf(model_key) is None:
            self._settings_status = f"{label} is selected but not downloaded. Press D to download it."
            return

        self._settings_status = f"Loading {label}..."
        prev_state, prev_status = self.state, self._status
        self._set_state(State.BOOT, f"Loading {label}...")
        try:
            self._llm = LLMEngine()
            self._settings_status = f"Current LLM changed to {label}."
        except Exception as e:
            self._settings_status = f"LLM load failed: {e}"
        finally:
            self._set_state(prev_state, prev_status)

    def _cycle_llm_model(self):
        keys = list(LLM_MODELS.keys())
        idx = keys.index(self.llm_model_key) if self.llm_model_key in keys else 0
        self._select_llm_model(keys[(idx + 1) % len(keys)])

    def _download_selected_assets(self):
        label = self._current_model_label()
        self._settings_status = f"Downloading/checking {label} and core assets..."
        try:
            results = {
                label: _download_llm(self.llm_model_key),
                "Kokoro": _download_kokoro_files(),
                "Whisper tiny": _download_whisper_model("tiny"),
                f"Whisper {WHISPER_MODEL}": _download_whisper_model(WHISPER_MODEL),
                "OpenWakeWord": _download_openwakeword_models(),
            }
            failed = [name for name, ok in results.items() if not ok]
            if failed:
                self._settings_status = "Download warnings: " + ", ".join(failed)
            else:
                self._settings_status = "Downloads complete. Press M to load the selected LLM if needed."
        except Exception as e:
            self._settings_status = f"Download failed: {e}"

    def _settings_count(self) -> int:
        if self._settings_view == "voice":
            return len(KOKORO_VOICES)
        if self._settings_view == "model":
            return len(LLM_MODELS)
        return 4

    def _settings_move(self, delta: int):
        count = self._settings_count()
        if count:
            self._settings_index = (self._settings_index + delta) % count

    def _settings_back(self):
        if self._settings_view == "main":
            self.settings_open = False
        else:
            self._settings_view = "main"
            self._settings_index = 0
        self._settings_status = ""

    def _settings_select(self):
        if self._settings_view == "voice":
            keys = list(KOKORO_VOICES.keys())
            selected = keys[self._settings_index]
            if selected == self.tts_voice:
                self._settings_status = f"{self._current_voice_label()} is already selected."
                return
            with self._settings_lock:
                self.tts_voice = selected
                self._save_runtime_config()
                self._settings_status = f"Voice changed to {self._current_voice_label()}. Reloading Kokoro..."
                try:
                    self._tts_cache.clear()
                    self._tts = self._make_tts(self.tts_backend)
                    self._tts_cache[self.tts_backend] = self._tts
                    self._settings_status = f"Voice changed to {self._current_voice_label()}."
                except Exception as e:
                    self._settings_status = f"Voice load failed: {e}"
            return

        if self._settings_view == "model":
            keys = list(LLM_MODELS.keys())
            selected = keys[self._settings_index]
            with self._settings_lock:
                self._select_llm_model(selected)
            return

        if self._settings_index == 0:
            keys = list(KOKORO_VOICES.keys())
            self._settings_view = "voice"
            self._settings_index = keys.index(self.tts_voice) if self.tts_voice in keys else 0
            self._settings_status = ""
        elif self._settings_index == 1:
            keys = list(LLM_MODELS.keys())
            self._settings_view = "model"
            self._settings_index = keys.index(self.llm_model_key) if self.llm_model_key in keys else 0
            self._settings_status = ""
        elif self._settings_index == 2:
            if self._settings_lock.acquire(blocking=False):
                def _run():
                    try:
                        self._download_selected_assets()
                    finally:
                        self._settings_lock.release()
                threading.Thread(target=_run, daemon=True).start()
        elif self._settings_index == 3:
            self.settings_open = False
            self._settings_status = ""

    # ── text input ───────────────────────────────────────────────────────────
    def _start_text_input(self):
        self._text_input_mode   = True
        self._text_input_buffer = ""
        if self._ptt:
            self._ptt.text_mode = True
        self._set_state(State.TYPING)
        self._live_text = "█"

    def _cancel_text_input(self):
        self._text_input_mode   = False
        self._text_input_buffer = ""
        if self._ptt:
            self._ptt.text_mode = False
        self._live_text = ""
        self._set_state(State.IDLE)

    def _handle_text_char(self, char: str):
        """Called from pynput thread for every keystroke while in text-input mode."""
        if char == "\n":
            text = self._text_input_buffer.strip()
            self._text_input_mode   = False
            self._text_input_buffer = ""
            if self._ptt:
                self._ptt.text_mode = False
            self._live_text = ""
            self._set_state(State.IDLE)
            if text:
                self._text_input_queue.put(text)
        elif char == "\x08":                          # backspace
            self._text_input_buffer = self._text_input_buffer[:-1]
            self._live_text = self._text_input_buffer + "█"
        else:
            self._text_input_buffer += char
            self._live_text = self._text_input_buffer + "█"

    def _handle_settings_key(self, key: str):
        # Text-input mode takes priority over everything else.
        if self._text_input_mode:
            if key == "esc":
                self._cancel_text_input()
            return
        if key == "t" and not self.settings_open:
            self._start_text_input()
            return
        if key == "s":
            self.settings_open = not self.settings_open
            self._settings_view = "main"
            self._settings_index = 0
            self._settings_status = ""
            return
        if key == "esc":
            if self.settings_open:
                self._settings_back()
            return
        if not self.settings_open:
            return
        if key == "up":
            self._settings_move(-1)
        elif key == "down":
            self._settings_move(1)
        elif key == "enter":
            self._settings_select()

    # ── conversation helpers ─────────────────────────────────────────────────
    def _add_message(self, role: str, text: str):
        with self._lock:
            self.messages.append(Message(role=role, text=text))

    def _set_state(self, s: str, status: str = ""):
        with self._lock:
            self.state   = s
            self._status = status

    def _set_live_text(self, text: str):
        with self._lock:
            self._live_text = text

    def _ui_snapshot(self) -> dict:
        with self._lock:
            return {
                "state": self.state,
                "messages": list(self.messages),
                "level": self._level,
                "status": self._status,
                "live_text": self._live_text,
                "tts": self.tts_backend,
                "settings_open": self.settings_open,
                "settings_status": self._settings_status,
                "current_model": self._current_model_label(),
                "current_voice": self._current_voice_label(),
                "settings_view": self._settings_view,
                "settings_index": self._settings_index,
            }

    # ── partial STT (runs in background while recording) ─────────────────────
    def _partial_transcribe(self, audio: np.ndarray):
        """Non-blocking: update live transcript while user is speaking."""
        # Skip if audio is mostly silence — prevents Whisper hallucinations
        if np.abs(audio).mean() < PARTIAL_STT_MIN_RMS:
            return
        if not self._stt_lock.acquire(blocking=False):
            return                              # previous call still running — skip
        try:
            segments, _ = self._whisper.transcribe(
                audio, language="en", beam_size=1,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 150, "threshold": 0.4},
            )
            text = " ".join(s.text for s in segments).strip()
            # Reject single-word hallucinations (Whisper often emits "the", ".", etc.)
            if text and len(text.split()) > 1:
                self._live_text = text
        except Exception:
            pass
        finally:
            self._stt_lock.release()

    # ── recording ─────────────────────────────────────────────────────────────
    def _record_until_silence(self, capture: AudioCapture) -> np.ndarray:
        """Wake-word path: record until SILENCE_SEC of quiet, with live STT."""
        self._set_state(State.RECORDING)
        self._live_text = ""
        capture.flush()                     # discard audio captured during wake detection
        frames: List[np.ndarray] = []
        silent_sec   = 0.0
        total_sec    = 0.0
        frame_sec    = CHUNK_SAMPLES / SAMPLE_RATE
        last_stt_t   = time.time()
        STT_INTERVAL = 0.4                  # partial transcript every 0.4 s

        while total_sec < MAX_RECORD_SEC:
            chunk = capture.read_chunk(timeout=0.15)
            if chunk is None:
                continue
            frames.append(chunk)
            total_sec  += frame_sec
            rms = float(np.abs(chunk).mean())
            self._level = rms
            silent_sec  = silent_sec + frame_sec if rms < RECORD_SILENCE_RMS else 0.0

            now = time.time()
            if now - last_stt_t >= STT_INTERVAL and len(frames) > 15:
                last_stt_t = now
                snap = np.concatenate(frames).copy()
                threading.Thread(target=self._partial_transcribe, args=(snap,), daemon=True).start()

            if total_sec >= MIN_RECORD_SEC and silent_sec >= SILENCE_SEC:
                break

        return np.concatenate(frames) if frames else np.zeros(SAMPLE_RATE, dtype=np.float32)

    def _record_ptt(self, capture: AudioCapture, ptt: "PushToTalk") -> np.ndarray:
        """Record while SPACE is held, showing a live partial transcript."""
        self._set_state(State.RECORDING)
        self._live_text = ""
        capture.flush()
        frames: List[np.ndarray] = []
        last_stt_t   = time.time()
        STT_INTERVAL = 0.4

        while ptt.is_held and self._running:
            chunk = capture.read_chunk(timeout=0.05)
            if chunk is None:
                continue
            frames.append(chunk)
            self._level = float(np.abs(chunk).mean())

            now = time.time()
            if now - last_stt_t >= STT_INTERVAL and len(frames) > 10:
                last_stt_t = now
                audio_snap = np.concatenate(frames).copy()
                threading.Thread(
                    target=self._partial_transcribe,
                    args=(audio_snap,),
                    daemon=True,
                ).start()

        return np.concatenate(frames) if frames else np.zeros(SAMPLE_RATE, dtype=np.float32)

    # ── main loop ─────────────────────────────────────────────────────────────
    def run(self):
        with quiet_terminal_input(), Live(
            build_ui(self.state, self.messages),
            console=console,
            refresh_per_second=10,
            screen=True,
        ) as live:

            def refresh():
                while self._running:
                    try:
                        snap = self._ui_snapshot()
                        bar = make_level_bar(min(snap["level"] * LEVEL_GAIN, 1.0))
                        live.update(build_ui(
                            snap["state"], snap["messages"],
                            level_bar=bar,
                            status_line=snap["status"],
                            live_text=snap["live_text"],
                            tts=snap["tts"],
                            settings_open=snap["settings_open"],
                            settings_status=snap["settings_status"],
                            current_model=snap["current_model"],
                            current_voice=snap["current_voice"],
                            settings_view=snap["settings_view"],
                            settings_index=snap["settings_index"],
                        ))
                    except Exception:
                        with (LOGS_DIR / "ui-refresh-error.log").open("a", encoding="utf-8") as f:
                            traceback.print_exc(file=f)
                    time.sleep(0.1)

            ui_thread = threading.Thread(target=refresh, daemon=True)
            ui_thread.start()

            try:
                self._set_state(State.BOOT, "Loading models…")
                self._load_engines()
                self._set_state(State.IDLE)

                capture = AudioCapture()
                capture.start()
                ptt = PushToTalk(on_key=self._handle_settings_key, on_text_char=self._handle_text_char)
                self._ptt = ptt
                ptt.start()

                def speak_interruptible(text: str):
                    """Speak text; PTT press stops both playback and generation."""
                    stop_ev = threading.Event()
                    done    = threading.Event()
                    self._set_state(State.SPEAKING)
                    self._live_text = text or ""

                    def _play():
                        try:
                            self._tts.speak(text, stop_event=stop_ev)
                        except Exception:
                            with (LOGS_DIR / "tts-error.log").open("a", encoding="utf-8") as f:
                                traceback.print_exc(file=f)
                        finally:
                            done.set()

                    threading.Thread(target=_play, daemon=True).start()
                    while not done.wait(timeout=0.04):
                        if text and not self._live_text:
                            self._live_text = text
                        if ptt.is_held:
                            stop_ev.set()       # tell generator to stop
                            sd.stop()           # cut current sentence immediately
                            done.wait(timeout=1)
                            break
                    # Flush mic so TTS bleed doesn't get transcribed
                    time.sleep(0.12)
                    capture.flush()

                try:
                    while self._running:
                        # ── idle: wake word OR space bar ──────────────────
                        self._set_state(State.IDLE)
                        self._live_text = ""
                        triggered_by = None

                        while self._running:
                            if self.settings_open or self._text_input_mode:
                                time.sleep(0.05)
                                continue
                            if not self._text_input_queue.empty():
                                triggered_by = "text"
                                break
                            if ptt.is_held:
                                triggered_by = "ptt"
                                break
                            chunk = capture.read_chunk(timeout=0.05)
                            if chunk is not None:
                                self._level = float(np.abs(chunk).mean())
                                if self._wake.feed(chunk):
                                    triggered_by = "wake"
                                    break

                        if not self._running:
                            break

                        # ── text input path: skip recording and STT ───────
                        if triggered_by == "text":
                            user_text = self._text_input_queue.get()
                            if not user_text or len(user_text.strip()) < 2:
                                continue
                            self._add_message("user", user_text)
                            self._live_text = ""
                            self._set_state(State.THINKING, "Generating response…")

                        else:
                            # ── record ────────────────────────────────────────
                            # Both paths go straight to recording — no ack delay.
                            # Wake word shows LISTENING label for 0.1 s so the
                            # state change is visible, then starts recording.
                            if triggered_by == "wake":
                                self._set_state(State.LISTENING)
                                time.sleep(0.1)
                                audio = self._record_until_silence(capture)
                            else:
                                self._set_state(State.RECORDING)
                                audio = self._record_ptt(capture, ptt)

                            if len(audio) < SAMPLE_RATE * 0.3:   # < 0.3 s — too short
                                continue

                            # ── final STT ─────────────────────────────────────
                            self._live_text = ""
                            self._set_state(State.THINKING, "Transcribing…")
                            user_text = self._stt.transcribe(audio)

                            if not user_text or len(user_text.strip()) < 2:
                                self._live_text = ""
                                self._set_state(State.SPEAKING)
                                speak_interruptible("I didn't catch that. Try again!")
                                continue

                            self._add_message("user", user_text)
                            self._live_text = ""
                            self._set_state(State.THINKING, "Generating response…")

                        # ── LLM (streaming tokens) ────────────────────────
                        def on_token(partial: str):
                            self._live_text = partial

                        def on_speak_plan(text: str):
                            """Fire plan TTS in background — return immediately so tools start in parallel."""
                            def _play():
                                try:
                                    self._tts.speak(text)
                                except Exception:
                                    pass
                            threading.Thread(target=_play, daemon=True).start()
                            self._set_state(State.THINKING, "Working on it…")
                            self._live_text = ""

                        reply = self._llm.chat_stream(user_text, on_token, on_speak=on_speak_plan)
                        self._add_message("bob", reply)

                        # ── TTS (interruptible) ───────────────────────────
                        self._live_text = reply
                        self._set_state(State.SPEAKING)
                        speak_interruptible(reply)
                        self._live_text = ""

                finally:
                    ptt.stop()
                    capture.stop()

            except KeyboardInterrupt:
                pass
            except Exception as e:
                self._set_state(State.ERROR, str(e))
                console.print_exception()
                time.sleep(3)
            finally:
                self._running = False


# ─────────────────────────────── Entry point ────────────────────────────────
TTS_CHOICES = ["kokoro"]

def run_builder_self_test():
    harness = ToolHarness()
    project = "builder-self-test"
    malformed_tool_json = (
        '{"status":"Creating an HTML file now.","tool_calls":[{"tool":"write_project_file",'
        '"args":{"project":"builder-self-test","path":"from-malformed-tool.json.html",'
        '"content":"<html><body><h1>Recovered tool call</h1></body></html>"}}]}'
    )
    screenshot_style_tool_json = (
        '{"status":"Creating the CSS file for the website.","tool_calls":[{"tool":"write_project_file",'
        '"args":{"path":"builder-self-test/style.css","content":"/* Global Styles */\\n'
        ':root {\\n  --primary-color: #6a11cb;\\n  --secondary-color: #2575fc;\\n}\\n'
        'body { font-family: Segoe UI, Tahoma, Geneva, Verdana, sans-serif; }\\n"}}]}'
    )
    recovered_calls = harness.parse_tool_calls(malformed_tool_json)
    recovered_calls += harness.parse_tool_calls(screenshot_style_tool_json)
    setup_calls = [
        {"tool": "delete_project_path", "args": {"path": f"{project}/recovered-tool-call.html"}},
        {"tool": "delete_project_path", "args": {"path": f"{project}/organized"}},
        {"tool": "make_project_directory", "args": {}},
        {"tool": "make_project_directory", "args": {"path": project}},
        {
            "tool": "write_project_file",
            "args": {
                "path": f"{project}/index.html",
                "content": (
                    "<!doctype html>\n"
                    "<html lang=\"en\">\n"
                    "<head><meta charset=\"utf-8\"><title>Builder Test</title><link rel=\"stylesheet\" href=\"style.css\"></head>\n"
                    "<body><h1>Builder Test Works</h1></body>\n"
                    "</html>\n"
                ),
            },
        },
        {
            "tool": "write_project_file",
            "args": {
                "path": f"{project}/main.py",
                "content": "print('Builder test works')\n",
            },
        },
        {"tool": "write_project_file", "args": {"path": f"{project}/loose-a.txt", "content": "A\n"}},
        {"tool": "write_project_file", "args": {"path": f"{project}/loose-b.txt", "content": "B\n"}},
        {"tool": "make_project_directory", "args": {"path": f"{project}/organized"}},
        {"tool": "move_project_paths", "args": {"paths": [f"{project}/loose-a.txt", f"{project}/loose-b.txt"], "destination": f"{project}/organized"}},
        {
            "tool": "edit_project_file",
            "args": {
                "path": f"{project}/main.py",
                "edits": [{"old": "Builder test works", "new": "Builder agent tools work"}],
            },
        },
        {"tool": "read_project_file", "args": {"path": f"{project}/index.html", "offset": 1, "limit": 6}},
        {"tool": "find_project_files", "args": {"path": project, "pattern": "*.html"}},
        {"tool": "grep_project", "args": {"path": project, "pattern": "Builder", "glob": "*.html"}},
    ] + recovered_calls
    review_calls = [
        {"tool": "review_web_project", "args": {"path": project}},
        {"tool": "rename_project_path", "args": {"old_path": f"{project}/from-malformed-tool.json.html", "new_path": f"{project}/recovered-tool-call.html"}},
        {"tool": "list_project_tree", "args": {"path": project}},
        {"tool": "run_project_command", "args": {"command": f"python {project}/main.py", "timeout": 10}},
    ]
    results = harness.run_calls(setup_calls + review_calls)
    ok = all(r.get("ok") for r in results) and harness.looks_like_tool_json(malformed_tool_json)
    path = PROJECTS_DIR / project
    style = "green" if ok else "red"
    console.print(Panel(
        f"[bold {style}]{'Builder test passed' if ok else 'Builder test failed'}[/]\n\n"
        f"Created files in:\n[cyan]{path}[/]\n\n"
        "Expected:\n"
        f"  {path / 'index.html'}\n"
        f"  {path / 'main.py'}\n"
        f"  {path / 'recovered-tool-call.html'}\n"
        f"  {path / 'organized' / 'loose-a.txt'}\n"
        f"  {path / 'organized' / 'loose-b.txt'}\n"
        f"  {path / 'style.css'}",
        title=f"BOB {BOB_BUILD_VERSION}",
        box=box.ROUNDED,
        border_style=style,
    ))

def run_offline_model_check():
    selected = selected_llm_key()
    checks = [
        (f"Selected LLM ({LLM_MODELS[selected]['label']})", local_llm_gguf(selected)),
        ("Whisper tiny", local_whisper_model("tiny")),
        (f"Whisper {WHISPER_MODEL}", local_whisper_model(WHISPER_MODEL)),
        ("Kokoro model", _HERE / "models" / "kokoro" / "kokoro-v1.0.onnx"),
        ("Kokoro voices", _HERE / "models" / "kokoro" / "voices-v1.0.bin"),
    ]
    rows = []
    all_ok = True
    for label, path in checks:
        ok = bool(path and Path(path).exists())
        all_ok = all_ok and ok
        rows.append(f"{'[green]✓[/]' if ok else '[red]✗[/]'} {label}: [dim]{path or 'missing'}[/]")

    style = "green" if all_ok else "yellow"
    console.print(Panel(
        "\n".join(rows) + "\n\n"
        f"Offline mode: [bold]{'ON' if not ALLOW_ONLINE else 'OFF, BOB_ALLOW_ONLINE=1'}[/]",
        title=f"BOB {BOB_BUILD_VERSION} Offline Model Check",
        box=box.ROUNDED,
        border_style=style,
    ))

def _download_llm(model_key: str) -> bool:
    meta = LLM_MODELS[model_key]
    if local_llm_gguf(model_key):
        console.print(f"[green]✓[/] {meta['label']} already cached")
        return True
    try:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        from huggingface_hub import hf_hub_download
        console.print(f"[yellow]→[/] Downloading [bold]{meta['label']}[/]")
        hf_hub_download(
            repo_id=meta["repo_id"],
            filename=meta["filename"],
            cache_dir=str(MODELS_DIR / "hub"),
            local_files_only=False,
        )
        return bool(local_llm_gguf(model_key))
    except Exception as e:
        console.print(f"[red]✗ LLM download failed: {e}[/]")
        return False


def _download_whisper_model(name: str) -> bool:
    if local_whisper_model(name):
        console.print(f"[green]✓[/] Whisper {name} already cached")
        return True
    try:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        from huggingface_hub import snapshot_download
        console.print(f"[yellow]→[/] Downloading [bold]Whisper {name}[/]")
        snapshot_download(
            repo_id=f"Systran/faster-whisper-{name}",
            cache_dir=str(MODELS_DIR / "hub"),
            local_files_only=False,
        )
        return bool(local_whisper_model(name))
    except Exception as e:
        console.print(f"[red]✗ Whisper {name} download failed: {e}[/]")
        return False


def _download_kokoro_files() -> bool:
    kokoro_dir = MODELS_DIR / "kokoro"
    kokoro_dir.mkdir(parents=True, exist_ok=True)
    model_dest = kokoro_dir / "kokoro-v1.0.onnx"
    voices_dest = kokoro_dir / "voices-v1.0.bin"
    ok = True

    def download_url(url: str, dest: Path):
        if shutil.which("curl"):
            subprocess.run(
                ["curl", "-L", "--fail", "--progress-bar", url, "-o", str(dest)],
                check=True,
            )
        else:
            import ssl
            import urllib.request
            context = ssl._create_unverified_context()
            with urllib.request.urlopen(url, context=context) as response:
                dest.write_bytes(response.read())

    if model_dest.exists():
        console.print("[green]✓[/] kokoro-v1.0.onnx already cached")
    else:
        model_url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
        try:
            # hexgrad/Kokoro-82M is the upstream PyTorch model. BOB uses
            # kokoro-onnx, so we download the compatible ONNX runtime files.
            console.print("[yellow]→[/] Downloading [bold]Kokoro ONNX model[/]")
            download_url(model_url, model_dest)
            console.print("[green]✓[/] Kokoro ONNX model saved")
        except Exception as e:
            console.print(f"[red]✗ Kokoro ONNX model download failed: {e}[/]")
            ok = False

    if voices_dest.exists():
        console.print("[green]✓[/] voices-v1.0.bin already cached")
    else:
        voice_url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
        try:
            console.print("[yellow]→[/] Downloading [bold]Kokoro voices[/]")
            download_url(voice_url, voices_dest)
            console.print("[green]✓[/] Kokoro voices saved")
        except Exception as e:
            console.print(f"[red]✗ Kokoro voices download failed: {e}[/]")
            ok = False

    return ok and model_dest.exists() and voices_dest.exists()


def _download_openwakeword_models() -> bool:
    try:
        os.environ.pop("HF_HUB_OFFLINE", None)
        from openwakeword.utils import download_models as oww_download_models
        console.print("[yellow]→[/] Downloading OpenWakeWord base models")
        oww_download_models()
        console.print("[green]✓[/] OpenWakeWord ready")
        return True
    except Exception as e:
        console.print(f"[yellow]![/] OpenWakeWord download skipped: {e}")
        return False


def setup_wizard(force: bool = False) -> bool:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

    console.print(Panel(
        "[bold]Welcome to BOB setup.[/]\n\n"
        "This wizard creates the models and projects folders, lets you choose the LLM, "
        "then downloads the LLM, Kokoro TTS, Whisper, and OpenWakeWord assets.",
        title=f"BOB {BOB_BUILD_VERSION}",
        box=box.ROUNDED,
        border_style="cyan",
    ))

    keys = list(LLM_MODELS.keys())
    console.print("[bold]Choose your local AI model:[/]")
    for i, key in enumerate(keys, start=1):
        meta = LLM_MODELS[key]
        console.print(f"  [cyan]{i}[/]. [bold]{meta['label']}[/] — {meta['strengths']}")

    default_index = keys.index(selected_llm_key()) + 1 if selected_llm_key() in keys else 1
    choice = input(f"\nModel number [{default_index}]: ").strip()
    if not choice:
        choice_index = default_index
    else:
        try:
            choice_index = max(1, min(len(keys), int(choice)))
        except ValueError:
            choice_index = default_index

    model_key = keys[choice_index - 1]
    if model_key == "qwen-27b":
        console.print("[yellow]Qwen3.6 27B is the best coding option here, but it is very large and slow on many Macs.[/]")
        confirm = input("Download/use Qwen3.6 27B anyway? Type yes to continue: ").strip().lower()
        if confirm != "yes":
            model_key = DEFAULT_LLM_MODEL
            console.print(f"[cyan]Using {LLM_MODELS[model_key]['label']} instead.[/]")

    config = load_config()
    config["llm_model"] = model_key
    config.setdefault("kokoro_voice", DEFAULT_KOKORO_VOICE)
    save_config(config)

    results = {
        LLM_MODELS[model_key]["label"]: _download_llm(model_key),
        "Kokoro TTS": _download_kokoro_files(),
        "Whisper tiny": _download_whisper_model("tiny"),
        f"Whisper {WHISPER_MODEL}": _download_whisper_model(WHISPER_MODEL),
        "OpenWakeWord": _download_openwakeword_models(),
    }
    ok = all(results.values())
    rows = "\n".join(f"{'[green]✓[/]' if result else '[red]✗[/]'} {name}" for name, result in results.items())
    console.print(Panel(
        rows,
        title="[green]Setup complete[/]" if ok else "[yellow]Setup finished with warnings[/]",
        box=box.ROUNDED,
        border_style="green" if ok else "yellow",
    ))
    return ok


def setup_needed() -> bool:
    config = load_config()
    model_key = config.get("llm_model", DEFAULT_LLM_MODEL)
    return (
        not CONFIG_PATH.exists()
        or local_llm_gguf(model_key) is None
        or local_whisper_model("tiny") is None
        or local_whisper_model(WHISPER_MODEL) is None
        or not (MODELS_DIR / "kokoro" / "kokoro-v1.0.onnx").exists()
        or not (MODELS_DIR / "kokoro" / "voices-v1.0.bin").exists()
    )

def relaunch_in_terminal_if_needed():
    """When bob.py is launched from Finder/IDLE, reopen it in macOS Terminal."""
    if sys.platform != "darwin":
        return
    if "--test-builder" in sys.argv or "--check-offline" in sys.argv or "--setup" in sys.argv:
        return
    if "--launched-in-terminal" in sys.argv:
        return
    if sys.stdin.isatty() and sys.stdout.isatty():
        return

    python_bin = _HERE / ".venv" / "bin" / "python"
    if not python_bin.exists():
        python_bin = Path(sys.executable)

    args = [str(python_bin), str(Path(__file__).resolve()), "--launched-in-terminal"]
    args.extend(arg for arg in sys.argv[1:] if arg != "--launched-in-terminal")
    shell_command = f"cd {shlex.quote(str(_HERE))} && {' '.join(shlex.quote(a) for a in args)}"
    osa = (
        'tell application "Terminal"\n'
        f'  do script {json.dumps(shell_command)}\n'
        '  activate\n'
        'end tell'
    )
    try:
        subprocess.run(["osascript", "-e", osa], check=True)
        sys.exit(0)
    except Exception:
        return

def main():
    relaunch_in_terminal_if_needed()

    import argparse
    parser = argparse.ArgumentParser(
        description="BOB — Local AI Voice Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "TTS:\n"
            "  Kokoro ONNX only — fast, local, lightweight\n"
        ),
    )
    parser.add_argument(
        "--tts",
        choices=TTS_CHOICES,
        default="kokoro",
        metavar="MODEL",
        help="TTS engine to use: kokoro (default)",
    )
    parser.add_argument("--launched-in-terminal", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--test-builder",
        action="store_true",
        help="Create a small HTML and Python test project, then exit.",
    )
    parser.add_argument(
        "--check-offline",
        action="store_true",
        help="Verify cached model paths for offline startup, then exit.",
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Run the first-launch setup wizard and download models.",
    )
    args = parser.parse_args()

    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold green]BOB {BOB_BUILD_VERSION}[/]  Project builder: [bold green]ON[/]  Folder: [cyan]{PROJECTS_DIR}[/]")

    if args.test_builder:
        run_builder_self_test()
        return
    if args.check_offline:
        run_offline_model_check()
        return
    if args.setup:
        setup_wizard(force=True)
        return

    if setup_needed() and sys.stdin.isatty():
        setup_wizard(force=False)

    # Quick pre-flight: verify llama-cpp-python is installed
    try:
        import llama_cpp  # noqa
    except ImportError:
        console.print(Panel(
            "[bold yellow]llama-cpp-python not installed![/]\n\n"
            "Run the setup script first:\n\n"
            "  [bold cyan]bash setup.sh[/]\n\n"
            "Then run the setup wizard:\n\n"
            "  [bold cyan]python bob.py --setup[/]\n\n"
            "Then start BOB:\n\n"
            "  [bold cyan]python bob.py[/]",
            title="[red]Setup Required[/]",
            box=box.ROUNDED,
            border_style="red",
        ))
        sys.exit(1)

    kokoro_model  = _HERE / "models" / "kokoro" / "kokoro-v1.0.onnx"
    kokoro_voices = _HERE / "models" / "kokoro" / "voices-v1.0.bin"
    if not kokoro_model.exists() or not kokoro_voices.exists():
        console.print(Panel(
            "[bold yellow]Kokoro model files not found![/]\n\n"
            "Run:\n\n"
            "  [bold cyan]python bob.py --setup[/]\n\n"
            "Expected files:\n"
            f"  [dim]{kokoro_model}[/]\n"
            f"  [dim]{kokoro_voices}[/]",
            title="[red]Kokoro Not Ready[/]",
            box=box.ROUNDED,
            border_style="red",
        ))
        sys.exit(1)
    console.print("[bold cyan]TTS: Kokoro[/]")
    voice = selected_kokoro_voice()
    console.print(f"[bold cyan]Voice: {KOKORO_VOICES.get(voice, voice)}[/]")

    bob = BOB(tts=args.tts, voice=voice)
    bob.run()


if __name__ == "__main__":
    main()
