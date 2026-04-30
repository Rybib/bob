#!/usr/bin/env python3
"""
BOB — Local AI Voice Assistant
Wake word → STT (Whisper) → LLM (GGUF) → TTS (Kokoro)
"""

import os, sys, time, queue, threading, re, warnings, tempfile, random, json, subprocess, shlex, shutil, contextlib
import numpy as np
from pathlib import Path

# ── Point HuggingFace cache at our local models/ folder ─────────────────────
_HERE = Path(__file__).parent
os.environ["HUGGINGFACE_HUB_CACHE"] = str(_HERE / "models" / "hub")
os.environ["HF_HOME"]               = str(_HERE / "models")
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
    },
    "gemma-e4b": {
        "label": "Gemma 3n E4B it Q4_K_M",
        "repo_id": "himkhati22/gemma-3n-E4B-it-Q4_K_M-GGUF",
        "filename": "gemma-3n-e4b-it-q4_k_m.gguf",
        "strengths": "Better reasoning and writing than E2B while still reasonably local.",
        "cache_name": "models--himkhati22--gemma-3n-E4B-it-Q4_K_M-GGUF",
    },
    "qwen-27b": {
        "label": "Qwen3.6 27B Q4_K_M",
        "repo_id": "sm54/Qwen3.6-27B-Q4_K_M-GGUF",
        "filename": "qwen3.6-27b-q4_k_m.gguf",
        "strengths": "Best coding choice, strongest for larger builds, but very heavy and slower.",
        "cache_name": "models--sm54--Qwen3.6-27B-Q4_K_M-GGUF",
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
    "If the user asks for code, a script, a website, an app, HTML, Python, "
    "or any build request, do not print code in chat; the app will route that "
    "through the project-file builder. "
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
- You may call multiple tools at once when they are independent.
- After tool results are returned, either call more tools or give a concise
  finished answer for the user.
- Never wrap tool-call JSON in markdown.

Available tools:
- create_or_select_project(name): create/select a project folder.
- list_projects(): list project folders.
- list_files(project, path="."): list files under a project path.
- read_file(project, path, max_chars=12000): read a UTF-8-ish text file.
- write_file(project, path, content): create or overwrite a file.
- append_file(project, path, content): append to a file.
- replace_in_file(project, path, old, new): replace exact text in a file.
- make_directory(project, path): create a directory.
- reference_workspace(max_files=120): list the main Bob repo and project files.
- read_workspace_file(path, max_chars=12000): read a text file from the main Bob repo.

Rules:
- Any request involving code, scripts, HTML, CSS, JavaScript, Python, apps,
  websites, or "build this" work MUST use the tool harness to create or edit
  real files in projects/. Do not answer those requests by printing source code
  in chat.
- Put user-requested builds in a sensible project folder under projects/.
- If the user does not name a project, use the next numbered project folder,
  such as project-1, project-2, and so on.
- Prefer small complete apps: include runnable files, README notes, and obvious
  entry points when useful.
- Before editing an existing file, read it unless the user clearly wants a fresh
  overwrite.
- After the files are created or edited, give a short high-level summary:
  say the work is complete, name the project folder, and list the important
  files. Do not include full source code unless the user explicitly asks to see
  a snippet.
- Never attempt to write outside projects/.
""".strip()

BUILD_MARKER = "@@BOB_BUILD@@"
BUILD_OUTPUT_SYSTEM = f"""
SYSTEM PROMPT:
You are BOB, a voice assistant. Act natural for normal conversation.

IMPORTANT CODING RULE:
If the user's request is about code, a website, an app, a script, HTML, CSS,
JavaScript, Python, files, folders, or any programming/build task, you are NOT
allowed to answer with code in chat. You must output a machine-readable build
signal so Bob can create real files in the projects folder.

OUTPUT FORMAT FOR CODING REQUESTS:
Output ONLY this exact marker, then valid JSON:

{BUILD_MARKER}
{{"project":"project-auto","files":[{{"path":"index.html","content":"<full file contents here>"}}]}}

Rules:
- Do not explain.
- Do not use markdown.
- Do not tell the user to copy and paste.
- Do not output anything before {BUILD_MARKER}.
- Generate complete file contents.
- The JSON must have a files array.
- Each file must have path and content.
- File paths must be relative paths, never absolute paths.
- The path decides the file type and extension.
- For simple websites, use index.html.
- For websites with styling or behavior, use index.html, styles.css, and script.js.
- For Python scripts, use main.py.
- For docs, use README.md.
- For nested folders, use paths like src/main.py or assets/style.css.
""".strip()

# ─────────────────────────────── State ──────────────────────────────────────
class State:
    BOOT      = "boot"
    IDLE      = "idle"
    LISTENING = "listening"
    RECORDING = "recording"
    THINKING  = "thinking"
    SPEAKING  = "speaking"
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
    State.IDLE:      ("blue",    "💤", "Say  BOB  — or hold  SPACE  to talk"),
    State.LISTENING: ("green",   "👂", "Go ahead…"),
    State.RECORDING: ("green",   "🔴", "Listening — release SPACE when done"),
    State.THINKING:  ("yellow",  "🧠", "Thinking…"),
    State.SPEAKING:  ("magenta", "🔊", "Speaking…"),
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
        table = Table(box=None, show_header=False, padding=(0, 1), expand=True)
        table.add_column("ts",   style="dim",   width=6, no_wrap=True)
        table.add_column("who",  style="bold",  width=5, no_wrap=True)
        table.add_column("text", style="white", ratio=1)

        visible = messages[-8:]
        for m in visible:
            if m.role == "user":
                table.add_row(m.ts, Text("YOU", style="bold green"),  Text(m.text))
            else:
                table.add_row(m.ts, Text("BOB", style=f"bold {color}"), Text(m.text, style="bright_white"))

        if not visible:
            table.add_row("", "", Text("No messages yet — say «BOB» to start.", style="dim italic"))

        chat_panel = Panel(
            table,
            title="[bold]Conversation[/]",
            border_style="dim",
            box=box.ROUNDED,
        )

    # ── footer ──
    footer = Text(
        "  Ctrl+C: quit  │  SPACE: push-to-talk  │  S: settings  ",
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
    """Watches the space bar globally via pynput (no terminal focus needed)."""

    def __init__(self, on_switch_tts=None, on_key=None):
        self._held           = False
        self._press_event    = threading.Event()
        self._release_event  = threading.Event()
        self._listener       = None
        self._on_switch_tts  = on_switch_tts
        self._on_key         = on_key

    def start(self):
        from pynput import keyboard

        def on_press(key):
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
            elif hasattr(key, "char") and key.char == "t" and self._on_switch_tts:
                threading.Thread(target=self._on_switch_tts, daemon=True).start()
            elif hasattr(key, "char") and key.char and self._on_key:
                threading.Thread(target=self._on_key, args=(key.char.lower(),), daemon=True).start()

        def on_release(key):
            if key == keyboard.Key.space:
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

    TEXT_EXTENSIONS = {
        ".py", ".html", ".css", ".js", ".json", ".md", ".txt", ".csv", ".tsv",
        ".xml", ".svg", ".yaml", ".yml", ".toml", ".ini", ".env", ".gitignore",
        ".sh", ".sql",
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

    def _safe_path(self, project: str, path: str = ".") -> Path:
        base = self._project_dir(project)
        target = (base / (path or ".")).resolve()
        if target != base and base not in target.parents:
            raise ValueError("Path escaped the selected project folder.")
        return target

    def _is_text_path(self, path: Path) -> bool:
        return path.suffix.lower() in self.TEXT_EXTENSIONS or path.name in {".env", ".gitignore"}

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

    def parse_tool_calls(self, text: str) -> List[dict]:
        data = self._extract_json(text)
        if not isinstance(data, dict):
            return []
        calls = data.get("tool_calls", [])
        return calls if isinstance(calls, list) else []

    def run_calls(self, calls: List[dict], on_update=None) -> List[dict]:
        results = []
        for call in calls:
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
            try:
                result = fn(**args)
                results.append({"tool": tool_name, "ok": True, "result": result})
            except Exception as e:
                results.append({"tool": tool_name, "ok": False, "error": str(e)})
        return results


# ─────────────────────────────── LLM ────────────────────────────────────────
class LLMEngine:
    def __init__(self):
        from llama_cpp import Llama
        self._model_key = selected_llm_key()
        self._model_meta = LLM_MODELS.get(self._model_key, LLM_MODELS[DEFAULT_LLM_MODEL])
        model_path = local_llm_gguf(self._model_key)
        if model_path is not None:
            console.print(f"[dim]Loading {self._model_meta['label']} from local cache: {model_path}[/]")
            self._llm = Llama(
                model_path=str(model_path),
                n_ctx=4096,
                n_gpu_layers=-1,   # offload all layers to GPU/Metal if available
                verbose=False,
            )
        elif ALLOW_ONLINE:
            console.print(f"[dim]Loading {self._model_meta['label']} from Hugging Face…[/]")
            self._llm = Llama.from_pretrained(
                repo_id=self._model_meta["repo_id"],
                filename=self._model_meta["filename"],
                n_ctx=4096,
                n_gpu_layers=-1,
                verbose=False,
            )
        else:
            raise RuntimeError(
                f"{self._model_meta['label']} is not cached locally. Run `python bob.py --setup` online once, "
                "or set BOB_ALLOW_ONLINE=1 to allow downloads."
            )
        self._history: List[dict] = []
        self._harness = ToolHarness()

    @property
    def model_label(self) -> str:
        return self._model_meta["label"]

    def _completion_text(self, messages: List[dict], max_tokens: int = 900, temperature: float = 0.35) -> str:
        response = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            stop=["<end_of_turn>", "<eos>"],
            stream=False,
        )
        return response["choices"][0]["message"].get("content", "").strip()

    def _should_use_tools(self, text: str) -> bool:
        return bool(re.search(
            r"\b(build|create|make|generate|design|edit|write|code|coding|program|file|folder|project|html|css|javascript|js|python|py|script|app|website|webpage|page|read|reference)\b",
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

            path = result.get("path")
            if not path:
                continue
            if entry.get("tool") not in {"write_file", "append_file", "replace_in_file"}:
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
                f"Done. I built it in {location}. "
                f"I created or updated {len(changed_files)} file"
                f"{'' if len(changed_files) == 1 else 's'}: {important}{extra}. "
                "The files are ready in your projects folder."
            )

        return "Done. I used the project harness and finished the requested work in your projects folder."

    def _has_file_write(self, results: List[dict]) -> bool:
        return any(
            entry.get("ok") and entry.get("tool") in {"write_file", "append_file", "replace_in_file"}
            for entry in results
        )

    def _next_project_name(self) -> str:
        root = self._harness.workspace.root
        existing = {p.name for p in root.iterdir() if p.is_dir()}
        n = 1
        while f"project-{n}" in existing:
            n += 1
        return f"project-{n}"

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
        files = self._detect_generated_code_files(model_text)
        if not files:
            return None
        on_token("I detected generated code, so I am saving it into project files…")
        summary = self._write_build_files(files, on_token)
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

    def _write_build_files(self, files: List[dict], on_token) -> str:
        project = self._next_project_name()
        on_token(f"Creating files in projects/{project}…")
        calls = [{"tool": "create_or_select_project", "args": {"name": project}}]
        for file_info in files:
            calls.append({
                "tool": "write_file",
                "args": {
                    "project": project,
                    "path": file_info["path"],
                    "content": file_info["content"],
                },
            })
        results = self._harness.run_calls(calls, on_update=on_token)
        return self._summarize_tool_results(results)

    def _fallback_build_with_files(self, user_text: str, draft: str, on_token) -> str:
        project = self._next_project_name()
        filename, content = self._extract_code_from_direct_answer(draft, user_text)
        on_token(f"Creating {project}/{filename} with the project harness…")
        results = self._harness.run_calls([
            {"tool": "create_or_select_project", "args": {"name": project}},
            {"tool": "write_file", "args": {"project": project, "path": filename, "content": content}},
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
            if reply and BUILD_MARKER not in reply and "<html" not in reply.lower() and "```" not in reply:
                return reply.strip()
        except Exception:
            pass
        return summary

    def chat_stream(self, user_text: str, on_token) -> str:
        """Stream reply tokens, calling on_token(partial_text) as they arrive."""
        self._history.append({"role": "user", "content": user_text})
        use_tools = self._should_use_tools(user_text)

        if use_tools:
            on_token("Building real files in the projects folder…")
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
                max_tokens=2200,
                temperature=0.2,
            )
            files = self._parse_build_marker(draft)
            if files:
                summary = self._write_build_files(files, on_token)
            else:
                detected_reply = self._materialize_code_if_present(user_text, draft, on_token)
                if detected_reply:
                    on_token(detected_reply)
                    self._history.append({"role": "assistant", "content": detected_reply})
                    return detected_reply
                summary = self._fallback_build_with_files(user_text, draft, on_token)

            reply = self._explain_completed_build(user_text, summary)
            on_token(reply)
            self._history.append({"role": "assistant", "content": reply})
            return reply

        messages = [{"role": "system", "content": LLM_SYSTEM}] + self._history[-10:]

        full_reply = ""
        try:
            stream = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=300,
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

        materialized_reply = self._materialize_code_if_present(user_text, full_reply, on_token)
        if materialized_reply:
            on_token(materialized_reply)
            self._history.append({"role": "assistant", "content": materialized_reply})
            return materialized_reply

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

        combined = np.concatenate(chunks)
        sd.stop()
        sd.play(combined, samplerate=out_sr, blocksize=2048)
        while sd.get_stream().active:
            if stop_event and stop_event.is_set():
                sd.stop()
                return
            time.sleep(0.02)
        sd.wait()


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

    def _handle_settings_key(self, key: str):
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
        self.state   = s
        self._status = status

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
                    bar = make_level_bar(min(self._level * LEVEL_GAIN, 1.0))
                    live.update(build_ui(
                        self.state, self.messages,
                        level_bar=bar,
                        status_line=self._status,
                        live_text=self._live_text,
                        tts=self.tts_backend,
                        settings_open=self.settings_open,
                        settings_status=self._settings_status,
                        current_model=self._current_model_label(),
                        current_voice=self._current_voice_label(),
                        settings_view=self._settings_view,
                        settings_index=self._settings_index,
                    ))
                    time.sleep(0.1)

            ui_thread = threading.Thread(target=refresh, daemon=True)
            ui_thread.start()

            try:
                self._set_state(State.BOOT, "Loading models…")
                self._load_engines()
                self._set_state(State.IDLE)

                capture = AudioCapture()
                capture.start()
                ptt = PushToTalk(on_switch_tts=self._switch_tts, on_key=self._handle_settings_key)
                ptt.start()

                def speak_interruptible(text: str):
                    """Speak text; PTT press stops both playback and generation."""
                    stop_ev = threading.Event()
                    done    = threading.Event()
                    self._set_state(State.SPEAKING)
                    self._live_text = text or ""

                    def _play():
                        self._tts.speak(text, stop_event=stop_ev)
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
                            if self.settings_open:
                                time.sleep(0.05)
                                continue
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

                        # ── LLM (streaming tokens) ────────────────────────
                        self._live_text = ""
                        self._set_state(State.THINKING, "Generating response…")

                        def on_token(partial: str):
                            self._live_text = partial

                        reply = self._llm.chat_stream(user_text, on_token)
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
    results = harness.run_calls([
        {"tool": "create_or_select_project", "args": {"name": project}},
        {
            "tool": "write_file",
            "args": {
                "project": project,
                "path": "index.html",
                "content": (
                    "<!doctype html>\n"
                    "<html lang=\"en\">\n"
                    "<head><meta charset=\"utf-8\"><title>Builder Test</title></head>\n"
                    "<body><h1>Builder Test Works</h1></body>\n"
                    "</html>\n"
                ),
            },
        },
        {
            "tool": "write_file",
            "args": {
                "project": project,
                "path": "main.py",
                "content": "print('Builder test works')\n",
            },
        },
    ])
    ok = all(r.get("ok") for r in results)
    path = PROJECTS_DIR / project
    style = "green" if ok else "red"
    console.print(Panel(
        f"[bold {style}]{'Builder test passed' if ok else 'Builder test failed'}[/]\n\n"
        f"Created files in:\n[cyan]{path}[/]\n\n"
        "Expected:\n"
        f"  {path / 'index.html'}\n"
        f"  {path / 'main.py'}",
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
