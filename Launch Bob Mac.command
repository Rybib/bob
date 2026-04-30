#!/bin/bash

cd "$(dirname "$0")" || exit 1

APP_NAME="BOB"
APP_VERSION="project-builder-v5-kokoro"
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/launcher.log"
VENV_DIR=".venv"
PY="$VENV_DIR/bin/python"
MARKER="$VENV_DIR/.bob_deps_installed"
PYTHON_CMD=""
PYTHON_MIN_MINOR="3.10"
PYTHON_MAX_MINOR="3.13"

mkdir -p "$LOG_DIR" models projects
: > "$LOG_FILE"

step=0

header() {
  clear
  echo "╔══════════════════════════════════════════════════════════╗"
  echo "║                       Launch BOB                         ║"
  echo "║             Local voice assistant installer              ║"
  echo "╚══════════════════════════════════════════════════════════╝"
  echo ""
  echo "Version: $APP_VERSION"
  echo "Folder:  $(pwd)"
  echo "Log:     $(pwd)/$LOG_FILE"
  echo ""
}

step() {
  step=$((step + 1))
  echo "[$step] $1"
}

pause_close() {
  echo ""
  read -r -p "Press Enter to close..."
}

fail() {
  echo ""
  echo "BOB could not finish this step:"
  echo "  $1"
  echo ""
  echo "The detailed log is here:"
  echo "  $(pwd)/$LOG_FILE"
  echo ""
  echo "After fixing the issue, double-click Launch Bob Mac.command again."
  pause_close
  exit 1
}

run_quiet() {
  echo "+ $*" >> "$LOG_FILE"
  "$@" >> "$LOG_FILE" 2>&1
}

find_brew() {
  if command -v brew >/dev/null 2>&1; then
    command -v brew
  elif [ -x "/opt/homebrew/bin/brew" ]; then
    echo "/opt/homebrew/bin/brew"
  elif [ -x "/usr/local/bin/brew" ]; then
    echo "/usr/local/bin/brew"
  fi
}

refresh_path() {
  export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
}

ensure_python() {
  refresh_path
  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)
PY
      then
        PYTHON_CMD="$(command -v "$candidate")"
        echo "    Found stable $($PYTHON_CMD --version 2>&1)"
        return 0
      fi
    fi
  done

  if command -v python3 >/dev/null 2>&1; then
    echo "    Found $(python3 --version 2>&1), but BOB uses Python 3.10-3.12 for native audio/model stability."
  fi

  BREW="$(find_brew)"
  if [ -n "$BREW" ] && [ -x "$BREW" ]; then
    echo "    Installing/checking Python 3.12 with Homebrew..."
    run_quiet "$BREW" install python@3.12 || fail "Python 3.12 installation failed."
    refresh_path
    if command -v python3.12 >/dev/null 2>&1; then
      PYTHON_CMD="$(command -v python3.12)"
      echo "    Found stable $($PYTHON_CMD --version 2>&1)"
      return 0
    fi
  fi

  if command -v python3 >/dev/null 2>&1; then
    if python3 - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)
PY
    then
      PYTHON_CMD="$(command -v python3)"
      echo "    Found stable $($PYTHON_CMD --version 2>&1)"
      return 0
    fi
  fi

  if command -v python3 >/dev/null 2>&1; then
    echo "    Python is installed, but not in BOB's stable range ($PYTHON_MIN_MINOR to before $PYTHON_MAX_MINOR)."
  else
    echo "    Python 3 is missing."
  fi

  if [[ "$OSTYPE" != "darwin"* ]]; then
    fail "Python 3.10, 3.11, or 3.12 is not installed. Install one of those versions, then run this launcher again."
  fi

  echo "    BOB can install Homebrew and Python 3.12 automatically."
  echo "    This may ask for your Mac password or Xcode Command Line Tools."
  echo ""
  read -r -p "Press Enter to install Python 3.12 automatically, or press Ctrl+C to stop..."

  if [ -z "$BREW" ]; then
    echo "    Installing Homebrew..."
    echo "+ official Homebrew installer" >> "$LOG_FILE"
    NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" >> "$LOG_FILE" 2>&1 || fail "Homebrew installation failed."
    refresh_path
    BREW="$(find_brew)"
  fi

  if [ -z "$BREW" ]; then
    fail "Homebrew installed, but the brew command was not found."
  fi

  echo "    Installing Python 3.12 with Homebrew..."
  run_quiet "$BREW" install python@3.12 || fail "Python 3.12 installation failed."
  refresh_path

  if command -v python3.12 >/dev/null 2>&1; then
    PYTHON_CMD="$(command -v python3.12)"
    echo "    Found stable $($PYTHON_CMD --version 2>&1)"
    return 0
  fi

  fail "Python 3.12 installation finished, but python3.12 was not found on PATH."
}

install_dependencies() {
  if [ -z "$PYTHON_CMD" ]; then
    fail "No stable Python command was selected."
  fi

  if [ -x "$PY" ]; then
    if ! "$PY" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)
PY
    then
      backup="$VENV_DIR.old-$(date +%Y%m%d-%H%M%S)"
      echo "    Existing environment uses an unstable Python for BOB. Moving it to $backup."
      mv "$VENV_DIR" "$backup" >> "$LOG_FILE" 2>&1 || fail "Could not move the old Python environment."
    fi
  fi

  if [ ! -x "$PY" ]; then
    echo "    Creating local Python environment..."
    run_quiet "$PYTHON_CMD" -m venv "$VENV_DIR" || fail "Could not create the local Python environment."
  else
    echo "    Local Python environment already exists."
  fi

  if [ ! -f "$MARKER" ] || [ "requirements.txt" -nt "$MARKER" ] || [ "bob.py" -nt "$MARKER" ]; then
    echo "    Installing/updating packages. This can take several minutes."
    echo "    Detailed install output is hidden in logs/launcher.log."

    run_quiet "$PY" -m pip install --upgrade pip wheel || fail "Could not upgrade pip."

    if [[ "$OSTYPE" == "darwin"* ]]; then
      echo "    Installing llama-cpp-python with Apple Metal support..."
      echo "+ CMAKE_ARGS=-DGGML_METAL=on $PY -m pip install llama-cpp-python --no-cache-dir" >> "$LOG_FILE"
      CMAKE_ARGS="-DGGML_METAL=on" "$PY" -m pip install llama-cpp-python --no-cache-dir >> "$LOG_FILE" 2>&1 || fail "Could not install llama-cpp-python."
    else
      run_quiet "$PY" -m pip install llama-cpp-python --no-cache-dir || fail "Could not install llama-cpp-python."
    fi

    run_quiet "$PY" -m pip install -r requirements.txt || fail "Could not install BOB's Python packages."
    touch "$MARKER"
    echo "    Dependencies are ready."
  else
    echo "    Dependencies are already ready."
  fi
}

install_terminal_command() {
  local target_dir="$HOME/.local/bin"
  local launcher="$(pwd)/bob"

  if [ ! -x "$launcher" ]; then
    chmod +x "$launcher" >> "$LOG_FILE" 2>&1 || true
  fi

  if [ ! -d "$target_dir" ]; then
    echo "    Creating user terminal command folder..."
    mkdir -p "$target_dir" >> "$LOG_FILE" 2>&1 || {
      echo "    Could not create $target_dir. You can still use ./bob in this folder."
      return 0
    }
  fi

  ln -sf "$launcher" "$target_dir/bob" >> "$LOG_FILE" 2>&1 || {
    echo "    Could not install terminal command. You can still use ./bob in this folder."
    return 0
  }
  ln -sf "$launcher" "$target_dir/Bob" >> "$LOG_FILE" 2>&1 || true

  if ! grep -qs 'HOME/.local/bin' "$HOME/.zshrc" 2>/dev/null; then
    {
      echo ""
      echo "# BOB terminal command"
      echo 'export PATH="$HOME/.local/bin:$PATH"'
    } >> "$HOME/.zshrc" 2>> "$LOG_FILE" || true
  fi

  export PATH="$HOME/.local/bin:$PATH"
  echo "    Terminal command ready: type 'bob' or 'Bob' from any new Terminal."
}

header

step "Checking Python"
ensure_python
echo ""

step "Preparing Bob"
install_dependencies
echo ""

step "Installing terminal command"
install_terminal_command
echo ""

step "Checking models and setup"
echo "    If this is the first launch, BOB will open the setup wizard."
echo "    The wizard downloads the selected LLM, Kokoro, Whisper, and OpenWakeWord."
echo ""

step "Starting BOB"
echo "    When setup is complete, the assistant opens in this Terminal."
echo ""
"$PY" bob.py || fail "BOB exited with an error."

echo ""
echo "BOB closed."
pause_close
