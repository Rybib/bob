#!/bin/bash

cd "$(dirname "$0")" || exit 1

APP_NAME="BOB"
APP_VERSION="project-builder-v5-kokoro"
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/launcher.log"
VENV_DIR=".venv"
PY="$VENV_DIR/bin/python"
MARKER="$VENV_DIR/.bob_deps_installed"

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
  export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
}

ensure_python() {
  refresh_path
  if command -v python3 >/dev/null 2>&1; then
    echo "    Found $(python3 --version 2>&1)"
    return 0
  fi

  if [[ "$OSTYPE" != "darwin"* ]]; then
    fail "Python 3 is not installed. Install Python 3, then run this launcher again."
  fi

  echo "    Python 3 is missing."
  echo "    BOB can install Homebrew and Python automatically."
  echo "    This may ask for your Mac password or Xcode Command Line Tools."
  echo ""
  read -r -p "Press Enter to install Python automatically, or press Ctrl+C to stop..."

  BREW="$(find_brew)"
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

  echo "    Installing Python with Homebrew..."
  run_quiet "$BREW" install python || fail "Python installation failed."
  refresh_path

  if ! command -v python3 >/dev/null 2>&1; then
    fail "Python installation finished, but python3 was not found on PATH."
  fi

  echo "    Found $(python3 --version 2>&1)"
}

install_dependencies() {
  if [ ! -x "$PY" ]; then
    echo "    Creating local Python environment..."
    run_quiet python3 -m venv "$VENV_DIR" || fail "Could not create the local Python environment."
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

header

step "Checking Python"
ensure_python
echo ""

step "Preparing Bob"
install_dependencies
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
