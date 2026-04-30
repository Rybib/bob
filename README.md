# BOB

BOB is a local-first voice assistant for macOS and Windows. It can listen for a wake word, transcribe your speech, run a local GGUF language model, speak back with Kokoro TTS, and build real code projects into a local `projects/` workspace.

The goal is to make a small Jarvis-style assistant that feels simple to launch and useful once it is running: double-click the launcher for your operating system, let the guided setup do the heavy lifting, then talk to BOB.

## Highlights

- Local voice assistant with wake word and push-to-talk support.
- macOS and Windows launchers.
- First-launch setup wizard that creates the needed folders and downloads models.
- Local speech-to-text with Faster Whisper.
- Local LLM inference with `llama-cpp-python` and GGUF models.
- Kokoro TTS for lightweight local speech output.
- In-app settings for changing voices, switching AI models, and downloading assets.
- Code-building harness that writes generated code into files instead of only replying in chat.
- Project workspace where BOB can create HTML, CSS, JavaScript, Python, and other files.
- Offline-first after setup: once models are downloaded, BOB runs from local files.

## Quick Start

Download or clone the BOB folder, then double-click the launcher for your operating system.

### macOS

Double-click:

```text
Launch Bob Mac.command
```

The macOS launcher opens Terminal, checks Python, creates a local `.venv`, installs dependencies, installs `llama-cpp-python` with Apple Metal support, and starts BOB.

If Python is missing, the launcher can offer to install Homebrew and Python.

### Windows

Double-click:

```text
Launch Bob Windows.bat
```

The Windows launcher opens Command Prompt, checks Python, creates a local `.venv`, installs dependencies, installs `llama-cpp-python`, and starts BOB.

If Python is missing, the launcher can try to install Python with `winget`. Native Windows is recommended instead of WSL because microphone and speaker access is much easier for a voice assistant.

## First Launch

On first launch, BOB guides you through setup.

The setup flow can:

1. Check for Python 3.
2. Create a local `.venv`.
3. Install Python dependencies.
4. Create local `models/`, `projects/`, and `logs/` folders.
5. Let you choose which LLM to download.
6. Download the selected LLM, Kokoro, Whisper, and OpenWakeWord assets.
7. Start the BOB interface.

After setup is complete, double-clicking the launcher starts the app directly.

The launchers also install a terminal shortcut when they can:

```bash
bob
```

On macOS, `Bob` also works. On Windows, open a new Command Prompt after first launch if `bob` is not recognized yet.

## Manual Start

If you prefer Terminal or Command Prompt:

```bash
./bob
python bob.py --setup
python bob.py
```

On some macOS installs, use `python3`:

```bash
python3 bob.py --setup
python3 bob.py
```

Useful commands:

```bash
python bob.py --check-offline
python bob.py --test-builder
```

BOB uses Python 3.10-3.12 for the most stable local audio/model runtime. The Mac launcher automatically avoids Python 3.13/3.14 because several native packages used by Kokoro, Whisper, and llama.cpp can crash the whole Python process on newer unreleased/bleeding-edge runtimes.

## Controls

Main interface:

| Key | Action |
| --- | --- |
| `Space` | Hold to talk |
| `S` | Open settings |
| `Ctrl+C` | Quit |

Wake word examples:

```text
Bob
Hey Bob
Okay Bob
Hi Bob
```

Settings interface:

| Key | Action |
| --- | --- |
| `Up` / `Down` | Move selection |
| `Enter` | Open or choose an item |
| `Esc` | Go back or close settings |
| `S` | Close settings |

## In-App Settings

BOB keeps setup simple and moves day-to-day choices into the app itself. Press `S` while BOB is running to open settings.

From settings, you can:

- Change the Kokoro voice.
- Switch the active LLM.
- Download or check model assets.
- Return to the main assistant screen.

The default voice is Echo.

## Model Choices

BOB currently supports these local GGUF model choices:

| Model | Best For | Notes |
| --- | --- | --- |
| Gemma 4 E2B Q4_K_M | Fast local assistant use | Lightest option and a good default |
| Gemma 3n E4B Q4_K_M | Better general assistant quality | Larger than E2B |
| Qwen3.6 27B Q4_K_M | Coding-heavy work | Strongest coding option, but much larger and slower |

Large local models can take a long time to download and may need significant RAM and disk space. Start with the smaller Gemma option unless you know your machine can handle the larger model.

## Project Builder

When you ask BOB to build code, websites, scripts, tools, or apps, it is designed to work like a small bounded coding agent inside the `projects/` folder.

Example requests:

```text
Build me a simple HTML website that says hello world.
Make a Python calculator script.
Create a small landing page with HTML, CSS, and JavaScript.
Change the colors on the weather dashboard.
Delete the old test page from the calculator project.
```

Example output structure:

```text
projects/
  hello-world-site/
    index.html
  calculator/
    main.py
  landing-page/
    index.html
    styles.css
    script.js
```

BOB can inspect existing project folders, remember the current project, create meaningful new folders, search files, read specific line ranges, write files, append files, make exact multi-edit replacements, move groups of files into folders, rename files or folders, delete files or folders, review website output, and run project commands with the working directory locked to `projects/`.

After file-changing tools run, BOB automatically checks the changed project folder so the model can see what actually exists before it gives the final response. For website projects, BOB also reviews the HTML/CSS and fills in missing stylesheet files so generated pages look like real responsive websites instead of bare HTML.

For follow-up requests like “fix it” or “make that project better,” BOB is designed to stay anchored to the existing project it just worked on instead of drifting into a new folder.

During long sessions, BOB automatically compacts older conversation and tool context into a short memory summary before the model hits its context limit. When that happens, the live status will say it is compacting context, then the agent continues working with the important details preserved.

BOB uses a dedicated tool harness for coding work plus a fallback detector for raw code output. If the model replies with HTML or Python directly, BOB attempts to capture that code and save it into an appropriate project file.

## Repository Layout

Files meant to be shared on GitHub:

```text
Launch Bob Mac.command
Launch Bob Windows.bat
bob.py
requirements.txt
README.md
.gitignore
```

Generated locally by the launcher or app:

```text
.venv/
models/
projects/
logs/
bob_config.json
__pycache__/
.DS_Store
```

Generated files are ignored by Git so the repository stays clean and lightweight.

## Requirements

BOB expects:

- macOS or Windows.
- Python 3.10 or newer.
- A microphone.
- Speakers or headphones.
- Enough disk space for the selected model downloads.
- Internet access during first setup.

After setup, BOB is intended to run locally from downloaded assets.

## Troubleshooting

Setup logs:

```text
logs/launcher.log
logs/launcher-windows.log
```

Common fixes:

- Re-run the launcher; the setup is designed to resume.
- On macOS, install Xcode Command Line Tools if prompted.
- On Windows, install Python from python.org if `winget` is unavailable.
- On Windows, install Microsoft C++ Build Tools if `llama-cpp-python` cannot build or install.
- Try the smaller Gemma model first if a large model is too slow.
- Run `python bob.py --setup` to repeat the setup wizard.
- Run `python bob.py --check-offline` after setup to verify local model paths.

If wake word detection feels too quiet, check your system microphone input level and make sure Terminal or Command Prompt has microphone permission.

## Privacy

BOB is designed to be local-first. It needs internet access to download models during setup, but after the required assets are present, speech transcription, language model inference, TTS, and project generation run from local files.

## Notes

This is an experimental assistant project. It is meant to be easy to launch, easy to inspect, and easy to extend. The generated `projects/` folder is where BOB does its building work, while the repository itself stays focused on the launchers, app code, and documentation.
