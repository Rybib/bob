"""
Workspace utility functions for the BOB agent.

All the helper logic for project discovery, mutation detection, and result
summarisation that was previously embedded in LLMEngine.  Extracted here so
the agent loop stays clean and these utilities can be tested independently.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional


# ── request classification ───────────────────────────────────────────────────

def requires_inspection(user_text: str) -> bool:
    """Return True if the request implies reading existing project files first."""
    return bool(re.search(
        r"\b(fix|repair|improve|update|change|modify|rename|delete|remove|debug"
        r"|error|issue|reference|look at|inspect|existing|current|that|it"
        r"|add|include|insert|enhance|upgrade|more|better|gradient|animation"
        r"|effect|content|information|section|layout|style|theme|feature)\b",
        user_text.lower(),
    ))


def requires_mutation(user_text: str) -> bool:
    """Return True if the request demands real file changes."""
    return bool(re.search(
        r"\b(fix|repair|improve|update|change|modify|rename|delete|remove"
        r"|build|create|make|generate|design|write|style|restyle|refactor"
        r"|add|include|insert|put|give|show|display|append|attach|enhance"
        r"|upgrade|polish|beautify|animate|gradient|animation|transition"
        r"|color|colours?|theme|layout|font|icon|image|hero|card|button"
        r"|nav|header|footer|section|feature|effect|element|detail|content"
        r"|information|text|copy|responsive|mobile|dark|light|better|more"
        r"|cool|nice|pretty|modern|clean|fresh|slick|bold|vibrant)\b",
        user_text.lower(),
    ))


# ── project name utilities ────────────────────────────────────────────────────

def _slug(text: str) -> str:
    """Convert arbitrary text to a filesystem-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60] if slug else "project"


def meaningful_project_name(user_text: str, workspace_root: Path) -> str:
    """
    Derive a human-readable project folder name from the user's request.
    Falls back to project-N if nothing useful can be inferred.
    """
    # Named/quoted candidate
    quoted = re.findall(r'"([^"]{3,60})"|\'([^\']{3,60})\'|`([^`]{3,60})`', user_text)
    for groups in quoted:
        candidate = next((g for g in groups if g), "")
        if candidate:
            return _slug(candidate)

    # Strip boilerplate words, then slug what remains
    text = re.sub(r"['\"`]", "", user_text.lower())
    text = re.sub(
        r"\b(?:please|can you|could you|would you|build|create|make|generate"
        r"|design|write|code|an?|the|for me|simple|basic|new|project|app"
        r"|website|webpage|page|script|python|html|css|javascript|js)\b",
        " ",
        text,
    )
    words = re.findall(r"[a-z0-9]+", text)
    stop = {
        "me", "my", "us", "i", "you", "your", "its", "a", "an",
        "that", "just", "says", "say", "with", "and", "or", "to", "in", "of",
        "inside", "folder", "file", "files", "want", "need", "called", "named",
        "some", "this", "these", "those", "there", "here", "very", "really",
    }
    words = [w for w in words if w not in stop]
    name = "-".join(words[:5]).strip("-")
    if name and len(name) >= 3:
        return name[:60]
    return _next_project_name(workspace_root)


def _next_project_name(workspace_root: Path) -> str:
    try:
        existing = {p.name for p in workspace_root.iterdir() if p.is_dir()}
    except Exception:
        existing = set()
    n = 1
    while f"project-{n}" in existing:
        n += 1
    return f"project-{n}"


# ── result parsing ────────────────────────────────────────────────────────────

_MUTATION_TOOLS = frozenset({
    "write_project_file", "append_project_file", "replace_in_project_file",
    "edit_project_file", "write_file", "append_file", "replace_in_file",
    "rename_project_path", "move_project_paths", "delete_project_path",
    "run_project_command", "make_project_directory",
})

_READ_TOOLS = frozenset({
    "list_project_tree", "find_project_files", "grep_project",
    "read_project_file", "read_file", "list_files",
})


def has_real_mutations(results: List[dict]) -> bool:
    """Return True if any tool result represents an actual file/dir change."""
    return any(
        r.get("ok") and r.get("tool") in _MUTATION_TOOLS
        for r in results
    )


def has_inspection(results: List[dict]) -> bool:
    """Return True if any result shows the model read project files."""
    return any(
        r.get("ok") and r.get("tool") in _READ_TOOLS
        for r in results
    )


def existing_projects_from_results(results: List[dict]) -> List[str]:
    """Parse list_project_tree results and return top-level project folder names."""
    projects: List[str] = []
    for entry in results:
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


def infer_existing_project(
    user_text: str,
    projects: List[str],
    active_project: Optional[str],
) -> Optional[str]:
    """
    Try to figure out which existing project the user is talking about.
    Returns the project name or None.
    """
    if not projects:
        return None
    lower = user_text.lower()
    for project in projects:
        readable = project.replace("-", " ").lower()
        if project.lower() in lower or readable in lower:
            return project
    if (
        active_project in projects
        and re.search(
            r"\b(it|that|this|current|existing|same|there)\b", lower
        )
    ):
        return active_project
    if len(projects) == 1 and requires_inspection(user_text):
        return projects[0]
    return active_project if active_project in projects else None


def changed_project_roots(results: List[dict]) -> List[str]:
    """Return the set of top-level project folders touched by the given results."""
    roots: List[str] = []
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
    return sorted(dict.fromkeys(r for r in roots if r and r != "."))


def summarize_tool_results(results: List[dict], projects_dir: Path) -> str:
    """
    Produce a short human-readable summary of what the tool calls accomplished.
    Used as the agent's final spoken reply when no clean text answer is available.
    """
    changed_files: List[str] = []
    projects: set = set()

    for entry in results:
        result = entry.get("result") if entry.get("ok") else None
        if not isinstance(result, dict):
            continue

        project = result.get("project")
        if project:
            projects.add(str(project))

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
        if entry.get("tool") not in _MUTATION_TOOLS:
            continue

        try:
            p = Path(str(path))
            if p.is_absolute() and projects_dir.resolve() in p.resolve().parents:
                rel = p.resolve().relative_to(projects_dir.resolve())
                parts = rel.parts
                if parts:
                    projects.add(parts[0])
                changed_files.append(str(rel))
            else:
                clean_rel = str(path).replace("\\", "/").strip("/")
                first_part = clean_rel.split("/", 1)[0]
                if first_part and first_part not in {".", str(projects_dir)}:
                    projects.add(first_part)
                changed_files.append(str(path))
        except Exception:
            changed_files.append(str(path))

    changed_files = sorted(dict.fromkeys(changed_files))
    sorted_projects = sorted(projects)

    if sorted_projects:
        location = (
            f"projects/{sorted_projects[0]}"
            if len(sorted_projects) == 1
            else "the projects folder"
        )
    else:
        location = "the projects folder"

    if changed_files:
        important = ", ".join(changed_files[:5])
        extra = (
            ""
            if len(changed_files) <= 5
            else f", plus {len(changed_files) - 5} more"
        )
        return (
            f"Done. I finished it in {location}. "
            f"I changed {len(changed_files)} "
            f"{'path' if len(changed_files) == 1 else 'paths'}: "
            f"{important}{extra}. "
            "The files are ready in your projects folder."
        )

    return (
        "Done. I used the project harness and finished the requested work "
        "in your projects folder."
    )
