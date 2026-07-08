"""Shared output and LaTeX helpers for report generators."""

import json
from pathlib import Path

from core.constants import REPO_ROOT, REPORTS_DIR


def resolve_repo_path(path):
    """Resolve CLI paths relative to the repository root."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def display_path(path):
    """Return a stable repository-relative path when possible."""
    path = Path(path)
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def report_paths(report_name, output_dir=REPORTS_DIR):
    """Return the canonical JSON, CSV, and LaTeX paths for a report."""
    report_dir = resolve_repo_path(output_dir) / report_name
    report_dir.mkdir(parents=True, exist_ok=True)
    return tuple(
        report_dir / f"{report_name}.{suffix}"
        for suffix in ("json", "csv", "tex")
    )


def write_json(path, payload):
    with open(path, "w", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def write_latex(path, latex):
    with open(path, "w", encoding="utf-8", newline="\n") as file:
        file.write(latex.rstrip() + "\n")


def latex_escape(value):
    """Escape ordinary text for use inside a LaTeX table cell."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def latex_number(value, precision=None):
    """Format a number in math mode with thesis-style thousands separators."""
    if value is None:
        return "--"
    if precision is None:
        rendered = f"{int(value):,}"
    else:
        rendered = f"{float(value):,.{precision}f}"
    return f"${rendered.replace(',', '{,}')}$"
