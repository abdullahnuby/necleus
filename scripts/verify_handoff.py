"""Validate the mandatory handoff-report structure.

Checks:
- docs/handoff/LATEST_REPORT.md exists and is non-empty.
- docs/handoff/history/ has at least one <YYYY-MM-DD>-TASK-<NNN>.md file.
- History task IDs are unique.
- LATEST and every history report contain the 11 required sections, in order,
  each with non-placeholder body text.
- LATEST mentions the newest history task ID.

Exit code 0 = PASS, 1 = failure. No project logic is modified.

Run:
    python scripts/verify_handoff.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SECTION_TITLES = (
    "هوية المهمة",
    "حالة المشروع قبل التنفيذ",
    "التغييرات المنفذة",
    "أوامر التنفيذ",
    "الاختبارات",
    "التجارب العلمية",
    "سلامة التقييم",
    "الأدلة والبصمات",
    "المشكلات والانحرافات",
    "الاستنتاج",
    "الخطوة المقترحة التالية",
)
HEADING = re.compile(r"^##\s+(\d+)\.\s*(.+?)\s*$", re.MULTILINE)
TASK_FILE = re.compile(r"^(\d{4}-\d{2}-\d{2})-TASK-(\d{3})\.md$")
PLACEHOLDERS = ("TODO", "TBD", "<TASK-ID>", "<command>", "<path>")
MIN_BODY_CHARS = 20


def _sections(text: str) -> list[tuple[int, str, str]]:
    matches = list(HEADING.finditer(text))
    out: list[tuple[int, str, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        out.append((int(match.group(1)), match.group(2).strip(), text[start:end].strip()))
    return out


def validate_report(path: Path, errors: list[str], *, label: str) -> None:
    if not path.is_file():
        errors.append(f"{label}: missing file {path}")
        return
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        errors.append(f"{label}: file is empty")
        return
    sections = _sections(text)
    numbers = [number for number, _, _ in sections]
    titles = [title for _, title, _ in sections]
    if numbers != list(range(1, 12)):
        errors.append(f"{label}: expected section numbers 1..11, found {numbers}")
    if titles and titles != list(SECTION_TITLES):
        errors.append(f"{label}: section titles/order mismatch: {titles}")
    for number, title, body in sections:
        if len(body) < MIN_BODY_CHARS:
            errors.append(f"{label}: section {number} ({title}) body is too short/empty")
        for token in PLACEHOLDERS:
            if token in body:
                errors.append(f"{label}: section {number} still contains placeholder {token!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="project root (default: cwd)")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    handoff = root / "docs" / "handoff"
    latest = handoff / "LATEST_REPORT.md"
    history = handoff / "history"
    errors: list[str] = []

    validate_report(latest, errors, label="LATEST_REPORT.md")
    if not history.is_dir():
        errors.append(f"missing history directory: {history}")
        history_files: list[Path] = []
    else:
        history_files = sorted(
            (p for p in history.iterdir() if p.is_file() and p.suffix == ".md"),
            key=lambda p: p.name,
        )
        if not history_files:
            errors.append("history directory contains no .md reports")

    ids: set[str] = set()
    for path in history_files:
        match = TASK_FILE.match(path.name)
        if not match:
            errors.append(f"history filename does not match YYYY-MM-DD-TASK-NNN.md: {path.name}")
            continue
        task_id = path.stem
        if task_id in ids:
            errors.append(f"duplicate history task id: {task_id}")
        ids.add(task_id)
        validate_report(path, errors, label=path.name)

    if history_files and latest.is_file() and not errors:
        newest = sorted(ids)[-1]
        if newest not in latest.read_text(encoding="utf-8"):
            errors.append(f"LATEST_REPORT.md does not mention the newest task id {newest}")

    if errors:
        print("HANDOFF VERIFICATION: FAIL")
        for error in errors:
            print(f" - {error}")
        return 1

    print("HANDOFF VERIFICATION: PASS")
    print(f" - latest: {latest.relative_to(root)}")
    print(f" - history reports: {len(history_files)}")
    print(f" - required sections: {len(SECTION_TITLES)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
