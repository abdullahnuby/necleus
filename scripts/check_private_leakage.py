"""Reject evaluator-private payloads from public data and build artifacts.

Usage:
    python scripts/check_private_leakage.py
    python scripts/check_private_leakage.py --root .

The guard scans JSON/JSONL/CSV artifacts outside the Git-ignored
``.local_private/`` tree. It rejects evaluator-private filenames and data keys
that contain target programs, query labels, or private audit metadata.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

FORBIDDEN_KEYS = frozenset({
    "target_program",
    "target_program_private",
    "query_y",
    "query_y_private",
    "classification_private",
    "audit_private",
})
FORBIDDEN_NAME_SUFFIXES = (
    "_evaluator_private.jsonl",
    "_evaluator_private.json",
)
SKIP_DIRS = frozenset({
    ".git", ".venv", "venv", "env", ".local_private", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".matplotlib", ".tmp",
    ".pip-cache",
})
DATA_SUFFIXES = frozenset({".json", ".jsonl", ".csv"})


def _iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        yield path


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        if FORBIDDEN_KEYS & set(value):
            return True
        return any(_contains_forbidden_key(v) for v in value.values())
    if isinstance(value, list):
        return any(_contains_forbidden_key(v) for v in value)
    return False


def _scan_json(path: Path) -> bool:
    try:
        return _contains_forbidden_key(json.loads(path.read_text(encoding="utf-8-sig")))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False


def _scan_jsonl(path: Path) -> bool:
    try:
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip() and _contains_forbidden_key(json.loads(line)):
                return True
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return False


def _scan_csv(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames and FORBIDDEN_KEYS & set(reader.fieldnames):
                return True
    except (UnicodeDecodeError, csv.Error):
        return False
    return False


def scan_repository(root: Path) -> dict[str, Any]:
    findings: list[dict[str, str]] = []
    scanned = 0
    for path in _iter_files(root):
        relative = path.relative_to(root).as_posix()
        if path.name.endswith(FORBIDDEN_NAME_SUFFIXES):
            findings.append({"path": relative, "reason": "evaluator-private filename"})
            continue
        if path.suffix.lower() not in DATA_SUFFIXES:
            continue
        scanned += 1
        if path.suffix.lower() == ".json":
            hit = _scan_json(path)
        elif path.suffix.lower() == ".jsonl":
            hit = _scan_jsonl(path)
        else:
            hit = _scan_csv(path)
        if hit:
            findings.append({"path": relative, "reason": "evaluator-private data key"})
    return {
        "status": "FAIL" if findings else "PASS",
        "scanned_data_files": scanned,
        "findings": findings,
        "local_private_tree": ".local_private/ (ignored and not scanned)",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    report = scan_repository(args.root.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
