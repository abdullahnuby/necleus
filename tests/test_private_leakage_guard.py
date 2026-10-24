from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from skillbench_v043.integrity import (
    EXPOSED_FAMILY_IDS,
    EXPOSED_TASK_FINGERPRINTS,
    EXPOSED_TASK_IDS,
    assert_fingerprint_not_disqualified,
    assert_not_disqualified,
    disqualification_report,
    official_test_family_ids,
)


ROOT = Path(__file__).resolve().parents[1]
SCANNER = ROOT / "scripts" / "check_private_leakage.py"


def _run_scanner(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCANNER), "--root", str(root)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def test_current_public_tree_has_no_private_payload() -> None:
    result = _run_scanner(ROOT)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "PASS"
    assert report["findings"] == []


def test_scanner_rejects_private_filename(tmp_path: Path) -> None:
    (tmp_path / "test_evaluator_private.jsonl").write_text('{"task_id":"x"}\n', encoding="utf-8")
    result = _run_scanner(tmp_path)
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert any("filename" in finding["reason"] for finding in report["findings"])


def test_scanner_rejects_private_key_in_nested_json(tmp_path: Path) -> None:
    (tmp_path / "public.json").write_text(
        json.dumps({"task": {"target_program": ["even", 0]}}), encoding="utf-8"
    )
    result = _run_scanner(tmp_path)
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert any("data key" in finding["reason"] for finding in report["findings"])


def test_scanner_accepts_public_payload(tmp_path: Path) -> None:
    (tmp_path / "public.jsonl").write_text(
        '{"task_id":"task-ok","support":[{"x":[1,2],"y":1}],"queries":[{"x":[3,4]}]}\n',
        encoding="utf-8",
    )
    result = _run_scanner(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["status"] == "PASS"


def test_exposed_ids_families_and_fingerprints_are_rejected() -> None:
    with pytest.raises(ValueError, match="task id"):
        assert_not_disqualified(
            task_id=next(iter(EXPOSED_TASK_IDS)),
            family_id="not-exposed",
            support=[{"x": [1, 2], "y": 1}],
            query_x=[[3, 4]],
            split="dev",
        )
    with pytest.raises(ValueError, match="family"):
        assert_not_disqualified(
            task_id="task-new",
            family_id=next(iter(EXPOSED_FAMILY_IDS)),
            support=[{"x": [1, 2], "y": 1}],
            query_x=[[3, 4]],
            split="test",
        )
    with pytest.raises(ValueError, match="fingerprint"):
        assert_fingerprint_not_disqualified(next(iter(EXPOSED_TASK_FINGERPRINTS)))


def test_official_test_family_filter_removes_exposed_families() -> None:
    candidates = sorted(EXPOSED_FAMILY_IDS) + ["family-new-a", "family-new-b"]
    assert official_test_family_ids(candidates) == ("family-new-a", "family-new-b")


def test_disqualification_report_is_public_safe() -> None:
    report = disqualification_report()
    serialized = json.dumps(report)
    assert "target_program" not in serialized
    assert "query_y" not in serialized
    assert report["raw_rows_published_again"] is False
    assert report["local_forensic_copy"].startswith(".local_private/")
