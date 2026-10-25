from __future__ import annotations

import json
from pathlib import Path

from skillbench_v043.cli import build_parser, main


def test_cli_inspect_keeps_official_test_closed(capsys) -> None:
    report = main(["inspect"])
    assert report["official_test_opened"] is False
    assert report["official_test_ready"] is False
    assert report["family_minimums_met"] is True
    assert "disqualification" in report
    assert "official_test_family_count_after_disqualification" in report
    captured = capsys.readouterr()
    assert json.loads(captured.out)["official_test_opened"] is False


def test_cli_smoke_keeps_evaluator_private_out_of_public_tree(tmp_path: Path, capsys) -> None:
    public = tmp_path / "public"
    private = tmp_path / "private"
    report = main([
        "smoke",
        "--output", str(public),
        "--private-output", str(private),
        "--seed", "430043",
    ])
    assert report["task_count"] == 3
    assert not list(public.glob("*_evaluator_private.jsonl"))
    assert len(list(private.glob("*_evaluator_private.jsonl"))) == 3
    assert (public / "stage1_smoke_report.json").exists()
    assert (private / "dev_evaluator_private.jsonl").exists()
    capsys.readouterr()


def test_cli_corpus_writes_a_manifest(tmp_path: Path, capsys) -> None:
    output = tmp_path / "corpus_manifest.json"
    report = main(["corpus", "--episodes", "8", "--output", str(output)])
    assert report["episodes"] == 8
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["actual_episodes"] == 8
    assert len(manifest["episode_sha256"]) == 64
    capsys.readouterr()


def test_cli_new_operation_writes_results(tmp_path: Path, capsys) -> None:
    output = tmp_path / "new_operation_results.json"
    report = main([
        "new-operation",
        "--seeds", "430043",
        "--output", str(output),
    ])
    assert report["official_test_opened"] is False
    assert "8" in report["by_budget"]
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["official_test_opened"] is False
    capsys.readouterr()


def test_cli_exposes_task010_commands_and_no_official_test_command() -> None:
    parser = build_parser()
    actions = [
        action for action in parser._actions
        if action.dest == "command" and action.choices is not None
    ]
    commands = set(actions[0].choices)
    assert {
        "dev-family", "validate-splits", "train-family", "diagnostics",
        "remediation", "retention-compare", "summary",
    } <= commands
    assert not any("official" in command for command in commands)


def test_cli_family_manifest_and_split_validation(tmp_path: Path, capsys) -> None:
    manifest_path = tmp_path / "dev_manifest.json"
    report = main(["dev-family", "--output", str(manifest_path)])
    assert report["official_test_opened"] is False
    assert report["task_count"] == 54
    assert report["subset_counts"] == {"calibration": 18, "evaluation": 18, "fit": 18}
    assert len(report["sha256"]) == 64

    audit = main(["validate-splits", "--manifest", str(manifest_path)])
    assert audit["official_test_opened"] is False
    assert audit["split_audit"]["family_overlap"] is False
    assert audit["split_audit"]["ood_operation_family_overlap"] is False
    capsys.readouterr()


def test_cli_summary_labels_missing_artifacts_not_run(tmp_path: Path, capsys) -> None:
    report = main(["summary", "--output", str(tmp_path)])
    assert report["official_test_opened"] is False
    assert report["official_test_ready"] is False
    assert report["artifacts"]["retention_comparison"]["status"] == "NOT RUN"
    capsys.readouterr()
