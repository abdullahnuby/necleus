from __future__ import annotations

import json
from pathlib import Path

from skillbench_v043.cli import main


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
