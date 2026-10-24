"""Permanent disqualification registry for exposed v0.43 smoke tasks.

The three stage-1 smoke evaluator files were once tracked in the public
repository. Their task IDs, family IDs, and support/query fingerprints are
therefore permanently disqualified from any official v0.43 evaluation.

Raw support rows and query labels are deliberately not repeated here. The
local forensic copies live under the Git-ignored ``.local_private/`` tree.
"""
from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Iterable, Mapping

EXPOSED_TASK_IDS: frozenset[str] = frozenset({
    "task-84d6edfe96ffd28e",
    "task-92940f0d430a70f6",
    "task-ff56a93a34464a02",
})

EXPOSED_FAMILY_IDS: frozenset[str] = frozenset({
    "001805d0b07b2fce",
    "00d96a49a204ff33",
    "00e3601b352f025e",
})

EXPOSED_TASK_FINGERPRINTS: frozenset[str] = frozenset({
    "27d9d364f5e410eb1ebeab2e4aca9e9297e2b135760068ba7aad0aa3dca9302f",
    "08dd63b7dcb69865ec828987295739983fda036afc28ccf033a1b7b0dd087881",
    "cc029704892116cc751f4546ac64d4a8ab43abcdda0b21057b93e207109d5045",
})

EXPOSED_SOURCE_FILES: Mapping[str, str] = {
    "dev_evaluator_private.jsonl": "210fd10dac069bb47cf56edd7b6adb544892a390b3a8efaceb31bb6ebd91b307",
    "test_evaluator_private.jsonl": "86444abd0e05107e825b63ba9d5b028716ca2b8d8777f3762c2e3bed08d62c4a",
    "train_evaluator_private.jsonl": "26a8b05fb206b21cd5565062105a42f65cc82feeec29e04bd6aec9d3a62051ce",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def task_fingerprint(support: Iterable[Mapping[str, Any]],
                     query_x: Iterable[Iterable[int]]) -> str:
    """Stable fingerprint of the exposed task's support/query input rows."""
    payload = {
        "support": list(support),
        "query_x": [list(x) for x in query_x],
    }
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def assert_fingerprint_not_disqualified(fingerprint: str) -> None:
    if fingerprint in EXPOSED_TASK_FINGERPRINTS:
        raise ValueError(f"disqualified exposed smoke task fingerprint: {fingerprint}")


def assert_not_disqualified(*, task_id: str, family_id: str,
                            support: Iterable[Mapping[str, Any]],
                            query_x: Iterable[Iterable[int]],
                            split: str | None = None) -> None:
    """Reject an exposed smoke task or family from official evaluation.

    Family-level exclusion is enforced for the ``test`` split. Exact task IDs
    and fingerprints are rejected in every split so accidental reuse is caught
    during development as well.
    """
    if task_id in EXPOSED_TASK_IDS:
        raise ValueError(f"disqualified exposed smoke task id: {task_id}")
    fingerprint = task_fingerprint(support, query_x)
    assert_fingerprint_not_disqualified(fingerprint)
    if split == "test" and family_id in EXPOSED_FAMILY_IDS:
        raise ValueError(f"disqualified exposed smoke family in test split: {family_id}")


def official_test_family_ids(all_test_family_ids: Iterable[str]) -> tuple[str, ...]:
    """Return test families that remain eligible for official evaluation."""
    return tuple(sorted(set(all_test_family_ids) - EXPOSED_FAMILY_IDS))


def disqualification_report() -> dict[str, Any]:
    return {
        "exposed_task_ids": sorted(EXPOSED_TASK_IDS),
        "exposed_family_ids": sorted(EXPOSED_FAMILY_IDS),
        "exposed_task_fingerprints": sorted(EXPOSED_TASK_FINGERPRINTS),
        "exposed_source_files": dict(sorted(EXPOSED_SOURCE_FILES.items())),
        "raw_rows_published_again": False,
        "local_forensic_copy": ".local_private/v043_stage1_smoke/",
        "official_test_allowed": False,
        "reason": "support/query rows and evaluator metadata were publicly tracked",
    }
