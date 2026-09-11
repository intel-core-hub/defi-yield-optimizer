import pandas as pd
import pytest

from phase_b import apply_phase_b_gate, load_manifest, phase_b_status_map

MANIFEST = {
    "pools": [
        {"pool_id": "approved-pool", "phase_b_status": "approved"},
        {"pool_id": "pending-pool", "phase_b_status": "pending"},
        {"pool_id": "rejected-pool", "phase_b_status": "rejected"},
    ]
}


def _ranked(pool_ids):
    return pd.DataFrame({"pool_id": pool_ids, "score": [10.0] * len(pool_ids)})


def test_phase_b_status_map():
    assert phase_b_status_map(MANIFEST) == {
        "approved-pool": "approved",
        "pending-pool": "pending",
        "rejected-pool": "rejected",
    }


def test_apply_phase_b_gate_keeps_only_approved():
    ranked = _ranked(["approved-pool", "pending-pool", "rejected-pool"])
    gated = apply_phase_b_gate(ranked, MANIFEST)

    approved_row = gated[gated["pool_id"] == "approved-pool"].iloc[0]
    pending_row = gated[gated["pool_id"] == "pending-pool"].iloc[0]
    rejected_row = gated[gated["pool_id"] == "rejected-pool"].iloc[0]

    assert approved_row["score"] == 10.0
    assert pd.isna(pending_row["score"])
    assert pd.isna(rejected_row["score"])


def test_apply_phase_b_gate_excludes_pools_not_in_manifest():
    ranked = _ranked(["approved-pool", "unknown-pool"])
    gated = apply_phase_b_gate(ranked, MANIFEST)

    unknown_row = gated[gated["pool_id"] == "unknown-pool"].iloc[0]
    assert unknown_row["phase_b_status"] == "not_reviewed"
    assert pd.isna(unknown_row["score"])


def test_load_manifest_rejects_invalid_status(tmp_path):
    bad_manifest = tmp_path / "bad.yaml"
    bad_manifest.write_text(
        "pools:\n  - pool_id: x\n    phase_b_status: maybe\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="phase_b_status"):
        load_manifest(bad_manifest)


def test_load_manifest_reads_real_file():
    manifest = load_manifest("docs/phase-b/2026-09-11.yaml")
    assert len(manifest["pools"]) == 7
    assert all(p["phase_b_status"] == "pending" for p in manifest["pools"])
