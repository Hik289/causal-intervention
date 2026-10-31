import hashlib

import pytest

from civic.checkpoint_selection import select_table_checkpoint


def candidate(epoch, auc, f1, fp):
    return {"epoch": epoch, "checkpoint_sha256": str(epoch % 10) * 64,
            "metrics": {"auroc": auc, "f1": f1, "fp": fp}}


def test_table_selection_does_not_splice_independent_metric_winners():
    rows = [candidate(2, .7, .9, .1), candidate(5, .8, .5, .3)]
    selected = select_table_checkpoint(rows, "auroc", {"auroc": "max", "f1": "max", "fp": "min"}, "test")
    assert selected["selected"] == rows[1]
    assert selected["selected"]["metrics"]["f1"] == .5
    assert selected["exploratory_test_selection"]


def test_primary_ties_respect_secondary_metric_direction():
    rows = [candidate(2, .8, .5, .4), candidate(5, .8, .7, .1)]
    selected = select_table_checkpoint(rows, "auroc", {"auroc": "max", "f1": "max", "fp": "min"})
    assert selected["selected_index"] == 1
    assert not selected["exploratory_test_selection"]


def test_mixed_epoch_cells_and_untrained_candidates_are_rejected():
    row = candidate(3, .8, .7, .2)
    row["metrics"]["f1"] = {"value": .9, "epoch": 8, "checkpoint_sha256": "8" * 64}
    with pytest.raises(ValueError, match="different weights or epochs"):
        select_table_checkpoint([row], "auroc", {"auroc": "max", "f1": "max"})
    with pytest.raises(ValueError, match="trained"):
        select_table_checkpoint([candidate(0, .5, .5, .5)], "auroc", {"auroc": "max"})


def test_weight_verification_rejects_changed_bytes(tmp_path):
    path = tmp_path / "state.pt"
    path.write_bytes(b"retained model bytes")
    row = candidate(3, .8, .7, .2)
    row.update(checkpoint_path=str(path), checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert select_table_checkpoint([row], "auroc", {"auroc": "max"}, verify_weights=True)
    path.write_bytes(b"different model bytes")
    with pytest.raises(ValueError, match="bytes do not match"):
        select_table_checkpoint([row], "auroc", {"auroc": "max"}, verify_weights=True)
