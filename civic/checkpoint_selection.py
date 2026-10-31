import hashlib
import math
from pathlib import Path


def _validate(candidate, directions, verify_weights):
    epoch = candidate.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
        raise ValueError("Each candidate must identify a trained, positive integer epoch")
    digest = candidate.get("checkpoint_sha256", "")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Each candidate needs its actual checkpoint SHA256")
    metrics = candidate.get("metrics", {})
    values = {}
    for name, direction in directions.items():
        if direction not in ("max", "min"):
            raise ValueError("Metric directions must be max or min")
        value = metrics.get(name)
        if isinstance(value, dict):
            if value.get("epoch") != epoch or value.get("checkpoint_sha256") != digest:
                raise ValueError("A table cannot combine metrics from different weights or epochs")
            value = value.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Every requested metric must have a finite measured value")
        values[name] = float(value)
    if verify_weights:
        path = Path(candidate["checkpoint_path"])
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(stream.read()).hexdigest()
        if actual != digest:
            raise ValueError("Checkpoint bytes do not match the recorded SHA256")
    return values


def select_table_checkpoint(candidates, primary, directions, selection_split="validation",
                            verify_weights=False):
    candidates = list(candidates)
    if not candidates or primary not in directions:
        raise ValueError("Provide measured candidates and an explicit primary metric")
    if selection_split not in ("validation", "test"):
        raise ValueError("Selection split must be validation or test")
    values = [_validate(c, directions, verify_weights) for c in candidates]
    signs = {name: 1 if direction == "max" else -1 for name, direction in directions.items()}

    def rank(index):
        primary_score = signs[primary] * values[index][primary]
        other_ranks = []
        for name in directions:
            if name == primary:
                continue
            score = signs[name] * values[index][name]
            other_ranks.append(sum(score >= signs[name] * row[name] for row in values) / len(values))
        secondary = sum(other_ranks) / max(len(other_ranks), 1)
        return primary_score, secondary, -candidates[index]["epoch"], -index

    selected = max(range(len(candidates)), key=rank)
    return {"selected": candidates[selected], "selected_index": selected,
            "primary_metric": primary, "directions": dict(directions),
            "selection_split": selection_split, "exploratory_test_selection": selection_split == "test",
            "single_checkpoint_for_all_metrics": True,
            "tie_break": "mean percentile rank of other metrics, earliest epoch, input order"}
