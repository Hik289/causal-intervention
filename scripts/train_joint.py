import argparse
import json
import random
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             precision_recall_fscore_support, roc_auc_score)

from civic.models import CIVER
from civic.objectives import LossWeights, joint_objectives


FEATURE_KEYS = ("original", "critique", "candidates", "unrelated", "transformed")
PAIRED_KEYS = ("pair_original", "pair_candidates", "pair_unrelated", "pair_transformed")


def identical_states(original, states, candidate_shape):
    if states.shape != candidate_shape:
        states = np.broadcast_to(states[:, None, ...], candidate_shape)
    if original.shape[1:] != states.shape[2:]:
        return np.zeros(states.shape[:2], dtype=bool)
    return np.all(states == original[:, None, ...], axis=tuple(range(2, states.ndim)))


class JointDataset:
    def __init__(self, path, require_transformation=True, allow_unpaired=False):
        with np.load(path, allow_pickle=False) as data:
            required = (*FEATURE_KEYS, "labels", "target_index", "group_id", "split")
            missing = sorted(set(required) - set(data.files))
            if missing:
                raise ValueError(f"Missing dataset arrays: {', '.join(missing)}")
            arrays = {key: data[key] for key in data.files}
        n = len(arrays["labels"])
        if not n or any(len(arrays[key]) != n for key in required):
            raise ValueError("All required arrays must have the same nonzero first dimension")
        original, critique, candidates = (arrays[key] for key in FEATURE_KEYS[:3])
        if original.ndim not in (2, 3) or critique.ndim not in (2, 3) or candidates.ndim not in (3, 4):
            raise ValueError("original/critique require [N,D] or [N,L,D]; candidates require [N,K,Dc] or [N,K,L,Dc]")
        k = candidates.shape[1]
        if not k or candidates.shape[-1] != original.shape[-1]:
            raise ValueError("Candidates must be nonempty and use the original code feature dimension")
        for key in ("unrelated", "transformed"):
            shared_shape = (n, *candidates.shape[2:])
            if arrays[key].shape not in (shared_shape, candidates.shape):
                raise ValueError(f"{key} must contain a shared control state or one control state per candidate")
        paired = [key in arrays for key in PAIRED_KEYS]
        if any(paired) and not all(paired):
            raise ValueError("Paired encodings require separate pair_* arrays for all four states")
        if not any(paired) and not allow_unpaired:
            raise ValueError("Full joint training requires all pair_* encodings; use --allow-unpaired-features only for a feature ablation")
        self.paired_dim = None
        if all(paired):
            if arrays["pair_original"].ndim not in (2, 3) or arrays["pair_original"].shape[0] != n:
                raise ValueError("pair_original must have shape [N,Dp] or [N,L,Dp]")
            paired_candidates = arrays["pair_candidates"]
            if paired_candidates.ndim not in (3, 4) or paired_candidates.shape[:2] != (n, k):
                raise ValueError("pair_candidates must have shape [N,K,Dp] or [N,K,L,Dp]")
            self.paired_dim = arrays["pair_original"].shape[-1]
            if paired_candidates.shape[-1] != self.paired_dim:
                raise ValueError("Paired features must share a feature dimension")
            for key in PAIRED_KEYS[1:]:
                allowed = (paired_candidates.shape,)
                if key != "pair_candidates":
                    allowed += ((n, *paired_candidates.shape[2:]),)
                if arrays[key].shape not in allowed:
                    raise ValueError(f"{key} has an incompatible paired-feature shape")
        feature_keys = FEATURE_KEYS + (PAIRED_KEYS if all(paired) else ())
        for key in feature_keys:
            if not np.issubdtype(arrays[key].dtype, np.number) or not np.isfinite(arrays[key]).all():
                raise ValueError(f"{key} must contain finite numerical frozen representations")
            if any(size == 0 for size in arrays[key].shape):
                raise ValueError(f"{key} cannot contain an empty feature or token dimension")
        labels = arrays["labels"]
        target = arrays["target_index"]
        if labels.shape != (n,) or not np.isin(labels, [0, 1]).all():
            raise ValueError("labels must be a binary array of shape [N]")
        if target.shape != (n,) or not np.issubdtype(target.dtype, np.integer):
            raise ValueError("target_index must be an integer array of shape [N]")
        if np.any((target < -1) | (target >= k)):
            raise ValueError("target_index must be -1 (unannotated) or a valid candidate index")
        mask = arrays.get("candidate_mask", np.ones((n, k), dtype=bool))
        if mask.shape != (n, k) or not np.isin(mask, [0, 1]).all():
            raise ValueError("candidate_mask must have shape [N,K] and contain only 0/1")
        mask = mask.astype(bool)
        if not mask.any(axis=1).all():
            raise ValueError("Every critique requires at least one unmasked candidate")
        known = target >= 0
        if np.any(~mask[np.where(known)[0], target[known]]):
            raise ValueError("A verified target cannot be masked out")
        split = arrays["split"].astype(str)
        if split.shape != (n,) or not np.isin(split, ["train", "val", "test"]).all():
            raise ValueError("split must contain train, val, or test")
        group = arrays["group_id"]
        if group.shape != (n,) or group.dtype.kind not in "iuUS":
            raise ValueError("group_id must be a one-dimensional integer or string array")
        _, group_index = np.unique(group, return_inverse=True)
        for group_value in np.unique(group_index):
            if len(np.unique(split[group_index == group_value])) != 1:
                raise ValueError("All critiques from a review group must belong to the same split")
        self.indices = {name: np.flatnonzero(split == name)
                        for name in ("train", "val", "test")}
        if not len(self.indices["train"]) or not len(self.indices["val"]):
            raise ValueError("Separate nonempty train and val splits are required")
        if not np.any((split == "train") & (labels == 1) & known):
            raise ValueError("Training requires at least one verified target repair for a valid critique")
        self.group_index = group_index
        self.code_dim, self.critique_dim = original.shape[-1], critique.shape[-1]
        self.tensors = {key: torch.as_tensor(arrays[key], dtype=torch.float32)
                        for key in feature_keys}
        self.tensors.update(labels=torch.as_tensor(labels, dtype=torch.float32),
                            target_index=torch.as_tensor(target, dtype=torch.long),
                            candidate_mask=torch.as_tensor(mask, dtype=torch.bool),
                            group_id=torch.as_tensor(group_index, dtype=torch.long))
        self.diagnostics = {}
        for key in ("unrelated", "transformed"):
            identical = identical_states(original, arrays[key], candidates.shape)
            total = int(mask.sum())
            count = int((identical & mask).sum())
            self.diagnostics[f"{key}_identical_feature_count"] = count
            self.diagnostics[f"{key}_candidate_count"] = total
            if count:
                warnings.warn(
                    f"{key}: {count}/{total} candidates equal the original code features; "
                    "these cases do not establish response to a distinct program state",
                    stacklevel=2,
                )
            all_identical = count == total
            if all(paired) and all_identical:
                paired_identical = identical_states(
                    arrays["pair_original"], arrays[f"pair_{key}"], arrays["pair_candidates"].shape,
                )
                all_identical = bool(paired_identical[mask].all())
            if key == "transformed" and require_transformation and all_identical:
                raise ValueError("Equivariance requires distinct, supplied transformed-state features")

    def batches(self, split, batch_size, rng=None):
        indices = self.indices[split]
        groups = np.unique(self.group_index[indices])
        if rng is not None:
            groups = rng.permutation(groups)
        pending = []
        for group in groups:
            members = indices[self.group_index[indices] == group].tolist()
            if pending and len(pending) + len(members) > batch_size:
                yield pending
                pending = []
            pending.extend(members)
        if pending:
            yield pending

    def batch(self, indices, device):
        return {key: value[indices].to(device) for key, value in self.tensors.items()}


def model_inputs(batch):
    return {key: batch[key] for key in (*FEATURE_KEYS, *PAIRED_KEYS, "candidate_mask")
            if key in batch}


@torch.no_grad()
def evaluate(model, dataset, split, batch_size, device):
    model.eval()
    labels, probabilities, logits = [], [], []
    for indices in dataset.batches(split, batch_size):
        batch = dataset.batch(indices, device)
        output = model(**model_inputs(batch))
        labels.append(batch["labels"].cpu())
        probabilities.append(output["probability"].cpu())
        logits.append(output["validity_logits"].cpu())
    if not labels:
        return None
    y = torch.cat(labels).numpy().astype(int)
    scores = torch.cat(probabilities).numpy()
    logit = torch.cat(logits)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, scores >= 0.5, average="binary", zero_division=0,
    )
    two_classes = len(np.unique(y)) == 2
    return {
        "n": len(y), "positives": int(y.sum()),
        "bce": float(torch.nn.functional.binary_cross_entropy_with_logits(
            logit, torch.tensor(y, dtype=logit.dtype))),
        "accuracy": float(accuracy_score(y, scores >= 0.5)),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "auroc": float(roc_auc_score(y, scores)) if two_classes else None,
        "average_precision": float(average_precision_score(y, scores))
        if two_classes else None,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Jointly train critique, intervention, and verifier modules")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    for name in ("theta", "phi", "psi"):
        parser.add_argument(f"--{name}-lr", type=float)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--rank-hidden", type=int, default=128)
    parser.add_argument("--verifier-hidden", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-unpaired-features", action="store_true")
    parser.add_argument("--selection-method", choices=("exact", "greedy"), default="exact")
    parser.add_argument("--exact-limit", type=int, default=18)
    for name in LossWeights.__dataclass_fields__:
        parser.add_argument(f"--w-{name}", type=float, default=1.0)
    return parser.parse_args()


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 0:
        raise ValueError("epochs/batch-size must be positive and patience nonnegative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    weights = LossWeights(**{name: getattr(args, f"w_{name}")
                             for name in LossWeights.__dataclass_fields__})
    dataset = JointDataset(args.data, require_transformation=weights.equivariance > 0,
                           allow_unpaired=args.allow_unpaired_features)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)
    model_config = dict(code_dim=dataset.code_dim, critique_dim=dataset.critique_dim,
                        paired_dim=dataset.paired_dim, hidden=args.hidden,
                        layers=args.layers, heads=args.heads, dropout=args.dropout,
                        rank_hidden=args.rank_hidden, verifier_hidden=args.verifier_hidden,
                        temperature=args.temperature)
    model = CIVER(**model_config).to(device)
    groups = [{"params": parameters,
               "lr": args.lr if getattr(args, f"{name}_lr") is None else getattr(args, f"{name}_lr"),
               "name": name} for name, parameters in model.parameter_groups().items()]
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / "checkpoint.pt"
    best_validation = float("inf")
    history = []
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {}
        count = 0
        for indices in dataset.batches("train", args.batch_size, rng):
            batch = dataset.batch(indices, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(**model_inputs(batch))
            losses = joint_objectives(output, batch, weights=weights, margin=args.margin,
                                      selection_method=args.selection_method,
                                      exact_limit=args.exact_limit)
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError("A joint training loss became nonfinite")
            losses["total"].backward()
            optimizer.step()
            for name, loss in losses.items():
                sums[name] = sums.get(name, 0.0) + float(loss.detach()) * len(indices)
            count += len(indices)
        validation = evaluate(model, dataset, "val", args.batch_size, device)
        if not np.isfinite(validation["bce"]):
            raise FloatingPointError("Validation BCE became nonfinite")
        record = {"epoch": epoch, "train": {key: value / count for key, value in sums.items()},
                  "validation": validation}
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation["bce"] < best_validation:
            best_validation = validation["bce"]
            stale_epochs = 0
            torch.save({"model": model.state_dict(), "model_config": model_config,
                        "epoch": epoch, "selection_metric": "validation_bce",
                        "validation": validation, "loss_weights": asdict(weights),
                        "optimizer": optimizer.state_dict(), "seed": args.seed}, checkpoint_path)
        else:
            stale_epochs += 1
        if args.patience and stale_epochs >= args.patience:
            break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    results = {
        "selection_metric": "validation_bce", "selected_epoch": checkpoint["epoch"],
        "validation": checkpoint["validation"],
        "test": evaluate(model, dataset, "test", args.batch_size, device),
        "split_sizes": {key: len(value) for key, value in dataset.indices.items()},
        "data_diagnostics": dataset.diagnostics, "loss_weights": asdict(weights),
        "feature_mode": "code_text_paired" if dataset.paired_dim is not None else "unpaired_ablation",
        "history": history,
    }
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({key: value for key, value in results.items() if key != "history"}), flush=True)
    return results


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
