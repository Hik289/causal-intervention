import argparse
import hashlib
import inspect
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from civic.models import CIVER
from civic.objectives import LossWeights, joint_objectives


FORWARD_KEYS = set(inspect.signature(CIVER.forward).parameters) - {"self"}
FLOAT_KEYS = {
    "original", "critique", "candidates", "unrelated", "transformed",
    "pair_original", "pair_candidates", "pair_unrelated", "pair_transformed",
    "transformed_critique", "candidate_edit_costs", "candidate_diff_features",
    "frozen_reviewer_logit", "labels", "location_labels", "support_labels",
    "transformed_location_map", "transformed_support_map",
}
INTEGER_KEYS = {"target_index", "reverse_index", "evaluation_target_index"}
METADATA_KEYS = {"case_id", "case_ids", "repo", "repo_ids", "split", "group_id",
                 "group_ids", "pr_id", "pr_ids", "pair_id", "role", "language"}
CANDIDATE_KEYS = {
    "candidates", "pair_candidates", "candidate_mask", "candidate_token_mask",
    "pair_candidates_mask", "candidate_edit_costs", "candidate_edit_cost_mask",
    "candidate_diff_features", "candidate_verified_mask",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def append_identity_candidate(arrays):
    original, candidates = arrays["original"], arrays["candidates"]
    count = candidates.shape[1]
    if candidates.shape[2:] != original.shape[1:]:
        raise ValueError("Identity candidates require the same original-state representation")
    additions = {
        "candidates": original[:, None],
        "candidate_mask": np.ones((len(original), 1), dtype=bool),
        "candidate_edit_costs": np.zeros((len(original), 1), dtype=np.float32),
        "candidate_edit_cost_mask": np.ones((len(original), 1), dtype=bool),
        "candidate_verified_mask": np.zeros((len(original), 1), dtype=bool),
    }
    for candidate_key, state_key in [("candidate_token_mask", "original_mask"),
                                     ("pair_candidates", "pair_original"),
                                     ("pair_candidates_mask", "pair_original_mask")]:
        if candidate_key in arrays:
            if state_key not in arrays:
                raise ValueError("An identity candidate needs its measured original view: " + state_key)
            state = arrays[state_key].reshape((len(original),) + arrays[candidate_key].shape[2:])
            additions[candidate_key] = state[:, None]
    if "candidate_diff_features" in arrays:
        mask = arrays.get("original_mask")
        if original.ndim == 3:
            weight = np.ones(original.shape[:2], dtype=np.float32) if mask is None else mask.astype(np.float32)
            pooled = (original * weight[..., None]).sum(1) / weight.sum(1, keepdims=True).clip(1)
        else:
            pooled = original
        additions["candidate_diff_features"] = (pooled - pooled)[:, None]
    for key in additions:
        if key not in arrays:
            if key == "candidate_mask":
                arrays[key] = np.ones((len(original), count), dtype=bool)
            elif key == "candidate_edit_cost_mask":
                arrays[key] = np.isfinite(arrays["candidate_edit_costs"])
            elif key == "candidate_verified_mask":
                verified = np.zeros((len(original), count), dtype=bool)
                target = arrays["target_index"]
                valid = (target >= 0) & (arrays["labels"] == 1)
                verified[np.flatnonzero(valid), target[valid]] = True
                arrays[key] = verified
    for key, value in additions.items():
        arrays[key] = np.concatenate([arrays[key], value], axis=1)
    return arrays


class FullDataset:
    def __init__(self, path, split_unit="repo", include_identity=False, weak_patch_supervision=False,
                 require_training_splits=True):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        aliases = {"case_ids": "case_id", "group_ids": "group_id", "pr_ids": "pr_id",
                   "repo_ids": "repo", "repos": "repo", "pair_ids": "pair_id", "pair_roles": "role",
                   "candidate_applicable_mask": "candidate_mask",
                   "unrelated_valid_mask": "unrelated_available_mask",
                   "transformation_valid_mask": "transformed_available_mask"}
        for old, new in aliases.items():
            if new not in arrays and old in arrays:
                arrays[new] = arrays[old]
        if "role" in arrays:
            group = arrays.get("group_id")
            if group is not None:
                arrays["group_id"] = np.asarray([str(a) + "::state=" + str(b)
                                                for a, b in zip(group, arrays["role"])])
        reviewer_available = arrays.get("frozen_reviewer_available")
        use_reviewer = "frozen_reviewer_logit" in arrays and (reviewer_available is None or bool(np.all(reviewer_available)))
        if not use_reviewer:
            arrays.pop("frozen_reviewer_logit", None)
        for key in ["pair_original", "pair_unrelated", "pair_transformed"]:
            mask_key = key + "_mask"
            if key in arrays and arrays[key].ndim == 2 and mask_key in arrays and arrays[mask_key].ndim == 1:
                arrays[mask_key] = arrays[mask_key][:, None]
        if include_identity:
            arrays = append_identity_candidate(arrays)
        if weak_patch_supervision:
            for key in ["weak_patch_location_labels", "weak_patch_location_mask"]:
                if key not in arrays:
                    raise ValueError("Explicit weak patch supervision requires measured patch spans: " + key)
            arrays["location_labels"] = arrays["weak_patch_location_labels"]
            arrays["location_supervision_mask"] = arrays["weak_patch_location_mask"] & (arrays["labels"] == 1)
        required = {"original", "critique", "candidates", "labels", "target_index",
                    "split", "case_id", "repo", "group_id", "candidate_edit_costs"}
        if required - arrays.keys():
            raise ValueError("Missing full-training fields: " + str(sorted(required - arrays.keys())))
        self.n = len(arrays["labels"])
        if self.n == 0 or any(len(value) != self.n for value in arrays.values()):
            raise ValueError("All archive arrays must have the same nonzero first dimension")
        if not np.isin(arrays["labels"], [0, 1]).all():
            raise ValueError("Full training requires binary labels with declared provenance")
        self.metadata = {key: arrays[key].astype(str) for key in METADATA_KEYS if key in arrays}
        if len(np.unique(self.metadata["case_id"])) != self.n:
            raise ValueError("Case identifiers must be unique")
        split = self.metadata["split"]
        if not np.isin(split, ["train", "val", "test"]).all():
            raise ValueError("Unknown split")
        self.indices = {name: np.flatnonzero(split == name) for name in ["train", "val", "test"]}
        for name in (["train", "val"] if require_training_splits else []):
            if not len(self.indices[name]) or len(np.unique(arrays["labels"][self.indices[name]])) != 2:
                raise ValueError(name + " must contain both classes")
        repo_sets = {name: set(self.metadata["repo"][indices]) for name, indices in self.indices.items()}
        repo_disjoint = not any(repo_sets[a] & repo_sets[b] for a, b in [("train", "val"), ("train", "test"), ("val", "test")])
        for key in (["repo"] if split_unit == "repo" else []) + ["group_id", "pair_id"]:
            if key not in self.metadata:
                continue
            sets = {name: set(self.metadata[key][indices]) for name, indices in self.indices.items()}
            if any(sets[a] & sets[b] for a, b in [("train", "val"), ("train", "test"), ("val", "test")]):
                raise ValueError(key + " overlaps across splits")
        self.k = arrays["candidates"].shape[1]
        if arrays["candidates"].ndim not in [3, 4] or self.k < 1:
            raise ValueError("Candidate states must be pooled vectors or token sequences")
        self.code_dim = arrays["original"].shape[-1]
        self.critique_dim = arrays["critique"].shape[-1]
        self.paired_dim = arrays["pair_original"].shape[-1] if "pair_original" in arrays else None
        if self.paired_dim is not None:
            for key in ["pair_candidates", "pair_unrelated", "pair_transformed"]:
                state = key.replace("pair_", "")
                if state in arrays and key not in arrays:
                    raise ValueError("Missing independently encoded paired view: " + key)
        _, groups = np.unique(self.metadata["group_id"], return_inverse=True)
        arrays["group_id"] = groups
        for name in ["unrelated", "transformed"]:
            arrays.setdefault(name + "_available_mask", np.zeros(self.n, dtype=bool))
            if name not in arrays and arrays[name + "_available_mask"].any():
                raise ValueError("A missing program state cannot be declared available")
        arrays.setdefault("candidate_mask", np.ones((self.n, self.k), dtype=bool))
        arrays.setdefault("candidate_edit_cost_mask", np.isfinite(arrays["candidate_edit_costs"]))
        self.tensors = {}
        for key, value in arrays.items():
            if key in METADATA_KEYS and key != "group_id":
                continue
            if key not in FORWARD_KEYS | FLOAT_KEYS | INTEGER_KEYS | {"group_id"} and not key.endswith("_mask"):
                continue
            if value.dtype.kind in "USO":
                continue
            if key in INTEGER_KEYS | {"group_id"}:
                dtype = torch.long
            elif key.endswith("_mask") or key == "candidate_mask":
                dtype = torch.bool
            else:
                dtype = torch.float32
                if key != "candidate_edit_costs" and not np.isfinite(value).all():
                    raise ValueError("Nonfinite frozen data: " + key)
            self.tensors[key] = torch.as_tensor(value, dtype=dtype)
        target = self.tensors["target_index"]
        if ((target < -1) | (target >= self.k)).any():
            raise ValueError("Target index outside candidate pool")
        known = target >= 0
        mask = self.tensors["candidate_mask"]
        if not mask.any(1).all() or not mask[known, target[known]].all():
            raise ValueError("Invalid candidate masks")
        self.audit = {
            "data_sha256": sha256(path), "n": self.n,
            "require_training_splits": require_training_splits,
            "split_counts": {key: len(value) for key, value in self.indices.items()},
            "positive_counts": {key: int(arrays["labels"][value].sum()) for key, value in self.indices.items()},
            "repo_disjoint": repo_disjoint, "group_disjoint": True, "split_unit": split_unit,
            "code_dim": self.code_dim, "critique_dim": self.critique_dim,
            "paired_dim": self.paired_dim, "candidates": self.k,
            "control_counts": {name: int(self.tensors[name + "_available_mask"].sum())
                               for name in ["unrelated", "transformed"]},
            "all_features_frozen": all(not value.requires_grad for value in self.tensors.values()),
            "frozen_reviewer_feature_used": use_reviewer,
            "review_groups_are_program_state_specific": "role" in arrays,
            "location_training_annotation": "official repair-span weak proxy, not human defect gold" if weak_patch_supervision else "archive-declared annotations only",
            "identity_candidate": {
                "included": include_identity,
                "definition": "unchanged measured original program; zero changed lines; never a repair or unrelated control",
                "diff_representation": "mean encoded candidate blocks minus mean encoded original blocks",
                "supervision": "unverified, not a ranking target, appended uniformly without labels",
            },
        }

    def batches(self, split, size, rng=None):
        indices = self.indices[split]
        group_array = self.tensors["group_id"].numpy()
        groups = np.unique(group_array[indices])
        if rng is not None:
            groups = rng.permutation(groups)
        pending = []
        for group in groups:
            members = indices[group_array[indices] == group].tolist()
            if pending and len(pending) + len(members) > size:
                yield np.asarray(pending, dtype=np.int64)
                pending = []
            pending.extend(members)
        if pending:
            yield np.asarray(pending, dtype=np.int64)

    def batch(self, indices, device, rng=None):
        batch = {key: value[indices].to(device) for key, value in self.tensors.items()}
        batch.setdefault("unrelated", None)
        batch.setdefault("transformed", None)
        if rng is not None:
            permutation = torch.as_tensor(np.stack([rng.permutation(self.k) for _ in indices]), device=device)
            row = torch.arange(len(indices), device=device).unsqueeze(1)
            for key in CANDIDATE_KEYS & batch.keys():
                batch[key] = batch[key][row, permutation]
            inverse = torch.argsort(permutation, dim=1)
            for key in INTEGER_KEYS & batch.keys():
                index = batch[key]
                batch[key] = torch.where(index >= 0, inverse.gather(1, index.clamp_min(0).unsqueeze(1)).squeeze(1), index)
            for key in ["unrelated", "transformed", "pair_unrelated", "pair_transformed"]:
                value = batch.get(key)
                if value is not None and value.ndim == batch["candidates"].ndim and value.shape[1] == self.k:
                    batch[key] = value[row, permutation]
            for key in ["unrelated_available_mask", "transformed_available_mask", "location_mapping_mask", "support_mapping_mask"]:
                value = batch.get(key)
                if value is not None and value.ndim == 2:
                    batch[key] = value[row, permutation]
            for key in ["unrelated_token_mask", "transformed_token_mask", "pair_unrelated_mask", "pair_transformed_mask"]:
                value = batch.get(key)
                if value is not None and value.ndim == 3:
                    batch[key] = value[row, permutation]
            for key in ["transformed_location_map", "transformed_support_map"]:
                value = batch.get(key)
                if value is not None and value.ndim == 4:
                    batch[key] = value[row, permutation]
        return batch


def model_inputs(batch):
    return {key: value for key, value in batch.items() if key in FORWARD_KEYS}


@torch.no_grad()
def predict(model, dataset, split, batch_size, device):
    model.eval()
    collected = {"indices": [], "labels": [], "logits": [], "state_scores": [],
                 "target_scores": [], "unrelated_scores": [], "transformed_scores": [],
                 "selection_probability": [], "ranking_scores": [], "target_index": [],
                 "unrelated_available": [], "transformed_available": [],
                 "location_probabilities": [], "transformed_location_probabilities": []}
    for indices in dataset.batches(split, batch_size):
        batch = dataset.batch(indices, device)
        output = model(**model_inputs(batch))
        target_index = batch.get("evaluation_target_index", batch["target_index"])
        safe_index = target_index.clamp_min(0)
        rows = torch.arange(len(indices), device=device)
        unrelated_mask = output["unrelated_available_mask"]
        transformed_mask = output["transformed_available_mask"]
        unrelated = (output["unrelated_scores"] * unrelated_mask).sum(1) / unrelated_mask.sum(1).clamp_min(1)
        transformed = (output["transformed_scores"] * transformed_mask).sum(1) / transformed_mask.sum(1).clamp_min(1)
        values = {
            "indices": indices, "labels": batch["labels"], "logits": output["validity_logits"],
            "state_scores": output["original_score"],
            "target_scores": output["target_scores"][rows, safe_index],
            "unrelated_scores": unrelated, "transformed_scores": transformed,
            "selection_probability": output["selection_probability"],
            "ranking_scores": output["ranking_logits"], "target_index": target_index,
            "unrelated_available": unrelated_mask.any(1), "transformed_available": transformed_mask.any(1),
            "location_probabilities": output["location_probabilities"],
        }
        states = output["state_predictions"]["transformed"]
        if states is None:
            values["transformed_location_probabilities"] = torch.zeros_like(output["location_probabilities"])
        else:
            values["transformed_location_probabilities"] = states["location_probabilities"].mean(1)
        for key, value in values.items():
            collected[key].append(value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value)
    if not collected["indices"]:
        return None
    result = {key: np.concatenate(value) for key, value in collected.items()}
    order = np.argsort(result["indices"])
    return {key: value[order] for key, value in result.items()}


def metrics(prediction, dataset, threshold=0.0):
    from civic.evaluation import binary_metrics, paired_metrics
    if prediction is None:
        return None
    indices = prediction["indices"]
    pr_ids = dataset.metadata.get("pr_id")
    result = binary_metrics(prediction["labels"].astype(int), prediction["logits"],
                            threshold=threshold, pr_ids=None if pr_ids is None else pr_ids[indices])
    logits64 = prediction["logits"].astype(np.float64)
    result["bce"] = float(np.mean(np.logaddexp(0, logits64) - prediction["labels"].astype(np.float64) * logits64))
    valid = (prediction["labels"] == 1) & (prediction["target_index"] >= 0) & prediction["unrelated_available"]
    if valid.any():
        result["paired"] = paired_metrics(prediction["state_scores"][valid], prediction["target_scores"][valid],
                                           prediction["unrelated_scores"][valid], threshold=0.0)
    else:
        result["paired"] = None
    return result


def selection_key(result, epoch, primary="auroc"):
    paired = result.get("paired") or {}
    auroc = result.get("auroc")
    if auroc is None:
        raise ValueError("Validation AUROC is undefined")
    pair_auroc = paired.get("pair_auroc", -1.0)
    if primary in ["pair_auroc", "joint"] and pair_auroc == -1.0:
        raise ValueError("Causal checkpoint selection requires actual validation unrelated controls")
    scores = (float(auroc), float(pair_auroc))
    if primary == "pair_auroc":
        scores = scores[::-1]
    elif primary == "joint":
        scores = (2 * float(auroc) * float(pair_auroc) / max(float(auroc) + float(pair_auroc), 1e-12), float(pair_auroc))
    return (*scores, -float(result["bce"]), -int(epoch))


def module_gradient_audit(model):
    return {name: {"parameters": sum(p.numel() for p in values),
                   "gradient_l1": float(sum(p.grad.detach().abs().sum().cpu().item()
                                            for p in values if p.grad is not None))}
            for name, values in model.parameter_groups().items()}


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    rng = np.random.default_rng(args.seed)
    dataset = FullDataset(args.data, split_unit=args.split_unit, include_identity=args.include_identity,
                          weak_patch_supervision=args.weak_patch_supervision)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    weights = LossWeights(**{name: getattr(args, "w_" + name) for name in LossWeights.__dataclass_fields__})
    config = {"code_dim": dataset.code_dim, "critique_dim": dataset.critique_dim,
              "paired_dim": dataset.paired_dim, "hidden": args.hidden, "layers": args.layers,
              "heads": args.heads, "dropout": args.dropout, "rank_hidden": args.rank_hidden,
              "verifier_hidden": args.verifier_hidden, "temperature": args.temperature}
    model = CIVER(**config).to(device)
    if args.initialize_from:
        pretrained = load_checkpoint(args.initialize_from, device)
        if pretrained["config"] != config:
            raise ValueError("Pretraining and adaptation must use the same module configuration")
        model.load_state_dict(pretrained["model"])
    optimizer = torch.optim.AdamW(
        [{"params": parameters, "name": name, "lr": args.lr}
         for name, parameters in model.parameter_groups().items()], weight_decay=args.weight_decay)
    module_counts = {name: sum(p.numel() for p in parameters) for name, parameters in model.parameter_groups().items()}
    manifest = {"config": config, "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "weights": asdict(weights), "data": dataset.audit, "module_parameters": module_counts,
                "total_parameters": sum(module_counts.values()),
                "initialization_sha256": sha256(args.initialize_from) if args.initialize_from else None,
                "primary_selection": "validation_" + args.primary_metric + "_then_other_validation_metric_then_bce_then_earliest_epoch",
                "test_selection": "exploratory_only_never_used_for_optimizer_hyperparameters_or_primary_checkpoint",
                "stopping_rule": "validation AUROC and validation Pair AUROC only; BCE cannot reset patience",
                "code_sha256": {str(path.relative_to(Path(__file__).parents[1])): sha256(path)
                                for path in [Path(__file__), Path(__file__).parents[1] / "civic/models.py",
                                             Path(__file__).parents[1] / "civic/objectives.py",
                                             Path(__file__).parents[1] / "civic/ranking.py",
                                             Path(__file__).parents[1] / "civic/evaluation.py"]}}
    save_json(args.output / "manifest.json", manifest)
    torch.save({"model": model.state_dict(), "config": config, "epoch": 0}, args.output / "initial.pt")
    best_key, best_epoch, best_validation = None, None, None
    best_stop_key, last_primary_improvement, epochs_trained = None, 0, 0
    started = time.monotonic()
    all_gradient_totals = {name: 0.0 for name in module_counts}
    history = args.output / "validation_history.jsonl"
    diagnostic = args.output / "exploratory_test_history.jsonl"
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals, seen = {}, 0
        last_gradient = None
        for indices in dataset.batches("train", args.batch_size, rng):
            batch = dataset.batch(indices, device, rng)
            optimizer.zero_grad(set_to_none=True)
            output = model(**model_inputs(batch))
            losses = joint_objectives(output, batch, weights=weights, margin=args.margin,
                                      selection_method="greedy", minimality_cost_scale=args.cost_scale,
                                      auxiliary_classification_weight=args.auxiliary_classification_weight)
            if not all(torch.isfinite(value) for value in losses.values()):
                raise FloatingPointError("Nonfinite joint objective")
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if epoch == 1:
                last_gradient = module_gradient_audit(model)
                for name, value in last_gradient.items():
                    all_gradient_totals[name] += value["gradient_l1"]
            optimizer.step()
            seen += len(indices)
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * len(indices)
        if epoch == 1:
            if not all(all_gradient_totals[name] > 0 for name in args.required_modules):
                raise RuntimeError("At least one trainable module received no gradient")
            save_json(args.output / "module_gradient_audit.json", {"passed": True, "totals": all_gradient_totals,
                                                                 "required_modules": args.required_modules,
                                                                 "last_batch": last_gradient})
        validation_prediction = predict(model, dataset, "val", args.eval_batch_size, device)
        validation = metrics(validation_prediction, dataset)
        key = selection_key(validation, epoch, args.primary_metric)
        stop_key = key[:2]
        if best_stop_key is None or stop_key > best_stop_key:
            best_stop_key, last_primary_improvement = stop_key, epoch
        if best_key is None or key > best_key:
            best_key, best_epoch, best_validation = key, epoch, validation
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config,
                        "epoch": epoch, "selection_key": key, "validation": validation},
                       args.output / "validation_selected.pt")
            np.savez_compressed(args.output / "selected_validation_predictions.npz", **validation_prediction)
        record = {"epoch": epoch, "train": {key: value / seen for key, value in totals.items()},
                  "validation": validation, "selected_epoch": best_epoch,
                  "elapsed_seconds": time.monotonic() - started}
        with history.open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        if args.test_every and not args.defer_test and (epoch % args.test_every == 0 or epoch == 1):
            prediction = predict(model, dataset, "test", args.eval_batch_size, device)
            exploratory = metrics(prediction, dataset)
            with diagnostic.open("a") as handle:
                handle.write(json.dumps({"epoch": epoch, "test": exploratory, "exploratory_only": True}, allow_nan=False) + "\n")
        if epoch == 1 or epoch % args.progress_every == 0:
            print(json.dumps({"epoch": epoch, "validation_auroc": validation["auroc"], "selected_epoch": best_epoch,
                              "seconds": round(time.monotonic() - started, 2)}), flush=True)
        epochs_trained = epoch
        if args.patience and epoch >= args.min_epochs and epoch - last_primary_improvement >= args.patience:
            print(json.dumps({"validation_plateau_stop": True, "epoch": epoch,
                              "last_primary_improvement": last_primary_improvement}), flush=True)
            break
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config,
                "epoch": epochs_trained}, args.output / "final.pt")
    checkpoint = args.output / "validation_selected.pt"
    save_json(args.output / "primary_selection_locked.json", {"epoch": best_epoch, "key": best_key,
              "checkpoint_sha256": sha256(checkpoint), "epochs_trained": epochs_trained,
              "test_diagnostics_not_used": True})
    model.load_state_dict(load_checkpoint(checkpoint, device)["model"])
    from civic.evaluation import validation_threshold
    selected_val = predict(model, dataset, "val", args.eval_batch_size, device)
    threshold_result = validation_threshold(selected_val["labels"].astype(int), selected_val["logits"], objective="f1")
    threshold = float(threshold_result["threshold"] if isinstance(threshold_result, dict) else threshold_result)
    save_json(args.output / "threshold_locked.json", {"threshold": threshold, "source": "validation_f1_only"})
    selected_test = None if args.defer_test else predict(model, dataset, "test", args.eval_batch_size, device)
    if selected_test is not None:
        np.savez_compressed(args.output / "selected_test_predictions.npz", **selected_test)
    test = metrics(selected_test, dataset, threshold)
    result = {"complete": True, "model_kind": args.model_kind,
              "epochs_trained": epochs_trained, "selected_epoch": best_epoch, "selection_key": best_key,
              "run_purpose": args.run_purpose, "test_deferred": args.defer_test,
              "selected_validation": best_validation, "threshold": threshold, "test": test,
              "checkpoint_sha256": sha256(checkpoint), "data_sha256": dataset.audit["data_sha256"],
              "module_parameters": module_counts, "loss_weights": asdict(weights),
              "elapsed_seconds": time.monotonic() - started}
    save_json(args.output / "results.json", result)
    print(json.dumps({"complete": True, "selected_epoch": best_epoch, "test": test}), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train and audit all three causal review modules")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--split-unit", choices=["repo", "group", "diff"], default="repo")
    parser.add_argument("--run-purpose", choices=["formal", "pilot", "smoke"], default="formal")
    parser.add_argument("--defer-test", action="store_true")
    parser.add_argument("--include-identity", action="store_true")
    parser.add_argument("--weak-patch-supervision", action="store_true")
    parser.add_argument("--required-modules", nargs="+", choices=["theta", "phi", "psi"], default=["theta", "phi", "psi"])
    parser.add_argument("--model-kind", default="full_three_trainable_modules")
    parser.add_argument("--primary-metric", choices=["auroc", "pair_auroc", "joint"], default="auroc")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--min-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--rank-hidden", type=int, default=256)
    parser.add_argument("--verifier-hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--cost-scale", type=float, default=10.0)
    parser.add_argument("--auxiliary-classification-weight", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--test-every", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    defaults = {"classification": 1.0, "flip": 0.5, "specificity": 0.1,
                "equivariance": 0.1, "ranking": 0.5, "minimality": 0.01, "selection": 0.05}
    for name, value in defaults.items():
        parser.add_argument("--w-" + name, type=float, default=value)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.eval_batch_size, args.cpu_threads) < 1 or args.test_every < 0:
        parser.error("Training sizes must be positive; test-every may be zero")
    if args.min_epochs < 1 or args.patience < 0:
        parser.error("min-epochs must be positive; patience may be zero")
    return args


if __name__ == "__main__":
    train(parse_args())
