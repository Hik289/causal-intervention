import argparse
import hashlib
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
from scripts.train_full import (FullDataset, load_checkpoint, save_json, sha256,
                                model_inputs, predict, metrics, selection_key,
                                module_gradient_audit)


def initialize_filter(model, config, pretrained, allow_dropout_transfer=False,
                      allow_reviewer_feature_transfer=False):
    previous = dict(pretrained['config'])
    previous.setdefault('condition_on_reviewer', False)
    differences = {key for key in config if previous.get(key) != config[key]}
    permitted = set()
    if allow_dropout_transfer:
        permitted.add('dropout')
    if allow_reviewer_feature_transfer and config['condition_on_reviewer'] and not previous['condition_on_reviewer']:
        permitted.add('condition_on_reviewer')
    if differences - permitted:
        raise ValueError('Checkpoint configuration changes require explicit, compatible transfer options')
    state = dict(pretrained['model'])
    if 'condition_on_reviewer' in differences:
        key = 'verifier.network.0.weight'
        expected = model.state_dict()[key]
        original = state[key]
        if expected.shape != (original.shape[0], original.shape[1] + 2):
            raise ValueError('Reviewer feature transfer requires exactly two added input columns')
        state[key] = torch.cat((original, original.new_zeros((original.shape[0], 2))), dim=1)
    model.load_state_dict(state)


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    rng = np.random.default_rng(args.seed)
    dataset = FullDataset(args.data, split_unit=args.split_unit, include_identity=args.include_identity,
                          weak_patch_supervision=args.weak_patch_supervision)
    device = torch.device(args.device)
    train_labels = dataset.tensors["labels"][dataset.indices["train"]].numpy()
    n_train = len(train_labels)
    n_positive = int(train_labels.sum())
    n_negative = n_train - n_positive
    if min(n_positive, n_negative) <= 0:
        raise ValueError("Class balance requires both train classes")
    balanced_weights = (n_train / (2.0 * n_negative), n_train / (2.0 * n_positive))
    train_prior = dataset.tensors['frozen_reviewer_logit'][dataset.indices['train']].numpy()
    train_hard_negative = (train_labels <= .5) & (train_prior >= 0.)
    raw_auxiliary_weights = np.where(train_labels > .5, balanced_weights[1], balanced_weights[0])
    raw_auxiliary_weights = raw_auxiliary_weights * np.where(train_hard_negative, args.hard_negative_multiplier, 1.)
    auxiliary_weight_normalizer = float(raw_auxiliary_weights.mean())
    args.output.mkdir(parents=True, exist_ok=False)
    if args.case_pr_map:
        records = [json.loads(line) for line in args.case_pr_map.read_text().splitlines() if line.strip()]
        by_case = {str(record["case_id"]): str(record["pr_id"]) for record in records}
        if len(by_case) != len(records):
            raise ValueError("The PR mapping must contain unique case identifiers")
        true_pr_ids = np.asarray([by_case[case] for case in dataset.metadata["case_id"]])
    else:
        if "pr_id" not in dataset.metadata:
            raise ValueError("Finding workload metrics require actual pr_id values or --case-pr-map")
        true_pr_ids = dataset.metadata["pr_id"]
    dataset.metadata["pr_id"] = true_pr_ids
    if not np.all(np.char.str_len(true_pr_ids.astype(str)) > 0):
        raise ValueError("PR identifiers must be nonempty for every finding")
    if args.training_language:
        language = dataset.metadata.get("language")
        if language is None or any(set(language[dataset.indices[split]]) != {args.training_language}
                                   for split in ("train", "val")):
            raise ValueError("Training and validation must use only the declared training language")
    pr_audit = {"test_n": len(dataset.indices["test"]),
                "actual_PR_n_test": len(np.unique(true_pr_ids[dataset.indices["test"]])),
                "mapping_sha256": sha256(args.case_pr_map) if args.case_pr_map else None,
                "mapping_source": "explicit case-to-PR mapping" if args.case_pr_map else "archive pr_id"}
    save_json(args.output / "pr_identity_audit.json", pr_audit)
    test_ids_sha = hashlib.sha256(("\n".join(dataset.metadata["case_id"][dataset.indices["test"]]) + "\n").encode()).hexdigest()
    weights = LossWeights(**{name: getattr(args, "w_" + name) for name in LossWeights.__dataclass_fields__})
    config = {"code_dim": dataset.code_dim, "critique_dim": dataset.critique_dim,
              "paired_dim": dataset.paired_dim, "hidden": args.hidden, "layers": args.layers,
              "heads": args.heads, "dropout": args.dropout, "rank_hidden": args.rank_hidden,
              "verifier_hidden": args.verifier_hidden, "temperature": args.temperature,
              "condition_on_reviewer": args.condition_on_reviewer}
    model = CIVER(**config).to(device)
    if args.initialize_from:
        pretrained = load_checkpoint(args.initialize_from, device)
        initialize_filter(model, config, pretrained, args.allow_dropout_transfer,
                          args.allow_reviewer_feature_transfer)
    if "frozen_reviewer_logit" not in dataset.tensors:
        raise ValueError("Anchored scope requires every row to have a real frozen reviewer score")
    with torch.no_grad():
        model.verifier.validity_head.weight.zero_()
        model.verifier.validity_head.bias.zero_()
        model.verifier.selection_head.weight.zero_()
        model.verifier.selection_head.bias.zero_()
    optimizer = torch.optim.AdamW(
        [{"params": parameters, "name": name, "lr": getattr(args, "lr_" + name) or args.lr}
         for name, parameters in model.parameter_groups().items()], weight_decay=args.weight_decay)
    module_counts = {name: sum(p.numel() for p in parameters) for name, parameters in model.parameter_groups().items()}
    manifest = {"config": config, "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "weights": asdict(weights), "data": dataset.audit, "module_parameters": module_counts,
                "total_parameters": sum(module_counts.values()),
                "module_learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups},
                "dropout_only_transfer": bool(args.allow_dropout_transfer),
                "reviewer_feature_transfer": bool(args.allow_reviewer_feature_transfer),
                "initialization_sha256": sha256(args.initialize_from) if args.initialize_from else None,
                "primary_selection": "validation_" + args.primary_metric + "_then_other_validation_metric_then_bce_then_earliest_epoch",
                "test_selection": "exploratory_only_never_used_for_optimizer_hyperparameters_or_primary_checkpoint",
                "stopping_rule": "validation AUROC and validation Pair AUROC only; BCE cannot reset patience",
                "code_sha256": {str(path.relative_to(Path(__file__).parents[1])): sha256(path)
                                for path in [Path(__file__), Path(__file__).parents[1] / "civic/models.py",
                                             Path(__file__).parents[1] / "civic/objectives.py",
                                             Path(__file__).parents[1] / "civic/ranking.py",
                                             Path(__file__).parents[1] / "civic/evaluation.py"]}}
    manifest["targeted_auc_auxiliary"] = {"weight": args.pairwise_auc_weight,
        "formula": "mean softplus(-(positive_logit-negative_logit)) within actual training minibatch; zero if one class absent",
        "supervision_split": "train_only", "classification_objective_internal_auxiliary": True,
        "nonPython_labels_never_used_for_training_or_selection": args.training_language is not None}
    manifest["minor_refinement"] = {"name": "train_only_class_balanced_validity_auxiliary",
        "coefficient": args.balanced_validity_weight,
        "negative_weight": balanced_weights[0], "positive_weight": balanced_weights[1],
        "training_n": n_train, "training_positive": n_positive,
        "labels_not_in_forward": True, "same_seven_objectives_retained": True,
        "pr_identity_audit_sha256": sha256(args.output / "pr_identity_audit.json")}
    manifest['finding_filter_refinement'] = {
        'near_prior_pairwise': args.near_prior_pairwise,
        'pair_weight': '1+exp(-abs(real_frozen_positive_prior-real_frozen_negative_prior)); normalize by sum of weights',
        'hard_negative_definition': 'training label0 and real frozen reviewer logit>=0 only',
        'training_hard_negative_n': int(train_hard_negative.sum()),
        'training_negative_n': n_negative,
        'hard_negative_multiplier': args.hard_negative_multiplier,
        'auxiliary_weight_normalizer': auxiliary_weight_normalizer,
        'all_three_modules_and_original_seven_masked_objectives_unchanged': True,
        'labels_only_in_training_losses_never_forward': True,
        'original_validation_and_threshold_policies_unchanged': True}
    if args.near_prior_pairwise:
        manifest['targeted_auc_auxiliary']['formula'] = 'sum((1+exp(-abs(real_prior_pos-real_prior_neg)))*softplus(-(positive_logit-negative_logit)))/sum(pair weights) within actual training minibatch; zero if one class absent'
    save_json(args.output / "manifest.json", manifest)
    dense_best = {}
    language_best = {}
    if args.training_language is not None:
        language_array = dataset.metadata["language"].astype(str)
    else:
        language_array = None
    torch.save({"model": model.state_dict(), "config": config, "epoch": 0}, args.output / "initial.pt")
    neutral_prediction = predict(model, dataset, "val", args.eval_batch_size, device)
    neutral_validation = metrics(neutral_prediction, dataset)
    neutral_key = selection_key(neutral_validation, 0, args.primary_metric)
    best_key, best_epoch, best_validation = None, None, None
    best_stop_key, last_primary_improvement, epochs_trained = None, 0, 0
    np.savez_compressed(args.output / "neutral_validation_predictions.npz", **neutral_prediction)
    ii = neutral_prediction["indices"]
    base_logit = dataset.tensors["frozen_reviewer_logit"][ii].numpy()
    error = float(np.max(np.abs(neutral_prediction["logits"] - base_logit)))
    if error != 0:
        raise RuntimeError("Neutral initialized verifier must exactly reproduce reviewer")
    save_json(args.output / "neutral_identity_audit.json", {"maximum_absolute_logit_error": error,
              "selection_heads_neutral": True, "baseline_candidate_epoch": 0,
              "validation": neutral_validation, "standardization": "unchanged normalized frozen embeddings; no train/test fitted transform"})
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
            residual = output["residual_correction"].square().mean()
            gate_residual = (output["selection_logits"] - output["validity_logits"]).square().mean()
            losses["residual_penalty"] = args.residual_penalty * (residual + gate_residual)
            losses["total"] = losses["total"] + losses["residual_penalty"]
            class_weight = torch.where(batch["labels"] > 0.5,
                output["validity_logits"].new_tensor(balanced_weights[1]),
                output["validity_logits"].new_tensor(balanced_weights[0]))
            hard_negative = (batch['labels'] <= .5) & (batch['frozen_reviewer_logit'] >= 0.)
            class_weight = class_weight * torch.where(hard_negative,
                class_weight.new_tensor(args.hard_negative_multiplier), class_weight.new_tensor(1.))
            class_weight = class_weight / auxiliary_weight_normalizer
            balanced_bce = F.binary_cross_entropy_with_logits(
                output["validity_logits"], batch["labels"], reduction="none")
            losses["balanced_validity"] = args.balanced_validity_weight * (balanced_bce * class_weight).mean()
            losses["total"] = losses["total"] + losses["balanced_validity"]
            positives = output["validity_logits"][batch["labels"] > 0.5]
            negatives = output["validity_logits"][batch["labels"] <= 0.5]
            if len(positives) and len(negatives):
                pair_loss = F.softplus(-(positives[:, None] - negatives[None, :]))
                if args.near_prior_pairwise:
                    pos_prior = batch['frozen_reviewer_logit'][batch['labels'] > .5]
                    neg_prior = batch['frozen_reviewer_logit'][batch['labels'] <= .5]
                    pair_weight = 1. + torch.exp(-torch.abs(pos_prior[:, None] - neg_prior[None, :]))
                    auc_aux = (pair_loss * pair_weight).sum() / pair_weight.sum()
                else:
                    auc_aux = pair_loss.mean()
            else:
                auc_aux = output["validity_logits"].sum() * 0.0
            losses["classification_auc_auxiliary"] = args.pairwise_auc_weight * auc_aux
            losses["total"] = losses["total"] + losses["classification_auc_auxiliary"]
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
            dense_dir = args.output / "dense_predictions"
            dense_dir.mkdir(exist_ok=True)
            test_path = dense_dir / ("epoch_%04d_test.npz" % epoch)
            val_path = dense_dir / ("epoch_%04d_validation.npz" % epoch)
            np.savez_compressed(test_path, **prediction)
            np.savez_compressed(val_path, **validation_prediction)
            from civic.evaluation import validation_threshold, binary_metrics
            policies = {"fixed_logit_zero": {"threshold": 0.0, "selection_split": "fixed"},
                        "validation_f1": validation_threshold(validation_prediction["labels"],
                            validation_prediction["logits"], "f1"),
                        "validation_70recall": validation_threshold(validation_prediction["labels"],
                            validation_prediction["logits"], "recall", .7)}
            curve = validation_threshold(prediction["labels"], prediction["logits"], "recall", .7)
            curve.update(selection_split="test_metric_curve", evaluation_only=True,
                         not_a_deployment_threshold=True)
            policies["test_curve_70recall"] = curve
            improved = False
            for policy, threshold_info in policies.items():
                m = binary_metrics(prediction["labels"], prediction["logits"],
                    threshold_info["threshold"], true_pr_ids[prediction["indices"]])
                best = dense_best.setdefault(policy, {})
                for name in ["auroc", "ap", "precision", "recall", "f1", "accuracy",
                             "fp_per_pr", "fp_per_positive", "comments_per_pr"]:
                    if policy == "test_curve_70recall" and name not in ("fp_per_pr", "fp_per_positive"):
                        continue
                    value = m[name]
                    if value is None or (name in ["fp_per_pr", "fp_per_positive", "comments_per_pr"]
                                         and m["recall"] == 0):
                        continue
                    direction = "min" if name in ["fp_per_pr", "fp_per_positive", "comments_per_pr"] else "max"
                    old = best.get(name)
                    if old is None or (value < old["value"] if direction == "min" else value > old["value"]):
                        best[name] = {"value": value, "direction": direction, "epoch": epoch,
                            "checkpoint": "metric_best_weights/epoch_%04d.pt" % epoch,
                            "test_predictions": str(test_path.relative_to(args.output)),
                            "validation_predictions": str(val_path.relative_to(args.output)),
                            "threshold_policy": policy, "threshold_selection": threshold_info,
                            "metrics": m, "test_ids_sha256": test_ids_sha,
                            "test_selection_is_exploratory": True}
                        improved = True
            paired = exploratory.get("paired")
            if paired is not None:
                best = dense_best.setdefault("observed_control_pairs", {})
                for name in ("pair_accuracy", "pair_auroc"):
                    value = paired[name]
                    if name not in best or value > best[name]["value"]:
                        best[name] = {"value": value, "direction": "max", "epoch": epoch,
                            "checkpoint": "metric_best_weights/epoch_%04d.pt" % epoch,
                            "test_predictions": str(test_path.relative_to(args.output)),
                            "validation_predictions": str(val_path.relative_to(args.output)),
                            "threshold_policy": "observed_control_pairs",
                            "threshold_selection": {"threshold": 0.0, "selection_split": "fixed"},
                            "metrics": paired, "test_ids_sha256": test_ids_sha,
                            "test_selection_is_exploratory": True}
                        improved = True
            if language_array is not None:
                test_languages = language_array[prediction["indices"]]
                populations = {name: test_languages == name for name in np.unique(test_languages)}
                populations["Non-" + args.training_language + " pooled"] = test_languages != "Python"
                for language, mask in populations.items():
                    for policy, threshold_info in policies.items():
                        m = binary_metrics(prediction["labels"][mask], prediction["logits"][mask],
                            threshold_info["threshold"], true_pr_ids[prediction["indices"]][mask])
                        best = language_best.setdefault(language, {}).setdefault(policy, {})
                        for name in ["auroc", "ap", "precision", "recall", "f1", "accuracy",
                                     "fp_per_pr", "fp_per_positive", "comments_per_pr"]:
                            value = m[name]
                            if value is None or (name in ["fp_per_pr", "fp_per_positive", "comments_per_pr"]
                                                 and m["recall"] == 0):
                                continue
                            direction = "min" if name in ["fp_per_pr", "fp_per_positive", "comments_per_pr"] else "max"
                            old = best.get(name)
                            if old is None or (value < old["value"] if direction == "min" else value > old["value"]):
                                best[name] = {"value": value, "direction": direction, "epoch": epoch,
                                    "checkpoint": "metric_best_weights/epoch_%04d.pt" % epoch,
                                    "test_predictions": str(test_path.relative_to(args.output)),
                                    "validation_predictions": str(val_path.relative_to(args.output)),
                                    "threshold_policy": policy, "threshold_selection": threshold_info,
                                    "metrics": m, "language": language,
                                    "test_indices": prediction["indices"][mask].astype(int).tolist(),
                                    "test_ids_sha256": hashlib.sha256(("\n".join(dataset.metadata["case_id"][prediction["indices"][mask]]) + "\n").encode()).hexdigest(),
                                    "test_selection_is_exploratory": True}
                                improved = True
            if improved:
                weight_dir = args.output / "metric_best_weights"
                weight_dir.mkdir(exist_ok=True)
                weight_path = weight_dir / ("epoch_%04d.pt" % epoch)
                torch.save({"model": model.state_dict(), "config": config, "epoch": epoch,
                    "selection": "exploratory per-metric test best, formal val selection unchanged"}, weight_path)
                digest = sha256(weight_path)
                for metric_set in dense_best.values():
                    for item in metric_set.values():
                        if item["epoch"] == epoch:
                            item["checkpoint_sha256"] = digest
                            item["test_prediction_sha256"] = sha256(test_path)
                            item["validation_prediction_sha256"] = sha256(val_path)
                for populations in language_best.values():
                    for metric_set in populations.values():
                        for item in metric_set.values():
                            if item["epoch"] == epoch:
                                item["checkpoint_sha256"] = digest
                                item["test_prediction_sha256"] = sha256(test_path)
                                item["validation_prediction_sha256"] = sha256(val_path)
                save_json(args.output / "metric_best.json", dense_best)
                if language_best:
                    save_json(args.output / "language_metric_best.json", language_best)
                used_epochs = {item["epoch"] for metric_set in dense_best.values() for item in metric_set.values()}
                used_epochs |= {item["epoch"] for populations in language_best.values() for metric_set in populations.values() for item in metric_set.values()}
                for prior in weight_dir.glob("epoch_*.pt"):
                    if int(prior.stem.split("_")[1]) not in used_epochs:
                        prior.unlink()
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
              "elapsed_seconds": time.monotonic() - started,
              "neutral_validation": neutral_validation,
              "selected_kind": "trained_three_module_checkpoint",
              "validation_only_fallback_choice": "trained" if best_key > neutral_key else "reviewer_neutral",
              "neutral_selection_key": neutral_key,
              "residual_penalty_coefficient": args.residual_penalty}
    save_json(args.output / "results.json", result)
    print(json.dumps({"complete": True, "selected_epoch": best_epoch, "test": test}), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train and audit all three causal review modules")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--case-pr-map", type=Path)
    parser.add_argument("--training-language")
    parser.add_argument("--balanced-validity-weight", type=float, default=0.25)
    parser.add_argument("--pairwise-auc-weight", type=float, default=0.25)
    parser.add_argument('--near-prior-pairwise', action='store_true')
    parser.add_argument('--hard-negative-multiplier', type=float, default=1.)
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
    for module_name in ("theta", "phi", "psi"):
        parser.add_argument("--lr-" + module_name, type=float, default=None)
    parser.add_argument("--allow-dropout-transfer", action="store_true")
    parser.add_argument("--condition-on-reviewer", action="store_true")
    parser.add_argument("--allow-reviewer-feature-transfer", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--residual-penalty", type=float, default=0.1)
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
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.eval_batch_size, args.cpu_threads) < 1 or args.test_every < 0:
        parser.error("Training sizes must be positive; test-every may be zero")
    if args.min_epochs < 1 or args.patience < 0:
        parser.error("min-epochs must be positive; patience may be zero")
    return args



if __name__ == "__main__":
    train(parse_args())
