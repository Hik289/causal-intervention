from collections.abc import Mapping

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _vector(value, name, *, finite=True):
    array = np.asarray(value)
    if array.ndim != 1 or not len(array):
        raise ValueError(f'{name} must be a nonempty one-dimensional array')
    if finite:
        array = array.astype(np.float64)
        if not np.isfinite(array).all():
            raise ValueError(f'{name} must contain finite values')
    return array


def _labels(value):
    labels = _vector(value, 'labels')
    if not np.isin(labels, [0, 1]).all():
        raise ValueError('labels must contain only zero and one')
    return labels.astype(np.int64)


def _threshold(value):
    threshold = float(value)
    if np.isnan(threshold):
        raise ValueError('threshold must not be NaN')
    return threshold


def validation_threshold(labels, scores, objective='f1', target_recall=0.7):
    labels, scores = _labels(labels), _vector(scores, 'scores')
    if labels.shape != scores.shape:
        raise ValueError('labels and scores must have the same shape')
    if objective not in {'f1', 'youden', 'recall'}:
        raise ValueError('objective must be f1, youden, or recall')
    if not 0 < target_recall <= 1:
        raise ValueError('target_recall must be in (0, 1]')
    positive = int(labels.sum())
    negative = len(labels) - positive
    if objective in {'youden', 'recall'} and not positive:
        raise ValueError(f'{objective} calibration requires positive validation labels')
    if objective == 'youden' and not negative:
        raise ValueError('youden calibration requires negative validation labels')
    order = np.argsort(-scores, kind='stable')
    sorted_scores, sorted_labels = scores[order], labels[order]
    end = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), len(scores) - 1]
    tp = np.r_[0, np.cumsum(sorted_labels)[end]].astype(np.float64)
    retained = np.r_[0, end + 1].astype(np.float64)
    fp, fn = retained - tp, positive - tp
    thresholds = np.r_[np.inf, sorted_scores[end]]
    if objective == 'f1':
        denominator = 2 * tp + fp + fn
        values = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
        selected = int(np.argmax(values))
        value = float(values[selected])
    elif objective == 'youden':
        values = tp / positive - fp / negative
        selected = int(np.argmax(values))
        value = float(values[selected])
    else:
        eligible = np.flatnonzero(tp / positive >= target_recall)
        selected = int(eligible[0])
        value = float(tp[selected] / positive)
    return {
        'threshold': float(thresholds[selected]),
        'objective': objective,
        'objective_value': value,
        'target_recall': float(target_recall) if objective == 'recall' else None,
        'selection_split': 'validation',
        'tie_break': 'highest_threshold',
        'comparison': 'score >= threshold',
        'n_validation': len(labels),
    }


def binary_metrics(labels, scores, threshold, pr_ids=None, all_pr_ids=None):
    labels, scores = _labels(labels), _vector(scores, 'scores')
    if labels.shape != scores.shape:
        raise ValueError('labels and scores must have the same shape')
    threshold = _threshold(threshold)
    predicted = scores >= threshold
    positive = labels == 1
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & ~positive))
    tn = int(np.sum(~predicted & ~positive))
    fn = int(np.sum(~predicted & positive))
    n_positive, n_negative = int(positive.sum()), int((~positive).sum())
    undefined = {}
    auroc = float(roc_auc_score(labels, scores)) if n_positive and n_negative else None
    if auroc is None:
        undefined['auroc'] = 'both label classes are required'
    ap = float(average_precision_score(labels, scores)) if n_positive else None
    if ap is None:
        undefined['ap'] = 'positive labels are required'
    recall = tp / n_positive if n_positive else None
    fp_positive = fp / n_positive if n_positive else None
    if not n_positive:
        undefined['recall'] = undefined['fp_per_positive'] = 'positive labels are required'
    n_pr, fp_pr, comments_pr = None, None, None
    if pr_ids is None:
        if all_pr_ids is not None:
            raise ValueError('pr_ids are required when all_pr_ids are provided')
        undefined['fp_per_pr'] = undefined['comments_per_pr'] = 'no PR identifiers supplied'
    else:
        case_pr = _vector(pr_ids, 'pr_ids', finite=False)
        if len(case_pr) != len(labels):
            raise ValueError('pr_ids must identify every prediction')
        if any(value is None or str(value) == '' for value in case_pr):
            raise ValueError('pr_ids must not contain missing identifiers')
        case_pr = {str(value) for value in case_pr}
        if all_pr_ids is None:
            cohort_pr = case_pr
        else:
            all_pr = _vector(all_pr_ids, 'all_pr_ids', finite=False)
            if any(value is None or str(value) == '' for value in all_pr):
                raise ValueError('all_pr_ids must not contain missing identifiers')
            cohort_pr = {str(value) for value in all_pr}
            if not case_pr <= cohort_pr:
                raise ValueError('all_pr_ids must include every prediction PR')
        n_pr = len(cohort_pr)
        fp_pr, comments_pr = fp / n_pr, int(predicted.sum()) / n_pr
    denominator = 2 * tp + fp + fn
    return {
        'n': len(labels), 'n_positive': n_positive, 'n_negative': n_negative,
        'threshold': threshold, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'accuracy': (tp + tn) / len(labels),
        'precision': tp / (tp + fp) if tp + fp else 0.0,
        'recall': recall,
        'f1': 2 * tp / denominator if denominator else 0.0,
        'auroc': auroc, 'ap': ap, 'n_pr': n_pr,
        'fp_per_pr': fp_pr, 'fp_per_positive': fp_positive,
        'comments_per_pr': comments_pr, 'undefined': undefined,
    }


def paired_metrics(original_scores, target_scores, unrelated_scores, threshold,
                   flip_margin=0.0):
    original = _vector(original_scores, 'original_scores')
    target = _vector(target_scores, 'target_scores')
    if original.shape != target.shape:
        raise ValueError('original_scores and target_scores must have the same shape')
    unrelated = np.asarray(unrelated_scores, dtype=np.float64)
    if unrelated.ndim == 1:
        unrelated = unrelated[:, None]
    if unrelated.ndim != 2 or unrelated.shape[0] != len(original) or not unrelated.shape[1]:
        raise ValueError('unrelated_scores must contain actual scored interventions for every pair')
    if not np.isfinite(unrelated).all():
        raise ValueError('unrelated_scores must contain finite observed scores')
    threshold = _threshold(threshold)
    flip_margin = float(flip_margin)
    if not np.isfinite(flip_margin) or flip_margin < 0:
        raise ValueError('flip_margin must be finite and nonnegative')
    target_drop = original - target
    unrelated_drop = original[:, None] - unrelated
    labels = np.r_[np.ones(len(target_drop)), np.zeros(unrelated_drop.size)]
    effects = np.r_[target_drop, unrelated_drop.ravel()]
    return {
        'n_pairs': len(original), 'n_unrelated': unrelated_drop.size,
        'pair_accuracy': float(np.mean(original > target)),
        'pair_auroc': float(roc_auc_score(labels, effects)),
        'positive_recall': float(np.mean(original >= threshold)),
        'trap_rejection': float(np.mean(target < threshold)),
        'joint_threshold_accuracy': float(np.mean((original >= threshold) & (target < threshold))),
        'target_flip_rate': float(np.mean(target_drop > flip_margin)),
        'mean_target_drop': float(np.mean(target_drop)),
        'mean_unrelated_drop': float(np.mean(unrelated_drop)),
        'specificity_leakage': float(np.mean(np.abs(unrelated_drop))),
        'target_minus_unrelated': float(np.mean(target_drop) - np.mean(np.abs(unrelated_drop))),
        'threshold': threshold, 'flip_margin': flip_margin,
        'pair_auroc_reference': 'observed_unrelated_intervention_logit_drops',
    }


def retrieval_metrics(pool_scores, target_index, mask=None):
    scores = np.asarray(pool_scores, dtype=np.float64)
    targets = _vector(target_index, 'target_index')
    if scores.ndim != 2 or not scores.shape[0] or not scores.shape[1]:
        raise ValueError('pool_scores must be a nonempty query-by-candidate matrix')
    if len(targets) != scores.shape[0] or not np.equal(targets, np.floor(targets)).all():
        raise ValueError('target_index must contain one integer candidate index per query')
    targets = targets.astype(np.int64)
    if np.any(targets < 0) or np.any(targets >= scores.shape[1]):
        raise ValueError('target_index is outside the candidate pool')
    if mask is None:
        mask = np.ones(scores.shape, dtype=bool)
    else:
        mask = np.asarray(mask)
        if mask.dtype != np.bool_ or mask.shape != scores.shape:
            raise ValueError('mask must be a boolean matrix matching pool_scores')
    if not mask[np.arange(len(targets)), targets].all():
        raise ValueError('target candidates must not be masked')
    if not np.isfinite(scores[mask]).all():
        raise ValueError('unmasked candidate scores must be finite')
    top1, recall5, mrr, ranks = [], [], [], []
    for row, target, valid in zip(scores, targets, mask):
        greater = int(np.sum(row[valid] > row[target]))
        equal = int(np.sum(row[valid] == row[target]))
        positions = np.arange(greater + 1, greater + equal + 1, dtype=np.float64)
        top1.append(float(np.mean(positions <= 1)))
        recall5.append(float(np.mean(positions <= 5)))
        mrr.append(float(np.mean(1.0 / positions)))
        ranks.append(float(np.mean(positions)))
    counts = mask.sum(1)
    return {
        'n_queries': len(targets), 'top1': float(np.mean(top1)),
        'recall5': float(np.mean(recall5)), 'mrr': float(np.mean(mrr)),
        'mean_rank': float(np.mean(ranks)),
        'min_candidates': int(counts.min()), 'max_candidates': int(counts.max()),
        'tie_policy': 'uniform_expected_rank_and_reciprocal_rank',
    }


def _sigmoid(scores):
    result = np.empty_like(scores)
    positive = scores >= 0
    result[positive] = 1 / (1 + np.exp(-scores[positive]))
    exponential = np.exp(scores[~positive])
    result[~positive] = exponential / (1 + exponential)
    return result


def _coordinate(value):
    if isinstance(value, (list, np.ndarray)):
        return tuple(_coordinate(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _mapped_outputs(gold, predicted, maps, n, name, *, sets=False):
    provided = [value is not None for value in (gold, predicted, maps)]
    if not any(provided):
        return None
    if not all(provided):
        raise ValueError(f'{name} requires gold annotations, actual model predictions, and independent maps')
    if any(len(value) != n for value in (gold, predicted, maps)):
        raise ValueError(f'{name} annotations, predictions, and maps must match the evaluated cases')
    results = []
    for original, output, mapping in zip(gold, predicted, maps):
        if original is None or output is None or mapping is None:
            continue
        if not isinstance(mapping, Mapping):
            raise ValueError(f'{name} maps must be mappings from original to transformed coordinates')
        normalized = {_coordinate(key): _coordinate(value) for key, value in mapping.items()}
        original_coordinates = original if sets else [original]
        for coordinate in original_coordinates:
            coordinate = _coordinate(coordinate)
            if coordinate not in normalized and isinstance(coordinate, int) and str(coordinate) in normalized:
                normalized[coordinate] = normalized[str(coordinate)]
        if sets:
            original_set = {_coordinate(item) for item in original}
            if not original_set <= normalized.keys():
                raise ValueError(f'{name} maps must contain every supplied gold coordinate')
            expected = {normalized[item] for item in original_set}
            actual = {_coordinate(item) for item in output}
            denominator = len(expected) + len(actual)
            f1 = 2 * len(expected & actual) / denominator if denominator else 1.0
            results.append((float(expected == actual), f1))
        else:
            original = _coordinate(original)
            if original not in normalized:
                raise ValueError(f'{name} maps must contain the supplied gold coordinate')
            results.append(float(_coordinate(output) == normalized[original]))
    return results


def transformation_metrics(labels, original_scores, transformed_scores, threshold,
                           *, gold_locations=None, predicted_locations=None,
                           location_maps=None, gold_evidence=None,
                           predicted_evidence=None, evidence_maps=None):
    labels = _labels(labels)
    original = _vector(original_scores, 'original_scores')
    transformed = _vector(transformed_scores, 'transformed_scores')
    if labels.shape != original.shape or original.shape != transformed.shape:
        raise ValueError('labels and original/transformed scores must have the same shape')
    threshold = _threshold(threshold)
    initial, changed = original >= threshold, transformed >= threshold
    negative = labels == 0
    n_negative = int(negative.sum())
    initial_fp = int(np.sum(negative & initial))
    transformed_fp = int(np.sum(negative & changed))
    undefined = {}
    classification = binary_metrics(labels, transformed, threshold)
    locations = _mapped_outputs(gold_locations, predicted_locations, location_maps,
                                len(labels), 'location')
    evidence = _mapped_outputs(gold_evidence, predicted_evidence, evidence_maps,
                               len(labels), 'evidence', sets=True)
    if not locations:
        undefined['location_accuracy'] = 'gold locations, model predictions, and independent maps are required'
    if not evidence:
        undefined['evidence_exact'] = undefined['evidence_f1'] = 'gold support, model predictions, and independent maps are required'
    if not n_negative:
        undefined['fp_change_rate'] = 'negative labels are required'
    if not initial_fp:
        undefined['fp_reduction'] = 'the original model surfaced no false positives'
    if classification['auroc'] is None:
        undefined['auroc'] = classification['undefined']['auroc']
    return {
        'n': len(labels), 'threshold': threshold,
        'auroc': classification['auroc'],
        'verdict_consistency': float(np.mean(initial == changed)),
        'equivariance_validity': float(1 - np.mean(np.abs(_sigmoid(original) - _sigmoid(transformed)))),
        'location_accuracy': float(np.mean(locations)) if locations else None,
        'n_location_evaluable': len(locations) if locations is not None else 0,
        'evidence_exact': float(np.mean([item[0] for item in evidence])) if evidence else None,
        'evidence_f1': float(np.mean([item[1] for item in evidence])) if evidence else None,
        'n_evidence_evaluable': len(evidence) if evidence is not None else 0,
        'n_negative': n_negative, 'original_fp': initial_fp, 'transformed_fp': transformed_fp,
        'fp_change_rate': float(np.sum(negative & ~initial & changed) / n_negative) if n_negative else None,
        'fp_reduction': (initial_fp - transformed_fp) / initial_fp if initial_fp else None,
        'undefined': undefined,
    }
