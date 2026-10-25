from dataclasses import dataclass, fields

import torch
from torch.nn import functional as F

from civic.ranking import select_critiques


@dataclass(frozen=True)
class LossWeights:
    classification: float = 1.0
    flip: float = 1.0
    specificity: float = 1.0
    equivariance: float = 1.0
    ranking: float = 1.0
    minimality: float = 1.0
    selection: float = 1.0

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if not torch.isfinite(torch.tensor(value)) or value < 0:
                raise ValueError(f"Loss weight {field.name} must be finite and nonnegative")


def _masked_mean(values, mask):
    weights = mask.to(values.dtype)
    return torch.where(mask, values, torch.zeros_like(values)).sum() / weights.sum().clamp_min(1)


def review_selection_loss(probability, group_ids, critique_features,
                          method="exact", exact_limit=18):
    if group_ids.ndim != 1 or group_ids.shape != probability.shape:
        raise ValueError("group_ids must identify the review containing each critique")
    utility = []
    for group in torch.unique(group_ids):
        indices = (group_ids == group).nonzero(as_tuple=False).flatten()
        if method == "exact" and indices.numel() > exact_limit:
            raise ValueError("Review exceeds the exact subset limit; select greedy explicitly")
        group_probability = probability[indices]
        features = critique_features[indices].detach()
        if features.ndim == 3:
            features = features.mean(dim=1)
        features = F.normalize(features, dim=-1)
        similarity = (features @ features.T).clamp(0, 1)
        selected = select_critiques(
            group_probability.detach(), similarity=similarity,
            method=method, exact_limit=exact_limit,
        )
        chosen = torch.as_tensor(selected["indices"], device=indices.device,
                                 dtype=torch.long)
        if chosen.numel():
            pair_similarity = similarity[chosen][:, chosen].triu(diagonal=1).sum()
            utility.append(group_probability[chosen].sum() - pair_similarity)
        else:
            utility.append(group_probability.sum() * 0)
    return -torch.stack(utility).mean()


def joint_objectives(outputs, batch, weights=None, margin=0.5,
                     selection_method="exact", exact_limit=18):
    if margin <= 0:
        raise ValueError("The target-response margin must be positive")
    weights = weights or LossWeights()
    required_outputs = {"validity_logits", "original_score", "target_scores",
                        "unrelated_scores", "transformed_scores", "ranking_logits",
                        "edit_distances", "probability"}
    required_batch = {"labels", "target_index", "group_id", "critique", "transformed"}
    if required_outputs - outputs.keys() or required_batch - batch.keys():
        raise ValueError("Joint objectives require all state scores, transformed features, labels, and review groups")
    labels = batch["labels"].to(outputs["validity_logits"].dtype)
    target_index = batch["target_index"].long()
    mask = batch.get("candidate_mask")
    if mask is None:
        mask = torch.ones_like(outputs["ranking_logits"], dtype=torch.bool)
    else:
        mask = mask.bool()
    known_target = (target_index >= 0) & (labels > 0.5)
    safe_index = torch.where(known_target, target_index, mask.long().argmax(dim=1))
    verified_target_score = outputs["target_scores"].gather(
        1, safe_index.unsqueeze(1)).squeeze(1)
    original = outputs["original_score"]
    original_grid = original.unsqueeze(1)
    ranking_logits = outputs["ranking_logits"].masked_fill(~mask, -torch.inf)
    ranking_log_probability = outputs.get("ranking_log_probabilities")
    if ranking_log_probability is None:
        ranking_log_probability = F.log_softmax(ranking_logits, dim=1)
    ranking_probability = ranking_log_probability.exp()
    ranking_nll = -ranking_log_probability.gather(
        1, safe_index.unsqueeze(1)).squeeze(1)
    losses = {
        "classification": F.binary_cross_entropy_with_logits(
            outputs["validity_logits"], labels),
        "flip": _masked_mean(
            F.relu(margin + verified_target_score - original), known_target),
        "specificity": _masked_mean(
            (outputs["unrelated_scores"] - original_grid).abs(), mask),
        "equivariance": _masked_mean(
            (outputs["transformed_scores"] - original_grid).abs(), mask),
        "ranking": _masked_mean(ranking_nll, known_target),
        "minimality": (
            ranking_probability * outputs["edit_distances"]
        ).sum(dim=1).mean(),
        "selection": review_selection_loss(
            outputs["probability"], batch["group_id"], batch["critique"],
            method=selection_method, exact_limit=exact_limit,
        ),
    }
    losses["total"] = sum(getattr(weights, name) * value
                          for name, value in losses.items())
    return losses
