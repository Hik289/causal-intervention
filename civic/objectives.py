from dataclasses import dataclass, fields

import torch
from torch.nn import functional as F


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
    mask = mask.to(device=values.device, dtype=torch.bool)
    weights = mask.to(values.dtype)
    return torch.where(mask, values, torch.zeros_like(values)).sum() / weights.sum().clamp_min(1)


def _grid_mask(value, reference, default=True):
    if value is None:
        return torch.full_like(reference, default, dtype=torch.bool)
    value = value.to(device=reference.device, dtype=torch.bool)
    if value.ndim == 1:
        value = value.unsqueeze(1).expand_as(reference)
    if value.shape != reference.shape:
        raise ValueError("Control and annotation masks must be [batch] or [batch, candidates]")
    return value


def review_selection_loss(probability, group_ids, critique_features,
                          method="exact", exact_limit=18, selection_logits=None,
                          labels=None, lambda_size=0.5, lambda_red=1.0):

    if group_ids.ndim != 1 or group_ids.shape != probability.shape:
        raise ValueError("group_ids must identify the review containing each critique")
    if method not in ("exact", "greedy", "auto"):
        raise ValueError("Unknown review selection method")
    if min(lambda_size, lambda_red) < 0:
        raise ValueError("Review selection penalties must be nonnegative")
    gate = probability if selection_logits is None else selection_logits.sigmoid()
    if selection_logits is not None and (labels is None or labels.shape != probability.shape):
        raise ValueError("A trainable selection gate requires validity labels")
    terms = []
    for group in torch.unique(group_ids):
        indices = (group_ids == group).nonzero(as_tuple=False).flatten()
        if method == "exact" and indices.numel() > exact_limit:
            raise ValueError("Review exceeds the exact subset limit; select greedy explicitly")
        features = critique_features[indices].detach()
        if features.ndim == 3:
            features = features.mean(dim=1)
        features = F.normalize(features, dim=-1)
        similarity = (features @ features.T).clamp(0, 1)
        selected = gate[indices]
        expected_pairs = (selected.unsqueeze(1) * selected.unsqueeze(0) * similarity).triu(1).sum()
        expected_utility = (selected * (probability[indices].detach() - lambda_size)).sum()
        terms.append((lambda_red * expected_pairs - expected_utility) / indices.numel())
    regularizer = torch.stack(terms).mean()
    if selection_logits is None:
        return regularizer
    return F.binary_cross_entropy_with_logits(selection_logits, labels.to(selection_logits.dtype)) + regularizer


def _line_supervision(logits, labels, available, token_mask):
    if labels is None or available is None:
        return logits.masked_fill(~token_mask, 0).sum() * 0
    available = available.to(device=logits.device, dtype=torch.bool)
    if available.shape != logits.shape[:1]:
        raise ValueError("Line supervision masks must have shape [batch]")
    labels = labels.to(logits.device)
    log_probability = F.log_softmax(logits, dim=-1)
    if labels.ndim == 1:
        known = available & (labels >= 0)
        safe = torch.where(known, labels, torch.zeros_like(labels)).long()
        if ((safe < 0) | (safe >= logits.shape[1])).any():
            raise ValueError("A supervised line index is outside the encoded program")
        if not token_mask.gather(1, safe.unsqueeze(1)).squeeze(1)[known].all():
            raise ValueError("A supervised line cannot be padding")
        values = -log_probability.gather(1, safe.unsqueeze(1)).squeeze(1)
    elif labels.shape == logits.shape:
        labels = labels.to(logits.dtype)
        if not torch.isfinite(labels).all() or (labels < 0).any():
            raise ValueError("Line target distributions must be finite and nonnegative")
        if (labels.masked_fill(token_mask, 0).sum(1)[available] > 0).any():
            raise ValueError("Supervised probability mass cannot be assigned to padding")
        mass = labels.sum(1)
        known = available & (mass > 0)
        distribution = labels / mass.unsqueeze(1).clamp_min(1e-12)
        values = -(distribution * log_probability.masked_fill(~token_mask, 0)).sum(1)
    else:
        raise ValueError("Line supervision must be indices [batch] or distributions [batch, lines]")
    return _masked_mean(values, known)


def _mapped_kl(original_probability, transformed_probability, mapping, available):
    if mapping is None:
        return original_probability.sum() * 0
    b, k, new_length = transformed_probability.shape
    old_length = original_probability.shape[1]
    if mapping.ndim == 3:
        mapping = mapping.unsqueeze(1).expand(-1, k, -1, -1)
    if mapping.shape != (b, k, old_length, new_length):
        raise ValueError("Line mappings must be [B,Lold,Lnew] or [B,K,Lold,Lnew]")
    mapping = mapping.to(device=original_probability.device, dtype=original_probability.dtype)
    if mapping.requires_grad or not torch.isfinite(mapping).all() or (mapping < 0).any():
        raise ValueError("Line mappings must be fixed, finite, and nonnegative")
    row_mass = mapping.sum(-1, keepdim=True)
    mapping = mapping / row_mass.clamp_min(1e-12)
    mapped = torch.einsum("bi,bkij->bkj", original_probability, mapping)
    mass = mapped.sum(-1)
    mapped = mapped / mass.unsqueeze(-1).clamp_min(1e-12)
    available = available & (mass > 0)
    kl = (mapped * (mapped.clamp_min(1e-12).log() -
                    transformed_probability.clamp_min(1e-12).log())).sum(-1)
    return _masked_mean(kl, available)


def joint_objectives(outputs, batch, weights=None, margin=0.5,
                     selection_method="exact", exact_limit=18,
                     lambda_size=0.5, lambda_red=1.0,
                     auxiliary_classification_weight=0.5,
                     localization_weight=1.0, support_weight=1.0,
                     minimality_cost_scale=1.0):
    if margin <= 0 or minimality_cost_scale <= 0:
        raise ValueError("The target margin and minimality cost scale must be positive")
    weights = weights or LossWeights()
    required_outputs = {"validity_logits", "original_score", "target_scores",
                        "unrelated_scores", "transformed_scores", "ranking_logits",
                        "edit_distances", "probability"}
    required_batch = {"labels", "target_index", "group_id", "critique", "transformed"}
    if required_outputs - outputs.keys() or required_batch - batch.keys():
        raise ValueError("Joint objectives require state scores, labels, target annotations, and review groups")
    labels = batch["labels"].to(outputs["validity_logits"].dtype)
    target_index = batch["target_index"].long()
    mask = _grid_mask(batch.get("candidate_mask"), outputs["ranking_logits"])
    if not mask.any(dim=1).all():
        raise ValueError("Each example requires an unmasked candidate")
    known_target = (target_index >= 0) & (labels > 0.5)
    safe_index = torch.where(known_target, target_index, mask.long().argmax(dim=1))
    if ((safe_index < 0) | (safe_index >= mask.shape[1])).any():
        raise ValueError("A verified target index is out of range")
    if not mask.gather(1, safe_index.unsqueeze(1)).squeeze(1).all():
        raise ValueError("A verified target index cannot be masked")
    verified_target_score = outputs["target_scores"].gather(1, safe_index.unsqueeze(1)).squeeze(1)
    original = outputs["original_score"]
    original_grid = original.unsqueeze(1)
    ranking_logits = outputs["ranking_logits"].masked_fill(~mask, -torch.inf)
    ranking_log_probability = outputs.get("ranking_log_probabilities")
    if ranking_log_probability is None:
        ranking_log_probability = F.log_softmax(ranking_logits, dim=1)
    ranking_probability = ranking_log_probability.exp()
    ranking_nll = -ranking_log_probability.gather(1, safe_index.unsqueeze(1)).squeeze(1)
    flip = _masked_mean(F.relu(margin + verified_target_score - original), known_target)
    reverse_index = batch.get("reverse_index")
    if reverse_index is not None:
        reverse = reverse_index.long()
        verified_reverse = batch.get("reverse_verified_mask", torch.zeros_like(labels, dtype=torch.bool)).bool()
        if reverse.shape != labels.shape or verified_reverse.shape != labels.shape:
            raise ValueError("Reverse intervention annotations must have shape [batch]")
        reverse_known = (labels < 0.5) & verified_reverse & (reverse >= 0)
        reverse_safe = torch.where(reverse_known, reverse, mask.long().argmax(1))
        if ((reverse_safe < 0) | (reverse_safe >= mask.shape[1])).any():
            raise ValueError("A verified reverse intervention index is out of range")
        if not mask.gather(1, reverse_safe.unsqueeze(1)).squeeze(1).all():
            raise ValueError("A reverse intervention cannot be masked")
        induced_score = outputs["target_scores"].gather(1, reverse_safe.unsqueeze(1)).squeeze(1)
        flip = flip + _masked_mean(F.relu(margin + original - induced_score), reverse_known)
    unrel_mask = mask & _grid_mask(outputs.get("unrelated_available_mask"), mask)
    eqv_mask = mask & _grid_mask(outputs.get("transformed_available_mask"), mask)
    unrel_mask &= _grid_mask(batch.get("unrelated_available_mask"), mask)
    eqv_mask &= _grid_mask(batch.get("transformed_available_mask"), mask)
    equivariance = _masked_mean((outputs["transformed_scores"] - original_grid).abs(), eqv_mask)
    states = outputs.get("state_predictions")
    if states is not None:
        before, transformed = states["original"], states["transformed"]
        for name, coefficient in (("location", localization_weight), ("support", support_weight)):
            supervised = _line_supervision(
                before[f"{name}_logits"], batch.get(f"{name}_labels"),
                batch.get(f"{name}_supervision_mask"), before["token_mask"],
            )
            mapped = original.sum() * 0
            if transformed is not None:
                mapping_mask = eqv_mask & _grid_mask(batch.get(f"{name}_mapping_mask"), mask, default=False)
                mapped = _mapped_kl(
                    before[f"{name}_probabilities"], transformed[f"{name}_probabilities"],
                    batch.get(f"transformed_{name}_map"), mapping_mask,
                )
            equivariance = equivariance + coefficient * (supervised + mapped)
    if outputs.get("edit_cost_source") == "program_edit_cost":
        verified = batch.get("candidate_verified_mask")
        if verified is None:
            verified = torch.zeros_like(mask)
            verified.scatter_(1, safe_index.unsqueeze(1), known_target.unsqueeze(1))
        else:
            verified = _grid_mask(verified, mask) & mask & (labels > 0.5).unsqueeze(1)
        cost = outputs["edit_distances"]
        if known_target.any() and not verified.gather(1, safe_index.unsqueeze(1)).squeeze(1)[known_target].all():
            raise ValueError("Verified target indices must be included in candidate_verified_mask")
        cost_available = _grid_mask(outputs.get("edit_cost_available_mask"), mask)
        if (verified & cost_available & (cost <= 0)).any():
            raise ValueError("A verified defect-removing repair cannot be a no-op")
        verified = verified & cost_available
        has_verified = verified.any(1)
        restricted_logits = ranking_logits.masked_fill(~verified, -torch.inf)
        restricted_logits = torch.where(has_verified.unsqueeze(1), restricted_logits,
                                        torch.zeros_like(restricted_logits))
        conditional_probability = restricted_logits.softmax(1) * verified.to(cost.dtype)
        invalid_mass = (ranking_probability * (mask & ~verified).to(cost.dtype)).sum(1)
        validity_barrier = cost.masked_fill(~verified, 0).max(1).values + margin
        minimality = _masked_mean(
            ((conditional_probability * cost).sum(1) + validity_barrier * invalid_mass)
            / minimality_cost_scale, has_verified,
        )
    else:
        minimality = (ranking_probability * outputs["edit_distances"]).sum(1).mean() / minimality_cost_scale
    losses = {
        "classification": F.binary_cross_entropy_with_logits(outputs["validity_logits"], labels) +
            auxiliary_classification_weight * F.binary_cross_entropy_with_logits(original, labels),
        "flip": flip,
        "specificity": _masked_mean((outputs["unrelated_scores"] - original_grid).abs(), unrel_mask),
        "equivariance": equivariance,
        "ranking": _masked_mean(ranking_nll, known_target),
        "minimality": minimality,
        "selection": review_selection_loss(
            outputs["probability"], batch["group_id"], batch["critique"],
            method=selection_method, exact_limit=exact_limit,
            selection_logits=outputs.get("selection_logits"), labels=labels,
            lambda_size=lambda_size, lambda_red=lambda_red,
        ),
    }
    losses["total"] = sum(getattr(weights, name) * value for name, value in losses.items())
    return losses
