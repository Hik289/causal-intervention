import torch
from torch import nn

class InterventionRanker(nn.Module):
    pass
    def __init__(self, dz, ddelta, hidden=256):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dz+ddelta,hidden),nn.ReLU(),nn.Linear(hidden,1))
    def forward(self, z, h_delta):
        return self.net(torch.cat([z,h_delta],-1)).squeeze(-1)


class InterventionModule(nn.Module):


    def __init__(self, context_dim, code_dim, hidden=128, temperature=1.0):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)
        self.net = nn.Sequential(
            nn.Linear(context_dim + code_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, context, candidate_deltas, candidate_mask=None):
        if context.ndim != 2 or candidate_deltas.ndim != 3:
            raise ValueError("Expected context [batch, hidden] and candidate deltas [batch, candidates, code_dim]")
        if context.shape[0] != candidate_deltas.shape[0] or candidate_deltas.shape[1] == 0:
            raise ValueError("Each example must have at least one candidate")
        expanded = context.unsqueeze(1).expand(-1, candidate_deltas.shape[1], -1)
        logits = self.net(torch.cat((expanded, candidate_deltas), dim=-1)).squeeze(-1)
        if candidate_mask is not None:
            if candidate_mask.shape != logits.shape:
                raise ValueError("candidate_mask must have shape [batch, candidates]")
            mask = candidate_mask.to(device=logits.device, dtype=torch.bool)
            if not mask.any(dim=-1).all():
                raise ValueError("Each example must contain an unmasked candidate")
            logits = logits.masked_fill(~mask, float("-inf"))
        log_probabilities = torch.log_softmax(logits / self.temperature, dim=-1)
        probabilities = log_probabilities.exp()
        selected_index = logits.argmax(dim=-1)
        if self.training:
            selection_weights = probabilities
        else:
            selection_weights = torch.nn.functional.one_hot(selected_index, logits.shape[-1]).to(logits.dtype)
        return {
            "ranking_logits": logits,
            "ranking_log_probabilities": log_probabilities,
            "ranking_probabilities": probabilities,
            "selected_index": selected_index,
            "selection_weights": selection_weights,
        }


def select_critiques(probabilities, similarity=None, critique_features=None,
                    method="exact", exact_limit=18, max_findings=None,
                    lambda_size=0.0, lambda_red=1.0):






    if probabilities.ndim != 1 or not torch.isfinite(probabilities).all():
        raise ValueError("probabilities must be a finite one-dimensional tensor")
    if not torch.isfinite(torch.tensor([lambda_size, lambda_red])).all() or min(lambda_size, lambda_red) < 0:
        raise ValueError("Selection size and redundancy costs must be finite and nonnegative")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("probabilities must lie in [0, 1]")
    count = probabilities.numel()
    if similarity is None:
        if critique_features is None:
            similarity = probabilities.new_zeros((count, count))
        else:
            if critique_features.ndim != 2 or critique_features.shape[0] != count:
                raise ValueError("critique_features must have shape [findings, dimensions]")
            normalized = torch.nn.functional.normalize(critique_features, dim=-1)
            similarity = (normalized @ normalized.transpose(0, 1)).clamp(min=0, max=1)
    if similarity.shape != (count, count) or not torch.isfinite(similarity).all():
        raise ValueError("similarity must be a finite square matrix with one row per critique")
    if (similarity < 0).any():
        raise ValueError("Review redundancy penalties must be nonnegative")
    if not torch.allclose(similarity, similarity.transpose(0, 1), atol=1e-6, rtol=1e-5):
        raise ValueError("similarity must be symmetric")
    if max_findings is None:
        max_findings = count
    if max_findings < 0:
        raise ValueError("max_findings must be nonnegative")
    max_findings = min(int(max_findings), count)
    if exact_limit < 0:
        raise ValueError("exact_limit must be nonnegative")
    if method == "auto":
        method = "exact" if count <= exact_limit else "greedy"
    if method not in ("exact", "greedy"):
        raise ValueError("method must be 'exact', 'greedy', or 'auto'")
    if method == "exact" and count > exact_limit:
        raise ValueError("Exact selection exceeds exact_limit; explicitly request greedy or auto selection")
    scores = (probabilities.detach() - lambda_size).cpu().tolist()
    penalties = (lambda_red * similarity.detach()).cpu().tolist()
    if method == "greedy":
        selected = []
        remaining = set(range(count))
        while remaining and len(selected) < max_findings:
            index = max(remaining, key=lambda i: (scores[i] - sum(penalties[i][j] for j in selected), -i))
            gain = scores[index] - sum(penalties[index][j] for j in selected)
            if gain <= 0:
                break
            selected.append(index)
            remaining.remove(index)
    else:
        selected = []
        best = 0.0
        suffix = [0.0] * (count + 1)
        for index in range(count - 1, -1, -1):
            suffix[index] = suffix[index + 1] + max(0.0, scores[index])

        def search(index, current, value):
            nonlocal selected, best
            if value > best:
                selected, best = list(current), value
            if index == count or len(current) >= max_findings or value + suffix[index] <= best:
                return
            gain = scores[index] - sum(penalties[index][j] for j in current)
            if gain > 0:
                current.append(index)
                search(index + 1, current, value + gain)
                current.pop()
            search(index + 1, current, value)

        search(0, [], 0.0)
    indices = torch.tensor(sorted(selected), device=probabilities.device, dtype=torch.long)
    mask = torch.zeros(count, device=probabilities.device, dtype=torch.bool)
    mask[indices] = True
    pair_mask = torch.triu(mask.unsqueeze(1) & mask.unsqueeze(0), diagonal=1)
    utility = probabilities[mask].sum() - lambda_red * similarity[pair_mask].sum() - lambda_size * mask.sum()
    return {"indices": indices, "mask": mask, "utility": utility, "method": method}
