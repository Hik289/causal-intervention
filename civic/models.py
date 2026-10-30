pass

import torch, torch.nn as nn

class MLPVerifier(nn.Module):
    def __init__(self, dc, dt, hidden=512, dropout=0.1):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dc+dt, hidden), nn.ReLU(), nn.Dropout(dropout),
                               nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.head=nn.Linear(hidden,1)
    def forward(self, hx, hc):
        z=self.net(torch.cat([hx,hc],-1)); return self.head(z).squeeze(-1), z

class BilinearVerifier(nn.Module):
    def __init__(self, dc, dt, hidden=512):
        super().__init__()
        self.px=nn.Linear(dc,hidden); self.pc=nn.Linear(dt,hidden)
        self.W=nn.Bilinear(hidden,hidden,1)
    def forward(self, hx, hc):
        zx=self.px(hx); zc=self.pc(hc); return self.W(zx,zc).squeeze(-1), zx*zc

class CrossAttnVerifier(nn.Module):
    pass

    def __init__(self, dc, dt, hidden=512, layers=4, heads=8, dropout=0.1):
        super().__init__()
        self.px=nn.Linear(dc,hidden); self.pc=nn.Linear(dt,hidden)
        self.type_emb=nn.Parameter(torch.randn(2,hidden)*0.02)
        enc=nn.TransformerEncoderLayer(hidden,heads,hidden*2,dropout,batch_first=True)
        self.tr=nn.TransformerEncoder(enc,layers)
        self.head=nn.Linear(hidden,1)
    def forward(self, hx, hc):
        zx=self.px(hx)+self.type_emb[0]; zc=self.pc(hc)+self.type_emb[1]
        seq=torch.stack([zx,zc],1)
        z=self.tr(seq)
        pooled=z.mean(1)
        return self.head(pooled).squeeze(-1), pooled

def build(name, dc, dt, **kw):
    if name=="mlp": return MLPVerifier(dc,dt,**{k:v for k,v in kw.items() if k in ('hidden','dropout')})
    if name=="bilinear": return BilinearVerifier(dc,dt,hidden=kw.get('hidden',512))
    if name=="crossattn": return CrossAttnVerifier(dc,dt,**{k:v for k,v in kw.items() if k in ('hidden','layers','heads','dropout')})
    raise ValueError(name)

def n_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


class CritiqueModule(nn.Module):


    def __init__(self, code_dim, critique_dim, paired_dim=None, hidden=128,
                 layers=2, heads=4, dropout=0.1):
        super().__init__()
        if hidden < 2 or hidden % heads:
            raise ValueError("hidden must be at least two and divisible by heads")
        self.paired_dim = paired_dim
        self.code_projection = nn.Linear(code_dim, hidden)
        self.critique_projection = nn.Linear(critique_dim, hidden)
        self.paired_projection = nn.Linear(paired_dim, hidden) if paired_dim is not None else None
        self.type_embedding = nn.Parameter(torch.randn(3, hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(
            hidden, heads, hidden * 2, dropout=dropout,
            activation="gelu", batch_first=True,
        )
        self.cross_attention = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.location_head = nn.Linear(hidden, 1)
        self.support_head = nn.Linear(hidden, 1)
        self.scorer = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1),
        )

    @staticmethod
    def _tokens(features, name):
        if features.ndim == 2:
            features = features.unsqueeze(1)
        if features.ndim != 3 or features.shape[1] == 0:
            raise ValueError(name + " must be [batch, dimensions] or [batch, tokens, dimensions]")
        if features.requires_grad:
            raise ValueError(name + " must contain frozen features without requires_grad")
        return features

    @staticmethod
    def _mask(mask, features, name):
        if mask is None:
            return torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        mask = mask.to(device=features.device, dtype=torch.bool)
        if mask.ndim == 1 and features.shape[1] == 1:
            mask = mask.unsqueeze(1)
        if mask.shape != features.shape[:2] or not mask.any(dim=1).all():
            raise ValueError(name + " must identify at least one valid token per example")
        return mask

    def forward(self, code_features, critique_features, paired_features=None,
                code_mask=None, critique_mask=None, paired_mask=None):
        code = self._tokens(code_features, "code_features")
        critique = self._tokens(critique_features, "critique_features")
        if code.shape[0] != critique.shape[0]:
            raise ValueError("Code and critique features must share their batch size")
        code_mask = self._mask(code_mask, code, "code_mask")
        critique_mask = self._mask(critique_mask, critique, "critique_mask")
        tokens = [self.code_projection(code) + self.type_embedding[0],
                  self.critique_projection(critique) + self.type_embedding[1]]
        masks = [code_mask, critique_mask]
        if self.paired_projection is not None:
            if paired_features is None:
                raise ValueError("paired_features are required when paired_dim is configured")
            paired = self._tokens(paired_features, "paired_features")
            if paired.shape[0] != code.shape[0]:
                raise ValueError("Paired features must share the code batch size")
            tokens.append(self.paired_projection(paired) + self.type_embedding[2])
            masks.append(self._mask(paired_mask, paired, "paired_mask"))
        elif paired_features is not None:
            raise ValueError("Configure paired_dim before supplying paired_features")
        mask = torch.cat(masks, dim=1)
        encoded = self.cross_attention(torch.cat(tokens, dim=1), src_key_padding_mask=~mask)
        code_encoded = encoded[:, :code.shape[1]]
        location_logits = self.location_head(code_encoded).squeeze(-1).masked_fill(~code_mask, -torch.inf)
        support_logits = self.support_head(code_encoded).squeeze(-1).masked_fill(~code_mask, -torch.inf)
        location_probabilities = torch.softmax(location_logits, dim=-1)
        support_probabilities = torch.softmax(support_logits, dim=-1)
        location_context = (location_probabilities.unsqueeze(-1) * code_encoded).sum(dim=1)
        support_context = (support_probabilities.unsqueeze(-1) * code_encoded).sum(dim=1)
        mean_context = (encoded * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        representation = (mean_context + location_context + support_context) / 3
        return {
            "score": self.scorer(representation).squeeze(-1),
            "representation": representation,
            "location_logits": location_logits,
            "location_probabilities": location_probabilities,
            "support_logits": support_logits,
            "support_probabilities": support_probabilities,
            "token_mask": code_mask,
        }


class VerifierModule(nn.Module):


    def __init__(self, hidden=128):
        super().__init__()
        self.response_encoder = nn.Sequential(nn.Linear(6, hidden), nn.GELU())
        self.network = nn.Sequential(
            nn.Linear(7 + hidden * 2, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.validity_head = nn.Linear(hidden, 1)
        self.selection_head = nn.Linear(hidden, 1)

    def forward(self, original_score, target_scores, unrelated_scores,
                transformed_scores, selection_weights, candidate_mask=None,
                unrelated_available_mask=None, transformed_available_mask=None,
                frozen_reviewer_logit=None):
        shape = target_scores.shape
        mask = torch.ones(shape, dtype=torch.bool, device=target_scores.device) \
            if candidate_mask is None else candidate_mask.bool()

        def availability(value):
            if value is None:
                return mask
            if value.ndim == 1:
                value = value.unsqueeze(1).expand_as(mask)
            if value.shape != mask.shape:
                raise ValueError("Response availability masks must be [batch] or [batch, candidates]")
            return value.bool() & mask

        unrel_mask = availability(unrelated_available_mask)
        eqv_mask = availability(transformed_available_mask)
        available = torch.stack((mask, unrel_mask, eqv_mask), dim=-1)
        raw_responses = torch.stack((original_score.unsqueeze(1) - target_scores,
                                    original_score.unsqueeze(1) - unrelated_scores,
                                    original_score.unsqueeze(1) - transformed_scores), dim=-1)
        responses = torch.where(available, raw_responses, torch.zeros_like(raw_responses))
        flags = available.to(responses.dtype)
        encoded = self.response_encoder(torch.cat((responses, flags), dim=-1))
        selected_responses = (selection_weights.unsqueeze(-1) * responses).sum(dim=1)
        selected_flags = (selection_weights.unsqueeze(-1) * flags).sum(dim=1)
        selected_context = (selection_weights.unsqueeze(-1) * encoded).sum(dim=1)
        pooled_context = (encoded * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        inputs = torch.cat((original_score.unsqueeze(-1), selected_responses, selected_flags,
                            selected_context, pooled_context), dim=-1)
        hidden = self.network(inputs)
        correction = self.validity_head(hidden).squeeze(-1)
        if frozen_reviewer_logit is None:
            logits = correction
        else:
            if frozen_reviewer_logit.shape != original_score.shape or frozen_reviewer_logit.requires_grad:
                raise ValueError("frozen_reviewer_logit must be a frozen [batch] tensor")
            if not torch.isfinite(frozen_reviewer_logit).all():
                raise ValueError("frozen_reviewer_logit must be finite")
            logits = frozen_reviewer_logit + correction
        gate_logits = logits + self.selection_head(hidden).squeeze(-1)
        return {
            "validity_logits": logits, "probability": torch.sigmoid(logits),
            "residual_correction": correction,
            "selection_logits": gate_logits, "selection_probability": torch.sigmoid(gate_logits),
            "target_response": selected_responses[:, 0],
            "unrelated_response": selected_responses[:, 1],
            "transformed_response": selected_responses[:, 2],
            "response_available": selected_flags,
        }


class CIVER(nn.Module):







    def __init__(self, code_dim, critique_dim, paired_dim=None, hidden=128,
                 layers=2, heads=4, dropout=0.1, rank_hidden=128,
                 verifier_hidden=128, temperature=1.0):
        super().__init__()
        from .ranking import InterventionModule
        self.code_dim = code_dim
        self.paired_dim = paired_dim
        self.critique = CritiqueModule(code_dim, critique_dim, paired_dim,
                                      hidden, layers, heads, dropout)
        self.intervention = InterventionModule(hidden, code_dim, rank_hidden, temperature)
        self.verifier = VerifierModule(verifier_hidden)

    @staticmethod
    def _repeat(features, count):
        if features is None:
            return None
        return features.unsqueeze(1).expand(-1, count, *features.shape[1:]).reshape(
            features.shape[0] * count, *features.shape[1:])

    @staticmethod
    def _expand_control(features, reference, name):
        if features is None:
            return None
        if features.ndim == reference.ndim - 1:
            features = features.unsqueeze(1).expand(-1, reference.shape[1], *features.shape[1:])
        if features.ndim != reference.ndim or features.shape[:2] != reference.shape[:2]:
            raise ValueError(name + " must contain a shared state or one state per candidate")
        if features.shape[-1] != reference.shape[-1]:
            raise ValueError(name + " feature dimensions do not match the candidates")
        return features

    @staticmethod
    def _candidate_mask(mask, batch, count, device, default=True):
        if mask is None:
            return torch.full((batch, count), default, dtype=torch.bool, device=device)
        mask = mask.to(device=device, dtype=torch.bool)
        if mask.ndim == 1:
            mask = mask.unsqueeze(1).expand(-1, count)
        if mask.shape != (batch, count):
            raise ValueError("Availability masks must be [batch] or [batch, candidates]")
        return mask

    @staticmethod
    def _pooled_code(features, mask=None):
        if features.ndim == 2:
            return features
        if mask is None:
            return features.mean(dim=-2)
        weights = mask.to(features.dtype).unsqueeze(-1)
        return (features * weights).sum(-2) / weights.sum(-2).clamp_min(1)

    def forward(self, original, critique, candidates, unrelated, transformed,
                pair_original=None, pair_candidates=None, pair_unrelated=None,
                pair_transformed=None, candidate_mask=None,
                transformed_critique=None, candidate_edit_costs=None,
                candidate_edit_cost_mask=None,
                candidate_diff_features=None, frozen_reviewer_logit=None,
                original_mask=None, critique_mask=None, candidate_token_mask=None,
                unrelated_token_mask=None, transformed_token_mask=None,
                transformed_critique_mask=None, pair_original_mask=None,
                pair_candidates_mask=None, pair_unrelated_mask=None,
                pair_transformed_mask=None, unrelated_available_mask=None,
                transformed_available_mask=None):
        if original.ndim not in (2, 3) or candidates.ndim not in (3, 4):
            raise ValueError("Original and candidate features must be pooled vectors or token sequences")
        batch, count = candidates.shape[:2]
        if batch != original.shape[0] or critique.shape[0] != batch or count == 0:
            raise ValueError("All states must share a batch size and include at least one candidate")
        if original.shape[-1] != self.code_dim or candidates.shape[-1] != self.code_dim:
            raise ValueError("Code feature dimensions do not match code_dim")
        unrelated = self._expand_control(unrelated, candidates, "unrelated")
        transformed = self._expand_control(transformed, candidates, "transformed")
        cmask = self._candidate_mask(candidate_mask, batch, count, original.device)
        unrel_available = self._candidate_mask(
            unrelated_available_mask, batch, count, original.device, unrelated is not None) & cmask
        eqv_available = self._candidate_mask(
            transformed_available_mask, batch, count, original.device, transformed is not None) & cmask
        if unrelated is None and unrel_available.any() or transformed is None and eqv_available.any():
            raise ValueError("A missing control state cannot be marked available")
        pairs = (pair_original, pair_candidates, pair_unrelated, pair_transformed)
        if self.paired_dim is not None:
            required = [pair_original, pair_candidates]
            if unrelated is not None:
                required.append(pair_unrelated)
            if transformed is not None:
                required.append(pair_transformed)
            if any(value is None for value in required):
                raise ValueError("Each program state needs its own paired features when paired_dim is configured")
            if pair_candidates.ndim not in (3, 4) or pair_candidates.shape[:2] != (batch, count):
                raise ValueError("pair_candidates must contain paired features for each candidate")
            if any(value.shape[-1] != self.paired_dim for value in required):
                raise ValueError("Paired feature dimensions do not match paired_dim")
            pair_unrelated = self._expand_control(pair_unrelated, pair_candidates, "pair_unrelated")
            pair_transformed = self._expand_control(pair_transformed, pair_candidates, "pair_transformed")
        elif any(value is not None for value in pairs):
            raise ValueError("Configure paired_dim before supplying paired features")

        def expand_token_mask(mask, features):
            if mask is None or features is None:
                return None
            if features.ndim == 3:
                if mask.shape == (batch, count):
                    return mask.reshape(batch * count, 1)
                if mask.shape == (batch, 1):
                    return self._repeat(mask, count)
            if features.ndim == 4:
                if mask.shape == (batch, features.shape[2]):
                    return self._repeat(mask, count)
                if mask.shape == features.shape[:3]:
                    return mask.reshape(batch * count, features.shape[2])
            raise ValueError("State token masks must match the supplied feature positions")

        original_output = self.critique(original, critique, pair_original,
                                        original_mask, critique_mask, pair_original_mask)
        repeated_critique = self._repeat(critique, count)
        repeated_critique_mask = self._repeat(critique_mask, count)

        def state_output(features, paired, token_mask, pair_mask, state_critique=None,
                         state_critique_mask=None, available=None):
            if features is None:
                return None
            flat = features.reshape(batch * count, *features.shape[2:])
            pair_flat = None if paired is None else paired.reshape(batch * count, *paired.shape[2:])
            current_critique = repeated_critique if state_critique is None else self._repeat(state_critique, count)
            current_critique_mask = repeated_critique_mask if state_critique is None else self._repeat(state_critique_mask, count)
            active = (cmask if available is None else available).reshape(-1)
            code_tokens = features.shape[2] if features.ndim == 4 else 1
            if not active.any():
                scalar = features.new_zeros(batch, count)
                positions = features.new_zeros(batch, count, code_tokens)
                return {
                    "score": scalar, "representation": features.new_zeros(batch, count, original_output["representation"].shape[-1]),
                    "location_logits": positions, "location_probabilities": positions,
                    "support_logits": positions, "support_probabilities": positions,
                    "token_mask": torch.zeros(batch, count, code_tokens, dtype=torch.bool, device=features.device),
                }
            code_tokens_mask = expand_token_mask(token_mask, features)
            pair_tokens_mask = expand_token_mask(pair_mask, paired)
            subset = lambda value: None if value is None else value[active]
            output = self.critique(flat[active], current_critique[active], subset(pair_flat),
                                   subset(code_tokens_mask), subset(current_critique_mask),
                                   subset(pair_tokens_mask))
            restored = {}
            for key, value in output.items():
                complete = value.new_zeros(batch * count, *value.shape[1:])
                complete[active] = value
                restored[key] = complete.reshape(batch, count, *value.shape[1:])
            return restored

        target_output = state_output(candidates, pair_candidates, candidate_token_mask, pair_candidates_mask)
        unrelated_output = state_output(unrelated, pair_unrelated, unrelated_token_mask,
                                        pair_unrelated_mask, available=unrel_available)
        transformed_output = state_output(transformed, pair_transformed, transformed_token_mask,
                                          pair_transformed_mask, transformed_critique,
                                          transformed_critique_mask, available=eqv_available)
        base_code = self._pooled_code(original, original_mask)
        candidate_code = candidates
        if candidates.ndim == 4:
            flat_mask = expand_token_mask(candidate_token_mask, candidates)
            pooled = self._pooled_code(candidates.reshape(batch * count, *candidates.shape[2:]), flat_mask)
            candidate_code = pooled.reshape(batch, count, -1)
        deltas = candidate_code - base_code.unsqueeze(1)
        if candidate_diff_features is not None:
            if candidate_diff_features.shape != deltas.shape or candidate_diff_features.requires_grad:
                raise ValueError("candidate_diff_features must be frozen [batch, candidates, code_dim]")
            rank_features = candidate_diff_features
        else:
            rank_features = deltas
        ranking = self.intervention(original_output["representation"], rank_features, cmask)
        unrel_scores = torch.zeros_like(target_output["score"]) if unrelated_output is None else unrelated_output["score"]
        eqv_scores = torch.zeros_like(target_output["score"]) if transformed_output is None else transformed_output["score"]
        verification = self.verifier(
            original_output["score"], target_output["score"], unrel_scores, eqv_scores,
            ranking["selection_weights"], cmask, unrel_available, eqv_available,
            frozen_reviewer_logit,
        )
        if candidate_edit_costs is None:
            edit_distances = deltas.abs().sum(dim=-1)
            edit_source = "representation_proxy"
            edit_available = cmask
        else:
            if candidate_edit_costs.shape != (batch, count) or candidate_edit_costs.requires_grad:
                raise ValueError("candidate_edit_costs must be frozen [batch, candidates]")
            edit_available = self._candidate_mask(candidate_edit_cost_mask, batch, count, original.device) & cmask
            if not torch.isfinite(candidate_edit_costs[edit_available]).all() or (candidate_edit_costs[edit_available] < 0).any():
                raise ValueError("Actual candidate edit costs must be finite and nonnegative")
            edit_distances = candidate_edit_costs.to(deltas.dtype).masked_fill(~edit_available, 0)
            edit_source = "program_edit_cost"
        weights = ranking["selection_weights"]
        selected_target = (weights * target_output["score"]).sum(dim=-1)
        selected_unrelated = (weights * unrel_scores).sum(dim=-1)
        selected_transformed = (weights * eqv_scores).sum(dim=-1)
        return {
            **ranking, **verification,
            "logits": verification["validity_logits"],
            "validity_probability": verification["probability"],
            "original_score": original_output["score"],
            "candidate_scores": target_output["score"], "target_scores": target_output["score"],
            "unrelated_scores": unrel_scores, "transformed_scores": eqv_scores,
            "selected_target_score": selected_target, "selected_unrelated_score": selected_unrelated,
            "selected_transformed_score": selected_transformed,
            "unrelated_score": selected_unrelated, "transformed_score": selected_transformed,
            "unrelated_available_mask": unrel_available, "transformed_available_mask": eqv_available,
            "edit_distances": edit_distances, "edit_cost_source": edit_source,
            "edit_cost_available_mask": edit_available,
            "selected_edit_distance": (weights * edit_distances).sum(dim=-1),
            "critique_features": original_output["representation"],
            "location_logits": original_output["location_logits"],
            "location_probabilities": original_output["location_probabilities"],
            "support_logits": original_output["support_logits"],
            "support_probabilities": original_output["support_probabilities"],
            "state_predictions": {"original": original_output, "target": target_output,
                                  "unrelated": unrelated_output, "transformed": transformed_output},
        }

    def parameter_groups(self):
        return {"theta": tuple(self.critique.parameters()),
                "phi": tuple(self.intervention.parameters()),
                "psi": tuple(self.verifier.parameters())}

    @staticmethod
    def select_critiques(probabilities, similarity=None, critique_features=None, **kwargs):
        from .ranking import select_critiques
        kwargs.setdefault("lambda_size", 0.5)
        return select_critiques(probabilities, similarity, critique_features, **kwargs)
