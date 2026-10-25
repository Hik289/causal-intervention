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
    """Shared state scorer over frozen code, text, and optional paired features.

    Token-level code features enable location and supporting-context predictions.
    With pooled code features these outputs have one context position, rather
    than claiming a source-line localization.
    """

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
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    @staticmethod
    def _tokens(features, name):
        if features.ndim == 2:
            features = features.unsqueeze(1)
        if features.ndim != 3 or features.shape[1] == 0:
            raise ValueError(name + " must be [batch, dimensions] or [batch, tokens, dimensions]")
        return features

    def forward(self, code_features, critique_features, paired_features=None):
        code = self._tokens(code_features, "code_features")
        critique = self._tokens(critique_features, "critique_features")
        if code.shape[0] != critique.shape[0]:
            raise ValueError("Code and critique features must share their batch size")
        tokens = [self.code_projection(code) + self.type_embedding[0],
                  self.critique_projection(critique) + self.type_embedding[1]]
        if self.paired_projection is not None:
            if paired_features is None:
                raise ValueError("paired_features are required when paired_dim is configured")
            paired = self._tokens(paired_features, "paired_features")
            if paired.shape[0] != code.shape[0]:
                raise ValueError("Paired features must share the code batch size")
            tokens.append(self.paired_projection(paired) + self.type_embedding[2])
        elif paired_features is not None:
            raise ValueError("Configure paired_dim before supplying paired_features")
        encoded = self.cross_attention(torch.cat(tokens, dim=1))
        code_encoded = encoded[:, :code.shape[1]]
        location_logits = self.location_head(code_encoded).squeeze(-1)
        support_logits = self.support_head(code_encoded).squeeze(-1)
        location_probabilities = torch.softmax(location_logits, dim=-1)
        support_probabilities = torch.softmax(support_logits, dim=-1)
        location_context = (location_probabilities.unsqueeze(-1) * code_encoded).sum(dim=1)
        support_context = (support_probabilities.unsqueeze(-1) * code_encoded).sum(dim=1)
        representation = (encoded.mean(dim=1) + location_context + support_context) / 3
        return {
            "score": self.scorer(representation).squeeze(-1),
            "representation": representation,
            "location_logits": location_logits,
            "location_probabilities": location_probabilities,
            "support_logits": support_logits,
            "support_probabilities": support_probabilities,
        }


class VerifierModule(nn.Module):
    """Combine base scores with targeted, unrelated, and transformed responses."""

    def __init__(self, hidden=128):
        super().__init__()
        self.response_encoder = nn.Sequential(nn.Linear(3, hidden), nn.GELU())
        self.network = nn.Sequential(
            nn.Linear(4 + hidden * 2, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1),
        )

    def forward(self, original_score, target_scores, unrelated_scores,
                transformed_scores, selection_weights, candidate_mask=None):
        responses = torch.stack((original_score.unsqueeze(1) - target_scores,
                                 original_score.unsqueeze(1) - unrelated_scores,
                                 original_score.unsqueeze(1) - transformed_scores), dim=-1)
        encoded = self.response_encoder(responses)
        selected_responses = (selection_weights.unsqueeze(-1) * responses).sum(dim=1)
        selected_context = (selection_weights.unsqueeze(-1) * encoded).sum(dim=1)
        if candidate_mask is None:
            pooled_context = encoded.mean(dim=1)
        else:
            mask = candidate_mask.to(device=encoded.device, dtype=encoded.dtype).unsqueeze(-1)
            pooled_context = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        inputs = torch.cat((original_score.unsqueeze(-1), selected_responses,
                            selected_context, pooled_context), dim=-1)
        logits = self.network(inputs).squeeze(-1)
        return {"validity_logits": logits, "probability": torch.sigmoid(logits),
                "target_response": selected_responses[:, 0],
                "unrelated_response": selected_responses[:, 1],
                "transformed_response": selected_responses[:, 2]}


class CIVER(nn.Module):
    """Joint trainable modules operating on separately supplied frozen features.

    Candidate states have shape [batch, candidates, dimensions], or
    [batch, candidates, tokens, dimensions]. Shared control states may omit the
    candidate axis. Training uses a differentiable candidate mixture; evaluation
    uses the highest-ranked candidate. Neither pretrained encoders nor an
    intervention generator is trained or invoked by this feature-level model.
    """

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
        return features.unsqueeze(1).expand(-1, count, *features.shape[1:]).reshape(
            features.shape[0] * count, *features.shape[1:])

    @staticmethod
    def _expand_control(features, reference, name):
        if features.ndim == reference.ndim - 1:
            features = features.unsqueeze(1).expand(-1, reference.shape[1], *features.shape[1:])
        if features.ndim != reference.ndim or features.shape[:2] != reference.shape[:2]:
            raise ValueError(name + " must contain a shared state or one state per candidate")
        if features.shape[-1] != reference.shape[-1]:
            raise ValueError(name + " feature dimensions do not match the candidates")
        return features

    @staticmethod
    def _pooled_code(features):
        return features.mean(dim=-2) if features.ndim > 2 else features

    def forward(self, original, critique, candidates, unrelated, transformed,
                pair_original=None, pair_candidates=None, pair_unrelated=None,
                pair_transformed=None, candidate_mask=None):
        if original.ndim not in (2, 3) or candidates.ndim not in (3, 4):
            raise ValueError("Original and candidate code features must be pooled vectors or token sequences")
        batch, count = candidates.shape[:2]
        if batch != original.shape[0] or critique.shape[0] != batch or count == 0:
            raise ValueError("All states must share a batch size and include at least one candidate")
        if original.shape[-1] != self.code_dim or candidates.shape[-1] != self.code_dim:
            raise ValueError("Code feature dimensions do not match code_dim")
        unrelated = self._expand_control(unrelated, candidates, "unrelated")
        transformed = self._expand_control(transformed, candidates, "transformed")
        pairs = (pair_original, pair_candidates, pair_unrelated, pair_transformed)
        if self.paired_dim is not None:
            if any(value is None for value in pairs):
                raise ValueError("Each program state needs its own paired features when paired_dim is configured")
            if pair_candidates.ndim not in (3, 4) or pair_candidates.shape[:2] != (batch, count):
                raise ValueError("pair_candidates must contain paired features for each candidate")
            if any(value.shape[-1] != self.paired_dim for value in pairs):
                raise ValueError("Paired feature dimensions do not match paired_dim")
            pair_unrelated = self._expand_control(pair_unrelated, pair_candidates, "pair_unrelated")
            pair_transformed = self._expand_control(pair_transformed, pair_candidates, "pair_transformed")
        elif any(value is not None for value in pairs):
            raise ValueError("Configure paired_dim before supplying paired features")
        original_output = self.critique(original, critique, pair_original)
        repeated_critique = self._repeat(critique, count)

        def state_output(features, paired):
            flat = features.reshape(batch * count, *features.shape[2:])
            pair_flat = None if paired is None else paired.reshape(batch * count, *paired.shape[2:])
            output = self.critique(flat, repeated_critique, pair_flat)
            return {key: value.reshape(batch, count, *value.shape[1:]) for key, value in output.items()}

        target_output = state_output(candidates, pair_candidates)
        unrelated_output = state_output(unrelated, pair_unrelated)
        transformed_output = state_output(transformed, pair_transformed)
        base_code = self._pooled_code(original)
        candidate_code = candidates.mean(dim=2) if candidates.ndim == 4 else candidates
        deltas = candidate_code - base_code.unsqueeze(1)
        ranking = self.intervention(original_output["representation"], deltas, candidate_mask)
        verification = self.verifier(original_output["score"], target_output["score"],
                                     unrelated_output["score"], transformed_output["score"],
                                     ranking["selection_weights"], candidate_mask)
        weights = ranking["selection_weights"]
        edit_distances = deltas.abs().sum(dim=-1)
        selected_target = (weights * target_output["score"]).sum(dim=-1)
        selected_unrelated = (weights * unrelated_output["score"]).sum(dim=-1)
        selected_transformed = (weights * transformed_output["score"]).sum(dim=-1)
        return {
            **ranking, **verification,
            "logits": verification["validity_logits"],
            "validity_probability": verification["probability"],
            "original_score": original_output["score"],
            "candidate_scores": target_output["score"],
            "target_scores": target_output["score"],
            "unrelated_scores": unrelated_output["score"],
            "transformed_scores": transformed_output["score"],
            "selected_target_score": selected_target,
            "selected_unrelated_score": selected_unrelated,
            "selected_transformed_score": selected_transformed,
            "unrelated_score": selected_unrelated,
            "transformed_score": selected_transformed,
            "edit_distances": edit_distances,
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
        return select_critiques(probabilities, similarity, critique_features, **kwargs)
