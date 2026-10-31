import pytest
import torch

from civic.models import CIVER
from scripts.train_finding_filter import confidence_tie_rank_loss


def inputs():
    return dict(original=torch.randn(2, 6), critique=torch.randn(2, 6),
                candidates=torch.randn(2, 2, 6), unrelated=None, transformed=None,
                candidate_mask=torch.ones(2, 2, dtype=torch.bool),
                candidate_edit_costs=torch.tensor([[0., 0.], [0., 3.]]),
                candidate_edit_cost_mask=torch.ones(2, 2, dtype=torch.bool),
                frozen_reviewer_logit=torch.tensor([-1., 1.]))


def test_identity_actions_remain_selectable_without_claiming_target_responses():
    torch.manual_seed(42)
    model = CIVER(6, 6, hidden=16, layers=1, heads=2, dropout=0.,
                  rank_hidden=12, verifier_hidden=12,
                  use_measured_target_availability=True).eval()
    data = inputs()
    result = model(**data)
    assert result['response_available'][0, 0].item() == 0.
    assert torch.isfinite(result['ranking_logits']).all()
    assert result['target_response'][0].item() == 0.
    with pytest.raises(ValueError, match='actual edit costs'):
        model(**dict(data, candidate_edit_costs=None))


def test_bounded_learned_correction_cannot_reverse_sufficiently_separated_priors():
    torch.manual_seed(42)
    model = CIVER(6, 6, hidden=16, layers=1, heads=2, dropout=0.,
                  rank_hidden=12, verifier_hidden=12, residual_logit_cap=.1).eval()
    with torch.no_grad():
        model.verifier.validity_head.weight.fill_(1e6)
        model.verifier.validity_head.bias.fill_(1e6)
    data = inputs()
    result = model(**data)
    assert (result['residual_correction'].abs() <= .1).all()
    assert result['validity_logits'][1] > result['validity_logits'][0]
    assert (result['raw_residual_correction'].abs() > 1).all()


def test_tie_objective_never_uses_cross_confidence_pairs():
    raw = torch.tensor([.2, .5, .7, .9], requires_grad=True)
    prior = torch.tensor([1., 1., 2., 3.])
    labels = torch.tensor([1., 0., 1., 0.])
    loss = confidence_tie_rank_loss(raw, prior, labels)
    loss.backward()
    assert raw.grad[0] < 0 and raw.grad[1] > 0
    assert torch.equal(raw.grad[2:], torch.zeros(2))
