import copy

import pytest
import torch
from torch.nn import functional as F

from civic.models import CIVER
from civic.objectives import joint_objectives, review_selection_loss


torch.set_num_threads(1)


def example(token_features=False):
    torch.manual_seed(317)
    n, k, dim = 4, 3, 6
    draw = lambda *shape: torch.randn(*shape)
    code = lambda: draw(n, 3, dim) if token_features else draw(n, dim)
    states = lambda: draw(n, k, 3, dim) if token_features else draw(n, k, dim)
    batch = {
        "original": code(), "critique": draw(n, dim), "candidates": states(),
        "unrelated": states(), "transformed": states(),
        "pair_original": draw(n, dim), "pair_candidates": draw(n, k, dim),
        "pair_unrelated": draw(n, k, dim), "pair_transformed": draw(n, k, dim),
        "candidate_edit_costs": torch.tensor([[3., 8., 0.]] * n),
        "candidate_mask": torch.ones(n, k, dtype=torch.bool),
        "labels": torch.tensor([1., 0., 1., 0.]),
        "target_index": torch.tensor([0, -1, 1, -1]),
        "group_id": torch.tensor([0, 0, 1, 1]),
    }
    model = CIVER(dim, dim, dim, hidden=16, layers=1, heads=2, dropout=0,
                  rank_hidden=12, verifier_hidden=12)
    return model, batch


def forward_inputs(batch):
    excluded = {"labels", "target_index", "group_id", "candidate_verified_mask",
                "reverse_index", "reverse_verified_mask", "location_labels", "support_labels",
                "location_supervision_mask", "support_supervision_mask",
                "transformed_location_map", "transformed_support_map",
                "location_mapping_mask", "support_mapping_mask"}
    return {key: value for key, value in batch.items() if key not in excluded}


def grad_total(module):
    return sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)


def test_classifier_baseline_scores_transported_critique_with_transformed_code():
    import numpy as np
    from scripts.train_classifier_baseline import FrozenFeatureMLP, predict

    _, batch = example()
    batch["transformed_critique"] = -3 * batch["critique"]
    batch["unrelated_available_mask"] = torch.ones(4, dtype=torch.bool)
    batch["transformed_available_mask"] = torch.ones(4, dtype=torch.bool)

    class Dataset:
        def batches(self, split, size):
            yield np.arange(4)

        def batch(self, indices, device):
            return batch

    model = FrozenFeatureMLP(6, 6, 6, hidden=12).eval()
    result = predict(model, Dataset(), "test", 4, "cpu")
    with torch.no_grad():
        transported = model(batch["transformed"], batch["transformed_critique"], batch["pair_transformed"])
        stale = model(batch["transformed"], batch["critique"], batch["pair_transformed"])
        original = model(batch["original"], batch["critique"], batch["pair_original"])
    assert np.allclose(result["transformed_scores"], transported.numpy())
    assert not np.allclose(result["transformed_scores"], stale.numpy())
    assert np.allclose(result["logits"], original.numpy())


def test_three_modules_receive_joint_gradients_with_program_edit_costs():
    model, batch = example(True)
    batch["candidate_verified_mask"] = torch.tensor(
        [[True, True, False], [False, False, False], [True, True, False], [False, False, False]])
    batch["location_labels"] = torch.tensor([0, -1, 2, -1])
    batch["support_labels"] = torch.tensor([[1., 0., 0.], [0., 0., 0.], [0., .5, .5], [0., 0., 0.]])
    for name in ("location", "support"):
        batch[f"{name}_supervision_mask"] = batch["labels"].bool()
        batch[f"{name}_mapping_mask"] = torch.ones(4, dtype=torch.bool)
        batch[f"transformed_{name}_map"] = torch.eye(3).expand(4, -1, -1)
    output = model(**forward_inputs(batch))
    losses = joint_objectives(output, batch)
    assert all(torch.isfinite(loss) for loss in losses.values())
    assert output["edit_cost_source"] == "program_edit_cost"
    losses["total"].backward()
    assert grad_total(model.critique) > 0
    assert grad_total(model.intervention) > 0
    assert grad_total(model.verifier) > 0
    assert model.critique.location_head.weight.grad.abs().sum() > 0
    assert model.critique.support_head.weight.grad.abs().sum() > 0
    assert model.verifier.selection_head.weight.grad.abs().sum() > 0
    assert all(not value.requires_grad for value in batch.values())


def test_gold_annotations_are_not_forward_features():
    model, batch = example()
    model.eval()
    initial = model(**forward_inputs(batch))
    changed = copy.deepcopy(batch)
    changed["labels"] = 1 - changed["labels"]
    changed["target_index"] = torch.tensor([-1, 2, -1, 1])
    changed["candidate_verified_mask"] = torch.ones(4, 3, dtype=torch.bool)
    changed["reverse_index"] = torch.tensor([1, -1, 1, -1])
    changed["reverse_verified_mask"] = torch.ones(4, dtype=torch.bool)
    repeated = model(**forward_inputs(changed))
    for key in ("original_score", "validity_logits", "selection_logits", "selected_index"):
        assert torch.equal(initial[key], repeated[key])
    with pytest.raises(TypeError):
        model(**forward_inputs(batch), labels=batch["labels"])


def test_actual_minimality_cannot_reward_a_noop_candidate():
    model, batch = example()
    output = model(**forward_inputs(batch))
    first = joint_objectives(output, batch)["minimality"]
    first.backward()
    assert grad_total(model.intervention) > 0
    modified = dict(output)
    logits = output["ranking_logits"].detach().clone()
    logits[:, 2] += 8
    modified["ranking_log_probabilities"] = F.log_softmax(logits, dim=-1)
    assert joint_objectives(modified, batch)["minimality"] > first.detach()
    batch["candidate_verified_mask"] = torch.zeros(4, 3, dtype=torch.bool)
    batch["candidate_verified_mask"][0, 0] = True
    batch["candidate_verified_mask"][2, 1] = True
    batch["candidate_verified_mask"][0, 2] = True
    with pytest.raises(ValueError, match="no-op"):
        joint_objectives(output, batch)


def test_reverse_flip_only_uses_verified_safe_negative_pairs():
    model, batch = example()
    output = model(**forward_inputs(batch))
    baseline = joint_objectives(output, batch)["flip"]
    batch["reverse_index"] = torch.tensor([-1, 0, -1, 0])
    batch["reverse_verified_mask"] = torch.zeros(4, dtype=torch.bool)
    assert torch.equal(joint_objectives(output, batch)["flip"], baseline)
    batch["reverse_verified_mask"][1] = True
    expected = F.relu(.5 + output["original_score"][1] - output["target_scores"][1, 0])
    assert torch.allclose(joint_objectives(output, batch)["flip"], baseline + expected)
    assert torch.allclose(joint_objectives(output, batch)["ranking"],
                          -output["ranking_log_probabilities"][[0, 2], [0, 1]].mean())


def test_missing_controls_are_masked_without_copying_original():
    model, batch = example()
    batch["unrelated"] = None
    batch["transformed"] = None
    batch["pair_unrelated"] = None
    batch["pair_transformed"] = None
    output = model(**forward_inputs(batch))
    assert output["state_predictions"]["unrelated"] is None
    assert output["state_predictions"]["transformed"] is None
    assert not output["unrelated_available_mask"].any()
    assert not output["transformed_available_mask"].any()
    losses = joint_objectives(output, batch)
    assert losses["specificity"] == 0
    assert losses["equivariance"] == 0
    with pytest.raises(ValueError, match="missing control"):
        model(**forward_inputs(batch), transformed_available_mask=torch.ones(4, dtype=torch.bool))


def test_unavailable_padded_controls_do_not_reach_the_encoder():
    model, batch = example(True)
    batch["unrelated_available_mask"] = torch.zeros(4, 3, dtype=torch.bool)
    batch["transformed_available_mask"] = torch.zeros(4, 3, dtype=torch.bool)
    batch["unrelated_token_mask"] = torch.zeros(4, 3, 3, dtype=torch.bool)
    batch["transformed_token_mask"] = torch.zeros(4, 3, 3, dtype=torch.bool)
    output = model(**forward_inputs(batch))
    assert torch.isfinite(output["validity_logits"]).all()
    losses = joint_objectives(output, batch)
    assert losses["specificity"] == 0 and losses["equivariance"] == 0


def test_transformed_critique_and_padding_masks_are_used():
    model, batch = example(True)
    model.eval()
    batch["original_mask"] = torch.tensor([[True, True, False]] * 4)
    batch["candidate_token_mask"] = torch.tensor([[[True, True, False]] * 3] * 4)
    output = model(**forward_inputs(batch))
    modified = copy.deepcopy(batch)
    modified["original"][:, -1] += 1000
    modified["candidates"][:, :, -1] -= 1000
    repeated = model(**forward_inputs(modified))
    assert torch.allclose(output["validity_logits"], repeated["validity_logits"], atol=1e-6)
    batch["transformed_critique"] = batch["critique"] * -3
    transformed = model(**forward_inputs(batch))
    assert not torch.allclose(output["transformed_scores"], transformed["transformed_scores"])
    assert torch.equal(output["original_score"], transformed["original_score"])


def test_mapped_equivariance_uses_the_location_map_and_availability():
    model, batch = example(True)
    output = model(**forward_inputs(batch))
    output["transformed_scores"] = output["original_score"].unsqueeze(1).expand(-1, 3)
    original = output["state_predictions"]["original"]
    transformed = output["state_predictions"]["transformed"]
    original["location_probabilities"] = torch.tensor([[.8, .1, .1]] * 4, requires_grad=True)
    transformed["location_probabilities"] = torch.tensor([[[.1, .8, .1]] * 3] * 4, requires_grad=True)
    mapping = torch.tensor([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])
    batch["transformed_location_map"] = mapping.expand(4, -1, -1)
    batch["location_mapping_mask"] = torch.ones(4, dtype=torch.bool)
    assert joint_objectives(output, batch)["equivariance"].abs() < 1e-6
    batch["transformed_location_map"] = torch.eye(3).expand(4, -1, -1)
    loss = joint_objectives(output, batch)["equivariance"]
    assert loss > 0
    loss.backward()
    assert original["location_probabilities"].grad.abs().sum() > 0
    assert transformed["location_probabilities"].grad.abs().sum() > 0
    batch["location_mapping_mask"] = torch.zeros(4, dtype=torch.bool)
    assert joint_objectives(output, batch)["equivariance"] == 0


def test_gate_does_not_reward_retaining_every_finding():
    probability = torch.tensor([.8, .8])
    gate_logits = torch.tensor([4., 4.], requires_grad=True)
    loss = review_selection_loss(probability, torch.tensor([0, 0]), torch.eye(2),
                                 selection_logits=gate_logits, labels=torch.tensor([1., 0.]))
    loss.backward()
    assert gate_logits.grad[1] > 0
    selected = CIVER.select_critiques(torch.tensor([.2, .8]), similarity=torch.zeros(2, 2))
    assert selected["indices"].tolist() == [1]


def test_frozen_reviewer_residual_is_not_an_output_floor():
    model, batch = example()
    model.eval()
    with torch.no_grad():
        model.verifier.validity_head.weight.zero_()
        model.verifier.validity_head.bias.fill_(-1)
    batch["frozen_reviewer_logit"] = torch.full((4,), 2.)
    output = model(**forward_inputs(batch))
    assert torch.allclose(output["validity_logits"], torch.ones(4))
    batch["original"].requires_grad_()
    with pytest.raises(ValueError, match="frozen features"):
        model(**forward_inputs(batch))


def test_missing_actual_edit_costs_do_not_become_zero_edit_supervision():
    model, batch = example()
    batch["candidate_edit_costs"] = torch.full((4, 3), float("nan"))
    batch["candidate_edit_cost_mask"] = torch.zeros(4, 3, dtype=torch.bool)
    output = model(**forward_inputs(batch))
    loss = joint_objectives(output, batch)["minimality"]
    assert loss == 0 and torch.isfinite(loss)
    assert not output["edit_cost_available_mask"].any()
