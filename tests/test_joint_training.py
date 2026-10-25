import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from civic.models import CIVER, build
from civic.objectives import joint_objectives, review_selection_loss
from civic.ranking import select_critiques
from civic.services import required_env
from scripts.train_joint import JointDataset


torch.set_num_threads(1)


def make_batch(token_features=False):
    torch.manual_seed(19)
    n, k, dc, dt, dp = 4, 3, 6, 5, 7
    batch = {
        'original': torch.randn(n, 3, dc) if token_features else torch.randn(n, dc),
        'critique': torch.randn(n, 2, dt) if token_features else torch.randn(n, dt),
        'candidates': torch.randn(n, k, 3, dc) if token_features else torch.randn(n, k, dc),
        'unrelated': torch.randn(n, k, 3, dc) if token_features else torch.randn(n, k, dc),
        'transformed': torch.randn(n, k, 3, dc) if token_features else torch.randn(n, k, dc),
        'pair_original': torch.randn(n, 2, dp) if token_features else torch.randn(n, dp),
        'pair_candidates': torch.randn(n, k, 2, dp) if token_features else torch.randn(n, k, dp),
        'pair_unrelated': torch.randn(n, k, 2, dp) if token_features else torch.randn(n, k, dp),
        'pair_transformed': torch.randn(n, k, 2, dp) if token_features else torch.randn(n, k, dp),
        'labels': torch.tensor([1., 0., 1., 0.]),
        'target_index': torch.tensor([0, -1, 1, -1]),
        'group_id': torch.tensor([0, 0, 1, 1]),
        'candidate_mask': torch.ones(n, k, dtype=torch.bool),
    }
    return batch


def make_model():
    return CIVER(6, 5, 7, hidden=16, layers=1, heads=2, dropout=0,
                 rank_hidden=12, verifier_hidden=10)


def inputs(batch):
    return {key: value for key, value in batch.items()
            if key not in {'labels', 'target_index', 'group_id'}}


@pytest.mark.parametrize('token_features', [False, True])
def test_all_objectives_and_module_gradients(token_features):
    batch, model = make_batch(token_features), make_model()
    outputs = model(**inputs(batch))
    losses = joint_objectives(outputs, batch)
    assert set(losses) == {'classification', 'flip', 'specificity', 'equivariance',
                           'ranking', 'minimality', 'selection', 'total'}
    assert all(torch.isfinite(value) for value in losses.values())
    assert losses['equivariance'].item() > 0
    losses['total'].backward()
    groups = model.parameter_groups()
    identities = [id(p) for group in groups.values() for p in group]
    assert len(set(identities)) == len(identities)
    assert set(identities) == {id(p) for p in model.parameters()}
    for parameters in groups.values():
        assert sum(p.grad.abs().sum().item() for p in parameters if p.grad is not None) > 0
    assert all(not value.requires_grad for value in batch.values())
    if token_features:
        assert model.critique.location_head.weight.grad.abs().sum() > 0
        assert model.critique.support_head.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('objective', ['classification', 'minimality'])
def test_intervention_receives_gradients_outside_ranking_loss(objective):
    batch, model = make_batch(), make_model()
    output = model(**inputs(batch))
    losses = joint_objectives(output, batch)
    losses[objective].backward()
    assert sum(p.grad.abs().sum().item() for p in model.intervention.parameters()
               if p.grad is not None) > 0


def test_masked_candidates_and_hard_inference():
    batch, model = make_batch(), make_model()
    batch['candidate_mask'][:, 2] = False
    output = model(**inputs(batch))
    assert torch.equal(output['ranking_probabilities'][:, 2], torch.zeros(4))
    model.eval()
    output = model(**inputs(batch))
    assert torch.equal(output['selection_weights'].sum(1), torch.ones(4))
    assert torch.all((output['selection_weights'] == 0) | (output['selection_weights'] == 1))
    batch['candidate_mask'][0] = False
    with pytest.raises(ValueError, match='unmasked'):
        model(**inputs(batch))


def test_each_state_requires_its_own_paired_representation():
    batch, model = make_batch(), make_model()
    del batch['pair_transformed']
    with pytest.raises(ValueError, match='Each program state'):
        model(**inputs(batch))


def test_subset_selection_matches_exhaustive_utility():
    torch.manual_seed(7)
    probabilities = torch.rand(6, requires_grad=True)
    raw = torch.rand(6, 6)
    similarity = (raw + raw.T) / 2
    result = select_critiques(probabilities, similarity=similarity)
    utilities = []
    for flags in itertools.product([False, True], repeat=6):
        chosen = [i for i, flag in enumerate(flags) if flag]
        value = sum(float(probabilities[i].detach()) for i in chosen)
        value -= sum(float(similarity[i, j]) for i, j in itertools.combinations(chosen, 2))
        utilities.append(value)
    assert float(result['utility'].detach()) == pytest.approx(max(utilities), abs=1e-6)
    result['utility'].backward()
    assert probabilities.grad.abs().sum() > 0
    with pytest.raises(ValueError, match='exact_limit'):
        select_critiques(probabilities.detach(), exact_limit=2)


def test_token_critique_selection_uses_semantic_pooling():
    batch = make_batch(True)
    probability = torch.tensor([.6, .7, .4, .8], requires_grad=True)
    token_loss = review_selection_loss(probability, batch['group_id'], batch['critique'])
    vector_loss = review_selection_loss(probability, batch['group_id'], batch['critique'].mean(1))
    assert torch.allclose(token_loss, vector_loss)
    token_loss.backward()
    assert probability.grad.abs().sum() > 0


@pytest.mark.parametrize('name', ['mlp', 'bilinear', 'crossattn'])
def test_original_verifier_interfaces_remain_available(name):
    model = build(name, 6, 5, hidden=16, layers=1, heads=2, dropout=0)
    score, representation = model(torch.randn(3, 6), torch.randn(3, 5))
    assert score.shape == (3,)
    assert representation.shape == (3, 16)


def test_service_configuration_has_no_implicit_values(monkeypatch):
    monkeypatch.delenv('CIVIC_API_KEY', raising=False)
    with pytest.raises(RuntimeError, match='CIVIC_API_KEY'):
        required_env('CIVIC_API_KEY')


def training_archive(path):
    rng = np.random.default_rng(22)
    n, k = 12, 3
    array = lambda *shape: rng.normal(size=shape).astype(np.float32)
    arrays = {
        'original': array(n, 3, 6), 'critique': array(n, 2, 5),
        'candidates': array(n, k, 3, 6), 'unrelated': array(n, k, 3, 6),
        'transformed': array(n, k, 3, 6), 'pair_original': array(n, 2, 7),
        'pair_candidates': array(n, k, 2, 7), 'pair_unrelated': array(n, k, 2, 7),
        'pair_transformed': array(n, k, 2, 7),
        'labels': np.tile([1, 0], n // 2), 'target_index': np.tile([0, -1], n // 2),
        'group_id': np.repeat(np.arange(n // 2), 2),
        'split': np.array(['train'] * 6 + ['val'] * 4 + ['test'] * 2),
    }
    np.savez(path, **arrays)
    return arrays


def test_training_selects_validation_checkpoint(tmp_path):
    path, output = tmp_path / 'features.npz', tmp_path / 'run'
    training_archive(path)
    command = [sys.executable, '-m', 'scripts.train_joint', '--data', str(path),
               '--output', str(output), '--epochs', '2', '--batch-size', '4',
               '--hidden', '16', '--layers', '1', '--heads', '2', '--dropout', '0',
               '--rank-hidden', '12', '--verifier-hidden', '10', '--device', 'cpu']
    import os
    environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                               env=environment, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    result = json.loads((output / 'results.json').read_text())
    selected = min(result['history'], key=lambda record: record['validation']['bce'])
    assert result['selected_epoch'] == selected['epoch']
    assert result['selection_metric'] == 'validation_bce'
    assert result['feature_mode'] == 'code_text_paired'
    assert result['test']['n'] == 2
    assert all('test' not in record for record in result['history'])
    assert (output / 'checkpoint.pt').exists()


def test_dataset_rejects_placeholder_transformations_and_split_leakage(tmp_path):
    path = tmp_path / 'features.npz'
    arrays = training_archive(path)
    arrays['transformed'] = np.repeat(arrays['original'][:, None], 3, axis=1)
    arrays['pair_transformed'] = np.repeat(arrays['pair_original'][:, None], 3, axis=1)
    np.savez(path, **arrays)
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError, match='distinct'):
            JointDataset(path)
    arrays = training_archive(path)
    arrays['group_id'][6] = arrays['group_id'][0]
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match='same split'):
        JointDataset(path)
