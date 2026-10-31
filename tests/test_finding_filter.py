import json

import numpy as np
import torch

from civic.models import CIVER
from scripts.train_finding_filter import parse_args, train
from scripts.train_full import load_checkpoint


def test_adaptation_keeps_real_metric_weights_and_validation_thresholds(tmp_path):
    rng = np.random.default_rng(42)
    n, dim, candidates = 12, 6, 2
    labels = np.tile([1., 0., 1., 0.], 3).astype(np.float32)
    split = np.repeat(['train', 'val', 'test'], 4)
    draw = lambda *shape: rng.normal(size=shape).astype(np.float32)
    bank = dict(original=draw(n, dim), critique=draw(n, dim),
                candidates=draw(n, candidates, dim), unrelated=draw(n, candidates, dim),
                transformed=draw(n, candidates, dim), pair_original=draw(n, dim),
                pair_candidates=draw(n, candidates, dim), pair_unrelated=draw(n, candidates, dim),
                pair_transformed=draw(n, candidates, dim), labels=labels,
                target_index=np.where(labels > 0, 0, -1), split=split,
                case_id=np.array([f'case-{i}' for i in range(n)]),
                group_id=np.array([f'group-{i}' for i in range(n)]), repo=split,
                pr_id=np.array([f'{split[i]}-pr-{i // 2}' for i in range(n)]),
                candidate_edit_costs=np.full((n, candidates), 3., dtype=np.float32),
                unrelated_available_mask=np.ones((n, candidates), dtype=bool),
                transformed_available_mask=np.ones((n, candidates), dtype=bool),
                frozen_reviewer_logit=draw(n), frozen_reviewer_available=np.ones(n, dtype=bool))
    path = tmp_path / 'features.npz'
    np.savez_compressed(path, **bank)
    config = dict(code_dim=dim, critique_dim=dim, paired_dim=dim, hidden=16,
                  layers=1, heads=2, dropout=.1, rank_hidden=12,
                  verifier_hidden=12, temperature=1.)
    parent = tmp_path / 'parent.pt'
    torch.save({'config': config, 'model': CIVER(**config).state_dict()}, parent)
    output = tmp_path / 'run'
    args = parse_args(['--data', str(path), '--output', str(output),
                       '--initialize-from', str(parent), '--epochs', '2', '--min-epochs', '1',
                       '--run-purpose', 'smoke', '--test-every', '1', '--batch-size', '4',
                       '--hidden', '16', '--layers', '1', '--heads', '2', '--rank-hidden', '12',
                       '--verifier-hidden', '12', '--dropout', '.3', '--allow-dropout-transfer',
                       '--lr-theta', '.00001', '--lr-phi', '.00003', '--lr-psi', '.001'])
    train(args)
    read = lambda name: json.loads((output / name).read_text())
    assert read('neutral_identity_audit.json')['maximum_absolute_logit_error'] == 0
    assert read('module_gradient_audit.json')['passed']
    assert read('manifest.json')['module_learning_rates'] == dict(theta=.00001, phi=.00003, psi=.001)
    assert read('threshold_locked.json')['source'] == 'validation_f1_only'
    assert set(read('metric_best.json')['observed_control_pairs']) == {'pair_accuracy', 'pair_auroc'}
    for policy, metrics in read('metric_best.json').items():
        for metric, entry in metrics.items():
            saved = load_checkpoint(output / entry['checkpoint'], 'cpu')
            assert saved['epoch'] == entry['epoch'] > 0
            assert (output / entry['test_predictions']).is_file()
            if policy == 'test_curve_70recall':
                assert entry['metrics']['recall'] >= .7
                assert entry['threshold_selection']['evaluation_only']
                assert entry['threshold_selection']['not_a_deployment_threshold']
            else:
                assert entry['threshold_selection']['selection_split'] in ('fixed', 'validation')
