import copy
import itertools

import numpy as np
import pytest

from civic.evaluation import (binary_metrics, paired_metrics, retrieval_metrics,
                              transformation_metrics, validation_threshold)
from tools.fill_experiment_tables import render_tables


def test_binary_metrics_keep_empty_PRs_and_positive_denominator_distinct():
    result = binary_metrics([1, 0, 1, 0], [.9, .8, .7, .1], .75,
                            ['a', 'a', 'b', 'b'], ['a', 'b', 'empty'])
    assert [result[key] for key in ['tp', 'fp', 'tn', 'fn']] == [1, 1, 1, 1]
    assert result['accuracy'] == result['precision'] == result['recall'] == result['f1'] == .5
    assert result['auroc'] == .75
    assert result['ap'] == pytest.approx(5 / 6)
    assert result['fp_per_pr'] == pytest.approx(1 / 3)
    assert result['comments_per_pr'] == pytest.approx(2 / 3)
    assert result['fp_per_positive'] == .5


def test_threshold_calibration_never_splits_tied_scores():
    result = validation_threshold([1, 0, 1, 0], [.9, .8, .7, .1])
    assert result['threshold'] == .7
    assert result['selection_split'] == 'validation'
    tied = validation_threshold([1, 0, 1, 0], [.9, .9, .8, .1], 'recall', .5)
    assert tied['threshold'] == .9
    assert tied['objective_value'] == .5
    highest_tie = validation_threshold([1, 0, 1, 0], [.9, .8, .7, .6], 'youden')
    assert highest_tie['threshold'] == .9


def test_undefined_metrics_are_explicit_and_not_fabricated():
    result = binary_metrics([0, 0], [.3, .1], .2)
    for key in ['auroc', 'ap', 'recall', 'fp_per_positive', 'fp_per_pr', 'comments_per_pr']:
        assert result[key] is None
        assert key in result['undefined']
    with pytest.raises(ValueError, match='include every'):
        binary_metrics([0, 1], [0, 1], .5, ['a', 'b'], ['a'])


def test_pair_AUROC_uses_real_unrelated_effects_instead_of_zero_reference():
    result = paired_metrics([3, 3], [2, 2], [1, 1], 2.5)
    assert result['pair_accuracy'] == 1
    assert result['pair_auroc'] == 0
    assert result['joint_threshold_accuracy'] == 1
    assert result['specificity_leakage'] == 2
    assert result['mean_target_drop'] == 1
    assert result['pair_auroc_reference'] == 'observed_unrelated_intervention_logit_drops'
    with pytest.raises(ValueError):
        paired_metrics([3, 3], [2, 2], [], 2.5)


def test_paired_metrics_distinguish_order_from_joint_threshold_correctness():
    result = paired_metrics([3, 2], [2, 1], [[2.5, 3.1], [1.5, 2.1]], .5)
    assert result['pair_accuracy'] == 1
    assert result['joint_threshold_accuracy'] == 0
    assert result['positive_recall'] == 1
    assert result['trap_rejection'] == 0
    assert result['n_unrelated'] == 4


def test_retrieval_ties_match_exhaustive_random_permutations():
    scores = np.array([[2., 1., 1., 1., 0., 0.]])
    actual = retrieval_metrics(scores, [1])
    positions = []
    for order in itertools.permutations([1, 2, 3]):
        positions.append(2 + order.index(1))
    assert actual['mrr'] == pytest.approx(np.mean(1 / np.array(positions)))
    assert actual['mean_rank'] == pytest.approx(np.mean(positions))
    assert actual['mrr'] != pytest.approx(1 / actual['mean_rank'])
    tied = retrieval_metrics(np.zeros((1, 5)), [0])
    permuted = retrieval_metrics(np.zeros((1, 5)), [4])
    assert tied == permuted
    assert tied['top1'] == .2
    assert tied['recall5'] == 1
    assert tied['mrr'] == pytest.approx(sum(1 / rank for rank in range(1, 6)) / 5)


def test_retrieval_masks_do_not_change_ties_or_admit_masked_targets():
    result = retrieval_metrics([[1, 1, np.nan]], [1], np.array([[True, True, False]]))
    assert result['top1'] == .5
    assert result['mrr'] == .75
    with pytest.raises(ValueError, match='must not be masked'):
        retrieval_metrics([[1, 2]], [1], np.array([[True, False]]))


def test_transformation_metrics_require_actual_mapped_predictions():
    result = transformation_metrics(
        [1, 0, 0], [2, 1, -1], [2, -1, 1], 0,
        gold_locations=[10, 20, 30], predicted_locations=[12, 99, 32],
        location_maps=[{10: 12}, {20: 22}, {30: 32}],
        gold_evidence=[[10, 11], [20], []], predicted_evidence=[[12, 13], [99], []],
        evidence_maps=[{10: 12, 11: 13}, {20: 22}, {}])
    assert result['verdict_consistency'] == pytest.approx(1 / 3)
    assert result['location_accuracy'] == pytest.approx(2 / 3)
    assert result['evidence_exact'] == result['evidence_f1'] == pytest.approx(2 / 3)
    assert result['fp_change_rate'] == .5
    assert result['fp_reduction'] == 0
    missing = transformation_metrics([1, 0], [2, -1], [2, -1], 0)
    assert missing['location_accuracy'] is missing['evidence_f1'] is None
    assert missing['fp_reduction'] is None
    with pytest.raises(ValueError, match='actual model predictions'):
        transformation_metrics([1, 0], [2, -1], [2, -1], 0, gold_locations=[1, 2])
    serialized_map = transformation_metrics([1, 0], [1, -1], [1, -1], 0,
                                           gold_locations=[1, 2], predicted_locations=[3, 4],
                                           location_maps=[{'1': 3}, {'2': 4}])
    assert serialized_map['location_accuracy'] == 1


TEMPLATE = r'''Front matter remains exactly the same.
\begin{table}[t]
% \caption{Old unused table}
% \end{table}

\begin{table}[t]
\caption{Actual table.}
\label{tab:test}
\begin{tabular}{lrr}
\toprule
Method & AUROC & FP/PR \\
\midrule
Base & \textbf{0.80} & 0.40 \\
\rowcolor{oursbg}
\ours{} & \textbf{TBD} & TBD \\
\bottomrule
\end{tabular}
\end{table}
Back matter remains exactly the same.
'''


def audited_results():
    return {
        'schema_version': 1, 'audit': {'passed': True, 'selection': 'validation'},
        'tables': {'tab:test': {
            'columns': ['auroc', 'fp_per_pr'], 'directions': ['max', 'min'], 'decimals': 2,
            'rows': [
                {'id': 'base', 'match': 'Base', 'values': [.8, .4], 'audited': True,
                 'selection': 'fixed_baseline', 'provenance': {'predictions_sha256': 'a' * 64}},
                {'id': 'ours', 'match': r'\ours{}', 'values': [.75, .2], 'audited': True,
                 'ours': True, 'provenance': {'predictions_sha256': 'b' * 64}}]}}}


def test_table_renderer_preserves_non_table_text_and_highlights_true_winners():
    rendered = render_tables(TEMPLATE, audited_results())
    assert rendered.startswith('Front matter remains exactly the same.\n')
    assert rendered.endswith('Back matter remains exactly the same.\n')
    assert '\n% \\begin{table}[t]\n% \\caption{Old unused table}' in rendered
    assert r'Base & \textbf{0.80} & 0.40' in rendered
    assert r'\ours{} & 0.75 & \textbf{0.20}' in rendered
    assert 'TBD' not in rendered


@pytest.mark.parametrize('bad', ['missing_value', 'test_selection', 'missing_audit', 'wrong_baseline', 'missing_baseline'])
def test_table_renderer_fails_without_complete_audited_same_cohort_results(bad):
    results = copy.deepcopy(audited_results())
    if bad == 'missing_value':
        results['tables']['tab:test']['rows'][1]['values'][0] = None
    elif bad == 'test_selection':
        results['audit']['selection'] = 'test_peak'
    elif bad == 'missing_audit':
        results['tables']['tab:test']['rows'][1]['audited'] = False
    elif bad == 'wrong_baseline':
        results['tables']['tab:test']['rows'][0]['values'][0] = .9
    else:
        results['tables']['tab:test']['rows'].pop(0)
    with pytest.raises(ValueError):
        render_tables(TEMPLATE, results)


def test_table_renderer_only_corrects_existing_values_when_explicitly_audited():
    results = audited_results()
    baseline = results['tables']['tab:test']['rows'][0]
    baseline['values'][0] = .7
    baseline['correct_existing'] = [True, False]
    rendered = render_tables(TEMPLATE, results)
    assert r'Base & 0.70 & 0.40' in rendered
    assert r'\ours{} & \textbf{0.75} & \textbf{0.20}' in rendered
