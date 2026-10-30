import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from civic.evaluation import retrieval_metrics
from civic.models import CIVER
from scripts.train_full import FullDataset, load_checkpoint, save_json, sha256


def load_bank(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


@torch.no_grad()
def encode_queries(model, dataset, bank, indices, device):
    model.eval()
    values = []
    for start in range(0, len(indices), 16):
        query_indices = bank['query_main_indices'][indices[start:start + 16]]
        batch = dataset.batch(query_indices, device)
        values.append(model.critique(batch['original'], batch['critique'], batch.get('pair_original'),
                      batch.get('original_mask'), batch.get('critique_mask'),
                      batch.get('pair_original_mask'))['representation'].cpu())
    return torch.cat(values)


@torch.no_grad()
def evaluate(model, contexts, bank, indices, device, feature_kind):
    model.eval()
    result, predictions = {}, {}
    for k in [4, 8, 16]:
        features = torch.as_tensor(bank[f'k{k}_' + feature_kind][indices], device=device)
        masks = torch.as_tensor(bank[f'k{k}_pool_mask'][indices], device=device)
        logits = model.intervention(contexts.to(device), features, masks)['ranking_logits']
        logits = logits.cpu().numpy()
        result[str(k)] = retrieval_metrics(logits, bank[f'k{k}_target_index'][indices], masks.cpu().numpy())
        predictions[f'k{k}_scores'] = logits
    return result, predictions


def selection_key(metrics, loss, epoch):
    return (float(np.mean([value['mrr'] for value in metrics.values()])),
            float(np.mean([value['top1'] for value in metrics.values()])), -float(loss), -epoch)


def train(args):
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    source = load_checkpoint(args.initialize_from, device)
    locked = json.loads((args.initialize_from.parent / 'primary_selection_locked.json').read_text())
    if locked['checkpoint_sha256'] != sha256(args.initialize_from):
        raise ValueError('Only a validation-locked pretrained three-module model may initialize retrieval')
    gradient_audit = json.loads((args.initialize_from.parent / 'module_gradient_audit.json').read_text())
    if not gradient_audit['passed'] or not all(gradient_audit['totals'][key] > 0 for key in ['theta', 'phi', 'psi']):
        raise ValueError('All three parent modules must already have been trained')
    model = CIVER(**source['config']).to(device)
    model.load_state_dict(source['model'])
    for name, parameters in model.parameter_groups().items():
        for parameter in parameters:
            parameter.requires_grad_(name == 'phi')
    dataset = FullDataset(args.source_data, include_identity=True)
    bank = load_bank(args.bank)
    if not np.array_equal(dataset.metadata['case_id'][bank['query_main_indices']], bank['query_case_ids']):
        raise ValueError('Retrieval query identifiers do not match the canonical frozen source')
    train_indices = np.flatnonzero(bank['split'] == 'train')
    val_indices = np.flatnonzero(bank['split'] == 'val')
    if set(bank['query_case_ids'][train_indices]) & set(bank['query_case_ids'][val_indices]):
        raise ValueError('Retrieval train and validation queries overlap')
    for split_indices, split_name in [(train_indices, 'train'), (val_indices, 'val')]:
        if not np.all(dataset.metadata['split'][bank['query_main_indices'][split_indices]] == split_name):
            raise ValueError('Query contexts cross the declared retrieval split')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {'model_kind': 'three_module_pretrained_with_intervention_retrieval_adaptation',
                'parent_checkpoint_sha256': sha256(args.initialize_from), 'source_data_sha256': sha256(args.source_data),
                'training_bank_sha256': sha256(args.bank), 'test_bank_sha256': None if args.defer_test else sha256(args.test_bank),
                'feature_kind': args.feature_kind, 'selection': 'validation_mean_mrr_then_mean_top1_then_ce_then_earliest',
                'adaptation': 'Only phi is updated; theta and psi retain their fully trained parent values. This checkpoint is for retrieval only.',
                'module_parameters': {key: sum(parameter.numel() for parameter in values) for key, values in model.parameter_groups().items()},
                'args': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                'code_sha256': sha256(__file__)}
    save_json(args.output / 'manifest.json', manifest)
    train_context = encode_queries(model, dataset, bank, train_indices, device)
    val_context = encode_queries(model, dataset, bank, val_indices, device)
    test_bank, test_context, test_indices = None, None, None
    if not args.defer_test:
        test_bank = load_bank(args.test_bank)
        test_indices = np.arange(len(test_bank['query_main_indices']))
        if not np.all(dataset.metadata['split'][test_bank['query_main_indices']] == 'test'):
            raise ValueError('Retrieval test contexts are not held out')
        test_context = encode_queries(model, dataset, test_bank, test_indices, device)
    optimizer = torch.optim.AdamW(model.intervention.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    features = torch.as_tensor(bank['k16_' + args.feature_kind][train_indices])
    masks = torch.as_tensor(bank['k16_pool_mask'][train_indices])
    targets = torch.as_tensor(bank['k16_target_index'][train_indices], dtype=torch.long)
    val_features = torch.as_tensor(bank['k16_' + args.feature_kind][val_indices], device=device)
    val_masks = torch.as_tensor(bank['k16_pool_mask'][val_indices], device=device)
    val_targets = torch.as_tensor(bank['k16_target_index'][val_indices], dtype=torch.long, device=device)
    best_key, best_primary, selected_epoch, last_improvement = None, None, None, 0
    gradient_sum = 0.0
    for epoch in range(1, args.epochs + 1):
        model.intervention.train()
        permutation = rng.permutation(len(train_indices))
        for start in range(0, len(permutation), args.batch_size):
            batch = permutation[start:start + args.batch_size]
            logits = model.intervention(train_context[batch].to(device), features[batch].to(device), masks[batch].to(device))['ranking_logits']
            loss = torch.nn.functional.cross_entropy(logits, targets[batch].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if epoch == 1:
                gradient_sum += sum(float(p.grad.abs().sum()) for p in model.intervention.parameters() if p.grad is not None)
            optimizer.step()
        validation, validation_predictions = evaluate(model, val_context, bank, val_indices, device, args.feature_kind)
        with torch.no_grad():
            val_loss = torch.nn.functional.cross_entropy(model.intervention(val_context.to(device), val_features, val_masks)['ranking_logits'], val_targets).item()
        key = selection_key(validation, val_loss, epoch)
        if best_key is None or key > best_key:
            best_key, selected_epoch = key, epoch
            torch.save({'model': model.state_dict(), 'config': source['config'], 'epoch': epoch,
                        'parent_checkpoint_sha256': manifest['parent_checkpoint_sha256'],
                        'retrieval_feature_kind': args.feature_kind}, args.output / 'validation_selected.pt')
            np.savez_compressed(args.output / 'selected_validation_predictions.npz', query_indices=val_indices, **validation_predictions)
        if best_primary is None or key[:2] > best_primary:
            best_primary, last_improvement = key[:2], epoch
        with (args.output / 'validation_history.jsonl').open('a') as handle:
            handle.write(json.dumps({'epoch': epoch, 'validation': validation, 'ce': val_loss, 'selected_epoch': selected_epoch}) + '\n')
        if not args.defer_test:
            diagnostic, _ = evaluate(model, test_context, test_bank, test_indices, device, args.feature_kind)
            with (args.output / 'exploratory_test_history.jsonl').open('a') as handle:
                handle.write(json.dumps({'epoch': epoch, 'test': diagnostic, 'exploratory_only': True}) + '\n')
        if epoch == 1 or epoch % 50 == 0:
            print(json.dumps({'epoch': epoch, 'validation_mean_mrr': key[0], 'selected_epoch': selected_epoch}), flush=True)
        if args.patience and epoch >= args.min_epochs and epoch - last_improvement >= args.patience:
            break
    if gradient_sum <= 0:
        raise ValueError('The intervention module did not learn during retrieval adaptation')
    save_json(args.output / 'intervention_gradient_audit.json', {'passed': True, 'gradient_l1': gradient_sum,
              'theta_and_psi_pretrained_and_frozen': True})
    checkpoint = args.output / 'validation_selected.pt'
    save_json(args.output / 'primary_selection_locked.json', {'epoch': selected_epoch,
              'checkpoint_sha256': sha256(checkpoint), 'selection': 'validation_only', 'epochs_trained': epoch})
    model.load_state_dict(load_checkpoint(checkpoint, device)['model'])
    validation, _ = evaluate(model, val_context, bank, val_indices, device, args.feature_kind)
    test = None
    if not args.defer_test:
        test, predictions = evaluate(model, test_context, test_bank, test_indices, device, args.feature_kind)
        np.savez_compressed(args.output / 'selected_test_predictions.npz', query_indices=test_indices, **predictions)
    save_json(args.output / 'results.json', {'complete': True, 'selection_key': best_key,
              'selected_epoch': selected_epoch, 'epochs_trained': epoch, 'validation': validation, 'test': test,
              'feature_kind': args.feature_kind, 'checkpoint_sha256': sha256(checkpoint)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-data', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--test-bank', type=Path)
    parser.add_argument('--initialize-from', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--feature-kind', choices=['patch_text_embeddings', 'deltas'], default='patch_text_embeddings')
    parser.add_argument('--defer-test', action='store_true')
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--min-epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--device', default='cpu')
    train(parser.parse_args())
