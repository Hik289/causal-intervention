import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from civic.evaluation import binary_metrics, paired_metrics, validation_threshold
from scripts.train_full import FullDataset, load_checkpoint, save_json, sha256


class FrozenFeatureMLP(nn.Module):
    def __init__(self, code_dim, critique_dim, paired_dim=None, hidden=256):
        super().__init__()
        self.paired_dim = paired_dim
        dimension = code_dim + critique_dim + (paired_dim or 0)
        self.network = nn.Sequential(nn.Linear(dimension, hidden), nn.GELU(),
                                     nn.Linear(hidden, hidden // 2), nn.GELU(),
                                     nn.Linear(hidden // 2, 1))

    @staticmethod
    def pool(value, mask=None):
        if value.ndim == 2:
            return value
        weight = torch.ones(value.shape[:2], device=value.device) if mask is None else mask.to(value.dtype)
        return (value * weight.unsqueeze(-1)).sum(1) / weight.sum(1, keepdim=True).clamp_min(1)

    def forward(self, code, critique, pair=None, code_mask=None, critique_mask=None, pair_mask=None):
        pieces = [self.pool(code, code_mask), self.pool(critique, critique_mask)]
        if self.paired_dim is not None:
            if pair is None:
                raise ValueError('Independent code-critique paired views are required')
            pieces.append(self.pool(pair, pair_mask))
        return self.network(torch.cat(pieces, -1)).squeeze(-1)


@torch.no_grad()
def predict(model, dataset, split, size, device):
    model.eval()
    collected = {key: [] for key in ['indices', 'labels', 'logits', 'state_scores', 'target_scores',
                                     'unrelated_scores', 'transformed_scores', 'target_index',
                                     'unrelated_available', 'transformed_available', 'location_probabilities']}
    for indices in dataset.batches(split, size):
        batch = dataset.batch(indices, device)
        critique = batch['critique']
        def score(code, pair, mask, pmask, state_critique=None, state_critique_mask=None):
            current_critique = critique if state_critique is None else state_critique
            current_mask = batch.get('critique_mask') if state_critique is None else state_critique_mask
            return model(code, current_critique, pair, mask, current_mask, pmask)
        original = score(batch['original'], batch.get('pair_original'), batch.get('original_mask'), batch.get('pair_original_mask'))
        target_index = batch.get('evaluation_target_index', batch['target_index'])
        rows = torch.arange(len(indices), device=device)
        choose = lambda key: None if key not in batch else batch[key][rows, target_index.clamp_min(0)]
        target = score(choose('candidates'), choose('pair_candidates'), choose('candidate_token_mask'), choose('pair_candidates_mask'))
        values = {'indices': indices, 'labels': batch['labels'], 'logits': original, 'state_scores': original,
                  'target_scores': target, 'target_index': target_index}
        for name in ['unrelated', 'transformed']:
            values[name + '_available'] = batch[name + '_available_mask']
            if batch.get(name) is None:
                values[name + '_scores'] = torch.zeros_like(original)
            else:
                values[name + '_scores'] = score(
                    batch[name], batch.get('pair_' + name), batch.get(name + '_token_mask'),
                    batch.get('pair_' + name + '_mask'),
                    batch.get('transformed_critique') if name == 'transformed' else None,
                    batch.get('transformed_critique_mask') if name == 'transformed' else None,
                )
        code = batch['original']
        if code.ndim == 2:
            location = torch.ones((len(indices), 1), device=device)
        else:
            blocks = code.shape[1]
            repeated = lambda x: None if x is None else x[:, None].expand(-1, blocks, *x.shape[1:]).reshape(len(indices) * blocks, *x.shape[1:])
            position_score = model(code.reshape(-1, code.shape[-1]), repeated(critique),
                                   repeated(batch.get('pair_original')), critique_mask=repeated(batch.get('critique_mask')),
                                   pair_mask=repeated(batch.get('pair_original_mask'))).reshape(len(indices), blocks)
            location = position_score.masked_fill(~batch['original_mask'], -torch.inf).softmax(-1)
        values['location_probabilities'] = location
        for key, value in values.items():
            collected[key].append(value.detach().cpu().numpy() if torch.is_tensor(value) else value)
    result = {key: np.concatenate(values) for key, values in collected.items()}
    order = np.argsort(result['indices'])
    return {key: value[order] for key, value in result.items()}


def summarize(prediction, dataset, threshold=0):
    result = binary_metrics(prediction['labels'].astype(int), prediction['logits'], threshold=threshold)
    valid = (prediction['labels'] == 1) & (prediction['target_index'] >= 0) & prediction['unrelated_available']
    result['paired'] = paired_metrics(prediction['state_scores'][valid], prediction['target_scores'][valid],
                                      prediction['unrelated_scores'][valid], threshold=threshold) if valid.any() else None
    result['bce'] = float(np.mean(np.logaddexp(0, prediction['logits'].astype(float)) - prediction['labels'] * prediction['logits']))
    return result


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    dataset = FullDataset(args.data, split_unit=args.split_unit, include_identity=True)
    device = torch.device(args.device)
    config = dict(code_dim=dataset.code_dim, critique_dim=dataset.critique_dim,
                  paired_dim=dataset.paired_dim, hidden=args.hidden)
    model = FrozenFeatureMLP(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / 'manifest.json', {'model_kind': 'frozen_feature_mlp', 'config': config,
              'data': dataset.audit, 'selection': 'validation_only', 'script_sha256': sha256(__file__),
              'args': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}})
    rng = np.random.default_rng(args.seed)
    best, best_epoch, last_improvement, best_stop = None, 0, 0, None
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for indices in dataset.batches('train', args.batch_size, rng):
            batch = dataset.batch(indices, device)
            logits = model(batch['original'], batch['critique'], batch.get('pair_original'),
                           batch.get('original_mask'), batch.get('critique_mask'), batch.get('pair_original_mask'))
            loss = nn.functional.binary_cross_entropy_with_logits(logits, batch['labels'])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        validation = predict(model, dataset, 'val', args.batch_size, device)
        scores = summarize(validation, dataset)
        pair_auc = (scores.get('paired') or {}).get('pair_auroc', -1)
        key = (scores['auroc'], pair_auc, -scores['bce'], -epoch)
        if best is None or key > best:
            best, best_epoch = key, epoch
            torch.save({'model': model.state_dict(), 'config': config, 'epoch': epoch}, args.output / 'validation_selected.pt')
            np.savez_compressed(args.output / 'selected_validation_predictions.npz', **validation)
        if best_stop is None or key[:2] > best_stop:
            last_improvement, best_stop = epoch, key[:2]
        with (args.output / 'validation_history.jsonl').open('a') as out:
            out.write(json.dumps({'epoch': epoch, 'validation': scores, 'selected_epoch': best_epoch}) + '\n')
        if not args.defer_test:
            diagnostic = summarize(predict(model, dataset, 'test', args.batch_size, device), dataset)
            with (args.output / 'exploratory_test_history.jsonl').open('a') as out:
                out.write(json.dumps({'epoch': epoch, 'test': diagnostic, 'exploratory_only': True}) + '\n')
        if epoch == 1 or epoch % 25 == 0:
            print(json.dumps({'epoch': epoch, 'validation_auroc': scores['auroc'], 'selected_epoch': best_epoch}), flush=True)
        if args.patience and epoch >= args.min_epochs and epoch - last_improvement >= args.patience:
            break
    checkpoint = args.output / 'validation_selected.pt'
    model.load_state_dict(load_checkpoint(checkpoint, device)['model'])
    validation = predict(model, dataset, 'val', args.batch_size, device)
    threshold = validation_threshold(validation['labels'].astype(int), validation['logits'], objective='f1')['threshold']
    save_json(args.output / 'primary_selection_locked.json', {'epoch': best_epoch, 'checkpoint_sha256': sha256(checkpoint), 'test_not_used': True})
    test = None
    if not args.defer_test:
        prediction = predict(model, dataset, 'test', args.batch_size, device)
        np.savez_compressed(args.output / 'selected_test_predictions.npz', **prediction)
        test = summarize(prediction, dataset, threshold)
    save_json(args.output / 'results.json', {'complete': True, 'selection_key': best, 'selected_epoch': best_epoch,
              'model_kind': 'frozen_feature_mlp', 'epochs_trained': epoch, 'test_deferred': args.defer_test,
              'threshold': threshold, 'test': test, 'elapsed_seconds': time.monotonic() - started,
              'checkpoint_sha256': sha256(checkpoint), 'data_sha256': dataset.audit['data_sha256']})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--min-epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split-unit', choices=['repo', 'group', 'diff'], default='repo')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--defer-test', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    train(parse_args())
