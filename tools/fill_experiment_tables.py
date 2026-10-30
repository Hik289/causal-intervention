import argparse
import hashlib
import json
import math
import re
from pathlib import Path


def _active(text):
    return re.sub(r'(?<!\\)%[^\n]*', lambda match: ' ' * len(match.group()), text)


def _balanced_end(text, start):
    if text[start] != '{':
        raise ValueError('expected a LaTeX brace group')
    depth = 0
    for index in range(start, len(text)):
        if text[index] == '{' and (index == 0 or text[index - 1] != '\\'):
            depth += 1
        elif text[index] == '}' and (index == 0 or text[index - 1] != '\\'):
            depth -= 1
            if not depth:
                return index + 1
    raise ValueError('unbalanced LaTeX brace group')


def _repair_orphan(text):
    pattern = re.compile(r'(?m)^\\begin\{table\}\[t\][ \t]*\n(?P<body>(?:[ \t]*%[^\n]*\n)+)')

    def repair(match):
        if re.search(r'(?m)^\s*%\s*\\end\{table\}', match['body']):
            return '% ' + match.group()
        return match.group()

    return pattern.sub(repair, text)


def _table_blocks(text):
    active = _active(text)
    pattern = re.compile(r'\\begin\{table\*?\}.*?\\end\{table\*?\}', re.S)
    blocks = {}
    for match in pattern.finditer(active):
        labels = re.findall(r'\\label\{([^}]+)\}', match.group())
        if len(labels) != 1:
            raise ValueError('every active table must have exactly one label')
        if labels[0] in blocks:
            raise ValueError(f'duplicate table label {labels[0]}')
        blocks[labels[0]] = (match.start(), match.end(), text[match.start():match.end()])
    return blocks


def _top_level_delimiters(text, kind):
    active = _active(text)
    depth, index = 0, 0
    while index < len(active):
        character = active[index]
        escaped = index > 0 and active[index - 1] == '\\'
        if character == '{' and not escaped:
            depth += 1
        elif character == '}' and not escaped:
            depth -= 1
        if depth < 0:
            raise ValueError('unbalanced tabular body')
        if depth == 0 and kind == 'cell' and character == '&' and not escaped:
            yield index, index + 1
        if depth == 0 and kind == 'row' and active[index:index + 2] == '\\\\':
            end = index + 2
            if end < len(active) and active[end] == '[':
                closing = active.find(']', end)
                if closing < 0:
                    raise ValueError('unclosed row spacing argument')
                end = closing + 1
            yield index, end
            index = end
            continue
        index += 1
    if depth:
        raise ValueError('unbalanced tabular body')


def _method_start(cell):
    prefix = re.compile(r'\s*(?:\\(?:toprule|midrule|bottomrule|hline)\b(?:\[[^]]*\])?|\\rowcolor\{[^}]+\}|\\cmidrule(?:\([^)]*\))?\{[^}]+\})\s*')
    index = 0
    while True:
        match = prefix.match(cell, index)
        if not match:
            break
        index = match.end()
    return index


def _method(cell):
    result = cell[_method_start(cell):].strip()
    if result.startswith('\\textbf{') and _balanced_end(result, 7) == len(result):
        result = result[8:-1]
    return re.sub(r'\s+', ' ', result).strip()


def _rows(block):
    active = _active(block)
    match = re.search(r'\\begin\{tabular\}\s*\{', active)
    if not match:
        raise ValueError('table must contain one tabular environment')
    begin = _balanced_end(block, match.end() - 1)
    end = active.find('\\end{tabular}', begin)
    if end < 0:
        raise ValueError('missing tabular end')
    body = block[begin:end]
    rows, previous = [], 0
    for row_end, terminator_end in _top_level_delimiters(body, 'row'):
        row = body[previous:row_end]
        delimiters = list(_top_level_delimiters(row, 'cell'))
        if delimiters:
            bounds, start = [], 0
            for left, right in delimiters:
                bounds.append((begin + previous + start, begin + previous + left))
                start = right
            bounds.append((begin + previous + start, begin + previous + len(row)))
            first = block[slice(*bounds[0])]
            rows.append({'match': _method(first), 'bounds': bounds})
        previous = terminator_end
    return rows


def _number_cell(cell):
    result = cell.strip()
    color = bool(re.search(r'\\cellcolor\{ours(?:good)?bg\}', result))
    result = re.sub(r'\\cellcolor\{[^}]+\}', '', result).strip()
    if result.startswith('\\textbf{') and _balanced_end(result, 7) == len(result):
        result = result[8:-1]
    if result == 'TBD':
        return None, color, 'TBD'
    if result.startswith('$') and result.endswith('$'):
        result = result[1:-1]
    if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', result):
        raise ValueError(f'unsupported numeric table cell: {cell!r}')
    return float(result), color, result


def _finite_values(values, n, label, row_id):
    if not isinstance(values, list) or len(values) != n:
        raise ValueError(f'{label}/{row_id}: exactly {n} numeric values are required')
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
        raise ValueError(f'{label}/{row_id}: missing, nonnumeric, or nonfinite results cannot be filled')
    return values


def _audit_row(entry, label):
    if entry.get('audited') is not True or not entry.get('provenance'):
        raise ValueError(f'{label}/{entry.get("id")}: audited results and provenance are required')
    selection = entry.get('selection', 'validation')
    if selection not in {'validation', 'fixed_baseline', 'analytical'}:
        raise ValueError(f'{label}/{entry.get("id")}: test-selected results are not formal table results')


def _best_cells(entries, spec, n_columns):
    mode = spec.get('best_mode', 'columns')
    best = set()
    directions = spec.get('directions', ['max'] * n_columns)
    if len(directions) != n_columns or any(value not in {'max', 'min', None} for value in directions):
        raise ValueError('directions must provide max, min, or null for every numeric column')
    if mode == 'row_pairs':
        for row_index, entry in enumerate(entries):
            for columns in spec.get('row_comparisons', []):
                if len(columns) < 2 or any(not isinstance(index, int) or not 0 <= index < n_columns for index in columns):
                    raise ValueError('row_comparisons must list valid numeric column indices')
                direction = directions[columns[0]]
                if direction is None or any(directions[index] != direction for index in columns):
                    raise ValueError('compared row cells must have the same direction')
                values = [entry['values'][index] for index in columns]
                optimum = (max if direction == 'max' else min)(values)
                best.update((row_index, index) for index in columns if math.isclose(entry['values'][index], optimum, rel_tol=0, abs_tol=1e-12))
    elif mode == 'columns':
        groups = {entry.get('comparison_group', 'default') for entry in entries if entry.get('compare', True)}
        for group in groups:
            indices = [index for index, entry in enumerate(entries) if entry.get('compare', True) and entry.get('comparison_group', 'default') == group]
            for column, direction in enumerate(directions):
                if direction is None or len(indices) < 2:
                    continue
                optimum = (max if direction == 'max' else min)(entries[index]['values'][column] for index in indices)
                best.update((index, column) for index in indices if math.isclose(entries[index]['values'][column], optimum, rel_tol=0, abs_tol=1e-12))
    else:
        raise ValueError('best_mode must be columns or row_pairs')
    return best


def _fill_table(block, spec, label):
    columns, entries = spec.get('columns'), spec.get('rows')
    if not isinstance(columns, list) or not columns or not isinstance(entries, list) or not entries:
        raise ValueError(f'{label}: columns and audited rows are required')
    candidates, selected, used = _rows(block), [], set()
    ids = set()
    for entry in entries:
        _audit_row(entry, label)
        row_id = entry.get('id')
        if not isinstance(row_id, str) or row_id in ids:
            raise ValueError(f'{label}: unique row ids are required')
        ids.add(row_id)
        _finite_values(entry.get('values'), len(columns), label, row_id)
        matches = [row for row in candidates if row['match'] == _method(entry.get('match', ''))]
        occurrence = entry.get('occurrence', 0)
        if not isinstance(occurrence, int) or occurrence < 0 or occurrence >= len(matches):
            raise ValueError(f'{label}/{row_id}: row match or occurrence was not found')
        row = matches[occurrence]
        if len(row['bounds']) - 1 != len(columns):
            raise ValueError(f'{label}/{row_id}: table/result column count differs')
        if row['bounds'][0] in used:
            raise ValueError(f'{label}: a table row was supplied twice')
        used.add(row['bounds'][0])
        selected.append(row)
    best = _best_cells(entries, spec, len(columns))
    edits = []
    for row_index, (entry, row) in enumerate(zip(entries, selected)):
        decimal_places = entry.get('decimals', spec.get('decimals', 4))
        decimals = decimal_places if isinstance(decimal_places, list) else [decimal_places] * len(columns)
        if len(decimals) != len(columns) or any(not isinstance(value, int) or not 0 <= value <= 8 for value in decimals):
            raise ValueError(f'{label}: decimals must specify integers between zero and eight')
        correction = entry.get('correct_existing', False)
        corrections = correction if isinstance(correction, list) else [correction] * len(columns)
        if len(corrections) != len(columns) or any(not isinstance(value, bool) for value in corrections):
            raise ValueError(f'{label}: correct_existing must be boolean or one boolean per column')
        green = entry.get('ours_cells', [])
        if any(not isinstance(index, int) or not 0 <= index < len(columns) for index in green):
            raise ValueError(f'{label}: ours_cells contains an invalid column index')
        for column, ((left, right), value) in enumerate(zip(row['bounds'][1:], entry['values'])):
            cell = block[left:right]
            current, colored, literal = _number_cell(cell)
            if current is not None and not corrections[column]:
                decimals_existing = len(literal.split('.')[1].split('e')[0].split('E')[0]) if '.' in literal else 0
                tolerance = 0.5 * 10 ** -decimals_existing + 1e-12
                if abs(current - value) > tolerance:
                    raise ValueError(f'{label}/{entry["id"]}/{columns[column]}: existing value disagrees with audited result; explicit correction is required')
                number = literal
            else:
                number = f'{value:.{decimals[column]}f}'
            content = '\\textbf{' + number + '}' if (row_index, column) in best else number
            if colored or column in green:
                content = '\\cellcolor{oursbg}' + content
            leading = cell[:len(cell) - len(cell.lstrip())]
            trailing = cell[len(cell.rstrip()):]
            edits.append((left, right, leading + content + trailing))
        if entry.get('ours'):
            left, right = row['bounds'][0]
            cell = block[left:right]
            if not re.search(r'\\rowcolor\{ours(?:good)?bg\}', cell):
                start = _method_start(cell)
                cell = cell[:start] + '\\rowcolor{oursbg}\n' + cell[start:]
                edits.append((left, right, cell))
    for row in candidates:
        if row['bounds'][0] in used:
            continue
        cells = [block[left:right] for left, right in row['bounds'][1:]]
        if any('TBD' in _active(cell) for cell in cells):
            raise ValueError(f'{label}: an unprovided table row still contains TBD')
        if len(cells) == len(columns):
            try:
                parsed = [_number_cell(cell) for cell in cells]
            except ValueError:
                continue
            if parsed:
                raise ValueError(f'{label}: every numeric baseline row requires same-cohort audited results')
    for left, right, replacement in sorted(edits, reverse=True):
        block = block[:left] + replacement + block[right:]
    if 'TBD' in _active(block):
        raise ValueError(f'{label}: unfilled TBD remains')
    return block


def render_tables(template, results, protocol=None):
    if results.get('schema_version') != 1 or results.get('audit', {}).get('passed') is not True:
        raise ValueError('schema_version=1 and a passed independent audit are required')
    if results.get('audit', {}).get('selection') != 'validation':
        raise ValueError('formal results must use validation-selected checkpoints')
    digest = hashlib.sha256(template.encode()).hexdigest()
    if protocol is not None and protocol.get('template_sha256') not in {None, digest}:
        raise ValueError('the template differs from the locked table protocol')
    text = _repair_orphan(template)
    blocks, specifications = _table_blocks(text), results.get('tables', {})
    if not isinstance(specifications, dict) or not specifications:
        raise ValueError('audited table specifications are required')
    unknown = set(specifications) - set(blocks)
    if unknown:
        raise ValueError(f'unknown table labels: {sorted(unknown)}')
    edits = []
    for label, (left, right, block) in blocks.items():
        if label not in specifications:
            if 'TBD' in _active(block):
                raise ValueError(f'{label}: missing complete audited results')
            continue
        filled = _fill_table(block, specifications[label], label)
        edits.append((left, right, filled))
    for left, right, replacement in sorted(edits, reverse=True):
        text = text[:left] + replacement + text[right:]
    if 'TBD' in _active(text):
        raise ValueError('the document still contains active TBD placeholders')
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path)
    args = parser.parse_args()
    if args.output.resolve() == args.template.resolve():
        raise ValueError('write a reviewable output first; replacing the source template is disallowed')
    template = args.template.read_text()
    results = json.loads(args.results.read_text())
    protocol = json.loads(args.protocol.read_text()) if args.protocol else None
    rendered = render_tables(template, results, protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(json.dumps({'output': str(args.output), 'sha256': hashlib.sha256(rendered.encode()).hexdigest(), 'status': 'complete_audited_tables'}))


if __name__ == '__main__':
    main()
