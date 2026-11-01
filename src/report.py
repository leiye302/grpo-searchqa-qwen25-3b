"""Export real per-question EM summaries only; never prefill scores."""
import argparse
import csv
import json
from pathlib import Path
from common import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/question_em.csv')
    args = parser.parse_args()
    rows = []
    for path in sorted((args.run_dir / 'evaluation').glob('update_*/attempt-*/summary.json')):
        value = json.loads(path.read_text(encoding='utf-8'))
        row = {'update': value['update'], 'attempt': path.parent.name, 'questions': value['questions'],
               'micro_em': value['question_micro_em'], 'macro_em': value['dataset_macro_em']}
        row.update({name + '_em': info['em'] for name, info in value['by_dataset'].items()})
        rows.append(row)
    if not rows:
        raise RuntimeError('No completed real evaluations found')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Actual question-EM results:', args.output)


if __name__ == '__main__':
    main()
