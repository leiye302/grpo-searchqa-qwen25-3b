"""Whole-test-set identity and per-question reporting, without changing RL."""
from collections import Counter, defaultdict
from common import digest, load


def assert_protocol(cfg):
    from omegaconf import OmegaConf
    expected = {'data.val_batch_size': 512, 'data.max_prompt_length': 4096,
        'data.max_response_length': 512, 'data.truncation': 'left',
        'actor_rollout_ref.rollout.val_kwargs.do_sample': False,
        'actor_rollout_ref.rollout.val_kwargs.temperature': 0,
        'actor_rollout_ref.rollout.val_kwargs.n': 1,
        'env.max_steps': 4, 'env.history_length': 4, 'env.search.topk': 3}
    for key, value in expected.items():
        if OmegaConf.select(cfg, key) != value:
            raise RuntimeError('Native Search-QA evaluation parameter differs: ' + key)
    data = load('data.json')
    if data['revision'] != 'b7d80abfee334a7a91cb377544f09180d58b34f6':
        raise RuntimeError('Use the shared pinned QA data revision')
    if data['splits']['test']['sha256'] != '30aa887b6d47e06e8c0f6f5307c88fe4e13461ac25a20ec0a5433ad7a4fe25dc':
        raise RuntimeError('Shared whole-test Parquet differs')


def summarize_questions(rows, expected_ids, expected_sources):
    """Count each input question once, independent of its trajectory length."""
    ordered = sorted(rows, key=lambda row: row['index'])
    if [r['index'] for r in ordered] != list(range(len(expected_ids))):
        raise RuntimeError('Evaluation missed or duplicated input question indices')
    if [r['question_id'] for r in ordered] != expected_ids:
        raise RuntimeError('Evaluation question identities/order changed')
    counts = Counter(r['data_source'] for r in ordered)
    if dict(counts) != expected_sources:
        raise RuntimeError('Evaluation source counts differ')
    scores = defaultdict(list)
    for row in ordered:
        if row['episode_reward'] not in (0., 1.):
            raise RuntimeError('Expected native binary normalized-EM episode reward')
        scores[row['data_source']].append(row['episode_reward'])
    by_source = {name: {'questions': len(values), 'correct': int(sum(values)),
                        'em': sum(values) / len(values)} for name, values in sorted(scores.items())}
    total = len(ordered)
    return {'questions': total, 'correct': int(sum(r['episode_reward'] for r in ordered)),
        'question_micro_em': sum(r['episode_reward'] for r in ordered) / total,
        'dataset_macro_em': sum(r['em'] for r in by_source.values()) / len(by_source),
        'by_dataset': by_source, 'ordered_question_ids_sha256': digest(expected_ids)}
