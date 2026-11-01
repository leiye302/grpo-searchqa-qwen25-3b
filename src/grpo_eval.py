"""Read-only evaluation/metrics and verified storage around native GRPO."""
import json
import os
from pathlib import Path
import uuid
from common import atomic_json, question_id
from checkpoint_io import NativeCheckpointMixin
from evaluation_contract import summarize_questions
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class SearchTrainer(NativeCheckpointMixin, RayPPOTrainer):
    def fit(self):
        from verl.utils.tracking import Tracking
        original = Tracking.log
        destination = Path(os.environ['GRPO_RUN_DIR']) / 'metrics.jsonl'

        def capture(tracker, data, step, backend=None):
            original(tracker, data=data, step=step, backend=backend)
            with destination.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'step': int(step), 'metrics': data},
                    default=lambda x: x.item(), allow_nan=False) + '\n')
                stream.flush()

        Tracking.log = capture
        try:
            return super().fit()
        finally:
            Tracking.log = original

    def _validate(self):
        run = Path(os.environ['GRPO_RUN_DIR'])
        receipt = json.loads(Path(os.environ['QA_DATA_RECEIPT']).read_text(encoding='utf-8'))
        expected = receipt['files']['test']['rows']
        destination = run / 'evaluation' / f'update_{self.global_steps:03d}' / ('attempt-' + uuid.uuid4().hex[:8])
        destination.mkdir(parents=True, exist_ok=False)
        original = self.traj_collector.multi_turn_loop
        questions, seen_ids = {}, []
        batches = 0

        def capture(*args, **kwargs):
            nonlocal batches
            incoming = kwargs['gen_batch']
            identities = [question_id(row) for row in incoming.non_tensor_batch['env_kwargs']]
            offset = len(seen_ids)
            if identities != expected['question_ids'][offset:offset + len(identities)]:
                raise RuntimeError('Actual validation batch differs from the fixed full test order')
            input_indices = [int(i) for i in incoming.non_tensor_batch['index']]
            if input_indices != list(range(offset, offset + len(identities))):
                raise RuntimeError('Native validation skipped, reordered or repeated question indices')
            input_ids = dict(zip(input_indices, identities))
            batch = original(*args, **kwargs)
            nt = batch.non_tensor_batch
            records = []
            for i in range(len(batch.batch['responses'])):
                # The unmodified collector resets `index` to item-within-batch.
                local_index = int(nt['index'][i])
                if not 0 <= local_index < len(input_indices):
                    raise RuntimeError('Output trajectory lost its input-question identity')
                index = input_indices[local_index]
                reward = float(nt['episode_rewards'][i])
                episode = {'index': index, 'question_id': input_ids[index],
                    'data_source': str(nt['data_source'][i]), 'episode_reward': reward,
                    'trajectory_id': str(nt['traj_uid'][i])}
                if index in questions and questions[index] != episode:
                    raise RuntimeError('Inconsistent episode reward, source or trajectory identity')
                questions[index] = episode
                ids = batch.batch['responses'][i]
                mask = batch.batch['attention_mask'][i, -ids.shape[-1]:].bool()
                tokens = ids[mask].cpu().tolist()
                records.append(dict(episode, step=int(nt['turn_step'][i]), token_ids=tokens,
                    text=self.tokenizer.decode(tokens, skip_special_tokens=False),
                    action_valid=bool(nt['is_action_valid'][i])))
            atomic_json(destination / f'batch_{batches:03d}.json', records)
            seen_ids.extend(identities)
            batches += 1
            return batch

        self.traj_collector.multi_turn_loop = capture
        try:
            native_metrics = super()._validate()
        finally:
            self.traj_collector.multi_turn_loop = original
        if seen_ids != expected['question_ids']:
            raise RuntimeError('Native validation did not traverse every test question')
        summary = summarize_questions(list(questions.values()), expected['question_ids'], expected['source_counts'])
        summary.update({'update': int(self.global_steps), 'batches': batches,
            'native_metrics_unchanged': native_metrics,
            'note': 'Per-question EM is separate from native decision-row/batch-averaged logs; no training metric was overwritten.'})
        atomic_json(destination / 'summary.json', summary)
        atomic_json(run / 'evaluation/latest.json', {'path': str(destination.relative_to(run)), 'update': int(self.global_steps)})
        return native_metrics
