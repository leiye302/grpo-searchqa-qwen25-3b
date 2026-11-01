# Minimal differences from the pinned public Search-QA GRPO launcher

Anchor: ZJU-REAL/SDAR `29dcb12a55d8fc17ba6ae03fd0a69048f43051f3`,
`examples/grpo_trainer/run_search.sh`, originally Qwen2.5-3B-Instruct/four GPUs.

## Method settings not changed

Native GRPO advantage computation, clipped policy loss, response masks,
token-mean aggregation, optimizer, precision, gradient accumulation, reference
KL0.001, entropy0.001, invalid-action penalty0.01,128x8 global episodes,
mini256/micro16,LR1e-6,warmup0.1,150 updates, TP1/vLLM0.5, prompt4096/
response512/left truncation, history4,max4 actions, top3 retrieval and native
training rollout sampling/validation decoding remain unchanged. The native environment manager,
collector, parser, memory, reward, data preprocessing and E5 retrieval server
are also unchanged and SHA-locked.

There is no distillation gate in pure GRPO. Do not add algorithm.sdar merely
to set beta5. The user-requested beta5 convention applies to later methods
that actually use the corresponding gate; KL0.001 must not become5.

## Named adaptations

1. Standalone public-source/model/data/retrieval downloads with fixed revisions,
   checksums, portable paths, explicit CPU/GPU assignments and isolated Ray temp.
   Remove upstream's private proxy/conda/logging examples; logger becomes console.
2. The separate1.5B repository selects Qwen2.5-1.5B-Instruct, while the3B
   repository keeps the original model family/size. All scientific configuration
   and full-test dataset files are identical across sizes. Entrypoints/model
   revision/output labels are the only backbone-selection differences.
3. Explicit evaluation adaptation: fixed448 questions (64/source) at0 and
   every20 updates through140; the full51,713-question test only at final150.
   Both sizes share the same committed manifest. Native generation/scoring
   and metric values remain unchanged; additional question-EM/time metrics
   use separate subset/full namespaces. This is not claimed as native cadence.
4. Save every20 updates and final150 (the original launcher has save_freq=-1).
   Reuse the native serializer/loader with verified staging, atomic publish,
   newest-two retention across resume and no pruning before a new save verifies.
   No optimizer-update or rollout logic is rewritten.
5. Start the unchanged retriever explicitly in its own documented environment/
   session rather than silently activating a private conda environment or
   killing a process. Default native GPU/FP16 E5+Flat retrieval is preserved.

Exactly one upstream line changes: main_ppo imports SearchTrainer as
RayPPOTrainer. SearchTrainer derives from the untouched native RayPPOTrainer,
calls super().fit / super()._validate, and uses the storage-only checkpoint
mixin. Full core/source locks and patch reverse-check prevent hidden edits.

## Faithful limitations, not hidden fixes

Original train/full-test split overlaps are reported, not filtered; the small
monitoring subset excludes known exact training-question matches. Original step/batch
aggregation remains in native logs, not silently replaced. Native parser
permissiveness remains as is. Environment/retriever state is not added to
FSDP checkpoints, so exact replay is not claimed. CPU tests do not certify
CUDA/FAISS/vLLM memory or performance. All such changes would need their own
explicit, matched experiment rather than a supposedly unchanged reproduction.
