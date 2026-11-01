# Search-QA evaluation: exact original behavior and explicit reporting

Source pin: SDAR `29dcb12a55d8fc17ba6ae03fd0a69048f43051f3`.
Relevant source files are SHA-locked in configs/upstream.json:
`examples/grpo_trainer/run_search.sh`, `verl/trainer/ppo/ray_trainer.py`,
SearchEnv, its utils.py scorer, environment manager and native collector.

## Task set and cadence

The original preprocessing script downloads train.parquet and test.parquet from
PeterJinGo/nq_hotpotqa_train. Its name does not mean the test split contains
only NQ/HotpotQA: the pinned test has51,713 questions from seven datasets.
See README for exact counts. Both backbone repositories use the same file
revision, full-file checksum, preprocessor, ordered question IDs and budgets.

Validation uses StatefulDataLoader shuffle=false/drop_last=false. Its batch
size512 is capacity/concurrency, not a task sampler. There are101 full batches
and one final singleton. The native loop iterates over every batch. Validation
runs at update0 and150 by default. Do not insert a randomly drawn512 subset or
the unrelated WebShop140 set and still call it the original evaluation.

Each question gets one greedy trajectory: do_sample=false, temperature0, n1,
max4 actions, history4,4096 prompt tokens,512 response tokens per action, top3
retrieved passages. No Teacher, skill or labels are passed to Actor prompts.
Ground truth is used internally by the environment's verifier only.

## Scoring and aggregation are different operations

Native SearchEnv computes binary reward at termination using the original
normalized exact-match function. It extracts the last complete answer block
from the executed interaction history, normalizes case/punctuation/articles/
whitespace and accepts any alias in ground_truth.target. No answer block is0.
An explanation that merely contains the correct short answer is not EM.
The original projection and action-valid flag are preserved without a new
strict-format execution gate.

The original reward manager repeats episode reward on each active decision
row. The native `_validate` then computes data-source `test_score` from these
rows; that statistic can be weighted by trajectory length. It separately
averages batch-level success statistics, without weighting the final singleton
like a full512 batch. Thus the raw keys must not be casually relabeled as
globally deduplicated per-question EM or a seven-dataset macro-average.

The adapter calls the original `_validate` and returns its metrics unchanged.
It additionally observes the native collector's output and reports exactly
one binary reward for each original test-question index:

\[
\mathrm{EM}_d=\frac{1}{N_d}\sum_{i\in d}r_i,\qquad
\mathrm{EM}_{\mathrm{macro}}=\frac{1}{7}\sum_d\mathrm{EM}_d,\qquad
\mathrm{EM}_{\mathrm{micro}}=\frac{1}{51713}\sum_i r_i.
\]

These additional metrics are read-only reports; they do not replace rewards,
losses, advantages, early stopping or native training log values. Reports keep
all three meanings distinct. Repeated decision rows are deduplicated by input
question index, not by question text (the source may contain duplicate text).
The native collector resets output `index` within each batch, so telemetry maps
local indices back to saved input indices before aggregating. A two-batch unit
test covers that boundary and the final partial batch.

Actual ordered IDs and source counts must match the pinned receipt. Missing,
duplicated or replaced question indices fail full-test reporting. Separate
attempt directories preserve interrupted/repeated evaluations rather than
overwriting their evidence. This does not make stochastic CUDA arithmetic
or training/resume behavior bitwise deterministic.

## Original train/test split overlap

The downloaded source files contain10 shared question strings under precisely
`str(question).casefold().strip()`. This is a limited exact-string audit, not
a semantic contamination assessment. The original169,615/51,713 split is
preserved and the overlap count is saved in data_receipt.json. There is no
claim that the original split is fully disjoint. A deduplicated recipe must be
explicitly named and matched across every comparator, not silently substituted.

## What has not been evaluated by packaging/CPU checks

No Actor benchmark results, full-index retrieval quality, A100 memory fit,
GPU throughput or150-update result follows from a source/data/HTTP-fixture
check. The repository includes no fabricated score table.
