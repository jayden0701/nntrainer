
# Qwen3 KV transfer evaluation

The native CPU runner scores every supplied continuation token with teacher
forcing. ARC-Challenge uses its test split, and HellaSwag uses its validation
split. Both use zero-shot completion prompts and select the answer by total
conditional log-likelihood divided by the number of Unicode characters in the
answer text. The score includes the leading delimiter, and the denominator
counts the answer itself. [ARC task configuration](https://github.com/EleutherAI/lm-evaluation-harness/tree/d6de81643928d653435c431bae19945d41d32520/lm_eval/tasks/arc)
[HellaSwag task and preprocessing](https://github.com/EleutherAI/lm-evaluation-harness/tree/d6de81643928d653435c431bae19945d41d32520/lm_eval/tasks/hellaswag)
[Harness scoring implementation](https://github.com/EleutherAI/lm-evaluation-harness/blob/d6de81643928d653435c431bae19945d41d32520/lm_eval/api/task.py)

The prepared manifest pins the exact context and candidate token IDs, selected
row indices, dataset revisions, and input hashes before scoring. Each choice is
tokenized together with its prompt. Preparation rejects an example if joining
any choice changes the context token IDs. WikiText-2 uses the raw test split
with its rows joined by blank lines, tokenized as one stream. The runner scores
the 1,024-token suffix in each sampled non-overlapping 2,048-token chunk.
[Paper, Appendix D](https://arxiv.org/html/2608.03893v1#A4)

## Prepare a sample

Use Python with NumPy and Hugging Face Tokenizers. For Parquet inputs also
install PyArrow. The dataset specification names a local JSONL, JSON array,
or Parquet file for each enabled task and records its upstream revision. The
relative file names below are resolved next to the specification.

~~~json
{
  "arc_challenge": {
    "file": "arc.jsonl",
    "source": "https://huggingface.co/datasets/allenai/ai2_arc",
    "revision": "210d026faf9955653af8916fad021475a3f00453",
    "split": "test"
  },
  "hellaswag": {
    "file": "hellaswag.jsonl",
    "source": "https://huggingface.co/datasets/Rowan/hellaswag",
    "revision": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
    "split": "validation"
  },
  "wikitext2": {
    "file": "wikitext.jsonl",
    "source": "https://huggingface.co/datasets/Salesforce/wikitext",
    "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
    "split": "test"
  }
}
~~~

Preserve the dataset row order when exporting JSONL. WikiText-2 reads the
raw test configuration and includes the empty rows in its input stream.

~~~sh
python tools/kv_transfer/benchmark.py prepare \
  artifacts/ridge-paper-data/bench-spec.json \
  artifacts/models/qwen3-1.7b-fp32/tokenizer.json \
  artifacts/ridge-paper-data/bench-manifest-128.json \
  --seed 202608 --arc 128 --hellaswag 128 --wikitext2 4
~~~

## Score the sample

Use a native CPU KV bridge build with the teacher-forcing scorer and the FP32
Qwen3 packs. Supply a mapping fitted on independent calibration data. Pass a
single hyphen in place of the mapping path to collect cold and exact-control
scores before the mapper is ready. Repeat the command with the mapping path
and the same output directory to finish the paired comparison.

~~~sh
python tools/kv_transfer/benchmark.py run \
  build-cpu/Applications/CausalLM/nntr_causallm \
  artifacts/models/qwen3-0.6b-fp32 artifacts/models/qwen3-1.7b-fp32 \
  path/to/mapping artifacts/ridge-paper-data/bench-manifest-128.json \
  artifacts/ridge-benchmark --threads 4 --score-block-rows 64 --batch-cases 16
~~~

For compatible FP32 Qwen3 packs, the optional score block size projects
multiple continuation rows in each native forward pass. The default block
size of one scores each continuation ID with a separate forward pass.

The cold target receives the full context. For cache import, the target
recomputes the final context token to obtain logits for the first continuation
token. The source and target exports cover the preceding tokens, so the
WikiText-2 import covers 1,023 cached tokens followed by one recomputed
context token. The exact target import controls run on the first two frozen
cases per task, or every available case if fewer were sampled. The runner
checks their maximum absolute per-token log-probability error against
`--exact-tolerance` (0.01 nats by default) and requires identical answer
predictions.

The output directory stores the run fingerprints, per-case token
log-probabilities, native logs, and `summary.json` with counts for each
mode and task. A rerun reuses complete case scores when their run
fingerprints match. `--tasks` selects which tasks to start, and `--limit`
caps the number of new cases for that invocation. `--batch-cases` lets cases
with the same input plane share a native process. Cache files are
temporary. Score durations are per process and include model loading.

For classification, the summary computes transfer accuracy divided by cold
target accuracy and floor-normalized retention using the paper's 25% floor.
Floor-normalized retention is omitted as a number when the paired cold target
scores at or below 25%. The paired correctness counts indicate how many
questions each condition gained or lost. The 95% paired bootstrap interval
resamples these question pairs with replacement. The summary records cases
where the top two normalized choice scores differ by less than 0.002 nats
per character. WikiText-2 aggregates negative log-likelihood over the scored
tokens before exponentiating.
[Paper, Section 4.1](https://arxiv.org/html/2608.03893v1#S4.SS1)

Qwen3-0.6B to Qwen3-1.7B is an additional model pair. The expanded
manifest selects 128 rows per classification task and four WikiText-2 test
chunks. It covers three of the paper's benchmarks and uses the native
target's FP32 weights and FP16 KV cache. Smaller sample limits with the same
seed select the corresponding initial rows of each ranked sample.

## Check close choice scores

After scoring, the scalar audit rechecks the modes flagged by the margin
threshold and scores every candidate for those cases. It compares the
resulting winners and log-probabilities with the block scorer and saves its
own summary under `audit-scalar/`. Rerun it after new benchmark cases complete
to check any additional flags.

~~~sh
python tools/kv_transfer/benchmark_audit.py \
  build-cpu/Applications/CausalLM/nntr_causallm \
  artifacts/ridge-paper-data/bench-manifest-128.json \
  artifacts/ridge-benchmark
~~~
