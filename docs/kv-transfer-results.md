# Ridge and CacheBridge experiment results

The local Qwen3-0.6B to Qwen3-1.7B CPU experiments used the KV manager
from [PR #4481](https://github.com/nntrainer/nntrainer/pull/4481) at
`5e97254e764aa30db4223c33d0dc3b0fded091b8`. The Ridge experiments ended
at `379cbae3` and the tracked CacheBridge experiments at `b5757e97`.
Later generation, agent and CUDA capacity probes used local harnesses.
The [implementation guide](../tools/kv_transfer/README.md) describes the
review branch's build, training and CLI.

## Models, mapper training and controls

Most experiments use batch-one Qwen3 FP32 weights and FP16 KV on WSL x86.
The Qwen3-0.6B model revision is `c1899de289a04d12100db370d81485cdf75e47ca`
and the Qwen3-1.7B revision is `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
Ridge concatenates all KV heads from the selected source layers. Compact
Ridge uses the corresponding source head from each layer. CacheBridge
starts with compact Ridge and calibrates V with the Appendix B readout
objective. [Ridge paper](https://arxiv.org/abs/2608.03893) · [CacheBridge v2](https://arxiv.org/html/2609.00891v2)

The matched fits use FineWeb-Edu revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`
with seed 20261117, 1,024 tokens per sequence, stride four, four source layers
per target layer and ridge penalty 0.01. The 120-sequence fit retains the
50 original token streams and KV snapshots and appends 70 sequences from
the same selection order. All layouts and both sample sizes use the layer
order ranked on the initial 50 sequences. QuALITY v1.0.1 train contributes
24 documents and 96 queries to V calibration. The independent dev evaluation
uses 24 different documents and 96 queries. The energy reference uses
4,096 sampled initial-fit KV states with a 1% residual-energy budget.

| Initial fit | Fit observations | QuALITY V readout SSE before → after | K after V calibration |
| --- | ---: | ---: | --- |
| 50 sequences | 12,800 | 590,282.22 → 118,533.77 (−79.9%) | Bitwise identical |
| 120 sequences | 30,720 | 553,690.59 → 120,839.22 (−78.2%) | Bitwise identical |

With 120 sequences, full-head fit took 453.5 s, compact fit 165.8 s and
V calibration 379.0 s. These were local measurements at different thread
settings, with overlapping fit workloads. Maximum stored update energy
divided by budget was 1.00000005. The benchmark artifacts used these exact
mapper files. The paper compares its CFM Ridge baseline and CacheBridge on the same
500 initial FineWeb-Edu sequences. Its model pairs and serving stack differ
from this CPU PoC.

## Ridge reproduction and numerical validation

The early eight-passage pilot fitted headwise Ridge on six synthetic
passages. Held-out relative L2 was 0.2075 for K and 0.7262 for V, against
0.4732/1.0497 and 0.4717/1.0465 for direct copy on the two held-out inputs.
A same-model seven-token prefix round trip generated the same 17 IDs in the
legacy, contiguous core, FP32 snapshot and native snapshot paths. A second
continuation produced the same six IDs after cold prefill and exact target
import. Mapped and direct-copy caches produced different IDs.

The paper-style sample used 128 ARC-Challenge test questions, 128
HellaSwag validation questions and 4,096 WikiText-2 suffix tokens. Choice
scores used log-likelihood per character. Six exact target-cache controls
matched cold scoring bit for bit. Scalar rescoring of 11 close-margin
case/mode pairs changed no winners, with a maximum token log-probability
difference of 0.002382 nat.

| Evaluation | Target re-prefill | Local full-head Ridge, fit-50 | Public Ridge, fit-120 |
| --- | ---: | ---: | ---: |
| ARC-Challenge accuracy | 59/128 | 42/128 | 44/128 |
| HellaSwag accuracy | 71/128 | 54/128 | 61/128 |
| WikiText-2 perplexity | 13.0364 | 14.7955 | 14.0856 |

The public mapper uses the external `k4-n200` artifact, with 120 sequences
assigned to fitting. Public-minus-local accuracy was +1.56 percentage
points on ARC and +5.47 on HellaSwag. Paired 95% bootstrap intervals were
[−4.69, +7.81] and [−1.56, +12.50] points, respectively. The later
matched-protocol run used the published 1,000 HellaSwag IDs, the published
candidate tokenizer and 16-token handoff boundary, and log-likelihood per
candidate token. The preliminary 128-question and public 1,000-question
samples overlap on 12 cases. [Public Ridge implementation](https://github.com/alesha-pro/vllm-cross-model-kv) · [Public mapper](https://huggingface.co/anonymousmaharaj/vllm-cross-model-kv)

| Same public HellaSwag 1,000 | Target re-prefill | Public Ridge | Retention |
| --- | ---: | ---: | ---: |
| nntrainer CPU FP32 model, FP16 KV | 58.2% | 55.1% | 94.67% |
| Transformers CPU BF16 | 58.2% | 55.5% | 95.36% |
| Public vLLM BF16 | 58.0% | 55.3% | 95.34% |

Changing the replay boundary alone raised the public mapper from 54 to
58 correct on the old 128-item sample with token normalization. Aligning
text and replay boundary on that sample gave nntrainer 72/58 correct and
Transformers 69/59 for re-prefill/transfer. Candidate choices agreed on
125/128 in each mode. On the public 1,000, FP32 Transformers matched
nntrainer on all 11 re-prefill disagreements and 14 of 15 transferred
disagreements found against BF16. For case 5444 the two backends agreed
with the same imported cache within 0.000119 nat/token. Each converted
model's 310 FP32 weight tensors matched the source Safetensors after
unpacking and transposition. Cross-runtime spot checks found maximum
K/V relative L2 of 0.0597%/0.1093% for 0.6B and 0.0494%/0.0951% for 1.7B.
Maximum token log-probability differences were 0.00417 and 0.00647 nat.

### Why the public and local 120-sequence maps differ

Both pipelines locate their data at the same FineWeb-Edu revision and use
length 1,024, stride four, k=4 and penalty 0.01. The public manifest uses a
streaming shuffle with seed 0 and buffer 10,000. The local fit shuffles rows
from one pinned `sample/350BT` shard with seed 20261117. The same number of
sequences therefore identifies different sampling procedures.

An OLS layer reselection on the local 120 sequences retained 111 of 112
source-layer slots from the 50-sequence list when order is ignored. The
120-sequence local selection and the public selection shared 73 of 112
slots and had identical sets at four of 28 target layers. On 12 separately
held-out FineWeb-Edu sequences, mean K/V R² was 0.72405/0.56727 for the
local fit, 0.72459/0.56885 for a local refit with the public layer list,
and 0.72563/0.56953 for the public coefficients. A single-layer solver
cross-check on the same 30,720 observations agreed to relative L2 of
1.03e−9 for K and 1.13e−8 for V. This isolates a small effect from layer
selection on the held-out reconstruction sample. Multiple corpus,
precision and task differences remain in the public comparison.

## CacheBridge accuracy

HellaSwag 200 uses positions 0–99 and 500–599 in the public ordering.
Of the 1,000 public-order examples, 968 have a transferable prefix. The
matched QuALITY columns use one-letter answer logits and the same dev
documents at both fit sizes.

| Method | QuALITY 96, fit-50 | QuALITY 96, fit-120 | HellaSwag 200, fit-50 | HellaSwag 200, fit-120 | HellaSwag 1,000 | Transferable 968 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Target re-prefill | 53 | 53 | 111 | 111 | 582 | 567 |
| 0.6B standalone | 43 | 43 | — | — | 448 | 436 |
| Public Ridge, fit-120 | — | — | — | 100 | 551 | 536 |
| Local full-head Ridge | 49 | 45 | 92 | 98 | 522 (fit-120) | 507 |
| Local compact Ridge | 50 | 51 | 98 | 99 | 518 (fit-120) | 503 |
| CacheBridge, V calibrated | 57 | 56 | 99 | 99 | 535 (fit-120) | 520 |
| CacheBridge, V calibrated, fit-50 | — | — | — | — | 538 | 523 |

The 0.6B standalone run used the same held-out questions and candidate
tokens. It retained 81.1% of the target correct count on QuALITY
and 77.0% on HellaSwag 1,000. The QuALITY score applies to both fit
conditions.

The early HellaSwag smoke test gave 9/16 for re-prefill, 8/16 for local
Ridge and 10/16 each for compact and calibrated. In the larger fit-120
runs, CacheBridge gained 11/96 on QuALITY over the local full-head Ridge
(+11.5 percentage points, document-paired 95% interval +4.2 to +19.8).
The corresponding HellaSwag improvements were +1/200 and +13/1,000
(+1.3 points, 95% interval −0.5 to +3.1 for 1,000). On 1,000, V calibration
gained 17 over its compact initializer (95% interval +0.3 to +3.1 points).
The fit-50 to fit-120 change was −1/96 on QuALITY and −3/1,000 on HellaSwag
for CacheBridge. The public Ridge artifact scored 16 higher than the local
fit-120 CacheBridge on HellaSwag, with its own sampling and coefficients.
Native accuracy runs recorded zero fallback events.

## CPU handoff latency

The table reports median TTFT in seconds across three warm, serial repeats
with models and mappers resident, five threads and blocks of 128 tokens.
The source prefix is ready when the clock starts. The target computes the
same suffix in each mode.

| History tokens | Target re-prefill | Full-head Ridge | Compact Ridge | CacheBridge |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 3.301 | 1.104 | 0.802 | 0.812 |
| 1,024 | 6.905 | 1.930 | 1.340 | 1.340 |
| 2,048 | 15.688 | 3.609 | 2.423 | 2.416 |
| 4,096 | 39.835 | 7.208 | 4.775 | 4.807 |

At 4,096 tokens the calibrated path had 8.29× faster time to the first token than re-prefill
and 33.3% lower TTFT than full-head Ridge. FP32 weights were 896 MiB for
full-head and 112 MiB for compact or calibrated, plus 224 KiB bias in
each case. Mapper scratch for the 128-token block was 2,688 KiB and
448 KiB, respectively. An injected FP16 overflow late in transfer
discarded the incomplete step and returned the same first token and
scores as re-prefill. On eight HellaSwag smoke cases, in-memory and
snapshot Ridge selected all the same candidates, with a maximum
difference of 0.000141 nat per candidate token.

## Generation and effective output budget

On a short capital-city prompt with thinking, 0.6B alone, 1.7B re-prefill
and CacheBridge generated 69, 161 and 157 tokens and all answered correctly.
With thinking off each produced two tokens. Three harder pilot prompts
hit the 320-token cap in all modes and yielded no scored answer. On the
2,581-token QuALITY prefix in that pilot, cached TTFT savings translated
to 64.09 s E2E against 81.20 s for re-prefill at the same generation cap.
The mapped arm's E2E clock starts after source preparation.

The GSM8K experiment used four fixed demonstrations, thinking on, a
497-token shared prefix, batch one, and per-case matched seeds with
temperature 0.6, top-k 20 and top-p 0.95. The first eight test IDs used
seed 20261124 and 768 output tokens. The 16 extra IDs, drawn without
replacement from the remaining test questions with seed 20261125, ran
all four paths serially with a 2,048-token cap. Their 768-token column
scores the first 768 IDs of each trace. Numeric grading recognizes
bold digits and an explicit last-line `Answer:` label alongside the
original final-answer formats.

| Mode | First 8 at 768 | All 24 at 768 | New 16, 768 → 2,048 | Cap hits on new 16, 768 → 2,048 | Mean tokens at 2,048 | Mean TTFT / E2E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.6B standalone | 4 | 14 | 10 → 10 | 4 → 3 | 695 | 1.86 / 51.4 s |
| 1.7B re-prefill | 5 | 16 | 11 → 13 | 4 → 1 | 644 | 4.20 / 104.1 s |
| Local Ridge fit-120 | 5 | 13 | 8 → 13 | 7 → 1 | 853 | 1.58 / 134.9 s |
| CacheBridge fit-120 | 3 | 10 | 7 → 13 | 8 → 1 | 795 | 1.17 / 124.7 s |

The 1.7B paths solved the same 13 questions at 2,048. Compared with
re-prefill, CacheBridge generated 151 more tokens on average and took
20.5 s longer E2E despite saving 3.03 s to the first token. Against
Ridge it saved 58 tokens and 10.2 s E2E on average, with a lower E2E
on 13/16 cases. The paired bootstrap interval for the mean E2E delta
against Ridge was [−31.3, +13.9] s. Aggregate decode speed was roughly
6.4 tokens/s for the 1.7B paths and 14.0 for standalone 0.6B. First-cohort
timings include concurrent runs and are omitted from the latency table.

The 2,048-token cap balances the PoC wall time against truncated answers.
An independent public Qwen3-1.7B evaluation reports a mean of 1,005
thinking tokens on correct GSM8K responses. The paper's own CacheBridge
GSM8K protocol uses eight-shot prompts with thinking off. [GSM8K](https://arxiv.org/abs/2110.14168) · [TRSD Table 7](https://arxiv.org/html/2603.13274v1) · [CacheBridge v2 Appendix A](https://arxiv.org/html/2609.00891v2#A1)

### Expanded GSM8K sample

The first eight questions were matched at 2,048 tokens by reusing 18
traces that had already reached EOS and extending the 14 traces capped
at 768 tokens. All 32 case/mode pairs matched their previous output
through the first 768 tokens or EOS. The resulting 24-question scores
were 14 for 0.6B, 19 for re-prefill, 19 for Ridge and 18 for CacheBridge.

Eight more test rows were sampled without replacement from the
remaining 1,295 questions using seed 20261126. The same four modes ran
serially with the same prompts, sampling settings and fit-120 mappers.
For the expanded 32-question sample, the 768-token scores and lengths
come from truncating the same traces. Generation counts include the
thinking and answer tokens. Retention divides a method's correct count
by the 1.7B re-prefill correct count at the same cap. Compact fit-120
later ran on all 32 frozen requests with the same settings. Final
number extraction handles escaped currency and trailing unit words
on explicit answer lines. This normalization was checked against
every previously scored case at both caps.

| Mode | Added 8 correct at 2,048 | All 32 correct at 2,048 (retention) | Mean generated tokens, 768 → 2,048 (change) | Cap hits, 768 → 2,048 |
| --- | ---: | ---: | ---: | ---: |
| 0.6B standalone | 5/8 | 19/32 (73.1%) | 515.1 → 858.6 (+343.5) | 12 → 7 |
| 1.7B re-prefill | 7/8 | 26/32 (100.0%) | 532.7 → 682.8 (+150.1) | 10 → 1 |
| Local Ridge fit-120 | 6/8 | 25/32 (96.2%) | 634.6 → 843.8 (+209.2) | 14 → 2 |
| Local Compact fit-120 | 7/8 | 25/32 (96.2%) | 673.3 → 960.5 (+287.2) | 17 → 3 |
| CacheBridge fit-120 | 7/8 | 25/32 (96.2%) | 635.5 → 914.3 (+278.8) | 17 → 4 |

CacheBridge generated 231.5 tokens more than re-prefill on average at
2,048, including 179.7 additional thinking tokens. The mean difference
against Ridge was +70.5 generated tokens. CacheBridge was shorter on
18/32 questions, and averaged 8.5 fewer tokens on the 28 questions
where both modes reached EOS. Each method solved one question that
the other missed. On the added eight serial runs, mean E2E was 84.9 s
for 0.6B, 103.1 s for re-prefill, 125.2 s for Ridge and 146.3 s for
CacheBridge. Across all 32, Compact generated 46.2 more tokens on
average than CacheBridge. Each solved one question that the other
missed.

### Matched GSM8K timing

Timing uses the 24 questions with serial measurements for every path.
The initial eight questions used a mix of saved and repeated runs.
Compact was measured in a later session with the same CPU settings.
For transfer, TTFT starts with the source KV ready and includes KV
mapping and installation. Subtracting the recorded prepare interval
leaves the target query and first-token sampling. The standalone and
re-prefill paths compute the full prompt. E2E runs through EOS or the
generation cap.

| Mode | TTFT mean / median (s) | Mean TTFT excluding KV preparation (s) | E2E mean / median (s) |
| --- | ---: | ---: | ---: |
| 0.6B standalone | 1.868 / 1.870 | 1.868 | 62.56 / 26.90 |
| 1.7B re-prefill | 4.202 / 4.201 | 4.202 | 103.80 / 74.73 |
| Local Ridge fit-120 | 1.580 / 1.567 | 0.694 | 131.65 / 113.74 |
| Local Compact fit-120 | 1.191 / 1.185 | 0.704 | 155.38 / 114.47 |
| CacheBridge fit-120 | 1.183 / 1.189 | 0.695 | 131.88 / 111.02 |

Mean KV preparation took 0.885 s for Ridge, 0.487 s for Compact and
0.488 s for CacheBridge. The latter two had similar TTFT. Compact
took 23.50 s longer on average than CacheBridge, with a 3.45 s
difference between medians. On its complete 32-question run,
Compact averaged 1.185 s TTFT and 154.62 s E2E.

### Review of capped outputs

In the original four-mode comparison, 14 outputs across nine questions
hit the 2,048-token cap. The table summarizes a manual review of their
reasoning traces and answer text. The likelihood of reaching a correct answer with more tokens is
a qualitative judgment. The two measured extensions used the same
prompts and seeds, and both matched the original first 2,048 tokens.
They were selected because CacheBridge hit the cap and both other
1.7B paths had already reached EOS.

| Test row | Capped mode | Observation |
| --- | --- | --- |
| 7, download restart | Ridge and CacheBridge | CacheBridge repeatedly computed the correct 160 minutes near the cap. Ridge also computed 160, then continued to alternate between a full restart and a 120-minute partial resume. |
| 20, drink mixture | 0.6B | Repeatedly retained the initial water total after spilling part of the orange drink. |
| 76, puppy food | 0.6B | Proposed the correct count of five bags with a 360-day year and inconsistent division. |
| 354, phone savings | CacheBridge | Computed the correct 45 dollars early, then compared a total including savings to the cost still outstanding. The 3,072-token run reached EOS at 2,438 tokens with an answer of zero. |
| 607, shampoo | 0.6B and CacheBridge | Both confused pumps per wash with the number of pumps in the bottle. At 3,072, CacheBridge had emitted 120 cents in the answer text and hit the cap. The expected answer was 10 cents. |
| 931, ages | All four | The question uses Jean and John. All four struggled with the missing constraint when treating them as distinct people. The reference answer of 50 follows if both names identify the same person. |
| 990, air conditioning | 0.6B | Treated high-mode cooling as heating near the cap. |
| 1241, egg sales | 0.6B | Computed the correct profit of eight dollars mid-trace, then used the wrong number of groups per dozen and returned to incorrect amounts. |
| 1301, cookies | 0.6B | Repeatedly computed the correct total of 23, then cut off during another reinterpretation. |

CacheBridge on row 7 and 0.6B on row 1301 showed the strongest evidence
of a correct answer within reach in the observed prefixes.
Row 931 accounts for the one re-prefill cap hit and for one cap hit
in each transfer mode. Compact reached the cap on rows 20, 607 and 931.

## Tool-use pilot

The retail sandbox ran eight deterministic cases. All handoff arms shared
the initial 0.6B tool call and observation. The harness used greedy
generation, thinking off, at most 96 tokens per turn and at most four
target turns.

| Mode | Task successes | Mean generated tokens | Mean tool calls | Mean E2E |
| --- | ---: | ---: | ---: | ---: |
| 0.6B standalone | 0/8 | 48.0 | 1.0 | 5.11 s |
| 1.7B re-prefill | 6/8 | 108.8 | 3.5 | 21.36 s |
| CacheBridge fit-120 | 0/8 | 112.5 | 1.0 | 17.28 s |

The cached arm stopped earlier because it made no successful subsequent
tool calls. The isolated next-token probe showed that cold and exact
target histories chose the tool-start token on all eight recovery
prompts and both valid-lookup prompts. Local Ridge, compact, CacheBridge
and public Ridge each chose it on zero of the ten. The independent
Python-mapped snapshot reproduced the low probability of that token.
For the first recovery probe, the cold/exact probability was about
0.99987 and calibrated CacheBridge was about 1.2e−12. Recomputing 128
target suffix tokens recovered the tool-start decision on all eight
recovery prompts but produced zero schema-correct lookup calls. A deeper
512-token replay recovered a schema-correct call on the one case tried.
The tool/chat structural tokens seen in the probe were absent from the
sampled FineWeb and QuALITY fit inputs.

## Additional controls

A distinct cross-family C2C pilot fused Qwen3-0.6B KV into SmolLM2-135M.
It preserved HellaSwag accuracy at 15/32 in both Transformers and native
CPU. WikiText-2 mean NLL on 32 passages changed from 2.981126 to 2.980801
in Transformers and from 2.981197 to 2.980820 in native CPU. The native
paired interval [−0.001047, +0.000276] includes zero. Six exact imports
matched native cold scoring bitwise. This pilot trained a different
mapper and ran fusion outside the native models.

A separate CUDA smoke test allocated Qwen3-1.7B context sizes of 4,096
and 32,768 on a GPU reporting 12,227 MiB total VRAM. It used FP16 body
weights and an FP32 tied head. Both runs executed a 64-token prefill and
four forced decode steps and reached cache position 68. Minimum sampled
free VRAM was 9,589 MiB and 6,433 MiB, respectively. These measurements
verify allocation and short execution at both capacities. Sustained
decoding near 32,768 tokens was outside the test.

## Evidence locations

Raw results are in the original experiment worktrees. The code branch carries
this report and the reproduction guide. Paths below are relative to the
respective original worktree.

| Evidence | Location |
| --- | --- |
| Ridge pilot, paper-style scores and HF/native parity | `docs/validation/kv-transfer/` at `379cbae3` |
| CacheBridge calibration, latency, QuALITY and HellaSwag | `docs/validation/cachebridge-*.json` at `b5757e97` |
| 50-to-120 training inputs, timing and layer/solver audit | `artifacts/fit120/training-summary.json` and `artifacts/fit120/audit/` |
| Generation pilots and four-path GSM8K audit | `artifacts/e2e-pilot/` and `artifacts/fit120/decode-agent/gsm8k24/` |
| GSM8K 32-question and targeted tail results | `summary-90min.json` and `summary-cachebridge-tail.json` in the GSM8K audit directory |
| Compact fit-120 GSM8K and matched timing | `summary-compact120-gsm8k32.json` and `existing-gsm8k24-timing-breakdown.json` in the GSM8K audit directory |
| 0.6B standalone QuALITY and HellaSwag | `artifacts/standalone-06b/summary.json` |
| Tool-use pilot and replay diagnostics | `artifacts/fit120/decode-agent/agent-summary.json` and `diagnostic/` in the same directory |
| CUDA capacity | `artifacts/gpu-decode-capacity/summary.json` |
