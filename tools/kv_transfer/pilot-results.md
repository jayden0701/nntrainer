# Qwen3 CPU pilot

The CPU pilot used Qwen3-0.6B as source and Qwen3-1.7B as target. Both
model binaries were built in FP32 layer-graph order using
`Applications/CausalLM/res/qwen3/qwen3-0.6b/weight_converter_layergraph.py`.
Both native KV caches used FP16 storage. The bridge exported those values
as float32 and rounded imported predictions back to FP16.

| Model | Hugging Face revision | Native FP32 binary SHA-256 |
| --- | --- | --- |
| Qwen3-0.6B | `c1899de289a04d12100db370d81485cdf75e47ca` | `88ffa34cf3c2d5900b62730416aac3f1912df9e8015c6c749f3fd37814bf70ef` |
| Qwen3-1.7B | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | `175b1ec912111c9b835b8fa8b95de95f30039a5c419c7de0c5f8a9d2830703bb` |

The eight passages are in `pilot_prompts.json`. Source and target cache
exports reported matching token IDs and positions for each passage. The
seed-0 group split selected luthier and rail-calendar as the held-out
prompts, with 130 and 114 tokens respectively. The six training prompts
were observatory (173), contour-map (165), barley-soup (166), library-bus
(161), snowpack (124) and glassblower (120), totaling 909 tokens.

The fit used headwise mapping, one source layer per target layer at the
same depth, an L2 penalty of 1 and an unpenalized intercept. Its held-out
aggregate relative L2 error was 0.207512 for post-RoPE keys and 0.726171 for
values. Direct copy uses the source planes unchanged with the target
snapshot metadata, which was possible because the layouts match.

| Held-out prompt | Fitted key error | Direct-copy key error | Fitted value error | Direct-copy value error |
| --- | ---: | ---: | ---: | ---: |
| luthier | 0.208904 | 0.473239 | 0.748611 | 1.049721 |
| rail-calendar | 0.205941 | 0.471740 | 0.700552 | 1.046490 |

A greedy continuation of the luthier passage used the same 18 suffix
tokens for all imports. The `num_to_generate=5` setting recorded six
IDs including the prefill seed. Cold target and exact native target import
produced the same IDs. The fitted and direct-copy caches differed from
the cold target at the first ID.

| Target run | Recorded IDs | Prefill tokens | Process wall time |
| --- | --- | ---: | ---: |
| Cold | 785, 326, 940, 1268, 87755, 264 | 148 | 8.568 s |
| Exact target import | 785, 326, 940, 1268, 87755, 264 | 18 | 7.340 s |
| Mapped source import | 151668, 271, 1249, 62079, 279, 21085 | 18 | 11.166 s |
| Direct-copy source import | 72, 198, 82, 198, 64, 271 | 18 | 6.291 s |

These are single runs with eight native threads and one NumPy OpenBLAS
thread. Source export process times ranged from 1.637 to 12.993 seconds
and target exports from 4.178 to 20.971 seconds. Fit took 2.049 seconds.
Application took 0.177 and 0.227 seconds on the two held-out prompts.
Some runs shared the CPU with a separate bridge validation, so the times
are observations of this run. The recorded continuation covers one prompt,
and the layout comparison covers the headwise fit and direct copy.
