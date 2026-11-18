
"""Regression checks for Ridge benchmark boundaries and metrics."""

import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from benchmark import (
    NativeScorer, choice_example, choice_margins, encode_choice, rank_sample, summarize,
    summarize_choices,
)


class FakeTokenizer:
    def __init__(self, encodings):
        self.encodings = encodings

    def encode(self, text, *, add_special_tokens):
        if add_special_tokens:
            raise AssertionError("completion scoring must not add special tokens")
        return SimpleNamespace(ids=self.encodings[text])


class BenchmarkTest(unittest.TestCase):
    def test_joint_tokenization_and_character_denominator(self):
        tokenizer = FakeTokenizer({"Q:": [1, 2], "Q:  é": [1, 2, 3],
                                   "Q:  xyz": [1, 2, 4, 5]})
        result = encode_choice(tokenizer, "Q: ", ["é", "xyz"], 0)
        self.assertEqual(result["context_ids"], [1, 2])
        self.assertEqual(result["candidates"], [
            {"text": "é", "characters": 1, "ids": [3]},
            {"text": "xyz", "characters": 3, "ids": [4, 5]},
        ])

    def test_reject_joined_encoding_that_changes_context(self):
        tokenizer = FakeTokenizer({"Q:": [1, 2], "Q: !!!": [1, 7, 8]})
        with self.assertRaisesRegex(ValueError, "across the context boundary"):
            encode_choice(tokenizer, "Q:", ["!!!", "second"], 0)

    def test_dataset_formats(self):
        arc = {"question": "Why?", "choices": {"label": ["1", "2"], "text": ["a", "b"]},
               "answerKey": "2"}
        self.assertEqual(choice_example("arc_challenge", arc),
                         ("Question: Why?\nAnswer:", ["a", "b"], 1))
        hella = {"activity_label": "Sketch", "ctx_a": "Intro [title] Part.",
                 "ctx_b": "a PERSON", "endings": [" one [aside] ", "two"], "label": "0"}
        self.assertEqual(choice_example("hellaswag", hella),
                         ("Sketch: Intro. Part. A person", ["one ", "two"], 0))

    def test_sampling_is_stable_and_prefix_preserving(self):
        sample = list(enumerate("abcdef"))
        first = rank_sample(sample, "hellaswag", 10, 4)
        self.assertEqual(first, rank_sample(reversed(sample), "hellaswag", 10, 4))
        self.assertEqual(first[:2], rank_sample(sample, "hellaswag", 10, 2))

    def test_persistent_batch_preserves_case_and_candidate_id_files(self):
        scorer = NativeScorer.__new__(NativeScorer)
        scorer.max_seq_len = 2048
        scorer.score_block_rows = 64
        scorer.spec = object()
        with TemporaryDirectory() as directory:
            work = Path(directory)
            cases = []
            for case_id, anchor, continuation in [("a", 2, [12, 13]), ("b", 3, [22, 23])]:
                scratch = work / case_id
                scratch.mkdir()
                example = {"id": case_id, "context_ids": [1, anchor],
                           "candidates": [{"ids": continuation}, {"ids": continuation[::-1]}]}
                cases.append((example, scratch, {"cold": None}))

            def execute(_model, arguments, _log, extra):
                self.assertEqual(arguments[0], "--requests")
                self.assertEqual(extra, {"NNTR_SCORE_BLOCK_ROWS": "64"})
                requests = json.loads(Path(arguments[1]).read_text())
                for index, request in enumerate(requests):
                    job = request["score"]
                    context = [int(item) for item in Path(job["input_ids"]).read_text().split()]
                    target = [int(item) for item in Path(job["target_ids"]).read_text().split()]
                    self.assertEqual(context, [1, 2 + index // 2])
                    expected = [12 + 10 * (index // 2), 13 + 10 * (index // 2)]
                    self.assertEqual(target, expected[::1 if index % 2 == 0 else -1])
                    Path(job["output"]).write_text(json.dumps({
                        "count": 2, "sum_logprob": -2 * (index + 1),
                        "token_logprobs": [-(index + 1)] * 2, "score_block_rows": 64,
                    }))
                return 0.25

            with patch.object(scorer, "model_config"), patch.object(scorer, "execute", execute):
                results, elapsed = scorer.score(work, work / "score.log", cases, persistent=True)
            self.assertEqual(elapsed, 0.25)
            self.assertEqual(list(results), ["a", "b"])
            self.assertEqual([score["sum_logprob"] for score in results["a"]["cold"]], [-2, -4])
            self.assertEqual([score["sum_logprob"] for score in results["b"]["cold"]], [-6, -8])

    def test_import_replays_complete_uncached_tail(self):
        scorer = NativeScorer.__new__(NativeScorer)
        scorer.max_seq_len, scorer.score_block_rows = 64, 1
        scorer.spec = object()
        with TemporaryDirectory() as directory:
            work = Path(directory)
            snapshot = SimpleNamespace(path=work / "snapshot", spec=scorer.spec,
                                       tokens=2, token_ids=(10, 11), positions=(0, 1))
            case = {"id": "a", "context_ids": [10, 11, 12, 13, 14],
                    "candidates": [{"ids": [15]}]}
            jobs = [(case, work, {"transfer": snapshot})]

            def execute(_model, arguments, _log, _extra):
                job = json.loads(Path(arguments[1]).read_text())[0]["score"]
                self.assertEqual(job["prefix_tokens"], 2)
                self.assertEqual(Path(job["input_ids"]).read_text().split(), ["12", "13", "14"])
                Path(job["output"]).write_text(json.dumps(
                    {"count": 1, "sum_logprob": -1, "token_logprobs": [-1]}))
                return 0

            with patch.object(scorer, "model_config"), patch.object(scorer, "execute", execute):
                scorer.score(work, work / "log", jobs, persistent=True)
                case["context_ids"] = [10, 11]
                with self.assertRaisesRegex(ValueError, "uncached"):
                    scorer.score(work, work / "log", jobs, persistent=True)
                case["context_ids"] = [11, 10, 12]
                with self.assertRaisesRegex(ValueError, "snapshot"):
                    scorer.score(work, work / "log", jobs, persistent=True)

    def test_retention_and_paired_correctness(self):
        cases = [{"gold": 0, "predictions": {"cold": cold, "transfer": transfer}}
                 for cold, transfer in [(0, 0), (0, 1), (1, 0), (1, 1)]]
        cases[0]["predictions"]["exact"] = 0
        result = summarize_choices(cases)
        self.assertEqual(result["accuracy"], {"cold": 0.5, "transfer": 0.5, "exact": 1.0})
        self.assertEqual(result["transfer_retention_pct"], 100)
        self.assertEqual(result["transfer_floor_normalized_retention_pct"], 100)
        self.assertEqual(set(result["cold_transfer_paired_counts"].values()), {1})

        cases[1]["predictions"]["cold"] = 1
        result = summarize_choices(cases)
        self.assertEqual(result["transfer_retention_pct"], 200)
        self.assertIsNone(result["transfer_floor_normalized_retention_pct"])

    def test_partial_transfer_uses_the_same_cases_as_cold(self):
        cases = [
            {"gold": 0, "predictions": {"cold": 0}},
            {"gold": 0, "predictions": {"cold": 1, "transfer": 0}},
        ]
        result = summarize_choices(cases)
        self.assertEqual(result["accuracy"]["cold"], 0.5)
        self.assertEqual(result["paired_cold_accuracy"], 0)
        self.assertEqual(result["transfer_minus_cold_pp"], 100)
        self.assertIsNone(result["transfer_retention_pct"])
        self.assertIsNone(result["paired_bootstrap_95_ci_pp"])

    def test_small_margin_is_measured_after_character_normalization(self):
        manifest = {"case": {"candidates": [{"characters": 10}, {"characters": 5}]}}
        result = {"id": "case", "scores": {
            "cold": [{"sum_logprob": -10.01}, {"sum_logprob": -5}],
            "transfer": [{"sum_logprob": -10.2}, {"sum_logprob": -5}],
        }}
        close = choice_margins(manifest, [result])
        self.assertEqual(list(close), ["cold"])
        self.assertAlmostEqual(close["cold"][0]["margin_nat_per_char"], 0.001)

    def test_perplexity_aggregates_token_nll_before_exponentiating(self):
        manifest = {"items": [{"task": "wikitext2"}] * 3}
        results = [
            {"task": "wikitext2", "exact_max_token_logprob_error": 0, "scores": {
                "cold": [{"count": 1, "sum_logprob": -2}],
                "exact": [{"count": 1, "sum_logprob": -2}],
                "transfer": [{"count": 1, "sum_logprob": -3}],
            }},
            {"task": "wikitext2", "scores": {
                "cold": [{"count": 3, "sum_logprob": -6}],
                "transfer": [{"count": 3, "sum_logprob": -9}],
            }},
        ]
        result = summarize(manifest, results, 0.01)["wikitext2"]
        self.assertEqual((result["scored"], result["planned"]), (2, 3))
        self.assertEqual(result["scored_tokens"], {"cold": 4, "exact": 1, "transfer": 4})
        self.assertEqual(result["perplexity"]["transfer"], math.exp(3))
        self.assertEqual(result["transfer_minus_cold_nll"], 1)
        self.assertTrue(result["exact_import_check_passed"])


if __name__ == "__main__":
    unittest.main()
