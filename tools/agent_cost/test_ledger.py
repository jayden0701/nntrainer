"""Tests for the local agent cost ledger."""

import json
from pathlib import Path
import tempfile
import unittest

from ledger import Ledger, LedgerError, append, import_rollout, report, rollout_snapshot, writer


class LedgerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.path = self.root / "ledger.jsonl"

    def add(self, kind, **fields):
        with writer(self.path) as ledger:
            append(self.path, ledger, kind, **fields)

    def task(self, name):
        self.add("task", id=name, task_class="documentation", acceptance="Check example commands")

    def attempt(self, name, task, role="implementation", **fields):
        self.add("attempt", id=name, task=task, actor=name, role=role, **fields)

    def usage(self, name, amount, basis="measured", complete=True):
        self.add("usage", attempt=name, basis=basis, unit="tokens", source="local-meter",
                 amount=str(amount), complete=complete)

    def validate(self, name="retry", validator="reviewer", revision="rev2", verdict="pass"):
        self.add("validation", task="one", attempt=name, validator=validator,
                 revision=revision, verdict=verdict, evidence="test commands passed")

    def summary(self):
        return report(Ledger.read(self.path))

    def test_entire_cohort_cost_includes_failed_tasks_and_review(self):
        self.task("one")
        self.attempt("first", "one", model_requested="model-small", effort_requested="low")
        self.add("outcome", attempt="first", status="failed")
        self.attempt("retry", "one", "rework", retry_of="first", model_resolved="model-large")
        self.add("outcome", attempt="retry", status="completed", revision="rev2")
        self.attempt("reviewer", "one", "review")
        self.add("outcome", attempt="reviewer", status="completed")
        self.validate()
        self.task("two")
        self.attempt("second", "two", model_requested="model-small", effort_requested="low")
        self.add("outcome", attempt="second", status="failed")
        self.add("closure", task="two", status="failed", evidence="Failed both acceptance checks")
        for attempt, tokens in (("first", 100), ("retry", 200), ("reviewer", 40), ("second", 120)):
            self.usage(attempt, tokens)
        self.usage("retry", 11, "counterfactual")
        cohort = self.summary()["by_initial_implementation"][0]
        self.assertEqual(cohort["model_requested"], "model-small")
        self.assertEqual(cohort["model_resolved"], "unknown")
        self.assertEqual(cohort["validated_successes"], 1)
        self.assertEqual(cohort["closed_failed_tasks"], 1)
        self.assertEqual(cohort["retries"], 1)
        self.assertEqual(cohort["roles"]["review"], 1)
        self.assertEqual(cohort["usage"][0]["per_validated_success"], "460")
        self.assertIsNone(cohort["usage"][1]["per_validated_success"])
        implementation = next(row for row in self.summary()["by_attempt"] if row["role"] == "implementation")
        self.assertEqual(implementation["outcomes"]["failed"], 2)

    def test_unknown_partial_and_estimated_usage_keep_separate(self):
        self.task("one")
        self.attempt("retry", "one")
        self.add("outcome", attempt="retry", status="completed", revision="rev2")
        self.validate()
        self.attempt("parent", "one", "orchestration")
        self.add("outcome", attempt="parent", status="completed")
        measured = self.summary()["by_initial_implementation"][0]["usage"][0]
        self.assertIsNone(measured["known_amount"])
        self.assertEqual(measured["unknown_attempts"], 2)
        self.usage("retry", 30, complete=False)
        self.usage("parent", 10, "estimated")
        measured, estimated = self.summary()["by_initial_implementation"][0]["usage"]
        self.assertEqual((measured["known_amount"], measured["unknown_attempts"], measured["partial_attempts"]),
                         ("30", 1, 1))
        self.assertEqual((estimated["known_amount"], estimated["unknown_attempts"]), ("10", 1))
        self.assertIsNone(measured["per_validated_success"])
        self.assertIsNone(estimated["per_validated_success"])

    def test_validation_requires_independence_and_exact_revision(self):
        self.task("one")
        self.attempt("retry", "one")
        with self.assertRaisesRegex(LedgerError, "completed implementation"):
            self.validate()
        self.add("outcome", attempt="retry", status="completed", revision="rev2")
        with self.assertRaisesRegex(LedgerError, "independent"):
            self.validate(validator="retry")
        with self.assertRaisesRegex(LedgerError, "recorded revision"):
            self.validate(revision="different")
        self.validate(verdict="fail")
        self.assertEqual(self.summary()["by_initial_implementation"][0]["validated_successes"], 0)

    def test_schema_and_invalid_records_leave_file_unchanged(self):
        self.task("one")
        self.attempt("first", "one")
        before = self.path.read_bytes()
        for amount in ("-1", "NaN", "1.2"):
            with self.assertRaises(LedgerError):
                self.usage("first", amount)
        with self.assertRaisesRegex(LedgerError, "invalid fields"):
            self.add("task", id="two", task_class="test", acceptance="ok", script="bad")
        self.assertEqual(self.path.read_bytes(), before)

    def rollout_file(self, amounts, mixed=False, sample=True):
        path = self.root / "rollout.jsonl"
        lines = [{"type": "session_meta", "payload": {
            "id": "private-session-id", "cwd": "C:/private/nntrainer", "secret": "private metadata"
        }}, {"type": "turn_context", "payload": {"model": "one", "effort": "medium"}},
            {"type": "response_item", "payload": {"text": "private prompt"}}]
        if mixed:
            lines.append({"type": "turn_context", "payload": {"model": "two", "effort": "high"}})
        for amount in amounts if sample else []:
            lines.append({"type": "event_msg", "payload": {"type": "token_count", "info": {
                "total_token_usage": {"input_tokens": amount - 1, "output_tokens": 1, "total_tokens": amount},
                "last_token_usage": {"total_tokens": 1000},
            }}})
        path.write_bytes("".join(json.dumps(line) + "\n" for line in lines).encode() + b'{"partial"')
        return path

    def load_rollout(self, path, attempt="first", final=False, scope="attempt"):
        snapshot = rollout_snapshot(path, r"C:\PRIVATE\NNTRAINER")
        with writer(self.path) as ledger:
            return import_rollout(self.path, ledger, attempt, snapshot, scope, final)

    def test_rollout_import_updates_repeated_cumulative_samples_once(self):
        self.task("one")
        self.attempt("first", "one")
        self.attempt("second", "one")
        path = self.rollout_file([10, 10, 22])
        self.load_rollout(path)
        before = self.path.read_bytes()
        self.assertIn("No new", self.load_rollout(path))
        self.assertEqual(self.path.read_bytes(), before)
        path = self.rollout_file([10, 10, 22, 40], mixed=True)
        self.load_rollout(path, final=True)
        with self.assertRaisesRegex(LedgerError, "already assigned"):
            self.load_rollout(path, "second")
        usage = Ledger.read(self.path).usage[("first", "measured", "tokens")]
        self.assertEqual(usage["amount"], "40")
        self.assertTrue(usage["complete"])
        self.assertEqual(usage["model_resolved"], "mixed")
        for private in (b"private prompt", b"private-session-id", b"private metadata", b"C:/private"):
            self.assertNotIn(private, self.path.read_bytes())

    def test_resumed_final_rollout_becomes_partial_when_usage_increases(self):
        self.task("one")
        self.attempt("first", "one")
        path = self.rollout_file([10])
        self.load_rollout(path, final=True)
        self.assertIn("No new", self.load_rollout(path))

        path = self.rollout_file([10, 22])
        self.load_rollout(path)
        usage = Ledger.read(self.path).usage[("first", "measured", "tokens")]
        self.assertEqual(usage["amount"], "22")
        self.assertFalse(usage["complete"])

        self.load_rollout(path, final=True)
        self.assertTrue(Ledger.read(self.path).usage[("first", "measured", "tokens")]["complete"])

    def test_session_scope_usage_never_counts_as_attempt_usage(self):
        self.task("one")
        self.attempt("first", "one")
        self.load_rollout(self.rollout_file([10]), scope="session")
        usage = self.summary()["by_initial_implementation"][0]["usage"][0]
        self.assertIsNone(usage["known_amount"])
        self.assertEqual(usage["unknown_attempts"], 1)
        self.assertEqual(usage["session_scope_amount"], "10")
        self.assertIsNone(usage["per_validated_success"])

    def test_new_revision_requires_new_validation(self):
        self.task("one")
        self.attempt("retry", "one")
        self.add("outcome", attempt="retry", status="completed", revision="rev2")
        self.validate()
        self.attempt("edit", "one", "rework")
        self.add("outcome", attempt="edit", status="completed", revision="rev3")
        self.assertEqual(self.summary()["by_initial_implementation"][0]["validated_successes"], 0)
        with self.assertRaisesRegex(LedgerError, "latest delivered"):
            self.validate()

    def test_rollout_rejects_wrong_checkout_or_decreasing_samples(self):
        path = self.rollout_file([10, 8])
        with self.assertRaisesRegex(LedgerError, "checkout"):
            rollout_snapshot(path, "C:/wrong")
        with self.assertRaisesRegex(LedgerError, "decreased"):
            rollout_snapshot(path, "C:/private/nntrainer")

    def test_rollout_without_samples_preserves_unknown(self):
        self.task("one")
        self.attempt("first", "one")
        before = self.path.read_bytes()
        self.assertIn("unknown", self.load_rollout(self.rollout_file([], sample=False), final=True))
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
