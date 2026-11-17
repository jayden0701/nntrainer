"""Append-only local ledger for delegated nntrainer work."""

import argparse
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys


ROLES = ("implementation", "review", "rework", "orchestration")
OUTCOMES = ("completed", "failed", "abandoned")
BASES = ("measured", "estimated", "counterfactual")
UNITS = ("tokens", "credits", "usd")
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra")
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
TOKEN_FIELDS = {
    "input_tokens", "cached_input_tokens", "cache_write_input_tokens",
    "output_tokens", "reasoning_output_tokens", "total_tokens",
}
FIELDS = {
    "task": ("id task_class acceptance", ""),
    "attempt": ("id task actor role", "model_requested model_resolved effort_requested effort_resolved retry_of"),
    "outcome": ("attempt status", "revision"),
    "validation": ("task attempt validator revision verdict evidence", ""),
    "closure": ("task status evidence", ""),
    "note": ("attempt evidence", ""),
    "usage": ("attempt basis unit source amount complete", "tokens model_resolved effort_resolved scope"),
}


class LedgerError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise LedgerError(message)


def get_amount(value):
    if not isinstance(value, str):
        raise LedgerError("usage amount must be a decimal string")
    try:
        amount = Decimal(value)
    except InvalidOperation:
        raise LedgerError("invalid usage amount") from None
    require(amount.is_finite() and amount >= 0, "usage amount must be finite and nonnegative")
    return amount


def check_tokens(tokens):
    require(isinstance(tokens, dict) and tokens.keys() <= TOKEN_FIELDS
            and "total_tokens" in tokens
            and all(type(value) is int and value >= 0 for value in tokens.values()),
            "invalid token counters")
    if {"input_tokens", "output_tokens"} <= tokens.keys():
        require(tokens["input_tokens"] + tokens["output_tokens"] == tokens["total_tokens"],
                "input plus output must equal total tokens")
    for subset, total in (("cached_input_tokens", "input_tokens"),
                          ("cache_write_input_tokens", "input_tokens"),
                          ("reasoning_output_tokens", "output_tokens")):
        if {subset, total} <= tokens.keys():
            require(tokens[subset] <= tokens[total], f"{subset} exceeds {total}")


class Ledger:
    def __init__(self):
        self.tasks = {}
        self.attempts = {}
        self.outcomes = {}
        self.deliveries = {}
        self.validations = {}
        self.closures = {}
        self.usage = {}
        self.rollout_owners = {}
        self.resolved = {}

    def add(self, record):
        require(isinstance(record, dict), "expected a record object")
        kind = record.get("type")
        require(isinstance(kind, str) and kind in FIELDS and type(record.get("v")) is int
                and record["v"] == 1, "unsupported record type or version")
        required, optional = (set(fields.split()) for fields in FIELDS[kind])
        keys = record.keys() - {"type", "v", "at"}
        require(required <= keys <= required | optional, f"invalid fields for {kind}")
        try:
            stamp = datetime.fromisoformat(record.get("at", "").replace("Z", "+00:00"))
            require(stamp.utcoffset() is not None, "record timestamp requires an offset")
        except (AttributeError, ValueError):
            raise LedgerError("invalid record timestamp") from None

        for key in keys - {"amount", "complete", "tokens", "acceptance", "evidence"}:
            require(isinstance(record[key], str) and ID.fullmatch(record[key]),
                    f"invalid {key}; use an identifier without spaces or paths")
        for key in keys & {"acceptance", "evidence"}:
            require(isinstance(record[key], str) and record[key].strip()
                    and "\n" not in record[key] and "\r" not in record[key],
                    f"{key} must be a short, single-line reference or summary")
        for key in keys & {"effort_requested", "effort_resolved"}:
            require(record[key] in EFFORTS or (key == "effort_resolved" and record[key] == "mixed"),
                    f"invalid {key}")
        getattr(self, f"_add_{kind}")(record)

    def _add_task(self, record):
        require(record["id"] not in self.tasks, "task already exists")
        self.tasks[record["id"]] = record

    def _add_attempt(self, record):
        attempt = record["id"]
        require(attempt not in self.attempts, "attempt already exists")
        require(record["task"] in self.tasks, "unknown task")
        require(record["role"] in ROLES, "invalid role")
        if "retry_of" in record:
            earlier = self.attempts.get(record["retry_of"])
            require(earlier and earlier["task"] == record["task"], "retry requires an earlier attempt on the same task")
        self.attempts[attempt] = record
        self.closures.pop(record["task"], None)
        self.resolved[attempt] = {key: record[key] for key in ("model_resolved", "effort_resolved") if key in record}

    def _add_outcome(self, record):
        attempt = record["attempt"]
        require(attempt in self.attempts and attempt not in self.outcomes, "outcome requires an open attempt")
        require(record["status"] in OUTCOMES, "invalid outcome")
        require(record["status"] != "completed" or "revision" in record or
                self.attempts[attempt]["role"] in {"review", "orchestration"},
                "completed implementation or rework requires a revision")
        self.outcomes[attempt] = record
        job = self.attempts[attempt]
        if record["status"] == "completed" and job["role"] in {"implementation", "rework"}:
            task = job["task"]
            self.deliveries[task] = attempt
            if self.validations.get(task, {}).get("attempt") != attempt:
                self.validations.pop(task, None)

    def _add_validation(self, record):
        attempt = self.attempts.get(record["attempt"])
        outcome = self.outcomes.get(record["attempt"])
        require(attempt and attempt["task"] == record["task"]
                and attempt["role"] in {"implementation", "rework"}
                and outcome and outcome["status"] == "completed"
                and outcome.get("revision") == record["revision"],
                "validate a completed implementation or rework at its recorded revision")
        require(self.deliveries[record["task"]] == record["attempt"], "validate the latest delivered revision")
        require(attempt["actor"] != record["validator"], "validation requires an independent actor")
        require(record["verdict"] in {"pass", "fail"}, "invalid validation verdict")
        self.validations[record["task"]] = record
        self.closures.pop(record["task"], None)

    def _add_note(self, record):
        require(record["attempt"] in self.attempts, "note requires a known attempt")

    def _add_closure(self, record):
        task = record["task"]
        require(task in self.tasks and task not in self.closures, "closure requires an open task")
        require(record["status"] in {"failed", "abandoned"}, "invalid closure status")
        require(self.validations.get(task, {}).get("verdict") != "pass",
                "the latest validation already passed")
        require(all(attempt["id"] in self.outcomes for attempt in self.attempts.values()
                    if attempt["task"] == task), "close all attempts before closing a task")
        self.closures[task] = record

    def _add_usage(self, record):
        attempt, basis, unit, source = (record[key] for key in ("attempt", "basis", "unit", "source"))
        require(attempt in self.attempts, "usage requires a known attempt")
        require(basis in BASES and unit in UNITS, "invalid usage basis or unit")
        require(type(record["complete"]) is bool, "complete must be a boolean")
        amount = get_amount(record["amount"])
        require(unit != "tokens" or amount == amount.to_integral_value(), "tokens must be integral")
        if "tokens" in record:
            check_tokens(record["tokens"])
            require(unit == "tokens" and amount == record["tokens"]["total_tokens"], "inconsistent usage amount")
        rollout = source.startswith("rollout-")
        if rollout:
            require(basis == "measured" and unit == "tokens" and "tokens" in record,
                    "rollout sources are reserved for measured token counters")
            require(record.get("scope") in {"session", "attempt"}, "rollout requires an explicit scope")
            require(record["scope"] != "session" or not record["complete"],
                    "session-scope usage is unallocated")
            require(self.rollout_owners.get(source, attempt) == attempt,
                    "rollout is already assigned to another attempt")
        else:
            require(not ({"model_resolved", "effort_resolved", "scope"} & record.keys()),
                    "only rollout metadata may update resolved configuration")

        key = (attempt, basis, unit)
        previous = self.usage.get(key)
        if previous:
            require(previous["source"] == source, "one source per attempt, basis and unit")
            if rollout:
                require(previous["scope"] == record["scope"], "rollout scope cannot change")
                require(amount >= get_amount(previous["amount"]) and
                        all(record["tokens"].get(field, value) >= value
                            for field, value in previous["tokens"].items()),
                        "cumulative rollout counters decreased")
            if previous["complete"] and not record["complete"]:
                require(rollout and counters_increased(previous["tokens"], record["tokens"]),
                        "complete usage cannot become partial without new rollout usage")
        resolved = self.resolved[attempt].copy()
        for field in ("model_resolved", "effort_resolved"):
            if field in record:
                prior = resolved.get(field)
                inferred = record[field]
                explicit = self.attempts[attempt].get(field)
                require(explicit is None or explicit == inferred, f"rollout conflicts with recorded {field}")
                require(prior in (None, inferred) or inferred == "mixed", f"rollout conflicts with earlier {field}")
                resolved[field] = inferred
        if rollout:
            self.rollout_owners[source] = attempt
        self.resolved[attempt] = resolved
        self.usage[key] = record

    @classmethod
    def read(cls, path):
        ledger = cls()
        if path.exists():
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    try:
                        ledger.add(json.loads(line))
                    except ValueError as exc:
                        raise LedgerError(f"ledger line {line_number}: {exc}") from None
        return ledger


@contextmanager
def writer(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # The operating system grants the lock to only one CLI process at a time.
    lock = path.with_suffix(path.suffix + ".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise LedgerError("ledger is locked by another writer") from None
    try:
        os.close(descriptor)
        yield Ledger.read(path)
    finally:
        lock.unlink()


def append(path, ledger, kind, **fields):
    record = {"v": 1, "type": kind, "at": datetime.now(timezone.utc).isoformat(), **fields}
    ledger.add(record)
    with path.open("ab") as stream:
        stream.write(json.dumps(record, sort_keys=True).encode("utf-8") + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def rollout_snapshot(path, checkout):
    expected_path = checkout.replace("\\", "/").rstrip("/").casefold()
    session_id = None
    counters = None
    models, efforts = set(), set()
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.endswith(b"\n"):
                break  # A live writer can leave the last line incomplete.
            try:
                row = json.loads(line)
            except (ValueError, UnicodeError):
                raise LedgerError(f"invalid rollout JSON at line {line_number}") from None
            if not isinstance(row, dict) or not isinstance(row.get("payload"), dict):
                require(line_number != 1, "rollout must begin with session metadata")
                continue
            payload = row["payload"]
            if line_number == 1:
                require(row.get("type") == "session_meta", "rollout must begin with session metadata")
                cwd = payload.get("cwd")
                require(isinstance(cwd, str), "rollout does not identify its checkout")
                actual_path = cwd.replace("\\", "/").rstrip("/").casefold()
                require(actual_path == expected_path, "rollout checkout does not match --checkout")
                session_id = payload.get("id")
                require(isinstance(session_id, str) and session_id, "rollout does not identify its session")
            elif row.get("type") == "turn_context":
                for key, observed in (("model", models), ("effort", efforts)):
                    value = payload.get(key)
                    if isinstance(value, str) and ID.fullmatch(value):
                        observed.add(value)
            elif row.get("type") == "event_msg" and payload.get("type") == "token_count":
                info = payload.get("info")
                if not isinstance(info, dict) or "total_token_usage" not in info:
                    continue
                value = info["total_token_usage"]
                require(isinstance(value, dict), "invalid rollout token counters")
                sample = {field: value[field] for field in TOKEN_FIELDS if field in value}
                check_tokens(sample)
                if counters:
                    require(all(sample.get(field, old) >= old for field, old in counters.items()),
                            f"cumulative rollout counters decreased at line {line_number}")
                counters = {**(counters or {}), **sample}
    require(session_id, "missing session metadata")
    fields = {"source": "rollout-" + sha256(session_id.encode("utf-8")).hexdigest()}
    for key, observed in (("model_resolved", models), ("effort_resolved", efforts)):
        if observed:
            fields[key] = next(iter(observed)) if len(observed) == 1 else "mixed"
    if counters is not None:
        check_tokens(counters)
        fields["tokens"] = counters
    return fields


def counters_increased(previous, current):
    return any(current.get(field, value) > value for field, value in previous.items())


def import_rollout(path, ledger, attempt, snapshot, scope, final=False):
    require(attempt in ledger.attempts, "import requires a known attempt")
    if "tokens" not in snapshot:
        return "No cumulative token sample found. Usage remains unknown."
    key = (attempt, "measured", "tokens")
    previous = ledger.usage.get(key, {})
    retain_final = previous.get("complete", False) and not counters_increased(
        previous.get("tokens", {}), snapshot["tokens"]
    )
    record = {
        "attempt": attempt, "basis": "measured", "unit": "tokens",
        "amount": str(snapshot["tokens"]["total_tokens"]),
        "scope": scope, "complete": final or retain_final,
        **snapshot,
    }
    if all(previous.get(key) == value for key, value in record.items()):
        return "No new cumulative token sample."
    append(path, ledger, "usage", **record)
    return "Imported cumulative token usage."


def usage_summary(ledger, attempts):
    entries = []
    for basis in BASES:
        for unit in UNITS:
            records = [ledger.usage[(attempt, basis, unit)] for attempt in attempts
                       if (attempt, basis, unit) in ledger.usage]
            if not records and (basis, unit) != ("measured", "tokens"):
                continue
            unallocated = [row for row in records if row.get("scope") == "session"]
            attributed = [row for row in records if row.get("scope") != "session"]
            entry = {
                "basis": basis,
                "unit": unit,
                "known_amount": str(sum(get_amount(row["amount"]) for row in attributed)) if attributed else None,
                "unknown_attempts": len(attempts) - len(attributed),
                "partial_attempts": sum(not row["complete"] for row in attributed),
            }
            if unallocated:
                entry["session_scope_attempts"] = len(unallocated)
                entry["session_scope_amount"] = str(sum(get_amount(row["amount"]) for row in unallocated))
            entries.append(entry)
    return entries


def report(ledger):
    grouped_attempts = defaultdict(list)
    grouped_tasks = defaultdict(list)
    by_task = defaultdict(list)
    for attempt in ledger.attempts.values():
        task = attempt["task"]
        by_task[task].append(attempt)
        case = ledger.tasks[task]["task_class"]
        config = ledger.resolved[attempt["id"]]
        cohort = tuple(config.get(key, "unknown") for key in ("model_resolved", "effort_resolved"))
        request = tuple(attempt.get(key, "unknown") for key in ("model_requested", "effort_requested"))
        grouped_attempts[(case, *cohort, *request, attempt["role"])].append(attempt["id"])

    for task, record in ledger.tasks.items():
        primary = next((attempt for attempt in by_task[task] if attempt["role"] == "implementation"), None)
        config = ledger.resolved[primary["id"]] if primary else {}
        request = tuple((primary or {}).get(key, "unknown") for key in ("model_requested", "effort_requested"))
        grouped_tasks[(record["task_class"], config.get("model_resolved", "unknown"),
                       config.get("effort_resolved", "unknown"), *request)].append(task)

    def group(key):
        return dict(zip(("task_class", "model_resolved", "effort_resolved", "model_requested",
                         "effort_requested"), key))

    tasks_report = []
    for key, tasks in sorted(grouped_tasks.items()):
        attempts = [attempt for task in tasks for attempt in by_task[task]]
        observed = [ledger.validations.get(task, {}).get("verdict") for task in tasks]
        successes = observed.count("pass")
        usage = usage_summary(ledger, [attempt["id"] for attempt in attempts])
        closed = all(verdict == "pass" or task in ledger.closures for task, verdict in zip(tasks, observed))
        closed = closed and all(attempt["id"] in ledger.outcomes for attempt in attempts)
        for entry in usage:
            complete = (closed and successes and not entry["unknown_attempts"]
                        and not entry["partial_attempts"] and entry["basis"] != "counterfactual")
            entry["per_validated_success"] = (
                str(Decimal(entry["known_amount"]) / successes) if complete else None
            )
        tasks_report.append({
            **group(key), "tasks": len(tasks), "validated_successes": successes,
            "latest_validation_failed": observed.count("fail"), "unvalidated_tasks": observed.count(None),
            "closed_failed_tasks": sum(ledger.closures.get(task, {}).get("status") == "failed" for task in tasks),
            "abandoned_tasks": sum(ledger.closures.get(task, {}).get("status") == "abandoned" for task in tasks),
            "attempts": len(attempts), "retries": sum("retry_of" in attempt for attempt in attempts),
            "roles": {role: sum(attempt["role"] == role for attempt in attempts) for role in ROLES},
            "usage": usage,
        })

    attempts_report = []
    for key, attempts in sorted(grouped_attempts.items()):
        statuses = [ledger.outcomes.get(attempt, {}).get("status", "unknown") for attempt in attempts]
        attempts_report.append({
            **group(key[:-1]), "role": key[-1], "attempts": len(attempts),
            "outcomes": {status: statuses.count(status) for status in (*OUTCOMES, "unknown")},
            "usage": usage_summary(ledger, attempts),
        })
    return {"by_initial_implementation": tasks_report, "by_attempt": attempts_report}


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=Path(".cache/agent-cost/ledger.jsonl"))
    commands = parser.add_subparsers(dest="command", required=True)

    task = commands.add_parser("task", help="register a task and acceptance criteria")
    task.add_argument("id")
    task.add_argument("task_class")
    task.add_argument("--acceptance", required=True)

    attempt = commands.add_parser("attempt", help="register work before starting it")
    attempt.add_argument("id")
    attempt.add_argument("task")
    attempt.add_argument("actor")
    attempt.add_argument("role", choices=ROLES)
    attempt.add_argument("--model-requested")
    attempt.add_argument("--model-resolved")
    attempt.add_argument("--effort-requested", choices=EFFORTS)
    attempt.add_argument("--effort-resolved", choices=EFFORTS)
    attempt.add_argument("--retry-of")

    outcome = commands.add_parser("outcome", help="record how an attempt ended")
    outcome.add_argument("attempt")
    outcome.add_argument("status", choices=OUTCOMES)
    outcome.add_argument("--revision")

    validation = commands.add_parser("validation", help="record an independent check")
    validation.add_argument("task")
    validation.add_argument("attempt")
    validation.add_argument("validator")
    validation.add_argument("revision")
    validation.add_argument("verdict", choices=("pass", "fail"))
    validation.add_argument("--evidence", required=True)

    note = commands.add_parser("note", help="record a check or observation during an attempt")
    note.add_argument("attempt")
    note.add_argument("--evidence", required=True)

    closure = commands.add_parser("closure", help="close a task without a validated success")
    closure.add_argument("task")
    closure.add_argument("status", choices=("failed", "abandoned"))
    closure.add_argument("--evidence", required=True)

    usage = commands.add_parser("usage", help="record or update an attempt's usage")
    usage.add_argument("attempt")
    usage.add_argument("basis", choices=BASES)
    usage.add_argument("unit", choices=UNITS)
    usage.add_argument("amount")
    usage.add_argument("--source", required=True)
    usage.add_argument("--complete", action="store_true")

    rollout = commands.add_parser("import-rollout", help="import local Codex cumulative token counts")
    rollout.add_argument("attempt")
    rollout.add_argument("rollout", type=Path)
    rollout.add_argument("--checkout", required=True, help="exact cwd listed in rollout metadata")
    rollout.add_argument("--scope", required=True, choices=("attempt", "session"),
                         help="use attempt only when the whole session belongs to this attempt")
    rollout.add_argument("--final", action="store_true", help="confirm the session is over and belongs entirely to the attempt")

    commands.add_parser("report", help="summarize task and attempt cohorts as JSON")
    return parser


def main():
    args = vars(make_parser().parse_args())
    path = args.pop("ledger")
    command = args.pop("command")
    try:
        if command == "report":
            print(json.dumps(report(Ledger.read(path)), indent=2))
            return 0
        if command == "import-rollout":
            snapshot = rollout_snapshot(args["rollout"], args["checkout"])
            with writer(path) as ledger:
                print(import_rollout(path, ledger, args["attempt"], snapshot, args["scope"], args["final"]))
        else:
            with writer(path) as ledger:
                append(path, ledger, command, **{key: value for key, value in args.items() if value is not None})
        return 0
    except (OSError, LedgerError) as exc:
        print(f"agent-cost: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
