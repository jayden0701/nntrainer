#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pair native Ridge scores with the fixed local Transformers reference."""

import argparse
import json
from pathlib import Path

import numpy as np


def compare(rows, refs, mode):
    if not rows:
        return None
    native = []
    reference = []
    errors = []
    flips = []
    for row in rows:
        ref = refs[row["index"]]
        if (row["label"], row["ending_tokens"], row["imported_tokens"]) != (
                ref["label"], ref["ending_tokens"], ref["imported_tokens"]):
            raise ValueError(f"mismatched reference for {row['index']}")
        length = np.asarray(row["ending_tokens"])
        scores = np.asarray(row[mode + "_logprobs"]) / length
        baseline = np.asarray(ref[mode + "_logprobs"]) / length
        if not np.isfinite([scores, baseline]).all():
            raise ValueError("non-finite candidate score")
        a, b = int(scores.argmax()), int(baseline.argmax())
        native.append(a == row["label"])
        reference.append(b == row["label"])
        errors.extend(np.abs(scores - baseline))
        if a != b:
            ordered = np.sort(baseline)
            flips.append({"index": row["index"], "native_choice": a, "hf_choice": b,
                          "gold": row["label"], "hf_margin_nat_per_token": ordered[-1] - ordered[-2]})
    n, h = np.asarray(native), np.asarray(reference)
    return {"n": len(n), "native_correct": int(n.sum()), "hf_correct": int(h.sum()),
            "native_only_correct": int((n & ~h).sum()), "hf_only_correct": int((h & ~n).sum()),
            "same_prediction": len(n) - len(flips), "changed_predictions": flips,
            "candidate_score_error_nat_per_token": dict(zip(("p50", "p95", "max"),
                                                             np.quantile(errors, [0.5, 0.95, 1]))),
            "native_minus_hf_pp": 100 * float(n.mean() - h.mean())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("native", type=Path)
    p.add_argument("hf", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    rows = json.loads(args.native.read_text())
    reference = json.loads(args.hf.read_text())
    ids = [r["index"] for r in rows]
    if len(ids) != len(set(ids)) or ids != [r["index"] for r in reference][:len(rows)]:
        raise ValueError("expected a prefix of the unique reference indices")
    refs = {r["index"]: r for r in reference}
    imported = [r for r in rows if r["imported_tokens"]]
    fallback = [r for r in rows if not r["imported_tokens"]]
    if any(r["standalone_logprobs"] != r["mapped_logprobs"] for r in fallback):
        raise ValueError("cold fallback differs from standalone")
    report = {}
    for group, subset in (("all", rows), ("transferred", imported), ("fallback", fallback)):
        report[group] = {m: compare(subset, refs, m) for m in ("standalone", "mapped")}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
