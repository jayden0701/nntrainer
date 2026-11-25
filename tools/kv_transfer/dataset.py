"""Load aligned prompt pairs and assign whole prompt groups to a holdout."""

import hashlib
import json
import math
from dataclasses import dataclass

from snapshot import Snapshot, read_json


@dataclass(frozen=True)
class Pair:
    source: Snapshot
    target: Snapshot
    digest: str
    group: str


def prompt_digest(token_ids):
    encoded = json.dumps(token_ids, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def read_pairs(path):
    manifest = read_json(path)
    raw_pairs = manifest["pairs"]
    if type(manifest.get("version")) is not int or manifest["version"] != 1:
        raise ValueError("unsupported dataset version")
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise ValueError("dataset needs at least one prompt pair")

    pairs = []
    seen = set()
    for item in raw_pairs:
        source = Snapshot(path.parent / item["source"])
        target = Snapshot(path.parent / item["target"])
        if (source.token_ids != target.token_ids
                or source.positions != target.positions):
            raise ValueError(f"{source.path} and {target.path} are not token-aligned")
        if pairs and (source.spec != pairs[0].source.spec
                      or target.spec != pairs[0].target.spec):
            raise ValueError("all pairs must have the same source and target model specs")
        digest = prompt_digest(source.token_ids)
        if digest in seen:
            raise ValueError(f"duplicate token sequence: {source.path}")
        seen.add(digest)
        if "group" in item:
            group = item["group"]
            if not isinstance(group, str) or not group.strip():
                raise ValueError("group must be a nonempty string")
            group = "group:" + group
        else:
            group = "prompt:" + digest
        pairs.append(Pair(source, target, digest, group))

    # If a dataset contains an entire prompt and its continuation, keeping
    # them in a group prevents the same causal prefix becoming a holdout.
    ordered = sorted(pairs, key=lambda pair: pair.source.token_ids)
    prefixes = []
    for pair in ordered:
        tokens = pair.source.token_ids
        while prefixes:
            prefix = prefixes[-1].source.token_ids
            if tokens[:len(prefix)] == prefix:
                break
            prefixes.pop()
        if prefixes and prefixes[-1].group != pair.group:
            raise ValueError("assign prefix-related prompts to the same explicit group")
        prefixes.append(pair)
    return sorted(pairs, key=lambda pair: pair.digest)


def split_pairs(pairs, fraction, seed):
    if not math.isfinite(fraction) or not 0 <= fraction < 1:
        raise ValueError("validation fraction must be at least zero and less than one")
    if fraction == 0:
        return pairs, []
    groups = {pair.group for pair in pairs}
    if len(groups) < 2:
        raise ValueError("dataset needs at least two independent prompt groups")

    def ranking(group):
        return hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest(), group

    count = min(len(groups) - 1, max(1, math.ceil(len(groups) * fraction)))
    validation_groups = set(sorted(groups, key=ranking)[:count])
    train = [pair for pair in pairs if pair.group not in validation_groups]
    validation = [pair for pair in pairs if pair.group in validation_groups]
    return train, validation


def sampled_count(tokens, stride):
    return (tokens + stride - 1) // stride


def describe_split(pairs, stride=1):
    return {
        "prompts": len(pairs), "groups": len({pair.group for pair in pairs}),
        "tokens": sum(pair.source.tokens for pair in pairs),
        "observations": sum(sampled_count(pair.source.tokens, stride) for pair in pairs),
        "prompt_digests": [pair.digest for pair in pairs],
    }
