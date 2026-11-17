#!/usr/bin/env python3
"""Fit, apply, and compare offline affine KV cache transfer."""

import argparse
import json
import os
from pathlib import Path

# Small per-head BLAS calls are inexpensive, and nested worker pools on some
# systems cost substantially more than the multiplication.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from dataset import read_pairs, split_pairs
from mapper import Geometry, Mapping, apply, compare, fit, select_layers
from selection import select_layers_by_r2
from snapshot import Snapshot, read_json


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    training = commands.add_parser("fit", help="fit a mapping with optional held-out diagnostics")
    training.add_argument("dataset", type=Path)
    training.add_argument("output", type=Path)
    training.add_argument("--layout", choices=["headwise", "all-head"], default="headwise")
    layer_selection = training.add_mutually_exclusive_group()
    layer_selection.add_argument("--k", type=int, default=1,
                                 help="source layers per target layer (default: 1)")
    layer_selection.add_argument("--layer-map", type=Path,
                                 help='JSON object with "source_layers" lists')
    training.add_argument("--layer-selection", choices=["nearest-depth", "ols-r2"],
                          help="rank source layers by depth (default) or calibration OLS R²")
    training.add_argument("--token-stride", type=int, default=1,
                          help="sample every Nth token starting at row zero of each prompt")
    training.add_argument("--alpha", type=float, default=1.0,
                          help="L2 penalty for coefficients in the summed SSE objective")
    training.add_argument("--validation-fraction", type=float, default=0.2,
                          help="fraction of prompt groups held out (0 uses all for fit)")
    training.add_argument("--seed", type=int, default=0)

    applying = commands.add_parser("apply", help="map a source KV snapshot")
    applying.add_argument("mapping", type=Path)
    applying.add_argument("source", type=Path)
    applying.add_argument("output", type=Path)

    comparing = commands.add_parser("compare", help="compare aligned target snapshots")
    comparing.add_argument("prediction", type=Path)
    comparing.add_argument("reference", type=Path)
    for command in (training, applying, comparing):
        command.add_argument("--block-tokens", type=int, default=128)
    for command in (training, applying):
        command.add_argument("--memory-mib", type=int, default=512)
    return result


def execute(args):
    if args.command == "fit":
        pairs = read_pairs(args.dataset)
        train, validation = split_pairs(pairs, args.validation_fraction, args.seed)
        source, target = train[0].source.spec, train[0].target.spec
        if args.layer_map:
            if args.layer_selection:
                raise ValueError("--layer-map and --layer-selection are mutually exclusive")
            layers = read_json(args.layer_map)["source_layers"]
            selection = {"method": "explicit"}
        elif args.layer_selection == "ols-r2":
            layers, selection = select_layers_by_r2(
                train, args.k, block_size=args.block_tokens, token_stride=args.token_stride,
                memory_mib=args.memory_mib,
            )
        else:
            layers = select_layers(source.layers, target.layers, args.k)
            selection = {"method": "nearest-depth"}
        return fit(
            args.output, train, validation, Geometry(source, target, args.layout, layers),
            alpha=args.alpha, block_size=args.block_tokens, memory_mib=args.memory_mib,
            token_stride=args.token_stride, selection=selection,
        )
    if args.command == "apply":
        apply(Mapping(args.mapping), Snapshot(args.source), args.output,
              block_size=args.block_tokens, memory_mib=args.memory_mib)
        return {"snapshot": str(args.output)}
    return compare(Snapshot(args.prediction), Snapshot(args.reference), args.block_tokens)


def main():
    arguments = parser()
    try:
        result = execute(arguments.parse_args())
    except (OSError, ValueError, KeyError, TypeError, OverflowError, np.linalg.LinAlgError) as error:
        arguments.exit(1, f"kv_transfer: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
