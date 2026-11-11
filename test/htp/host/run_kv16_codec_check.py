#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cross-check the production C++ codec against a frozen Python wire oracle.

No CPU/GPU model inference, NPU, SDK or NumPy is used. The original PoC
codec.py SHA256 was dcd63fd10c20f190fc259c0bace2aece4e4f90539f2ff19b6bf9a4792f9d58ed.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import struct
import subprocess
import sys
import time
import zlib


def run(command, expected=0, timeout=60):
    result = subprocess.run([str(part) for part in command], capture_output=True, text=True, timeout=timeout)
    if result.returncode != expected:
        raise RuntimeError(f"exit {result.returncode}, expected {expected}: {command}\n{result.stdout}\n{result.stderr}")
    return result


def words(values):
    return struct.pack(f"<{len(values)}H", *values)


def rechecksum(data):
    data = bytearray(data)
    struct.pack_into("<I", data, 48, zlib.crc32(data[52:], zlib.crc32(data[:48])))
    return bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[3]
    parser.add_argument("--repo", type=Path, default=root)
    parser.add_argument("--output", type=Path, default=root / "build/kv16-codec")
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args()
    root, output = args.repo.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "RUNNING", "hardware_validated": False, "model_inference": False,
              "sanitize": args.sanitize, "python": sys.version.split()[0]}
    start = time.perf_counter()
    try:
        reference = root / "tools/kv16/codec_reference.py"
        spec = importlib.util.spec_from_file_location("kv16_reference", reference)
        codec = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(codec)
        source = root / "Applications/CausalLM/kv16_lossless.cpp"
        header = source.with_suffix(".h")
        helper = root / "test/htp/host/kv16_codec_check.cpp"
        report["source_sha256"] = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in (reference, source, header, helper)}
        report["compiler"] = run([args.cxx, "--version"]).stdout.splitlines()[0]
        binary = output / "codec_check"
        command = [args.cxx, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                   "-I", source.parent, source, helper, "-o", binary]
        if args.sanitize:
            command += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
        run(command, timeout=120)
        report["selfcheck"] = run([binary]).stdout.strip()

        rng = random.Random(4481)
        gammas = (0, 0x80000000, 0x3f800000, 0xbf800000, 0x3f000000,
                  0x3f8020c5, 0xc18a0000, 1, 0x007fffff, 0x7f7fffff,
                  0x7f800000, 0xff800000, 0x7fc12345)
        predictor_input, predictor_expected = bytearray(), bytearray()
        for gamma in gammas:
            for value in range(65536):
                predictor_input.extend(struct.pack("<HI", value, gamma))
                predictor_expected.extend(struct.pack("<H", codec.predict_fp16(value, gamma)))
        for _ in range(20000):
            value, gamma = rng.randrange(65536), rng.randrange(1 << 32)
            predictor_input.extend(struct.pack("<HI", value, gamma))
            predictor_expected.extend(struct.pack("<H", codec.predict_fp16(value, gamma)))
        (output / "predictor.in").write_bytes(predictor_input)
        run([binary, "predict", output / "predictor.in", output / "predictor.out"])
        if (output / "predictor.out").read_bytes() != predictor_expected:
            raise RuntimeError("C++/Python predictor mismatch")
        report["predictor_pairs"] = len(predictor_expected) // 2

        fixtures = []
        fixtures.append(("empty", b"", b"", (0, 2, 4), [0x3f800000] * 4, [0, 2], 3))
        k, v = [0x3c00] * 51, [0x3c00] * 51
        k[1] ^= 0xffff
        for d in range(17):
            if d not in (0, 8):
                k[17 + d] ^= 1
                k[34 + d] ^= 0xffff
        fixtures.append(("three-modes", words(k), words(v), (3, 1, 17), [0x3f800000] * 17, [0, 8], 1))
        v = list(range(65536))
        fixtures.append(("all-fp16-original", words(v[32768:] + v[:32768]), words(v),
                         (256, 1, 256), [0xbf800000] * 256, [0, 1, 126, 128, 129, 254], 1))
        k = []
        for offset in range(0, 65536, 1024):
            k.extend(range(offset, offset + 1024))
            k.extend([0] * 3072)
        fixtures.append(("all-fp16-sparse", words(k), words([0] * len(k)), (64, 1, 4096),
                         [0x3f800000] * 4096, [], 1))
        v = [rng.randrange(0x3000, 0x4000) for _ in range(9 * 2 * 16)]
        gamma = [0x3f800000] * 16 + [0xbfa00000] * 16
        k = [codec.predict_fp16(value, gamma[i % 32]) ^ (i % 4) for i, value in enumerate(v)]
        fixtures.append(("per-head-tail", words(k), words(v), (9, 2, 16), gamma, [0, 1, 8, 9], 5))
        for width in range(16):
            fixtures.append((f"width-{width}", words([0x3c00 ^ ((1 << width) - 1)] * 21),
                             words([0x3c00] * 21), (7, 1, 3), [0x3f800000] * 3, [], 3))
        for index in range(20):
            t, h, d = rng.randrange(12), rng.randrange(1, 5), rng.randrange(1, 34)
            k = words([rng.randrange(65536) for _ in range(t * h * d)])
            v = words([rng.randrange(65536) for _ in range(t * h * d)])
            gamma = [rng.randrange(1 << 32) for _ in range(d * (h if index % 2 else 1))]
            rotary = sorted(rng.sample(range(d), rng.randrange(d + 1)))
            fixtures.append((f"random-{index}", k, v, (t, h, d), gamma, rotary, rng.randrange(1, 8)))
        wire_cases = []
        blobs = []
        for name, k, v, shape, gamma, rotary, rows in fixtures:
            blob = codec.encode(k, v, shape, gamma, rotary, rows)
            path = output / (name + ".kv16")
            path.write_bytes(blob)
            run([binary, "roundtrip", path, output / "reencoded.kv16", output / "key.bin",
                 output / "value.bin", output / "status.txt"])
            if ((output / "reencoded.kv16").read_bytes() != blob or
                    (output / "key.bin").read_bytes() != k or (output / "value.bin").read_bytes() != v):
                raise RuntimeError(f"C++/Python wire or original bytes mismatch: {name}")
            metadata = codec.inspect_blob(blob)
            wire_cases.append({"name": name, "bytes": len(blob), "raw_blocks": metadata["raw_blocks"],
                               "fixed_blocks": metadata["bitpacked_blocks"], "sparse_blocks": metadata["sparse_blocks"]})
            blobs.append(blob)
        report["wire_cases"] = wire_cases

        sample = blobs[1]
        invalid_inputs = [sample[:end] for end in range(len(sample))]
        for offset in range(len(sample)):
            broken = bytearray(sample)
            broken[offset] ^= 1
            invalid_inputs.append(bytes(broken))
        for index in range(200):
            broken = bytearray(sample)
            for _ in range(1 + index % 3):
                broken[rng.randrange(len(broken))] ^= 1 << rng.randrange(8)
            broken = rechecksum(broken)
            try:
                codec.inspect_blob(broken)
            except ValueError:
                invalid_inputs.append(broken)
        for blob in invalid_inputs:
            (output / "invalid.kv16").write_bytes(blob)
            run([binary, "inspect", output / "invalid.kv16"], expected=2)
        report["rejected_malformed_cases"] = len(invalid_inputs)
        report["status"] = "PASS"
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = str(error)
    report["seconds"] = time.perf_counter() - start
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "hardware_validated", "seconds")}, indent=2))
    if report["status"] != "PASS":
        print(report["error"], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
