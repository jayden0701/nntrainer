#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build real KV16 prefix/Flash checks against a matching Meson host engine.

This runs storage/reference checks only, with no model inference or DSP. It
does not install dependencies, alter the engine build or download files.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time


COMMON = ["Applications/CausalLM/kv_cache_manager.cpp",
          "Applications/CausalLM/kv16_lossless.cpp",
          "Applications/CausalLM/prefix_flash_store.cpp"]
CHECKS = ["kv16_prefix_check", "prefill_prefix_check", "prefix_flash_check",
          "kv16_flash_check"]
ABI_HEADERS = ["api/ccapi/include/tensor_dim.h", "nntrainer/engine.h",
               "nntrainer/mem_allocator.h", "nntrainer/tensor/tensor.h",
               "nntrainer/tensor/tensor_base.h", "nntrainer/tensor/memory_data.h",
               "nntrainer/tensor/memory_pool.h"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matching_digest(path):
    # A Windows checkout can use CRLF while the matching Linux source uses LF.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def run_step(report, output, name, command, cwd, timeout=300):
    step = {"name": name, "command": [str(x) for x in command], "cwd": str(cwd),
            "log": name + ".log"}
    report["steps"].append(step)
    print(name, flush=True)
    begin = time.monotonic()
    with (output / step["log"]).open("w", encoding="utf-8") as stream:
        try:
            result = subprocess.run(step["command"], cwd=cwd, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=timeout)
            step["returncode"] = result.returncode
        except subprocess.TimeoutExpired:
            step["timed_out"] = True
            raise RuntimeError(f"{name} timed out; see {step['log']}")
        finally:
            step["seconds"] = round(time.monotonic() - begin, 3)
    if result.returncode:
        raise RuntimeError(f"{name} failed ({result.returncode}); see {step['log']}")


def prepare(args, report):
    entries = json.loads((args.engine_build / "compile_commands.json").read_text())
    entry = next((x for x in entries if x["file"].replace("\\", "/").endswith(
        "/nntrainer/kv_cache/kv_cache_manager.cpp")), None)
    if entry is None:
        raise RuntimeError("Engine build lacks the production core KV manager")
    engine_source = Path(entry["file"]).resolve().parents[2]
    source_hashes = {}
    for item in entries:
        path = Path(item["file"]).resolve()
        try:
            relative = path.relative_to(engine_source)
        except ValueError:
            continue
        if path.suffix not in (".c", ".cpp"):
            continue
        counterpart = args.repo / relative
        if not counterpart.is_file() or matching_digest(path) != matching_digest(counterpart):
            raise RuntimeError(f"Engine compiled source differs from checkout: {relative}")
        source_hashes[str(relative)] = matching_digest(path)
    headers = ABI_HEADERS + [str(p.relative_to(engine_source)) for p in
                             (engine_source / "nntrainer/kv_cache").glob("*.h")]
    for relative in headers:
        path, counterpart = engine_source / relative, args.repo / relative
        if not path.is_file() or not counterpart.is_file() or matching_digest(path) != matching_digest(counterpart):
            raise RuntimeError(f"Engine ABI header differs from checkout: {relative}")
        source_hashes[relative] = matching_digest(path)
    # Reject stale builds rather than comparing a newer source to an older .so.
    library = args.engine_build / "nntrainer/libnntrainer.so"
    if not library.is_file():
        raise RuntimeError("Build nntrainer/libnntrainer.so in the supplied engine build first")
    if any(Path(item["file"]).is_file() and Path(item["file"]).stat().st_mtime > library.stat().st_mtime
           for item in entries):
        raise RuntimeError("Engine sources are newer than its library; rebuild the engine")
    try:
        revision = subprocess.check_output(["git", "-C", str(engine_source),
                                            "rev-parse", "HEAD"], text=True,
                                           stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    report.update(engine_source=str(engine_source), engine_revision=revision,
                  engine_library_sha256=digest(library),
                  matching_engine_inputs=source_hashes,
                  compile_commands_sha256=digest(args.engine_build / "compile_commands.json"))
    options = args.engine_build / "meson-info/intro-buildoptions.json"
    if options.is_file():
        report["engine_options"] = {item["name"]: item["value"] for item in json.loads(options.read_text())
                                    if item["section"] == "user"}
    raw = entry.get("arguments") or shlex.split(entry["command"])
    base = [raw[0], "-I" + str(args.repo / "Applications/CausalLM")]
    skip = False
    for part in raw[1:]:
        if skip:
            skip = False
            continue
        if part in ("-o", "-MF", "-MQ"):
            skip = True
            continue
        if part in ("-c", "-MD") or part == entry["file"]:
            continue
        base.append(part)
    return base, library


def execute(args, report, output):
    base, library = prepare(args, report)
    extra = ["-O1"]
    if args.sanitize:
        extra += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
    if args.json_include:
        if not (args.json_include / "json.hpp").is_file():
            raise RuntimeError("--json-include must contain json.hpp")
        base += ["-I" + str(args.json_include)]
        report["json_header_sha256"] = digest(args.json_include / "json.hpp")
    sources = list(COMMON)
    if args.json_include:
        sources += ["Applications/CausalLM/baked_prefix.cpp"]
    objects = []
    for index, relative in enumerate(sources):
        obj = output / f"common-{index}.o"
        run_step(report, output, "build-" + Path(relative).stem,
                 base + extra + ["-c", str(args.repo / relative), "-o", str(obj)], args.engine_build)
        objects.append(str(obj))
    checks = list(CHECKS)
    if args.json_include:
        checks += ["baked_prefix_check"]
    else:
        report["not_run"] = ["baked_prefix_check: supply --json-include to test asset metadata"]
    for check in checks:
        source = args.repo / "test/htp/host" / (check + ".cpp")
        obj, exe = output / (check + ".o"), output / check
        run_step(report, output, "build-" + check,
                 base + extra + ["-c", str(source), "-o", str(obj)], args.engine_build)
        run_step(report, output, "link-" + check,
                 [base[0], *objects, str(obj), *extra, "-L" + str(library.parent),
                  "-lnntrainer", "-Wl,-rpath," + str(library.parent), "-pthread", "-ldl", "-lm",
                  "-o", str(exe)], args.engine_build)
        run_step(report, output, "run-" + check, [str(exe)], args.repo, timeout=180)
        report["checks"].append({"name": check, "status": "PASS"})
    tracked = sources + ["Applications/CausalLM/kv16_lossless.h", "Applications/CausalLM/kv_cache_manager.h",
                         "Applications/CausalLM/prefix_flash_store.h",
                         "test/htp/host/run_kv16_integration_checks.py"]
    tracked += ["test/htp/host/" + check + ".cpp" for check in checks]
    report["source_sha256"] = {relative: digest(args.repo / relative) for relative in tracked}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--engine-build", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sanitize", action="store_true", help="Instrument the new storage code/checks with ASan/UBSan")
    parser.add_argument("--json-include", type=Path, help="Optional json.hpp include directory; also runs baked asset metadata checks")
    args = parser.parse_args()
    args.repo, args.engine_build = args.repo.resolve(), args.engine_build.resolve()
    args.json_include = args.json_include.resolve() if args.json_include else None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = args.output.resolve() if args.output else args.repo / "build/kv16-integration" / stamp
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "FAIL", "started_utc": stamp, "repo": str(args.repo),
              "engine_build": str(args.engine_build), "sanitizers": args.sanitize,
              "engine_instrumented": False, "hardware_validated": False,
              "model_inference_validated": False, "platform": platform.platform(),
              "python": platform.python_version(), "steps": [], "checks": []}
    try:
        execute(args, report, output)
        report["status"] = "PASS"
    except (Exception, KeyboardInterrupt) as error:
        report["error"] = str(error) or type(error).__name__
        print(report["error"], file=sys.stderr)
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status']}: {output / 'report.json'}", flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
