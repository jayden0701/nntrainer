"""Count successful host gather work and compare local optimization experiments.

This builds an unchanged production gather alongside generated, instrumented
copies of its body. It neither changes the checkout nor exercises HTP hardware.
Requires a C++17 compiler and the repository's nlohmann json.hpp dependency.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
import subprocess
import tempfile


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"Expected exactly one source site: {old!r}")
    return text.replace(old, new, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("--json-include", type=Path)
    parser.add_argument("--cxx", default="c++")
    parser.add_argument("--commit", help="Optional commit ID for cross-OS worktrees")
    parser.add_argument("--sanitize", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    here = Path(__file__).resolve().parent
    output = args.output or here / "result.json"
    source = repo / "Applications/CausalLM/kv_cache_manager.cpp"
    full = source.read_text(encoding="utf-8")
    start = full.index("std::shared_ptr<const KVCacheManager::PrefixSnapshot>\nKVCacheManager::gatherSparsePrefix(")
    end = full.index("\nvoid KVCacheManager::restorePrefix(", start)
    original = full[start:end]
    body = original.replace("KVCacheManager::", "Manager::")
    body = replace_once(body, "page->key.resize(bytes);", "auditResize(page->key, bytes);")
    body = replace_once(body, "page->value.resize(bytes);", "auditResize(page->value, bytes);")
    body = replace_once(body, "std::memcpy(out, input->keyData() + offset, row_bytes);",
                        "auditCopy(true, out, input->keyData() + offset, row_bytes);")
    body = replace_once(body, "std::memcpy(page->value.data() + output_row % 64 * row_bytes,",
                        "auditCopy(false, page->value.data() + output_row % 64 * row_bytes,")
    body = replace_once(body, "if (pos < dst.base)\n          continue;",
                        "if (counting) ++counts.visited_rows;\n"
                        "        if (pos < dst.base) {\n"
                        "          if (counting) ++counts.skipped_rows;\n"
                        "          continue;\n        }")
    variants = [("auditBaseline", body)]
    body = replace_once(body, "        auditCopy(true, out, input->keyData() + offset, row_bytes);\n", "")
    body = replace_once(body, "if (src == pos || rope[l].frequencies.empty())\n          continue;",
                        "if (src == pos || rope[l].frequencies.empty()) {\n"
                        "          auditCopy(true, out, input->keyData() + offset, row_bytes);\n"
                        "          continue;\n        }")
    body = replace_once(body, "auto *head = out + h * dim * result->element_bytes;",
                        "auto *head = out + h * dim * result->element_bytes;\n"
                        "          const auto *input_head = input->keyData() + offset + h * dim * result->element_bytes;")
    body = replace_once(body, "std::memcpy(&key[d], head + d * 4, 4);", "std::memcpy(&key[d], input_head + d * 4, 4);")
    body = replace_once(body, "std::memcpy(&bits, head + d * 2, 2);", "std::memcpy(&bits, input_head + d * 2, 2);")
    variants.append(("auditNoKeyPrecopy", body))
    body = replace_once(body, "dst.base = length > retained ? static_cast<unsigned>(length - retained) : 0;",
                        "dst.base = length > retained ? static_cast<unsigned>(length - retained) : 0;\n"
                        "    dst.pages.reserve((length - dst.base + 63) / 64);")
    variants.append(("auditReserve", body))
    body = replace_once(body, "for (size_t t = 0; t < run.count; ++t)",
                        "for (size_t t = dst.base > run.destination ? dst.base - run.destination : 0; t < run.count; ++t)")
    variants.append(("auditSkipRows", body))
    body = replace_once(body, "const size_t output_row = pos - dst.base;",
                        "const size_t output_row = pos - dst.base;\n"
                        "        if (src == pos && output_row % 64 == 0 && row % 64 == 0 &&\n"
                        "            run.count - t >= 64 && length - pos >= 64 &&\n"
                        "            input->keySize() == 64 * row_bytes &&\n"
                        "            input->valueSize() == 64 * row_bytes) {\n"
                        "          dst.pages.push_back(input);\n"
                        "          if (counting) counts.shared_rows += 64;\n"
                        "          t += 63;\n          continue;\n        }")
    variants.append(("auditShareExact", body))
    generated = '#include "audit_support.h"\n#include <fp16.h>\n#include <limits>\n'
    generated += "\n".join(replace_once(code, "Manager::gatherSparsePrefix(", name + "(") for name, code in variants)
    (here / "generated.cpp").write_text(generated, encoding="utf-8")
    includes = [here] + [repo / p for p in (
        "Applications/CausalLM", "nntrainer", "nntrainer/tensor", "nntrainer/utils",
        "api", "api/ccapi/include", "api/capi/include")]
    includes += sorted({p.parent for p in (repo / "nntrainer").rglob("*.h")})
    if args.json_include:
        includes.append(args.json_include.resolve())
    flags = ["-std=c++17", "-O2", "-fno-fast-math", "-ffunction-sections", "-fdata-sections"]
    if args.sanitize:
        flags += ["-fsanitize=address,undefined", "-fno-sanitize-recover=all", "-fno-omit-frame-pointer"]
    flags += [v for p in dict.fromkeys(includes) for v in ("-I", str(p))]
    golden_source = source
    if args.sanitize:
        # Full KV source retains unrelated allocator vtables under sanitizer
        # instrumentation. Extract the golden method verbatim, without stubs.
        golden_source = here / "production_gather.cpp"
        golden_source.write_text(
            '#include <kv_cache_manager.h>\n#include <fp16.h>\n#include <limits>\n'
            'namespace causallm {\n' + original + '\n}\n', encoding="utf-8")
    sources = [here / "audit.cpp", here / "generated.cpp", golden_source, repo / "nntrainer/utils/fp16.cpp"]
    with tempfile.TemporaryDirectory(prefix="nntr-normal-gather-") as folder:
        build = Path(folder)

        def compile_one(item):
            index, path = item
            obj = build / f"{index}.o"
            subprocess.run([args.cxx, *flags, "-c", str(path), "-o", str(obj)], check=True)
            return obj

        with ThreadPoolExecutor(max_workers=4) as pool:
            objects = list(pool.map(compile_one, enumerate(sources)))
        binary = build / "normal-audit"
        link_flags = ["-fsanitize=address,undefined"] if args.sanitize else []
        subprocess.run([args.cxx, *link_flags, "-Wl,--gc-sections", *map(str, objects), "-o", str(binary)], check=True)
        result = subprocess.run([str(binary), str(output)], text=True, capture_output=True)
        output.with_suffix(".stderr.txt").write_text(result.stderr, encoding="utf-8")
        if result.returncode or (args.sanitize and result.stderr.strip()):
            raise RuntimeError(f"Audit failed ({result.returncode}):\n{result.stdout}\n{result.stderr}")
    report = json.loads(output.read_text(encoding="utf-8"))
    report["source_sha256"] = hashlib.sha256(full.encode()).hexdigest()
    report["function_sha256"] = hashlib.sha256(original.encode()).hexdigest()
    commit = args.commit
    if not commit:
        git = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, capture_output=True)
        commit = git.stdout.strip() if git.returncode == 0 else None
    report["checkout_commit"] = commit
    report["compiler"] = subprocess.check_output([args.cxx, "--version"], text=True).splitlines()[0]
    report["sanitizers"] = args.sanitize
    report["golden_build_scope"] = "verbatim extracted original method" if args.sanitize else "unchanged full production source"
    if args.sanitize:
        report["sanitizer_scope"] = "ASan/UBSan on gather functions and fixtures; full engine excluded"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
