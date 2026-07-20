#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

"""Verify that Gradle packaged the selected Android NDK's C++ runtime."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


PRIMARY_EXCEPTION_SYMBOL = "__cxa_init_primary_exception"


class ValidationError(RuntimeError):
    """Raised when an Android native package is internally inconsistent."""


def read_ndk_revision(ndk: Path) -> str:
    source_properties = ndk / "source.properties"
    if not source_properties.is_file():
        raise ValidationError(f"NDK metadata is missing: {source_properties}")

    for line in source_properties.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "Pkg.Revision":
            revision = value.strip()
            if revision:
                return revision
    raise ValidationError(f"Pkg.Revision is missing from {source_properties}")


def find_ndk_tools(ndk: Path) -> tuple[Path, Path]:
    prebuilt_root = ndk / "toolchains" / "llvm" / "prebuilt"
    matches: list[tuple[Path, Path]] = []
    host_directories = sorted(prebuilt_root.iterdir()) if prebuilt_root.is_dir() else []
    for host_dir in host_directories:
        readelf_candidates = [
            host_dir / "bin" / "llvm-readelf",
            host_dir / "bin" / "llvm-readelf.exe",
        ]
        readelf = next(
            (candidate for candidate in readelf_candidates if candidate.is_file()),
            None,
        )
        runtime = (
            host_dir
            / "sysroot"
            / "usr"
            / "lib"
            / "aarch64-linux-android"
            / "libc++_shared.so"
        )
        if readelf is not None and runtime.is_file():
            matches.append((readelf, runtime))

    if len(matches) != 1:
        raise ValidationError(
            f"Expected one host toolchain under {prebuilt_root}, found {len(matches)}"
        )
    return matches[0]


def run_readelf(readelf: Path, option: str, binary: Path) -> str:
    result = subprocess.run(
        [str(readelf), option, "--wide", str(binary)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise ValidationError(
            f"llvm-readelf failed for {binary}: {result.stderr.strip()}"
        )
    return result.stdout


def read_build_id(readelf: Path, binary: Path) -> str:
    output = run_readelf(readelf, "--notes", binary)
    match = re.search(r"Build ID:\s*([0-9a-fA-F]+)", output)
    if not match:
        raise ValidationError(f"ELF Build ID is missing: {binary}")
    return match.group(1).lower()


def symbol_states(readelf: Path, binary: Path, symbol: str) -> tuple[bool, bool]:
    output = run_readelf(readelf, "--dyn-syms", binary)
    undefined = False
    defined = False
    for line in output.splitlines():
        fields = line.split()
        if len(fields) < 8:
            continue
        name = fields[7].split("@", 1)[0]
        if name != symbol:
            continue
        binding = fields[4]
        visibility = fields[5]
        section = fields[6]
        if section == "UND" and binding == "GLOBAL":
            undefined = True
        elif (
            section != "UND"
            and binding in {"GLOBAL", "WEAK"}
            and visibility in {"DEFAULT", "PROTECTED"}
        ):
            defined = True
    return undefined, defined


def normalized_path(value: str | Path) -> str:
    spelling = str(value).strip().replace("\\:", ":").replace("\\", "/")
    try:
        candidate = Path(spelling)
        if candidate.exists():
            spelling = candidate.resolve().as_posix()
    except OSError:
        pass
    spelling = re.sub(r"/+", "/", spelling).rstrip("/")
    if re.match(r"^[A-Za-z]:/", spelling):
        spelling = spelling[0].lower() + spelling[1:]
        spelling = spelling.casefold()
    return spelling


def read_cmake_cache(cache: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        declaration, value = line.split("=", 1)
        key = declaration.split(":", 1)[0]
        values[key] = value
    return values


def validate_cmake_caches(
    cache_root: Path, expected_ndk: Path, expected_revision: str
) -> None:
    caches = [
        cache
        for cache in cache_root.rglob("CMakeCache.txt")
        if "arm64-v8a" in cache.parts
    ]
    if not caches:
        raise ValidationError(
            f"No arm64-v8a Gradle CMake cache found under {cache_root}"
        )

    expected_path = normalized_path(expected_ndk)
    for cache in caches:
        values = read_cmake_cache(cache)
        actual_ndk = values.get("CMAKE_ANDROID_NDK") or values.get("ANDROID_NDK")
        if not actual_ndk:
            raise ValidationError(f"Gradle NDK path is missing from {cache}")
        if normalized_path(actual_ndk) != expected_path:
            raise ValidationError(
                f"Gradle used a different NDK in {cache}: "
                f"{actual_ndk} != {expected_ndk}"
            )

        actual_revision = values.get("NNTRAINER_ANDROID_NDK_REVISION")
        if actual_revision != expected_revision:
            raise ValidationError(
                f"Gradle NDK revision mismatch in {cache}: "
                f"{actual_revision} != {expected_revision}"
            )


def extract_member(archive: zipfile.ZipFile, member: str, destination: Path) -> None:
    with archive.open(member) as source, destination.open("wb") as output:
        shutil.copyfileobj(source, output)


def validate_archive(
    label: str,
    archive_path: Path,
    native_prefix: str,
    readelf: Path,
    expected_runtime_build_id: str,
    temporary_directory: Path,
) -> None:
    expected_runtime = f"{native_prefix}/arm64-v8a/libc++_shared.so"
    expected_causallm = f"{native_prefix}/arm64-v8a/libcausallm.so"

    with zipfile.ZipFile(archive_path) as archive:
        names = [entry.filename for entry in archive.infolist()]
        runtime_entries = [
            name
            for name in names
            if PurePosixPath(name).name == "libc++_shared.so"
        ]
        if runtime_entries != [expected_runtime]:
            raise ValidationError(
                f"{label} must contain exactly {expected_runtime}; "
                f"found {runtime_entries}"
            )
        if names.count(expected_causallm) != 1:
            raise ValidationError(
                f"{label} must contain exactly one {expected_causallm}"
            )

        runtime = temporary_directory / f"{label}-libc++_shared.so"
        causallm = temporary_directory / f"{label}-libcausallm.so"
        extract_member(archive, expected_runtime, runtime)
        extract_member(archive, expected_causallm, causallm)

    packaged_build_id = read_build_id(readelf, runtime)
    if packaged_build_id != expected_runtime_build_id:
        raise ValidationError(
            f"{label} libc++ Build ID {packaged_build_id} does not match the "
            f"selected NDK runtime {expected_runtime_build_id}"
        )

    requires_symbol, _ = symbol_states(
        readelf, causallm, PRIMARY_EXCEPTION_SYMBOL
    )
    if requires_symbol:
        _, runtime_defines_symbol = symbol_states(
            readelf, runtime, PRIMARY_EXCEPTION_SYMBOL
        )
        if not runtime_defines_symbol:
            raise ValidationError(
                f"{label} libcausallm.so requires {PRIMARY_EXCEPTION_SYMBOL}, "
                "but its packaged libc++_shared.so does not provide it"
            )

    requirement = "required and provided" if requires_symbol else "not required"
    print(
        f"{label}: one selected-NDK libc++ runtime; "
        f"{PRIMARY_EXCEPTION_SYMBOL} {requirement}"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ndk", required=True, type=Path)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--aar", required=True, type=Path)
    parser.add_argument("--apk", required=True, type=Path)
    parser.add_argument("--cmake-cache-root", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    ndk = arguments.ndk.resolve()
    actual_revision = read_ndk_revision(ndk)
    if actual_revision != arguments.expected_revision:
        raise ValidationError(
            f"Selected NDK revision {actual_revision} does not match "
            f"{arguments.expected_revision}"
        )

    readelf, ndk_runtime = find_ndk_tools(ndk)
    expected_build_id = read_build_id(readelf, ndk_runtime)
    validate_cmake_caches(
        arguments.cmake_cache_root.resolve(), ndk, actual_revision
    )

    with tempfile.TemporaryDirectory(prefix="nntrainer-android-runtime-") as temp:
        temporary_directory = Path(temp)
        validate_archive(
            "AAR",
            arguments.aar,
            "jni",
            readelf,
            expected_build_id,
            temporary_directory,
        )
        validate_archive(
            "APK",
            arguments.apk,
            "lib",
            readelf,
            expected_build_id,
            temporary_directory,
        )

    print(f"Verified Android NDK {actual_revision} ({ndk})")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValidationError, zipfile.BadZipFile) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
