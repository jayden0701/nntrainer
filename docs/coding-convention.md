---
title: Coding Convention
...

# Coding Convention

## C headers (.h)

You may indent differently from what clang-format does. You may also break the 80-column rule with header files.

Except for those two allowances, follow the general coding styles mandated by clang-format.
Pull requests are checked by the `C++ Format Checker`, which runs clang-format 14 on changed lines only.

## C/C++ files (.cpp, .c)

Use .h for headers and .cpp / .c for source.
You have to use clang-format 14 with the given [.clang-format](https://github.com/nntrainer/nntrainer/blob/main/.clang-format) file.
Run `clang-format-14 -i <changed .c/.cpp/.h files>` before submitting C/C++ changes.


## Other files

Project-specific Java, Python, and Bash conventions are not documented yet.
Follow the style of nearby files and keep scripts readable and portable.


# File Locations

## Directory structure of nntrainer.git

- **api**: API definitions and implementations
    - **capi**: C-APIs (Tizen and others)
    - **ccapi**: C++-APIs
- **Applications**: Examples for NNTrainer
- **debian**: Debian/Ubuntu packaging files
- **docs**: Documentation
- **jni**: Android/Java build scripts.
- **nnstreamer**: NNStreamer sub-filter code for NNTrainer
- **nntrainer**: All core NNTrainer code is located here
- **packaging**: Tizen RPM build scripts. openSUSE/Red Hat Linux may reuse this.
- **test**: Unit tests, grouped by subdirectory. Most tests use GTest.
- **tools**: Various developmental tools and scripts of NNTrainer.

## Related git repositories

- [NNStreamer](https://github.com/nnstreamer/nnstreamer)
- [TAOS-CI, CI Service for On-Device AI Systems](https://github.com/nnstreamer/TAOS-CI)
