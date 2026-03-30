# nntrainer cloud build requirements (for Codex)

This file summarizes only the following sections from `docs/getting-started.md`:
- `Prerequisites`
- `Linux Self-Hosted Build -> Build with meson`

The goal is to let an automated coding agent (e.g. Codex) provision a Linux cloud environment and build `nntrainer` with Meson.

## Target

Build `nntrainer` from source on Debian/Ubuntu-like Linux using Meson.

## Minimum prerequisites

Install or ensure availability of:

- `gcc` / `g++` >= 7
  - `C++17` is required.
  - `gcc/g++ >= 13` is recommended if fp16 support is needed.
- `meson >= 0.55.0`
- OpenBLAS development package
- TensorFlow Lite >= 2.3.0
- `iniparser`
- `jsoncpp` (needed if OpenAI-related functionality is used)
- `libcurl3` >= 7.47 (needed if OpenAI-related functionality is used)
- `gtest` >= 1.10 (for testing)

## Apt repository setup

Some dependencies come from the NNStreamer PPA.

```bash
sudo add-apt-repository ppa:nnstreamer/ppa
sudo apt-get update
```

## Required apt packages for Meson build

Install Meson/Ninja first:

```bash
sudo apt install -y meson ninja-build
```

Install the packages explicitly listed in the upstream document:

```bash
sudo apt install -y \
  gcc g++ pkg-config \
  libopenblas-dev \
  libiniparser-dev \
  libjsoncpp-dev \
  libcurl3-dev \
  tensorflow2-lite-dev \
  nnstreamer-dev \
  libglib2.0-dev \
  libgstreamer1.0-dev \
  libgtest-dev \
  ml-api-common-dev \
  flatbuffers-compiler \
  ml-inference-api-dev
```

## Repository preparation

At the repository root, make sure submodules are initialized:

```bash
git submodule sync && git submodule update --init --depth 1
```

## Build commands (Meson)

Run from the git repository root:

```bash
meson build
ninja -C build install
```

## Install locations

The upstream document states that the Meson install step will:

- install libraries to `{prefix}/{libdir}`
- install common header files to `{prefix}/{includedir}`

## Recommended Codex workflow

Use the following order:

1. Clone repository.
2. Add the NNStreamer PPA.
3. `apt-get update`.
4. Install all required apt packages listed above.
5. Initialize git submodules.
6. Run `meson build`.
7. Run `ninja -C build install`.

## Example bootstrap script

```bash
set -euxo pipefail

# repo should already be cloned before this step
sudo add-apt-repository -y ppa:nnstreamer/ppa
sudo apt-get update

sudo apt install -y \
  meson ninja-build \
  gcc g++ pkg-config \
  libopenblas-dev \
  libiniparser-dev \
  libjsoncpp-dev \
  libcurl3-dev \
  tensorflow2-lite-dev \
  nnstreamer-dev \
  libglib2.0-dev \
  libgstreamer1.0-dev \
  libgtest-dev \
  ml-api-common-dev \
  flatbuffers-compiler \
  ml-inference-api-dev

git submodule sync
git submodule update --init --depth 1

meson build
ninja -C build install
```

## Optional note from troubleshooting

If the build fails due to a missing FlatBuffers header such as:

- `flatbuffers/flatbuffers.h: No such file or directory`

then the upstream document suggests installing:

```bash
sudo apt install -y libflatbuffers-dev
```

## Source basis

This file was derived from:
- `docs/getting-started.md` -> `Prerequisites`
- `docs/getting-started.md` -> `Linux Self-Hosted Build` -> `Build with meson`
