---
title: How to run examples
...

# How to run examples

## Preparing NNTrainer for execution

### Use PPA

If you don't want to build binaries, you can directly download from PPA with daily releases.

```bash
sudo add-apt-repository ppa:nnstreamer/ppa
sudo apt-get update
sudo apt-get install nntrainer
```

Note that this may install TensorFlow Lite packaged by us.

## Build examples (Ubuntu)

See [Getting Started](getting-started.md) for more information.

Install related packages before building nntrainer and examples.

1. gcc/g++ >= 8 ( std=c++17 is used )
    - (note) >= 13 is recommended to enable fp16 support
2. meson >= 0.55.0
3. libopenblas-dev and base
4. tensorflow-lite >= 2.3.0
5. libiniparser
6. libjsoncpp >=0.6.0 (if you want to use OpenAI/Gym)
7. libcurl >=7.47 (if you want to use OpenAI/Gym)
8. libgtest >=1.10 (for testing)

Important build options (meson)

1. platform : default none, set target platform (-Dplatform=tizen)
2. enable-blas : default true, add option to enable blas (-Denable-blas=true)
3. enable-app : default true, add option to enable Applications (-Denable-app=true)
4. install-app : default true, add option to install Applications (-Dinstall-app=true)
5. use_gym : default false, add option to use OpenAI Gym (-Duse_gym=false)
6. enable-capi : default auto, add option to install C-API (-Denable-capi=enabled)
7. enable-test : default true, add option to test (-Denable-test=true)
8. enable-logging : default true, add option to do logging (-Denable-logging=true)
9. enable-tizen-feature-check : default true, add option to enable tizen feature check (-Denable-tizen-feature-check=true)

For example, to build and install NNTrainer and C-API,

```bash
meson setup build --prefix=${NNTRAINER_ROOT} --sysconfdir=${NNTRAINER_ROOT} --libdir=lib --bindir=bin --includedir=include -Denable-capi=enabled
```

Build source code

```bash
# Set your own path to install libraries and header files
$ sudo vi ~/.bashrc

export NNTRAINER_ROOT=$HOME/nntrainer
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NNTRAINER_ROOT/lib
# Include NNStreamer headers and libraries
export C_INCLUDE_PATH=$C_INCLUDE_PATH:$NNTRAINER_ROOT/include
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$NNTRAINER_ROOT/include
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$NNTRAINER_ROOT/lib/pkgconfig

$ source ~/.bashrc

# Download source, then compile it.
# Build and install nntrainer
$ git clone https://github.com/nntrainer/nntrainer.git nntrainer.git
$ meson setup build --prefix=${NNTRAINER_ROOT} --sysconfdir=${NNTRAINER_ROOT} --libdir=lib --bindir=bin --includedir=include
$ ninja -C build install
$ cd ..
```
