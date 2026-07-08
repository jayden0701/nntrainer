---
title: Android NNTrainer PicoGPT Application Sample
...

# Android NNTrainer PicoGPT Application Sample
This is a practical demonstration of the Android NNTrainer PicoGPT inference application.

## How to run
Build nntrainer with `${NNTRAINER_HOME}/tools/package_android.sh` as in [Document](https://github.com/nntrainer/nntrainer/blob/main/docs/how-to-run-example-android.md)

```bash
$ ls
api                 CONTRIBUTING.md  index.md  MAINTAINERS.md     nnstreamer        nntrainer.pc.in  RELEASE.md
Applications        debian           jni       meson.build        nntrainer         packaging        test
CODE_OF_CONDUCT.md  docs             LICENSE   meson_options.txt  nntrainer.ini.in  README.md        tools
$
$ ./tools/package_android.sh
$ ls builddir/android_build_result
Android.mk  conf  examples  include  lib
$ ls builddir/android_build_result/libs/arm64-v8a
libcapi-nntrainer.so  libccapi-nntrainer.so  libc++_shared.so  libnnstreamer-native.so  libnntrainer.so
```

Build the JNI with `${APP_HOME}/app/src/main/jni/prepare_android_deps.sh`
```bash
$ cd ${APP_HOME}/app/src/main/jni

./prepare_android_deps.sh 
${APP_HOME}/app/src/main/jni/nntrainer
[arm64-v8a] Prebuilt       : libccapi-nntrainer.so <= jni/nntrainer/lib/arm64-v8a/
[arm64-v8a] Install        : libccapi-nntrainer.so => libs/arm64-v8a/libccapi-nntrainer.so
[arm64-v8a] Prebuilt       : libnntrainer.so <= jni/nntrainer/lib/arm64-v8a/
[arm64-v8a] Install        : libnntrainer.so => libs/arm64-v8a/libnntrainer.so
[arm64-v8a] Compile++      : picogpt_jni <= picogpt.cpp
[arm64-v8a] Compile++      : picogpt_jni <= picogpt_jni.cpp
[arm64-v8a] Prebuilt       : libc++_shared.so <= <NDK>/sources/cxx-stl/llvm-libc++/libs/arm64-v8a/
[arm64-v8a] SharedLibrary  : libpicogpt_jni.so
[arm64-v8a] Install        : libpicogpt_jni.so => libs/arm64-v8a/libpicogpt_jni.so
[arm64-v8a] Install        : libc++_shared.so => libs/arm64-v8a/libc++_shared.so
```

Prepare the model and tokenizer assets. The app expects `pico_gpt.bin`, `merges.txt`, and `vocab.json` in the Android assets directory.

```bash
$ cd ${APP_HOME}/app/src/main/assets
$ ls
merges.txt  pico_gpt.bin  vocab.json

```


Build Application with gradlew.

``` bash
$ cd ${APP_HOME}
$ ./gradlew build

> Configure project :app

> Task :app:stripDebugDebugSymbols
Unable to strip the following libraries, packaging them as they are: libc++_shared.so, libccapi-nntrainer.so, libnntrainer.so, libpicogpt_jni.so.

...

BUILD SUCCESSFUL in 10s
83 actionable tasks: 81 executed, 2 up-to-date

```

Install the application and run

``` bash
$ adb install ${APP_HOME}/app/build/outputs/apk/debug/app-debug.apk

```

After installing the application, run PicoGPT from the Android device.
