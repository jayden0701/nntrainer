---
title: How to Run Test Cases
...


# How to Run Test Cases

- Use the built library

- Unit tests

For GTest-based test cases (common library)

```bash
$ cd build
$ ninja test
...
```

### Run test cases on Android

- To run unit tests on Android, follow [How to Run Android Examples](how-to-run-example-android.md) to set up the environment.
- Then, you can run the unit tests on Android as follows:

```
(nntrainer) $ ./tools/android_test.sh
(nntrainer) $ adb shell
(adb) $ cd /data/local/tmp/nntr_android_test
(adb) $ export LD_LIBRARY_PATH=.
(adb) $ ./unittest_layers
```

- For more information, please refer to [tools](../tools/README.md)
- [**Note**] The Android unit test script builds NNTrainer with GPU support by default.
