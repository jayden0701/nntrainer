# Layers

In this example, we demonstrate training Single Layers model with dummy dataset.  
You just Run Simple Layer example with ```.ini``` file.  (it automatically config input&output size.)  
if you want to modify input&output size you just modify ```.ini``` file


## Layer List

| No  | Layer Type              | Model Summary                 |
| --- | ----------------------- | ----------------------------- |
| 1   | Linear(Fully connected) | -> FC ->                      |
| 2   | Convolution             | -> Conv ->                    |
| 3   | LSTM                    | -> LSTM ->                    |
| 4   | Model_A_Linear          | -> FC -> FC -> FC ->          |
| 5   | Model_A_Conv            | -> Conv -> Conv -> Conv ->    |
| 6   | Model_C_Linear          | -> FC -> RELU -> Flatten ->   |
| 7   | Model_C_Conv            | -> Conv -> RELU -> Flatten -> |

## How to Run

### 1. NNTrainer
Build with meson, ninja

In "nntrainer dir"
```.bash
meson build
```

In "nntrainer/build dir"
```.bash
ninja
```

In "nntrainer/build dir"
```.bash
./Applications/Layers/jni/nntrainer_Layers ../Applications/Layers/res/{ini file}
```

### 2. PyTorch, TensorFlow

We provide PyTorch and TensorFlow examples with the same model code. You can test the model in the ```./PyTorch``` and ```./Tensorflow``` directories and run with:
```.bash
python3 ./PyTorch/{LayerName}.py
python3 ./Tensorflow/{LayerName}.py
```
