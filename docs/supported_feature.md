### Supported Optimizers

NNTrainer provides

| Keyword | Optimizer Name | Description |
|:-------:|:---:|:---:|
| sgd | Stochastic Gradient Descent | - |
| adam | Adaptive Moment Estimation | - |
| adamw | Adam with decoupled weight decay regularization | - |
| lion | Lion Optimizer | - |

| Keyword | Learning Rate | Description |
|:-------:|:---:|:---:|
| exponential | exponential learning rate decay | - |
| constant | constant learning rate | - |
| step | step learning rate | - |
| cosine | cosine annealing learning rate | - |
| linear | linear learning rate decay | - |

### Supported Loss Functions

NNTrainer provides

| Keyword | Class Name | Description |
|:-------:|:---:|:---:|
| cross_sigmoid | CrossEntropySigmoidLossLayer | Cross entropy sigmoid loss layer |
| cross_softmax | CrossEntropySoftmaxLossLayer | Cross entropy softmax loss layer |
| constant_derivative | ConstantDerivativeLossLayer | Constant derivative loss layer |
| mse | MSELossLayer | Mean square error loss layer |

### Supported Activation Functions

NNTrainer provides

| Keyword | Activation Name | Description |
|:-------:|:---:|:---|
| tanh | tanh function | set as layer property |
| sigmoid | sigmoid function | set as layer property |
| softmax | softmax function | set as layer property |
| relu | relu function | set as layer property |
| leaky_relu | leaky_relu function | set as layer property |
| swish | swish function | set as layer property |
| gelu | gelu function | set as layer property |
| tanh_gelu | tanh gelu function | set as layer property |
| sigmoid_gelu | sigmoid gelu function | set as layer property |
| elu | elu function | set as layer property |
| selu | selu function | set as layer property |
| softplus | softplus function | set as layer property |
| mish | mish function | set as layer property |
| none | no activation | set as layer property |

### Tensor

Tensor is responsible for calculation of a layer. It executes several operations such as addition, division, multiplication, dot product, data averaging and so on. In order to accelerate calculation speed, CBLAS (C-Basic Linear Algebra: CPU) and CUBLAS (CUDA: Basic Linear Algebra) for PC (Especially NVIDIA GPU) are implemented for some of the operations. Later, these calculations will be optimized.
Currently, we support lazy calculation mode to reduce complexity for copying tensors during calculations.

| Keyword | Description |
|:-------:|:---:|
| 4D Tensor | B, C, H, W|
| Add/sub/mul/div | - |
| sum, average, argmax | - |
| Dot, Transpose | - |
| normalization, standardization | - |
| save, read | - |

### Others

NNTrainer provides

| Keyword | Feature Name | Description |
|:-------:|:---:|:---|
| weight_initializer | Weight Initialization | Zeros, Ones, Xavier(Normal/Uniform), LeCun(Normal/Uniform), He(Normal/Uniform), None |
| weight_regularizer | Weight decay (l2norm only) | Set weight_regularizer=l2norm and weight_regularizer_constant |
