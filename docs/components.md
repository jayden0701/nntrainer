## Components

### Supported Layers

This component defines layers which consist of a neural network model. Layers have their own properties to be set.

| Keyword | Layer Class Name | Description |
|:-------:|:---:|:---|
| add | AddLayer | Add tensors |
| subtract | SubtractLayer | Subtract tensors |
| multiply | MultiplyLayer | Multiply tensors |
| divide | DivideLayer | Divide tensors |
| pow | PowLayer | Power operation layer |
| sqrt | SQRTLayer | Square-root operation layer |
| sin | SineLayer | Sine operation layer |
| cos | CosineLayer | Cosine operation layer |
| tan | TangentLayer | Tangent operation layer |
| matmul | MatMulLayer | Matrix multiplication layer |
| cast | CastLayer | Cast tensor data type |
| gather | GatherLayer | Gather layer |
| slice | SliceLayer | Slice layer |
| negative | NegativeLayer | Negative operation layer |
| weight | WeightLayer | Weight layer |
| conv1d | Conv1DLayer | Convolution 1-Dimensional Layer |
| conv2d | Conv2DLayer | Convolution 2-Dimensional Layer |
| conv2dtranspose | Conv2DTransposeLayer | Transposed convolution 2-D layer |
| pooling2d | Pooling2DLayer | Pooling 2-Dimensional Layer. Supports average / max / global average / global max pooling |
| flatten | FlattenLayer | Flatten layer |
| fully_connected | FullyConnectedLayer | Fully connected layer |
| input | InputLayer | Input Layer.  This is not always required. |
| batch_normalization | BatchNormalizationLayer | Batch normalization layer |
| layer_normalization | LayerNormalizationLayer | Layer normalization layer |
| activation | ActivationLayer | Set by layer property |
| addition | AdditionLayer | Add input layers |
| attention | AttentionLayer | Attention layer |
| mol_attention | MoLAttentionLayer | Mixture of logits attention layer |
| centroid_knn | CentroidKNN | Centroid K-nearest neighbor layer |
| channel_shuffle | ChannelShuffle | Channel shuffle layer |
| concat | ConcatLayer | Concatenate input layers |
| multiout | MultiOutLayer | Multi-Output Layer |
| backbone_nnstreamer | NNStreamerLayer | Encapsulate NNStreamer layer |
| backbone_tflite | TfLiteLayer | Encapsulate tflite as a layer |
| permute | PermuteLayer | Permute layer for transpose |
| preprocess_flip | PreprocessFlipLayer | Preprocess random flip layer |
| preprocess_l2norm | PreprocessL2NormLayer | Preprocess simple l2norm layer to normalize |
| preprocess_translate | PreprocessTranslateLayer | Preprocess translate layer |
| reshape | ReshapeLayer | Reshape tensor dimension layer |
| reduce_mean | ReduceMeanLayer | Reduce mean layer |
| reduce_sum | ReduceSumLayer | Reduce sum layer |
| split | SplitLayer | Split layer |
| dropout | DropOutLayer | Dropout Layer |
| embedding | EmbeddingLayer | Embedding Layer |
| positional_encoding | PositionalEncodingLayer | Positional Encoding Layer |
| identity | IdentityLayer | Identity layer |
| upsample2d | Upsample2dLayer | Upsample 2-D layer |
| rnn | RNNLayer | Recurrent Layer |
| rnncell | RNNCellLayer | Recurrent Cell Layer |
| gru | GRULayer | Gated Recurrent Unit Layer |
| grucell | GRUCellLayer | Gated Recurrent Unit Cell Layer |
| lstm | LSTMLayer | Long Short-Term Memory Layer |
| lstmcell | LSTMCellLayer | Long Short-Term Memory Cell Layer |
| zoneout_lstmcell | ZoneoutLSTMCellLayer | Zoneout Long Short-Term Memory Cell Layer |
| time_dist | TimeDistLayer | Time distributed Layer |
| multi_head_attention | MultiHeadAttentionLayer | Multi Head Attention Layer |
| mse | MSELossLayer | Mean squared error loss layer |
| cross_sigmoid | CrossEntropySigmoidLossLayer | Cross entropy sigmoid loss layer |
| cross_softmax | CrossEntropySoftmaxLossLayer | Cross entropy softmax loss layer |
| constant_derivative | ConstantDerivativeLossLayer | Constant derivative loss layer |
