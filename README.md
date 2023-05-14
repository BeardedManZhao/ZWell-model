# ZWell-model

A deep learning model library that supports various deep network models and transformations to libraries such as Keras.
With this tool, it is easy to construct neural network objects that can be used for any API, saving additional API
learning time.

# 使用示例

There are many neural network implementations in this library that support conversion to various library model objects.
The following are relevant examples.

## 残差神经网络

You can obtain the general object of the residual neural network object from ZWell model in the following way.

```python
import zWell_model

# Obtaining residual neural network
resNet = zWell_model.res_net.ResNet(
    # Specify the number of residual blocks as 4 TODO defaults to 4
    model_layers_num=4,
    # Specify the number of output channels in the four residual blocks
    k=[12, 12, 12, 12],
    # Specify the step size in the four residual blocks
    stride=[1, 2, 1, 2],
    # Specify the input dimension of the residual neural network
    input_shape=(32, 32, 3),
    # Specify classification quantity
    classes=10
)
```

将获取到的残差神经网络对象直接转换成为 Keras 中的神经网络模型对象，使得其能够被keras支持。

```python
# This is an example Python script.
import numpy as np
from keras.datasets import cifar10
from keras.optimizers import Adam
from livelossplot import PlotLossesKeras

import zWell_model

# Obtaining a dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# data standardization
x_train, x_test = x_train.astype(np.float32) / 255., x_test.astype(np.float32) / 255.

# Obtaining residual neural network
resNet = zWell_model.res_net.ResNet(
    # Specify the number of residual blocks as 4 TODO defaults to 4
    model_layers_num=4,
    # Specify the number of output channels in the four residual blocks
    k=[12, 12, 12, 12],
    # Specify the step size in the four residual blocks
    stride=[1, 2, 1, 2],
    # Specify the input dimension of the residual neural network
    input_shape=(32, 32, 3),
    # Specify classification quantity
    classes=10
)

# Converting to the network model of Keras
model = resNet.to_keras_model()
model.summary()

# Start building the model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['acc']
)

# Start training the model
model.fit(
    x=x_train, y=y_train,
    validation_data=(x_test, y_test),
    batch_size=32, epochs=30,
    callbacks=[PlotLossesKeras()],
    verbose=1
)
```
