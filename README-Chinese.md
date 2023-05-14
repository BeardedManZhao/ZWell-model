# ZWell-model

深度学习模型库，其支持各种深度网络模型，支持向keras等库进行转换，通过此工具能够轻松的构建出能用于任何API的神经网络对象，节省额外的API学习时间。

# 使用示例

在本库中有很多的神经网络实现，其支持转换到各种不同的库模型对象，接下来就是有关的示例。

## 残差神经网络

您可以通过下面的方式，从ZWell-model中获取到残差神经网络对象的通用对象。

```python
import zWell_model

# 获取到残差神经网络
resNet = zWell_model.res_net.ResNet(
    # 指定残差块数量为 4 TODO 默认为4
    model_layers_num=4,
    # 指定四个残差块中的输出通道数量
    k=[12, 12, 12, 12],
    # 指定四个残差块中的步长
    stride=[1, 2, 1, 2],
    # 指定残差神经网络的输入维度
    input_shape=(32, 32, 3),
    # 指定分类数量
    classes=10
)
```

将获取到的残差神经网络对象直接转换成为 Keras 中的神经网络模型对象，使得其能够被keras支持。

```python
# 这是一个示例 Python 脚本。
import numpy as np
from keras.datasets import cifar10
from keras.optimizers import Adam
from livelossplot import PlotLossesKeras

import zWell_model

# 获取到数据集
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# 数据标准化
x_train, x_test = x_train.astype(np.float32) / 255., x_test.astype(np.float32) / 255.

# 获取到残差神经网络
resNet = zWell_model.res_net.ResNet(
    # 指定残差块数量为 4 TODO 默认为4
    model_layers_num=4,
    # 指定四个残差块中的输出通道数量
    k=[12, 12, 12, 12],
    # 指定四个残差块中的步长
    stride=[1, 2, 1, 2],
    # 指定残差神经网络的输入维度
    input_shape=(32, 32, 3),
    # 指定分类数量
    classes=10
)

# 转换到 keras 的网络模型
model = resNet.to_keras_model()
model.summary()

# 开始构建模型
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['acc']
)

# 开始训练模型
model.fit(
    x=x_train, y=y_train,
    validation_data=(x_test, y_test),
    batch_size=32, epochs=30,
    callbacks=[PlotLossesKeras()],
    verbose=1
)
```
