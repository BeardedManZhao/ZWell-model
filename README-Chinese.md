# ![image](https://github.com/BeardedManZhao/ZWell-model/assets/113756063/87aedd4f-092f-4ff7-9e4a-3749cffedf0d) ZWell-model

# 介绍

深度学习模型库，其支持各种深度网络模型，支持向keras等库进行转换，通过此工具能够轻松的构建出能用于任何API的神经网络对象，节省额外的API学习时间。

## 获取方式

通过 pip 工具进行库的安装，也可以通过GitHub中的源码手动装载。

```shell
pip install zWell-model
```

## 当前发展

这里展示的是当前 zWell-model 支持的深度学习模型，以及其支持接入的第三方库等更详细的情况。

| 神经网络名称    | 引用方式                   | 支持接入keras | 支持版本            |
|-----------|------------------------|-----------|-----------------|
| 基本卷积第一版   | zWell_model.ConvNetV1  | yes       | v0.0.1.20230514 |
| 基本卷积第二版   | zWell_model.ConvNetV2  | yes       | v0.0.1.20230514 |
| 残差神经网络第一版 | zWell_model.ResNetV1   | yes       | v0.0.1.20230514 |
| 稠密神经网络第一版 | zWell_model.DenseNetV1 | yes       | v0.0.2.2023xxxx |

# 使用示例

在本库中有很多的神经网络实现，其支持转换到各种不同的库模型对象，接下来就是有关的示例。

## 基本卷积神经网络

基本卷积神经网络包含最基本的网络结构，其学习速度与精确度折中，共有两个版本。

### 第一版

以学习速度为主要核心的卷积神经网络，针对大量特征几乎相同的数据，该网络的学习能够提升学习速度。

```python
import zWell_model

# 获取到第一种卷积神经网络
resNet = zWell_model.ConvNetV1(
    # 指定基础架构之上要额外增加的卷积层数量为 4 TODO 默认为1
    model_layers_num=4,
    # 指定四个卷积层中的步长
    stride=[1, 2, 1, 2],
    # 指定卷积神经网络的输入维度
    input_shape=(None, 32, 32, 3),
    # 指定分类数量
    classes=10
)
```

### 第二版

以学习速度与防止过拟合为核心的卷积神经网络，针对大量特征多样化的数据，该网络的学习能够提升学习速度与模型精度。

```python
import zWell_model

# 获取到第二种卷积神经网络
resNet = zWell_model.ConvNetV2(
    # 指定基础架构之上要额外增加的卷积层数量为 1 TODO 默认为1
    model_layers_num=1,
    # 指定1个卷积层中的步长
    stride=[1],
    # 指定卷积神经网络的输入维度
    input_shape=(32, 32, 3),
    # 指定分类数量
    classes=10
)
```

### 基本卷积神经网络使用示例

在这里展示的就是将基本卷积神经网络系列，转换成为keras中的模型并进行调用的操作。

```python
# 这是一个示例 Python 脚本。
import numpy as np
from keras.datasets import cifar10
from keras.optimizers import Adam
from livelossplot import PlotLossesKeras

import zWell_model

# 获取到数据集
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# 数据标准化以及维度拓展
x_train = x_train.astype(np.float32).reshape(-1, 32, 32, 3) / 255.
x_test = x_test.astype(np.float32).reshape(-1, 32, 32, 3) / 255.

# 获取到第二种卷积神经网络
resNet = zWell_model.ConvNetV2(
    # 指定卷积层数量为 4 TODO 默认为4
    model_layers_num=4,
    # 指定四个卷积层中的步长
    stride=[1, 2, 1, 2],
    # 指定卷积神经网络的输入维度
    input_shape=(None, 32, 32, 3),
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
    # 30批模型训练 每次梯度下降使用 32 个样本
    batch_size=32, epochs=30,
    # 使用回调函数实时打印模型情况
    callbacks=[PlotLossesKeras()],
    verbose=1
)
```

## 残差神经网络

您可以通过下面的方式，从ZWell-model中获取到残差神经网络对象的通用对象。

```python
import zWell_model

# 获取到残差神经网络
resNet = zWell_model.res_net1.ResNetV1(
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
# 数据标准化以及维度拓展
x_train = x_train.astype(np.float32).reshape(-1, 32, 32, 3) / 255.
x_test = x_test.astype(np.float32).reshape(-1, 32, 32, 3) / 255.

# 获取到残差神经网络
resNet = zWell_model.ResNetV1(
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

print(resNet)

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
    # 30批模型训练 每次梯度下降使用 32 个样本
    batch_size=32, epochs=6,
    # 使用回调函数实时打印模型情况
    callbacks=[PlotLossesKeras()],
    verbose=1
)
```

## 稠密神经网络

您可以通过下面的方式从 ZWell-mode 库中获取到稠密神经网络模型。

```python
import zWell_model

# 获取到稠密神经网络
resNet = zWell_model.dense_net1.DenseNetV1(
    # 指定稠密块数量为 3 TODO 默认为4
    model_layers_num=3,
    # 指定 2 个稠密块后的过渡层中的卷积步长
    stride=[1, 1, 1],
    # 指定稠密神经网络的输入维度
    input_shape=(32, 32, 3),
    # 指定分类数量
    classes=10
)
```

将获取到的稠密神经网络对象转换成为 keras 库中的深度神经网络对象。

```python
# 这是一个示例 Python 脚本。
import numpy as np
from keras.datasets import cifar10
from keras.optimizers import Adam
from livelossplot import PlotLossesKeras

import zWell_model

# 获取到数据集
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# 数据标准化以及维度拓展
x_train = x_train.astype(np.float32).reshape(-1, 32, 32, 3) / 255.
x_test = x_test.astype(np.float32).reshape(-1, 32, 32, 3) / 255.

# 获取到稠密神经网络
resNet = zWell_model.DenseNetV1(
    # 指定稠密块数量为 3 TODO 默认为4
    model_layers_num=3,
    # 指定 3 个稠密块后的过渡层中的卷积步长
    stride=[1, 1, 1],
    # 指定稠密神经网络的输入维度
    input_shape=(32, 32, 3),
    # 指定分类数量
    classes=10
)

print(resNet)

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

<hr>

## 更多操作

### 使用简称获取神经网络

从 0.0.2 版本开始，我们可以使用模型的别名获取到模型类，其操作有利于简单的模型实例化，减少代码量，不同的模型简称与实例化具体操作如下所示。

| 模型原名      | 模型简称  |
|-----------|-------|
| ConvNetV1 | Cnn1  |
| ConvNetV2 | Cnn2  |
| ResNetV1  | RCnn1 |

```python
# 这是一个示例 Python 脚本。
import zWell_model as zModel
from zWell_model.allModel import AllModel

# 在这通过简称获取到了 ConvNetV1 的模型对象 TODO Cnn1 是简称
z_model: AllModel = zModel.Cnn1(
    # 指定基础架构之上要额外增加的卷积层数量为 4 TODO 默认为1
    model_layers_num=4,
    # 指定四个卷积层中的步长
    stride=[1, 2, 1, 2],
    # 指定卷积神经网络的输入维度
    input_shape=(None, 32, 32, 3),
    # 指定分类数量
    classes=10
)
print(z_model)
print(z_model.to_keras_model())
```
