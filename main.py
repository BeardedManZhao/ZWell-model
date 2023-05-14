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
resNet = zWell_model.conv_net2.ConvNetV2(
    # 指定基础架构之上要额外增加的卷积层数量为 1 TODO 默认为4
    model_layers_num=1,
    # 指定2个卷积层中的步长
    stride=[1],
    # 指定卷积神经网络的输入维度
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
    # 30批模型训练 每次梯度下降使用 32 个样本
    batch_size=32, epochs=6,
    # 使用回调函数实时打印模型情况
    callbacks=[PlotLossesKeras()],
    verbose=1
)
