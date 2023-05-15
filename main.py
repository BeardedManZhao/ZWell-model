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
    batch_size=32, epochs=30,
    # 使用回调函数实时打印模型情况
    callbacks=[PlotLossesKeras()],
    verbose=1
)
