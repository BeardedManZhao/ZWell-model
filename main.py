# 这是一个示例 Python 脚本。
from keras.datasets import cifar10
from keras.optimizers import Adam

import zWell_model

# 获取到数据集
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# 数据类型转换
x_train = x_train.astype(float)
x_test = x_test.astype(float)

# 获取到第二版的残差神经网络
resNet = zWell_model.ResNetV2(
    # 指定残差块数量为 4 TODO 默认为4
    model_layers_num=4,
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

# 开始构建模型
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(learning_rate=0.002),
    metrics=['acc']
)

# 开始训练模型
model.fit(
    x=x_train, y=y_train,
    validation_data=(x_test, y_test),
    # 4 周期模型训练 每次梯度下降使用 64 个样本
    batch_size=64, epochs=4,
)
model.summary()
