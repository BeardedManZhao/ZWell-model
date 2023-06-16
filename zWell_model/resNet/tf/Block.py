# -*- coding: utf-8 -*-
# @Time : 2023/6/15 18:10
# @Author : zhao
# @Email : liming7887@qq.com
# @File : Block.py
# @Project : ZWell-model
import tensorflow as tf


class BlockV2(tf.keras.layers.Layer):
    """
    残差块的类 该类能够被第二版残差神经网络集成。
    """

    def __init__(self, filter_num, k_size=(3, 3), stride=1, name='res_block'):
        """
        :param filter_num: 当前残差块中的滤波器的个数。
        :param k_size: 当前残差块中的卷积层中的卷积核尺寸。
        :param stride: 当前残差块中的卷积步长。
        :param name: 当前残差块的名称
        """
        super(BlockV2, self).__init__()
        self.b_name = name
        # 准备 c1 第一个卷积层
        self.c1 = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=k_size, strides=stride, padding='same')
        # 准备批处理与激活函数
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dr1 = tf.keras.layers.Dropout(0.5)
        self.af1 = tf.keras.layers.Activation('relu')
        # 准备 c2 第二个卷积层 TODO 需要注意的是，在这里的步长应为1 否则两次不同的步长会导致残差块中的输出与我们使用时的预期不一致
        # 这是因为我们在很多层对象中的步长指定为2的时候，其输出的数据只会为步长为1时的 1/2 我们的习惯导致我们更加喜欢这类层对象
        self.c2 = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=k_size, strides=1, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.af2 = tf.keras.layers.Activation('relu')
        # 准备下采样模型 由于新的特征是被卷积了两次单独，其维度是有了变化的，为了最后的求和，我们需要添加1x1卷积核
        # 因此就可以使用这个卷积方式针对支路中的恒定数值进行一个相同步长的卷积，实现维度一致
        self.down_sample = tf.keras.models.Sequential()
        self.down_sample.add(tf.keras.layers.Conv2D(filters=filter_num, kernel_size=1, strides=stride))

    def call(self, inputs, training=None, mask=None):
        """
        调用残差块进行前向传播操作
        :param inputs: 残差块要接收的数据。
        :param training: 训练时是否使用
        :param mask:
        :return:当前残差块的前向传播结果
        """
        # 进行下采样
        x = self.down_sample(inputs)
        # 开始第一个卷积层的前向传播操作
        inputs = self.c1(inputs)
        inputs = self.bn1(inputs)
        inputs = self.dr1(inputs)
        inputs = self.af1(inputs)
        # 开始第二个卷积层的前向传播操作
        inputs = self.c2(inputs)
        inputs = self.bn2(inputs)
        inputs = self.af2(inputs)
        # 准备计算残差块的输出结果 这里是卷积之后的数据 + 原数据
        inputs = tf.keras.layers.add([inputs, x])
        # 最后将结果进行激活
        return tf.nn.relu(inputs)
