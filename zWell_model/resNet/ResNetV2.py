# -*- coding: utf-8 -*-
# @Time : 2023/6/15 18:06
# @Author : zhao
# @Email : liming7887@qq.com
# @File : ResNetV2.py
# @Project : ZWell-model
from zWell_model.resNet.resNetWork import ResNet


class ResNetV2(ResNet):

    def __init__(self, stride, input_shape, classes, k_size=3, chan_dim=-1, red=True, reg=0.0001, bn_eps=2e-5,
                 bn_mom=0.9,
                 model_layers_num=4, ckp=2, init_k_len=64):
        super().__init__([n for n in range(model_layers_num)], stride, input_shape, classes, chan_dim, red, reg, bn_eps,
                         bn_mom, model_layers_num, ckp,
                         init_k_len, dense_len=classes)
        self.k_size = k_size

    def to_keras_model(self, add_fully_connected=True, **args):
        super().to_keras_model(add_fully_connected, **args)
        import tensorflow as tf
        from zWell_model.resNet.tf.Block import BlockV2
        # 开始根据指定的残差块数量进行残差块的添加
        model = tf.keras.models.Sequential([
            # 预处理层 分别是 卷积 标准化 激活 池化
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=2, strides=1, padding='same', input_shape=self.input_shape
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D()
        ])
        for index in range(self.model_layers_num):
            # 将当前层的残差块添加到模型中
            model.add(BlockV2(
                filter_num=self.init_k_len,
                k_size=self.k_size,
                stride=self.stride[index],
                name=f'resBlock{index}'
            ))
            index += 1
            self.init_k_len *= self.ckp
        if add_fully_connected:
            # 首先使用 均值池化的方式进行扁平化
            model.add(tf.keras.layers.GlobalAveragePooling2D())
            model.add(tf.keras.layers.Dense(self.classes, activation='softmax'))
        # 返回最终的模型对象
        return model
