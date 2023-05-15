# -*- coding: utf-8 -*-
# @Time : 2023/5/13 20:17
# @Author : zhao
# @Email : liming7887@qq.com
# @File : __init__.py.py
# @Project : Keras-model
from zWell_model import resNet, convNet, denseNet
from zWell_model.convNet.ConvNetV1 import ConvNetV1
from zWell_model.convNet.ConvNetV2 import ConvNetV2
from zWell_model.denseNet.DenseNetV1 import DenseNetV1
from zWell_model.resNet.ResNetV1 import ResNetV1

conv_net1 = convNet.ConvNetV1
conv_net2 = convNet.ConvNetV2

res_net1 = resNet.ResNetV1

dense_net1 = denseNet.DenseNetV1
