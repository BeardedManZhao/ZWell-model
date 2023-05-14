# -*- coding: utf-8 -*-
# @Time : 2023/5/13 20:17
# @Author : zhao
# @Email : liming7887@qq.com
# @File : __init__.py.py
# @Project : Keras-model
from zWell_model import allModel
from zWell_model.convNet import ConvNetV1, ConvNetV2
from zWell_model.convNet import convNetWork
from zWell_model.resNet import resNetWork

resNet = "resNet"
convNet1 = 'convNet1'
convNet2 = 'convNet2'

dict_nn: dict[str: allModel] = {
    "resNet": resNetWork,
    "convNet1": ConvNetV1,
    "convNet2": ConvNetV2
}

res_net = dict_nn[resNet]
conv_net1 = dict_nn[convNet1]
conv_net2 = dict_nn[convNet2]


def __getattr__(name):
    return dict_nn[name]
