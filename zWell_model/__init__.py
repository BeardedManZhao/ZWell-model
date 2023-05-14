# -*- coding: utf-8 -*-
# @Time : 2023/5/13 20:17
# @Author : zhao
# @Email : liming7887@qq.com
# @File : __init__.py.py
# @Project : Keras-model
from zWell_model import resNetWork, allModel

resNet = "resNet"

dict_nn: dict[str: allModel] = {
    "resNet": resNetWork,
}

res_net = dict_nn[resNet]


def __getattr__(name):
    return dict_nn[name]
