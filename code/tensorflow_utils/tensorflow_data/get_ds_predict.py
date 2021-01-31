#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-01-31 15:39
    
Author:
    huayang
    
Subject:
    
"""
import os

import tensorflow as tf

try:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
except:
    import keras
    import keras.backend as K

from .utils import pre_process_image, get_file_list


def get_ds_predict_image(data_path,
                         is_sort=True,
                         image_size=(224, 224),
                         is_normalize=True,
                         batch_size=32,
                         n_limit=None):
    """生成 ds_predict

    目录结构
    data_path
        |
        |-- 4813.jpg
        |-- 54546.jpg
        |-- 375.jpg
        |-- ...
        \-- xxx.jpg
    """
    process_image = lambda file_path: pre_process_image(file_path, image_size, is_normalize)

    # 1. 获取文件名列表
    file_path_ls, file_name_ls = get_file_list(data_path, is_sort=is_sort, batch_size=batch_size, n_limit=n_limit)

    # 2. 构建 ds（此时的 ds 内部是字符串 file_path）
    ds_predict = tf.data.Dataset.from_tensor_slices(file_path_ls)

    # 3. 对 ds 进行 map 操作，将对应 file_path 下的图片转换为 tensor
    ds_predict = ds_predict.map(process_image)

    # 4. 其他设置
    if batch_size:
        ds_predict = ds_predict.batch(batch_size)

    return ds_predict, file_path_ls, file_name_ls
