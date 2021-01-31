#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-01-31 15:53
    
Author:
    huayang
    
Subject:
    
"""
import os

import numpy as np
import tensorflow as tf

try:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
except:
    import keras
    import keras.backend as K


ALLOW_FORMATS = ('.bmp', '.gif', '.jpeg', '.jpg', '.png')


def pre_process_image(image_path, image_size, is_normalize):
    """从 image_path 读取图片并处理成 tensor

    测试版本：tensorflow==2.4，不保证其他版本
    """
    img_raw = tf.io.read_file(image_path)
    img_tensor = tf.image.decode_image(img_raw, channels=3, expand_animations=False)
    img_final = tf.image.resize(img_tensor, image_size)
    if is_normalize:
        img_final = img_final / 255.0
    return img_final


def get_file_list(data_path, with_prefix=True, is_sort=True, batch_size=None, n_limit=None):
    """获取 data_path 下的文件列表
    该方法会同时返回两个列表，分别为
        - 文件路径
        - 文件名_不含后缀

    Returns:
        file_ls, file_name_ls
    """
    file_ls = [file_name for file_name in os.listdir(data_path)
               if file_name.lower().endswith(ALLOW_FORMATS)]

    assert batch_size is None or batch_size <= len(file_ls), 'batch_size 要小于文件数'

    if is_sort:
        file_ls = sorted(file_ls)

    file_ls = file_ls[:n_limit]
    file_name_ls = [os.path.splitext(file_name)[0] for file_name in file_ls]

    if with_prefix:
        file_ls = [os.path.join(data_path, file_name) for file_name in file_ls]

    return file_ls, file_name_ls


def ds_image_show(ds: tf.data.Dataset, file_name_ls=None, with_label=True, label_mode='int', batch_size=None, n_show=9):
    """一个 3*3 的画布展示样例
    """
    import matplotlib.pyplot as plt

    def _loop(n_loop, func):
        tmp_image_ls = []
        tmp_label_ls = []
        ds_repeat = ds.repeat()
        for it in ds_repeat.take(n_loop):
            if with_label:
                tmp_image_ls.append(K.get_value(it[0]))
                tmp_label_ls.append(K.get_value(it[1]))
            else:
                tmp_image_ls.append(it)
        tmp_image_ls = func(tmp_image_ls, axis=0)
        if with_label:
            tmp_label_ls = func(tmp_label_ls, axis=0)
        if label_mode != 'int':
            tmp_label_ls = [np.argmax(tmp_label) for tmp_label in tmp_label_ls]

        return tmp_image_ls, tmp_label_ls

    if batch_size:
        image_ls, label_ls = _loop(n_show // batch_size + 1, K.concatenate)
    else:
        image_ls, label_ls = _loop(n_show, K.stack)

    image_ls = image_ls[:n_show]
    label_ls = label_ls[:n_show]

    tmp = int(n_show ** 0.5)
    if tmp**2 == n_show:
        n_row = n_col = tmp
    else:
        n_row = n_col = tmp + 1

    title_ls = [''] * n_show
    if file_name_ls:
        file_name_ls = file_name_ls[:n_show]
        if len(file_name_ls) < n_show:
            file_name_ls = file_name_ls * (n_show // len(file_name_ls) + 1)
            file_name_ls = file_name_ls[:n_show]
        for i, file_name in enumerate(file_name_ls):
            if i < n_show:
                title_ls[i] = title_ls[i] + file_name
            else:
                break

    if with_label:
        for i, label in enumerate(K.get_value(label_ls)[:n_show]):
            title_ls[i] = title_ls[i] + str(label)

    for i in range(n_show):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(image_ls[i])
        plt.title(title_ls[i])
        plt.axis("off")

    plt.show()
