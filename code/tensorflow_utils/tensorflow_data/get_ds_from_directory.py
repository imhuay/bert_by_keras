#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-01-26 11:51
    
Author:
    huayang
    
Subject:
    tensorflow==2.4 下的 keras 引入了 tf.keras.preprocessing.image_dataset_from_directory 方法很好用
    但是低版本不支持，这里写一个类似的

    TODO: 注意：不知道哪里写的有问题，生成的 ds_train 相比原版，训练时收敛速度慢很多，batch_size 也没有生效，原因待查
"""
import os

import numpy as np
import tensorflow as tf

ALLOW_FORMATS = ('.bmp', '.gif', '.jpeg', '.jpg', '.png')

from .utils import ds_image_show


def get_dataset_from_directory(dir_path,
                               label_mode='int',
                               val_percent=0.2,
                               image_size=(224, 224),
                               batch_size=32,
                               is_normalize=True,
                               random_seed=1):
    """

    Args:
        dir_path: 图片所在目录，其中每个类别的图片单独存放在一个文件夹下，文件夹名为类别名。
        label_mode: 'int' or 'categorical'(one_hot)
        val_percent: 验证集比例，如果为 0 表示不划分验证集
        image_size:
        batch_size:
        is_normalize: 是否归一化到 [0,1] 范围
        random_seed:

    Returns:
        (ds_train, ds_val) 或者 ds_train
    """

    def get_dataset(samples, labels, n_class):
        ds_sample = tf.data.Dataset.from_tensor_slices(samples)
        ds_sample = ds_sample.map(lambda image_path: pre_process_image(image_path, image_size, is_normalize))

        ds_label = tf.data.Dataset.from_tensor_slices(labels)
        if label_mode != 'int':
            ds_label = ds_label.map(lambda x: tf.one_hot(x, n_class))

        ds = tf.data.Dataset.zip((ds_sample, ds_label))
        ds = ds.shuffle(buffer_size=batch_size * 8, seed=random_seed)
        if batch_size is not None:
            ds = ds.batch(batch_size)
        return ds

    class_files_dict, class_label_dict = get_class_files_dict(dir_path, random_seed)
    n_class = len(class_label_dict.keys())

    val_samples = []
    val_labels = []
    train_samples = []
    train_labels = []
    for class_name, samples in class_files_dict.items():
        n_val_samples = int(val_percent * len(samples))
        val_samples += samples[:-n_val_samples]
        val_labels += [class_label_dict[class_name]] * len(samples[:-n_val_samples])
        train_samples += samples[-n_val_samples:]
        train_labels += [class_label_dict[class_name]] * len(samples[-n_val_samples:])

    # print(train_samples[:3])
    ds_train = get_dataset(train_samples, train_labels, n_class)

    if len(val_samples) <= 0:
        return ds_train

    ds_val = get_dataset(val_samples, val_labels, n_class)
    return ds_train, ds_val


def pre_process_image(image_path, image_size, is_normalize):
    """"""
    img_raw = tf.io.read_file(image_path)
    img_tensor = tf.image.decode_image(img_raw, channels=3, expand_animations=False)
    img_final = tf.image.resize(img_tensor, image_size)
    if is_normalize:
        img_final = img_final / 255.0
    return img_final


def get_class_files_dict(dir_path, random_seed):
    """"""
    from multiprocessing.pool import ThreadPool

    class_names, class_label_dict = get_class_index(dir_path)

    pool = ThreadPool()
    results = []
    for class_name in class_names:
        results.append(
            pool.apply_async(get_class_file_names,
                             (dir_path, class_name)))

    class_files_dict = dict()
    for res in results:
        class_file_names, class_name = res.get()

        if random_seed is not None:
            rng = np.random.RandomState(random_seed)
            rng.shuffle(class_file_names)

        class_files_dict[class_name] = class_file_names

    return class_files_dict, class_label_dict


def get_class_file_names(dir_path, class_name):
    """"""
    class_file_names = []
    for root, _, files in os.walk(os.path.join(dir_path, class_name)):
        for file_name in files:
            if file_name.lower().endswith(ALLOW_FORMATS):
                class_file_names.append(os.path.join(root, file_name))

    return class_file_names, class_name


def get_class_index(dir_path):
    """
    生成类目名到 id 的字典: {class_a: 0, class_b: 1, ...}
    """
    # 文件夹名为类名
    class_names = []
    for subdir in sorted(os.listdir(dir_path)):
        if os.path.isdir(os.path.join(dir_path, subdir)):
            class_names.append(subdir)

    # 类目名到 id 的字典: {class_a: 0, class_b: 1, ...}
    class_label_dict = dict((class_name, label) for label, class_name in enumerate(class_names))
    # print(class_label_dict)
    return class_names, class_label_dict


# def ds_image_show(ds):
#     """"""
#     import matplotlib.pyplot as plt
#
#     # 一个 3*3 的画布展示样例
#     plt.figure(figsize=(10, 10))
#     for images, labels in ds.take(1):
#         n = tf.shape(images)[0]
#         n = 9 if n > 9 else n
#         for i in range(n):
#             plt.subplot(3, 3, i + 1)
#             plt.imshow(images[i])
#             plt.title(str(labels[i]))
#             plt.axis("off")
#
#     plt.show()
