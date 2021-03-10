#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-01-07 19:45
    
Author:
    huayang
    
Subject:
    图像模型 fine-tuning 一般流程 demo

References:
    Image classification via fine-tuning with EfficientNet | https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/

Requirements:
    tensorflow==2.4
"""
import os
import time
import requests

from multiprocessing.pool import ThreadPool
from tqdm import tqdm

import tensorflow as tf  # 2.4

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers.preprocessing import image_preprocessing

from tensorflow.keras.applications import EfficientNetB1, Xception


# 数据增强
image_augmentation = Sequential([
    image_preprocessing.RandomRotation(factor=0.1),  # 随机旋转
    image_preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),  # 随机平移
    image_preprocessing.RandomFlip(),  # 随机翻转
    image_preprocessing.RandomContrast(factor=0.1),  # 随机改变对比度
    image_preprocessing.RandomZoom(height_factor=0.1, width_factor=0.1),  # 随机缩放
    # image_preprocessing.RandomHeight(factor=0.1),  # 随机改变高度
    # image_preprocessing.RandomWidth(factor=0.1),  # 随机改变宽度
    # image_preprocessing.RandomCrop(height, width),  # 随机裁剪
    # image_preprocessing.CenterCrop(height, width),  # 中心裁剪
], name="img_augmentation", )


def get_time_suffix(suffix_len=15):
    """获取一个时间后缀"""
    import time
    time_suffix = str(time.perf_counter())
    time_suffix = time_suffix.replace('.', '')
    time_suffix.rjust(suffix_len, '0')

    return time_suffix


def file_download(args):
    """数据下载"""
    url, save_path = args
    response = requests.get(url)
    open(save_path, 'wb').write(response.content)

    return save_path


def data_download(main_path, url_ls, label_ls, name_ls=None, n_thread=5):
    """数据下载（多线程）

    Args:
        main_path:
        url_ls:
        label_ls:
        name_ls:
        n_thread:
    """
    assert len(url_ls) == len(label_ls)
    os.makedirs(main_path, exist_ok=True)

    label_st = set(label_ls)
    for label in label_st:
        os.makedirs(os.path.join(main_path, label), exist_ok=True)

    if name_ls is None:
        # 编号作为文件名
        name_ls = []
        label2cnt_dt = dict()
        for label in label_ls:
            if label not in label2cnt_dt:
                label2cnt_dt[label] = 1
            else:
                label2cnt_dt[label] += 1
            name_ls.append(label2cnt_dt[label])

        # 时间戳作为文件名
        # name_ls = []
        # for _ in label_ls:
        #     name_ls.append(get_time_suffix())

    args_ls = [(url, os.path.join(main_path, label, '%s.jpg' % id_))
               for id_, url, label in zip(name_ls, url_ls, label_ls)]
    ret = ThreadPool(n_thread).imap_unordered(file_download, args_ls)
    ret = list(ret)
    # print(ret)

    return ret


def create_data_set(data_path,
                    label_mode='categorical',
                    validation_split=0.2,
                    image_size=(224, 224),
                    batch_size=32,
                    random_seed=1):
    """构建训练集与验证集

    Args:
        data_path: 图片所在目录，其中每个类别的图片单独存放在一个文件夹下，文件夹名为类别名。
        label_mode:
            - 'int' for 'sparse_categorical_crossentropy';
            - 'categorical' for 'categorical_crossentropy';
            - 'binary' for 'binary_crossentropy'
        validation_split: 验证集比例，默认 0.2；如果小于等于 0 或为 None，表示不划分验证集
        image_size:
        batch_size:
        random_seed: 随机数种子

    Returns:
        (ds_train, ds_val) 或者 ds_train
    """
    from tensorflow.keras.preprocessing import image_dataset_from_directory

    if validation_split and validation_split <= 0:
        validation_split = None

    ds_train = image_dataset_from_directory(
        data_path,
        label_mode=label_mode,
        validation_split=validation_split,
        subset="training",
        seed=random_seed,
        image_size=image_size,
        batch_size=batch_size,
    )

    if validation_split is not None:
        ds_val = image_dataset_from_directory(
            data_path,
            label_mode=label_mode,
            validation_split=validation_split,
            subset="validation",
            seed=random_seed,
            image_size=image_size,
            batch_size=batch_size,
        )

        return ds_train, ds_val

    return ds_train


def ds_image_show(ds):
    """"""
    import matplotlib.pyplot as plt

    # 一个 3*3 的画布展示样例
    plt.figure(figsize=(10, 10))
    for images, labels in ds.take(1):
        n = tf.shape(images)[0]
        n = 9 if n > 9 else n
        for i in range(n):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(int(labels[i]))
            plt.axis("off")

    plt.show()


def get_base_model(inputs, is_fine_tune=False, fine_tune_layers=None):
    """"""
    from tensorflow.keras.applications import EfficientNetB1

    model = EfficientNetB1(include_top=False, input_tensor=inputs, weights="imagenet")

    # base model 是否 fine tune
    model.trainable = is_fine_tune if fine_tune_layers is None else False

    # 如果仅需要部分层参与 fine tune（BatchNormalization 层不需要）
    # if fine_tune_layers is not None:
    #     model_layers = [model.layers[i] for i in fine_tune_layers]
    #     for layer in model_layers:  # 前 20 层中的非 BatchNormalization 层参与训练
    #         if not isinstance(layer, layers.BatchNormalization):
    #             layer.trainable = True

    return model


def build_model(n_class, IMG_SIZE, is_augment=True, is_scale=False):
    """"""
    x = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    if is_augment:
        x = image_augmentation(x)

    if is_scale:
        x = image_preprocessing.Rescaling(1./255)(x)

    # 一个基础模型
    model = get_base_model(x)

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    # x = layers.Dense(768, activation='relu', name='dense')(x)
    # x = layers.Dropout(0.5)(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(n_class, activation="softmax", name="pred")(x)

    model = tf.keras.Model(x, outputs, name="EfficientNet")

    return model


def test_data_download():
    """"""
    url_ls = ['https://qcloud.dpfile.com/pc/cUFh4ZqMzqRrdoekDGFd1hxMZBptbSsldhPAvT1-OfdNZxyXTLdXGdUngAKY3L7n.jpg'] * 9
    label_ls = ['a', 'b', 'c'] * 3
    main_path = '../-test/download'
    data_download(main_path, url_ls, label_ls, n_thread=3)
    # ds_train, ds_val = create_data_set(main_path)


def test_data_augment():
    main_path = '../-test/download'
    ds_train, ds_val = create_data_set(main_path)
    # rt = image_preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1)
    rt = image_preprocessing.RandomFlip()
    # rt = image_preprocessing.RandomRotation(factor=0.1)

    # ds_train.map(rt)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    for images, labels in ds_train.take(1):
        images = rt(images)
        n = tf.shape(images)[0]
        n = 9 if n > 9 else n
        for i in range(n):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            # plt.title(int(labels[i]))
            plt.axis("off")

    plt.show()


if __name__ == '__main__':
    """"""
    # test_data_download()
    test_data_augment()
