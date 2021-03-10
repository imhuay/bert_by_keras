#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-03-09 10:42
    
Author:
    huayang
    
Subject:
    图片的序列化与反序列化

Note:
    基于 base64 的方法详见：python_utils/basic_utils/serialize.py

    - 库版本
        - tensorflow >= 2.0

References:
    keras.preprocessing.image
"""
import io

import numpy as np
from PIL import Image


def load_image(image, color_mode='RGB'):
    """
    加载原始图像，返回 PIL.Image 对象

    Args:
        image: 图像路径，或二进制数据
        color_mode: 颜色模式，支持 {"L","RGB","RGBA"} 种类型，对应的 shape 分别为 (w, h)、(w, h, 3)、(w, h, 4)

    Returns:
        Image.Image
    """
    if color_mode not in {"L", "RGB", "RGBA"}:
        raise ValueError('Unsupported color_mode: %s, it must be one of {"L", "RGB", "RGBA"}' % color_mode)

    if isinstance(image, bytes):
        img = Image.open(io.BytesIO(image))
    else:
        with open(image, 'rb') as f:
            img = Image.open(io.BytesIO(f.read()))

    if img.mode != color_mode:
        img = img.convert(color_mode)

    return img


def image_to_tensor(image, dtype='float32'):
    """
    将 PIL.Image 对象转为 numpy 数组

    Args:
        image:
        dtype:

    Returns:

    """
    x = np.asarray(image, dtype=dtype)

    if len(x.shape) == 2:
        x = x.reshape((x.shape[0], x.shape[1], 1))

    return x


def tensor_to_image(x, scale=True, dtype='float32'):
    """
    将 numpy 数组转为 PIL.Image 对象

    Args:
        x:
        scale:
        dtype:

    Returns:

    """
    x = np.asarray(x, dtype=dtype)
    if len(x.shape) != 3:
        raise ValueError('Unsupported tensor dim: %s, it must be 3' % len(x.shape))

    n_channel = x.shape[2]
    if n_channel not in {1, 3, 4}:
        raise ValueError('Unsupported channel number: %s, it must be one of {1, 3, 4}' % n_channel)

    if scale:
        x = x - np.min(x)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255

    if n_channel == 3:  # RGB
        return Image.fromarray(x.astype('uint8'), 'RGB')
    elif n_channel == 4:  # RGBA
        return Image.fromarray(x.astype('uint8'), 'RGBA')
    else:  # L
        if np.max(x) > 255:
            return Image.fromarray(x[:, :, 0].astype('int32'), 'I')
        return Image.fromarray(x[:, :, 0].astype('uint8'), 'L')


if __name__ == '__main__':
    """"""
    obj = open('../_test_data/pok.jpg', 'rb').read()
    img = load_image(obj, color_mode='RGB')
    print(isinstance(img, Image.Image))

    x = image_to_tensor(img)
    print(x.shape)

    img = tensor_to_image(x)
    print(isinstance(img, Image.Image))
