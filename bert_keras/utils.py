#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-01-23 20:14
    
Author:
    huayang
    
Subject:
    
"""
import numpy as np
import tensorflow as tf

try:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
except:
    import keras
    import keras.backend as K


def gelu(x):
    """An approximation of gelu.

    See: https://arxiv.org/pdf/1606.08415.pdf
    """
    approximation = 0.7978845608028654  # math.sqrt(2.0 / math.pi) 的近似值
    return 0.5 * x * (1.0 + K.tanh(approximation * (x + 0.044715 * x ** 3)))


def to_array(*args):
    """批量转numpy的array
    """
    results = [np.array(a) for a in args]
    if len(args) == 1:
        return results[0]
    else:
        return results


def sequence_masking(x, mask, mode='add', axis=-1):
    """序列 mask

    Args:
        x: 2D 或 2D 以上张量，如 [batch_size, seq_len, input_size]
        mask: 形如  (batch_size, seq_len) 的 0/1矩阵
        mode: 有 'mul' 和 'add' 两种：
            mul 会将 pad 部分置零，一般用于全连接层之前；
            add 会把 pad 部分减去一个大的常数，一般用于 softmax 之前。
        axis: 需要 mask 的序列所在轴

    Returns:
        tensor with shape same as x
    """
    if mask is None:
        return x

    assert mode in ['add', 'mul'], mode

    if axis < 0:
        axis = K.ndim(x) + axis

    for _ in range(axis - 1):
        mask = K.expand_dims(mask, axis=1)

    for _ in range(K.ndim(x) - axis - 1):
        mask = K.expand_dims(mask, axis=-1)

    if mode == 'mul':
        return x * mask
    return x - (1 - mask) * 1e12
