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
