#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-01-31 00:31
    
Author:
    huayang
    
Subject:
    自定义 Add 层，主要是重写 compute_mask
    说明：keras 自带的 Add 层也可以用，这里重写是为了加深理解 keras 中的 mask 的机制
    
"""
try:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
except:
    import keras
    import keras.backend as K


class CustomAdd(keras.layers.Add):
    """Embedding layer with weights returned."""

    def compute_mask(self, inputs, mask=None):
        return mask[0]
